# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in vLLM 0.19 rejection sampler for unconstrained strict greedy requests.

Preserves the original target logits and FP32 argmax. Only integer candidate
selection is fused; target/head GEMMs and attention are untouched. Requests with
logprobs, active processors, penalties, or masks retain the upstream sampler.
Importing this module installs no hooks.
"""

import functools
import importlib.metadata

import torch
import triton
import triton.language as tl

COUNTS = {"fast_calls": 0, "fallback_calls": 0}


@triton.jit
def _select_greedy_tokens(
    argmax_ptr,
    draft_ptr,
    cu_drafts_ptr,
    target_indices_ptr,
    bonus_indices_ptr,
    output_ptr,
    OUTPUT_COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = 0 if req == 0 else tl.load(cu_drafts_ptr + req - 1)
    end = tl.load(cu_drafts_ptr + req)
    num_drafts = end - start
    positions = tl.arange(0, BLOCK)
    has_draft = positions < num_drafts
    target_rows = tl.load(target_indices_ptr + start + positions, has_draft, other=0)
    target_ids = tl.load(argmax_ptr + target_rows, has_draft, other=0)
    draft_ids = tl.load(draft_ptr + start + positions, has_draft, other=0)
    first_rejection = tl.min(
        tl.where(has_draft & (target_ids != draft_ids), positions, OUTPUT_COLUMNS),
        axis=0,
    )
    accepted_or_recovered = has_draft & (positions <= first_rejection)
    bonus_row = tl.load(bonus_indices_ptr + req)
    bonus_id = tl.load(argmax_ptr + bonus_row)
    append_bonus = (first_rejection >= num_drafts) & (positions == num_drafts)
    values = tl.where(accepted_or_recovered, target_ids, -1)
    values = tl.where(append_bonus, bonus_id, values)
    tl.store(
        output_ptr + req * OUTPUT_COLUMNS + positions,
        values,
        positions < OUTPUT_COLUMNS,
    )


def eligible(metadata, logits, sampling_metadata):
    from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

    sm = sampling_metadata
    if (
        not sm.all_greedy
        or sm.all_random
        or sm.max_num_logprobs is not None
        or not sm.no_penalties
        or sm.allowed_token_ids_mask is not None
        or bool(sm.bad_words_token_ids)
        or logits is None
        or not logits.is_cuda
        or logits.ndim != 2
        or logits.dtype not in (torch.bfloat16, torch.float32)
        or not 0 <= metadata.max_spec_len <= 4
        or not metadata.num_draft_tokens
    ):
        return False
    # vLLM always creates an instance of this processor for spec decode,
    # even when min_tokens=0. Checking for an empty processor list is too strict.
    return all(
        type(processor) is MinTokensLogitsProcessor and not processor.min_toks
        for processor in sm.logitsprocs.non_argmax_invariant
    )


def try_greedy_rejection_sample(metadata, logits, sampling_metadata):
    """Return SamplerOutput, or None when the original sampler must be used."""
    from vllm.v1.outputs import SamplerOutput

    if not eligible(metadata, logits, sampling_metadata):
        return None
    # Match upstream FP32 argmax semantics, including NaNs and first-index ties.
    # Casting BF16 is exact. Combining all rows eliminates vocabulary-sized
    # bonus/target index_select copies and preservation clones.
    argmax = logits.to(torch.float32).argmax(dim=-1)
    output_columns = metadata.max_spec_len + 1
    output = torch.empty(
        (len(metadata.num_draft_tokens), output_columns),
        device=logits.device,
        dtype=torch.int32,
    )
    _select_greedy_tokens[(len(metadata.num_draft_tokens),)](
        argmax,
        metadata.draft_token_ids,
        metadata.cu_num_draft_tokens,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        output,
        OUTPUT_COLUMNS=output_columns,
        BLOCK=triton.next_power_of_2(output_columns),
        num_warps=4,
    )
    return SamplerOutput(sampled_token_ids=output, logprobs_tensors=None)


def _uses_native_sampler(instance, rejection_type, sampler_type):
    """Do not bypass customized nested sampler calls or their module hooks."""
    from torch.nn.modules import module as nn_module

    if (
        type(instance) is not rejection_type
        or type(instance.sampler) is not sampler_type
    ):
        return False
    if nn_module._global_forward_pre_hooks or nn_module._global_forward_hooks:
        return False
    for module in (instance, instance.sampler):
        if module._forward_pre_hooks or module._forward_hooks:
            return False
        if getattr(module, "_compiled_call_impl", None) is not None:
            return False
    # Exact class identity alone does not exclude instance-level overrides.
    methods = (
        "forward",
        "sample",
        "greedy_sample",
        "apply_logits_processors",
        "apply_penalties",
    )
    return not any(
        name in vars(module)
        for module in (instance, instance.sampler)
        for name in methods
    )


def _make_wrapper(original, rejection_type, sampler_type):

    @functools.wraps(original)
    def forward(self, metadata, draft_probs, logits, sampling_metadata):
        if _uses_native_sampler(self, rejection_type, sampler_type):
            result = try_greedy_rejection_sample(metadata, logits, sampling_metadata)
            if result is not None:
                COUNTS["fast_calls"] += 1
                return result
        COUNTS["fallback_calls"] += 1
        return original(self, metadata, draft_probs, logits, sampling_metadata)

    forward._paddleocr_vllm019_greedy = True
    return forward


def install():
    """Explicitly install the guarded fast path; never called at import time."""
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("Greedy sampler adapter requires vLLM 0.19.0")
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    from vllm.v1.sample.sampler import Sampler

    original = RejectionSampler.forward
    if getattr(original, "_paddleocr_vllm019_greedy", False):
        return
    RejectionSampler.forward = _make_wrapper(original, RejectionSampler, Sampler)
