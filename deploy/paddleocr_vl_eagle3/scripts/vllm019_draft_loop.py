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

"""Experimental B1/K4 EAGLE3 tail graph for the pinned vLLM 0.19 runtime.

Call ``install()`` from an experimental worker, after installing the complete
private INT8 top-1 head. This does not modify the installed vLLM package.
Prefill, batching and unsupported configurations retain the upstream proposer.

Only the final three autoregressive draft steps are grouped. The first step
retains upstream input preparation and its existing compiled forward. The tail
uses the same padded matrix shape, model weights, complete vocabulary and greedy
head. A fixed attention length *upper bound* allows one graph to serve changing
contexts; actual causal lengths and KV slots remain dynamic GPU tensors. This
may change draft rounding/acceptance and must pass end-to-end target token/stop
regressions before deployment. Target verification and sampling are untouched.
"""
from collections import Counter
from functools import wraps
import importlib.metadata
from types import SimpleNamespace

import torch

COUNTS = Counter()
_INSTALLED = False


def _runtime():
    # Keep module import and the CPU contract tests independent of CUDA/vLLM.
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.v1.spec_decode.utils import (
        eagle_step_update_slot_mapping_and_metadata,
    )

    return SimpleNamespace(
        no_graph=CUDAGraphMode.NONE,
        context=set_forward_context,
        update=eagle_step_update_slot_mapping_and_metadata,
    )


def _fallback_reason(proposer, cad, rejected, mm_inputs):
    """All admission checks are CPU metadata; never synchronize a GPU scalar."""
    if (
        proposer.method != "eagle3"
        or proposer.num_speculative_tokens != 4
        or proposer.parallel_drafting
        or not proposer.pass_hidden_states_to_model
        or proposer.needs_extra_input_slots
    ):
        return "nonstandard_drafter"
    if (
        cad.batch_size() != 1
        or not 1 <= cad.num_actual_tokens <= 5
        or not 1 <= cad.max_query_len <= 5
        or rejected is None
        or rejected.numel() != 1
        # Multimodal targets also pass ([], false_mask) during text decode.
        # Checking the Python embedding list avoids synchronizing the mask.
        or (mm_inputs is not None and len(mm_inputs[0]) != 0)
    ):
        return "prefill_or_non_b1"
    if (
        proposer.uses_mrope
        or (proposer.uses_xdrope_dim and proposer.draft_uses_xdrope_dim)
        or not cad.causal
        or proposer.block_size <= 0
    ):
        return "positions_or_attention"
    cfg = proposer.vllm_config
    if (
        (
            cfg.scheduler_config.async_scheduling
            and cfg.scheduler_config.max_num_seqs != 1
        )
        or cfg.parallel_config.tensor_parallel_size != 1
        or cfg.parallel_config.pipeline_parallel_size != 1
        or cfg.parallel_config.data_parallel_size != 1
        or cfg.parallel_config.decode_context_parallel_size != 1
        or proposer.speculative_config.enforce_eager
    ):
        return "scheduler_or_parallelism"
    if (
        not proposer.use_local_argmax_reduction
        or type(proposer.model).__name__ != "Eagle3LlamaForCausalLM"
        or not getattr(type(proposer.model), "_round3_full_private_int8_top1", False)
        or proposer.model.config.num_hidden_layers != 1
        or proposer.hidden_size != 1024
        or proposer.dtype != torch.bfloat16
    ):
        return "draft_model_or_head"
    if (
        len(proposer.draft_attn_groups) != 1
        or type(proposer.draft_attn_groups[0].get_metadata_builder()).__name__
        != "FlashAttentionMetadataBuilder"
        or proposer.draft_attn_groups[0].get_metadata_builder().aot_schedule
    ):
        return "non_fa2_attention"
    if not proposer.input_ids.is_cuda:
        return "non_cuda"
    return None


def _first_step(
    proposer,
    target_token_ids,
    target_positions,
    target_hidden_states,
    next_token_ids,
    token_indices_to_sample,
    common_attn_metadata,
    num_rejected_tokens_gpu,
):
    """The upstream sequential EAGLE3 first pass, without its Python tail."""
    rt = _runtime()
    target_hidden_states = proposer.model.combine_hidden_states(target_hidden_states)
    num_tokens, indices, cad = proposer.set_inputs_first_pass(
        target_token_ids=target_token_ids,
        next_token_ids=next_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=token_indices_to_sample,
        cad=common_attn_metadata,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    )
    attention = {}
    for group in proposer.draft_attn_groups:
        metadata = group.get_metadata_builder().build_for_drafting(
            common_attn_metadata=cad, draft_index=0
        )
        attention.update({name: metadata for name in group.layer_names})
    mode, padded, across_dp = proposer._determine_batch_execution_and_padding(
        num_tokens
    )
    if proposer.supports_mm_inputs:
        proposer.inputs_embeds[:num_tokens] = proposer.model.embed_input_ids(
            proposer.input_ids[:num_tokens],
            multimodal_embeddings=None,
            is_multimodal=None,
        )
        input_ids = None
        embeds = proposer.inputs_embeds[:padded]
    else:
        input_ids = proposer.input_ids[:padded]
        embeds = None
    with rt.context(
        attention,
        proposer.vllm_config,
        num_tokens=padded,
        num_tokens_across_dp=across_dp,
        cudagraph_runtime_mode=mode,
        slot_mapping=proposer._get_slot_mapping(padded, cad.slot_mapping),
    ):
        last_hidden, hidden = proposer.model(
            input_ids=input_ids,
            positions=proposer._get_positions(padded),
            inputs_embeds=embeds,
            hidden_states=proposer.hidden_states[:padded],
        )
    # Advanced indexing owns these inputs: replay/capture may overwrite the
    # proposer's static input buffers without corrupting the next reset.
    return (
        proposer._greedy_sample(last_hidden[indices]),
        proposer.positions[indices],
        hidden[indices],
        cad,
    )


class _DraftLoop:
    """One immutable-shape graph and persistent metadata per proposer."""

    def __init__(self, proposer, cad, padded):
        self.proposer = proposer
        self.padded = padded
        self.rt = _runtime()
        self.graph = None
        self.captures = 0
        self.replays = 0
        self.input_ids = proposer.input_ids[:padded]
        self.positions = proposer.positions[:padded]
        self.hidden = proposer.hidden_states[:padded]
        self.slots = proposer._slot_mapping_buffer[:padded]
        self.seq_lens = torch.empty_like(cad.seq_lens)
        self.block_table = torch.empty_like(cad.block_table_tensor)
        self.output = torch.empty(
            (1, 4), dtype=torch.int64, device=self.input_ids.device
        )
        self.cad = cad.replace(
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=proposer.max_model_len,
            query_start_loc=proposer.arange[:2],
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=self.seq_lens,
            block_table_tensor=self.block_table,
            slot_mapping=self.slots[:1],
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
        )
        self.attention = {}
        for group in proposer.draft_attn_groups:
            metadata = group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=self.cad, draft_index=1
            )
            self.attention.update({name: metadata for name in group.layer_names})
        self.slot_mapping = {
            name: self.slots for name in proposer._draft_attn_layer_names
        }

    def matches(self, cad, padded):
        return (
            padded == self.padded
            and cad.seq_lens.shape == self.seq_lens.shape
            and cad.seq_lens.dtype == self.seq_lens.dtype
            and cad.block_table_tensor.shape == self.block_table.shape
            and cad.block_table_tensor.dtype == self.block_table.dtype
            and cad.block_table_tensor.device == self.block_table.device
        )

    def _reset(self, first, positions, hidden, cad, rejected):
        self.output[:, 0].copy_(first)
        self.input_ids[:1].copy_(first)
        self.positions[:1].copy_(positions)
        self.hidden[:1].copy_(hidden)
        self.seq_lens.copy_(cad.seq_lens)
        self.seq_lens.sub_(rejected)
        self.block_table.copy_(cad.block_table_tensor)

    def _run_steps(self):
        proposer = self.proposer
        for index in range(3):
            self.rt.update(
                positions_1d=self.positions[:1],
                block_table_tensor=self.block_table,
                seq_lens=self.seq_lens,
                block_size=proposer.block_size,
                max_model_len=proposer.max_model_len,
                out_clamped_positions=self.positions[:1],
                out_slot_mapping=self.slots,
                input_batch_size=self.padded,
            )
            # NONE bypasses the inner piecewise graph only. The model retains
            # its compiled kernels, captured together in this outer graph.
            with self.rt.context(
                self.attention,
                proposer.vllm_config,
                num_tokens=self.padded,
                cudagraph_runtime_mode=self.rt.no_graph,
                slot_mapping=self.slot_mapping,
            ):
                if proposer.supports_mm_inputs:
                    proposer.inputs_embeds[:1] = proposer.model.embed_input_ids(
                        self.input_ids[:1]
                    )
                    input_ids = None
                    embeds = proposer.inputs_embeds[: self.padded]
                else:
                    input_ids = self.input_ids
                    embeds = None
                last_hidden, hidden = proposer.model(
                    input_ids=input_ids,
                    positions=self.positions,
                    hidden_states=self.hidden,
                    inputs_embeds=embeds,
                )
            token = proposer._greedy_sample(last_hidden[:1])
            self.output[:, index + 1].copy_(token)
            if index != 2:
                self.input_ids[:1].copy_(token)
                self.hidden[:1].copy_(hidden[:1])

    def run(self, first, positions, hidden, cad, rejected):
        args = first, positions, hidden, cad, rejected
        if self.graph is None:
            # Warm exactly these graph inputs/kernels on a side stream. Each
            # reset restores the starting causal state. Repeated KV writes are
            # to the same slots, and future draft slots are masked by seq_lens.
            current = torch.cuda.current_stream(self.input_ids.device)
            capture_stream = torch.cuda.Stream(device=self.input_ids.device)
            capture_stream.wait_stream(current)
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self._reset(*args)
                    self._run_steps()
            current.wait_stream(capture_stream)
            current.synchronize()
            self._reset(*args)
            capture_stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                self._run_steps()
            current.wait_stream(capture_stream)
            self.graph = graph
            self.captures += 1
            COUNTS["captures"] += 1
        self._reset(*args)
        self.graph.replay()
        self.replays += 1
        COUNTS["replays"] += 1
        # Preserve the upstream common-metadata mutations. Input/position/slot
        # buffers already alias the proposer, exactly as in its ordinary tail.
        cad.seq_lens.copy_(self.seq_lens)
        cad.num_actual_tokens = 1
        cad.max_query_len = 1
        cad.query_start_loc = self.cad.query_start_loc
        cad.query_start_loc_cpu = self.cad.query_start_loc_cpu
        cad.slot_mapping = self.slots[:1]
        cad.max_seq_len = min(cad.max_seq_len + 3, self.proposer.max_model_len)
        cad._seq_lens_cpu = None
        cad._num_computed_tokens_cpu = None
        # A caller may keep a proposal while the next replay reuses its graph
        # storage. Match upstream stack() ownership rather than leaking a view.
        return self.output.clone()


def _make_wrapper(original):
    @wraps(original)
    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        slot_mappings=None,
    ):
        reason = _fallback_reason(
            self, common_attn_metadata, num_rejected_tokens_gpu, mm_embed_inputs
        )
        plan = getattr(self, "_vllm019_draft_loop", None)
        if reason is None:
            _, padded, _ = self._determine_batch_execution_and_padding(1)
            if padded != 5 or (plan and not plan.matches(common_attn_metadata, padded)):
                reason = "graph_shape_changed"
        if reason is not None:
            COUNTS["fallback_" + reason] += 1
            return original(
                self,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
                common_attn_metadata,
                sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )
        first, positions, hidden, cad = _first_step(
            self,
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            num_rejected_tokens_gpu,
        )
        if plan is None:
            plan = self._vllm019_draft_loop = _DraftLoop(self, cad, padded)
        return plan.run(first, positions, hidden, cad, num_rejected_tokens_gpu)

    return propose


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("The draft tail graph requires exactly vLLM 0.19.0")
    from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

    SpecDecodeBaseProposer.propose = _make_wrapper(SpecDecodeBaseProposer.propose)
    _INSTALLED = True


def metadata(proposer=None):
    plan = getattr(proposer, "_vllm019_draft_loop", None)
    return {
        "installed": _INSTALLED,
        "scope": "B1/K4 greedy EAGLE3 decode; last three steps; FA2; padded size 5",
        "draft_vocab": "unchanged complete private INT8 vocabulary",
        "target_verification": "strict greedy acceptance with the guarded sampler adapter",
        "draft_context_bound": None if plan is None else plan.cad.max_seq_len,
        "counts": dict(COUNTS),
        "captures": 0 if plan is None else plan.captures,
        "replays": 0 if plan is None else plan.replays,
    }
