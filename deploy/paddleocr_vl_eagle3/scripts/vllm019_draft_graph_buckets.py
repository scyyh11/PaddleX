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

"""Bounded-cache draft CUDA graphs with CPU-selected FA2 context bounds.

The vLLM 0.19 runner's *runtime* CommonAttentionMetadata.max_seq_len comes from
optimistic_seq_lens_cpu; only its startup target graph capture uses the model
limit. No GPU scalar is inspected here. Each bucket covers that upper bound
plus all three additional draft positions. Actual causal lengths/KV slots and
the context-overflow clamp remain unchanged. Target attention is unaffected.

At most eight graphs are retained per proposer, captured lazily once each.
First-use capture costs and extra CUDA graph memory must be measured alongside
steady-state speed; this module makes no claim of a free cold-start improvement.
"""
from collections import Counter
from functools import wraps

import vllm019_draft_full_graph as full
import vllm019_draft_loop as tail

BOUNDS = (128, 256, 512, 1024, 2048, 4096, 8192, 12288)
COUNTS = Counter()
_INSTALLED = False


def context_bound(max_seq_len, max_model_len):
    """Select a safe length bound using only CPU integer metadata."""
    if type(max_seq_len) is not int or type(max_model_len) is not int:
        raise TypeError(
            "CPU integer context bounds required; GPU scalar sync forbidden"
        )
    if not 0 < max_seq_len <= max_model_len <= BOUNDS[-1]:
        raise ValueError("Unsupported context bound")
    required = min(max_seq_len + 3, max_model_len)
    return min(next(bound for bound in BOUNDS if bound >= required), max_model_len)


class _BoundedFullGraph(full._FullDraftGraph):
    def __init__(self, proposer, cad, ids, positions, hidden, next_ids, indices, bound):
        super().__init__(proposer, cad, ids, positions, hidden, next_ids, indices)
        self.bound = bound
        # All values are changed before the first warmup/capture. Metadata's
        # dynamic GPU tensor references are unchanged. FA2 has no AOT scheduler.
        self.cad.max_seq_len = bound
        self.first_cad.max_seq_len = bound
        for metadata in (*self.attention.values(), *self.first_attention.values()):
            metadata.max_seq_len = bound


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
        reason = tail._fallback_reason(
            self, common_attn_metadata, num_rejected_tokens_gpu, mm_embed_inputs
        )
        if reason is None:
            reason = full._input_contract(
                self,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
                common_attn_metadata,
            )
        if reason is None and not (
            type(common_attn_metadata.max_seq_len) is int
            and 0 < common_attn_metadata.max_seq_len <= self.max_model_len <= BOUNDS[-1]
        ):
            reason = "unsupported_bound"
        if reason is None:
            _, padded, _ = self._determine_batch_execution_and_padding(5)
            if padded != 5:
                reason = "padding"
        if reason is None:
            bound = context_bound(common_attn_metadata.max_seq_len, self.max_model_len)
            cache = getattr(self, "_vllm019_draft_graph_buckets", None)
            if cache is None:
                cache = self._vllm019_draft_graph_buckets = {}
            plan = cache.get(bound)
            if plan is None:
                plan = cache[bound] = _BoundedFullGraph(
                    self,
                    common_attn_metadata,
                    target_token_ids,
                    target_positions,
                    target_hidden_states,
                    next_token_ids,
                    token_indices_to_sample,
                    bound,
                )
            # The full adapter still validates input shapes/dtypes/addresses
            # and owns graph replay. Switching buckets never changes target KV.
            self._vllm019_draft_full_graph = plan
            COUNTS[str(bound)] += 1
        else:
            COUNTS["fallback_" + reason] += 1
            if reason == "unsupported_bound":
                # A previous bucket is not a valid bound for an unclassified
                # context. Let the full adapter create its model-limit plan.
                self._vllm019_draft_full_graph = None
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

    return propose


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    full.install()
    from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

    SpecDecodeBaseProposer.propose = _make_wrapper(SpecDecodeBaseProposer.propose)
    _INSTALLED = True


def metadata(proposer=None):
    cache = getattr(proposer, "_vllm019_draft_graph_buckets", {})
    return {
        "installed": _INSTALLED,
        "max_graphs": len(BOUNDS),
        "bounds": list(BOUNDS),
        "selection": "CPU optimistic max_seq_len + 3; no GPU scalar read",
        "counts": dict(COUNTS),
        "graphs": {
            str(bound): {"captures": plan.captures, "replays": plan.replays}
            for bound, plan in sorted(cache.items())
        },
        "full": full.metadata(proposer),
    }
