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

"""Experimental complete Q5/B1 EAGLE3 proposal graph for vLLM 0.19.0.

Install after ``vllm019_draft_loop.install()``. Only a stable five-token target
verification block is eligible. Every other input retains that tail adapter's
fallback, including the native prefill path. The graph includes auxiliary FC,
native first-input preparation, all four draft forwards and full-vocabulary
INT8 greedy heads. Target verification/rejection sampling remains external.

As with the tail adapter, a conservative fixed draft attention length bound can
affect draft floating-point rounding/acceptance; end-to-end token and stop-reason
regressions are required before this experimental candidate is deployed.
"""
from collections import Counter
from functools import wraps
import importlib.metadata

import torch

import vllm019_draft_loop as tail

COUNTS = Counter()
_INSTALLED = False


def _position_row(proposer, positions):
    # Mirror both native conversions: set_inputs_first_pass handles target
    # xD-RoPE, and _set_positions handles target M-RoPE for a text-only draft.
    # The PaddleOCR-VL-1.6 checkpoint uses M-RoPE (three target axes), not xD.
    if proposer.uses_xdrope_dim and not proposer.draft_uses_xdrope_dim:
        positions = positions[0]
    if proposer.vllm_config.model_config.uses_mrope:
        positions = positions[0]
    return positions


def _input_contract(proposer, ids, positions, hidden, next_ids, indices, cad):
    if cad.num_actual_tokens != 5 or cad.max_query_len != 5:
        return "non_q5"
    # Native padded drafting supplies an explicit GPU sample index. The None
    # case belongs to native first-pass preparation and keeps that fallback.
    if indices is None:
        return "missing_sample_indices"
    target_mrope = proposer.vllm_config.model_config.uses_mrope
    if target_mrope and proposer.uses_xdrope_dim:
        return "unsupported_position_config"
    expected_positions = (
        (3, 5)
        if target_mrope
        else ((proposer.uses_xdrope_dim, 5) if proposer.uses_xdrope_dim else (5,))
    )
    if positions.shape != expected_positions:
        return "position_shape"
    if (
        ids.shape != (5,)
        or hidden.shape != (5, 3 * proposer.hidden_size)
        or next_ids.shape != (1,)
        or indices.shape != (1,)
        or cad.slot_mapping.shape != (5,)
        or hidden.dtype != torch.bfloat16
    ):
        return "input_shape_or_dtype"
    if any(
        tensor.device != ids.device
        for tensor in (positions, hidden, next_ids, indices, cad.slot_mapping)
    ):
        return "input_device"
    return None


class _FullDraftGraph(tail._DraftLoop):
    def __init__(self, proposer, cad, ids, positions, hidden, next_ids, indices):
        super().__init__(proposer, cad, 5)
        self.target_ids = torch.empty_like(ids)
        self.target_positions = torch.empty_like(_position_row(proposer, positions))
        self.target_hidden = torch.empty_like(hidden)
        self.next_ids = torch.empty_like(next_ids)
        self.indices = torch.empty_like(indices)
        self.rejected = torch.empty_like(cad.seq_lens)
        self.target_lengths = torch.empty_like(cad.seq_lens)
        self.target_blocks = torch.empty_like(cad.block_table_tensor)
        self.target_slots = torch.empty_like(cad.slot_mapping)
        self.first_cad = cad.replace(
            max_seq_len=proposer.max_model_len,
            seq_lens=self.target_lengths,
            block_table_tensor=self.target_blocks,
            slot_mapping=self.slots,
            query_start_loc=torch.tensor(
                [0, 5], device=self.input_ids.device, dtype=torch.int32
            ),
            query_start_loc_cpu=torch.tensor([0, 5], dtype=torch.int32),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
        )
        self.first_attention = {}
        for group in proposer.draft_attn_groups:
            metadata = group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=self.first_cad, draft_index=0
            )
            self.first_attention.update({name: metadata for name in group.layer_names})

    def matches_inputs(self, ids, positions, hidden, next_ids, indices, rejected, cad):
        tensors = (
            (ids, self.target_ids),
            (_position_row(self.proposer, positions), self.target_positions),
            (hidden, self.target_hidden),
            (next_ids, self.next_ids),
            (indices, self.indices),
            (rejected, self.rejected),
            (cad.slot_mapping, self.target_slots),
        )
        return self.matches(cad, 5) and all(
            source.shape == dest.shape
            and source.dtype == dest.dtype
            and source.device == dest.device
            for source, dest in tensors
        )

    def _stage(self, ids, positions, hidden, next_ids, indices, cad, rejected):
        self.target_ids.copy_(ids)
        self.target_positions.copy_(_position_row(self.proposer, positions))
        self.target_hidden.copy_(hidden)
        self.next_ids.copy_(next_ids)
        self.indices.copy_(indices)
        self.target_lengths.copy_(cad.seq_lens)
        self.target_blocks.copy_(cad.block_table_tensor)
        self.target_slots.copy_(cad.slot_mapping)
        self.rejected.copy_(rejected)

    def _run_full(self):
        proposer = self.proposer
        combined = proposer.model.combine_hidden_states(self.target_hidden)
        self.input_ids[:4].copy_(self.target_ids[1:])
        self.input_ids[self.indices] = self.next_ids
        self.positions.copy_(self.target_positions)
        self.hidden.copy_(combined)
        self.slots.copy_(self.target_slots)
        if proposer.supports_mm_inputs:
            proposer.inputs_embeds[:5] = proposer.model.embed_input_ids(self.input_ids)
            input_ids = None
            embeds = proposer.inputs_embeds[:5]
        else:
            input_ids = self.input_ids
            embeds = None
        with self.rt.context(
            self.first_attention,
            proposer.vllm_config,
            num_tokens=5,
            cudagraph_runtime_mode=self.rt.no_graph,
            slot_mapping=self.slot_mapping,
        ):
            last_hidden, hidden = proposer.model(
                input_ids=input_ids,
                positions=self.positions,
                hidden_states=self.hidden,
                inputs_embeds=embeds,
            )
        first = proposer._greedy_sample(last_hidden[self.indices])
        positions = self.positions[self.indices]
        hidden = hidden[self.indices]
        self._reset(first, positions, hidden, self.first_cad, self.rejected)
        self._run_steps()

    def run_full(self, ids, positions, hidden, next_ids, indices, cad, rejected):
        self._stage(ids, positions, hidden, next_ids, indices, cad, rejected)
        if self.graph is None:
            current = torch.cuda.current_stream(self.input_ids.device)
            capture_stream = torch.cuda.Stream(device=self.input_ids.device)
            capture_stream.wait_stream(current)
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self._run_full()
            current.wait_stream(capture_stream)
            current.synchronize()
            capture_stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                self._run_full()
            current.wait_stream(capture_stream)
            self.graph = graph
            self.captures += 1
            COUNTS["captures"] += 1
        self.graph.replay()
        self.replays += 1
        COUNTS["replays"] += 1
        cad.seq_lens.copy_(self.seq_lens)
        cad.num_actual_tokens = 1
        cad.max_query_len = 1
        cad.query_start_loc = self.cad.query_start_loc
        cad.query_start_loc_cpu = self.cad.query_start_loc_cpu
        cad.slot_mapping = self.slots[:1]
        cad.max_seq_len = min(cad.max_seq_len + 3, self.proposer.max_model_len)
        cad._seq_lens_cpu = None
        cad._num_computed_tokens_cpu = None
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
        reason = tail._fallback_reason(
            self, common_attn_metadata, num_rejected_tokens_gpu, mm_embed_inputs
        )
        if reason is None:
            reason = _input_contract(
                self,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
                common_attn_metadata,
            )
        plan = getattr(self, "_vllm019_draft_full_graph", None)
        if reason is None:
            _, padded, _ = self._determine_batch_execution_and_padding(5)
            if padded != 5 or (
                plan
                and not plan.matches_inputs(
                    target_token_ids,
                    target_positions,
                    target_hidden_states,
                    next_token_ids,
                    token_indices_to_sample,
                    num_rejected_tokens_gpu,
                    common_attn_metadata,
                )
            ):
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
        if plan is None:
            plan = self._vllm019_draft_full_graph = _FullDraftGraph(
                self,
                common_attn_metadata,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
            )
        return plan.run_full(
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            num_rejected_tokens_gpu,
        )

    return propose


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("The complete draft graph requires exactly vLLM 0.19.0")
    from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

    tail.install()
    SpecDecodeBaseProposer.propose = _make_wrapper(SpecDecodeBaseProposer.propose)
    _INSTALLED = True


def metadata(proposer=None):
    plan = getattr(proposer, "_vllm019_draft_full_graph", None)
    return {
        "installed": _INSTALLED,
        "scope": "B1/K4, Q5 decode; complete four-step draft proposal",
        "draft_context_bound": None if plan is None else plan.cad.max_seq_len,
        "counts": dict(COUNTS),
        "captures": 0 if plan is None else plan.captures,
        "replays": 0 if plan is None else plan.replays,
        "fallback": tail.metadata(proposer),
    }
