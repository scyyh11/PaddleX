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

"""One lossless staging launch for Q5/B1 full draft graphs.

Every tensor retains its exact dtype and complete contents. BF16 auxiliary
states use uint16 pointer loads/stores, including NaN payloads and signed zero.
This module changes neither model arithmetic nor graph/attention metadata.
``install()`` explicitly wraps only the full graph's existing ``_stage`` method;
unsupported layouts retain its nine native tensor copies.
"""
from collections import Counter
from functools import wraps

import torch
import triton
import triton.language as tl

COUNTS = Counter()


@triton.jit
def _stage_kernel(
    ids,
    positions,
    hidden,
    next_ids,
    indices,
    lengths,
    blocks,
    slots,
    rejected,
    out_ids,
    out_positions,
    out_hidden,
    out_next_ids,
    out_indices,
    out_lengths,
    out_blocks,
    out_slots,
    out_rejected,
    HIDDEN_ELEMENTS: tl.constexpr,
    TABLE_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    offsets = program * BLOCK + tl.arange(0, BLOCK)
    # Reinterpretation is required: floating-point load/cast/store could alter
    # BF16 NaN payload bits. No floating-point arithmetic is performed here.
    hidden_bits = hidden.to(tl.pointer_type(tl.uint16))
    out_hidden_bits = out_hidden.to(tl.pointer_type(tl.uint16))
    values = tl.load(hidden_bits + offsets, offsets < HIDDEN_ELEMENTS, other=0)
    tl.store(out_hidden_bits + offsets, values, offsets < HIDDEN_ELEMENTS)
    table = tl.load(blocks + offsets, offsets < TABLE_ELEMENTS, other=0)
    tl.store(out_blocks + offsets, table, offsets < TABLE_ELEMENTS)
    if program == 0:
        token = tl.load(ids + offsets, offsets < 5, other=0)
        position = tl.load(positions + offsets, offsets < 5, other=0)
        slot = tl.load(slots + offsets, offsets < 5, other=0)
        tl.store(out_ids + offsets, token, offsets < 5)
        tl.store(out_positions + offsets, position, offsets < 5)
        tl.store(out_slots + offsets, slot, offsets < 5)
        tl.store(out_next_ids, tl.load(next_ids))
        tl.store(out_indices, tl.load(indices))
        tl.store(out_lengths, tl.load(lengths))
        tl.store(out_rejected, tl.load(rejected))


def _layout_reason(sources, destinations):
    if len(sources) != 9 or len(destinations) != 9:
        return "tensor_count"
    expected_shapes = ((5,), (5,), (5, 3072), (1,), (1,), (1,), None, (5,), (1,))
    expected_dtypes = (
        torch.int32,
        torch.int64,
        torch.bfloat16,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int64,
        torch.int32,
    )
    device = sources[0].device
    for source, dest, shape, dtype in zip(
        sources, destinations, expected_shapes, expected_dtypes
    ):
        if shape is None:
            if source.ndim != 2 or source.shape[0] != 1 or source.shape[1] < 1:
                return "block_table_shape"
        elif source.shape != shape:
            return "shape"
        if dest.shape != source.shape:
            return "destination_shape"
        if source.dtype != dtype or dest.dtype != dtype:
            return "dtype"
        if not source.is_contiguous() or not dest.is_contiguous():
            return "strides"
        if source.device != device or dest.device != device:
            return "device"
    if not sources[0].is_cuda:
        return "non_cuda"
    return None


def stage_tensors(sources, destinations):
    """Copy all nine tensors in one kernel, returning False for native fallback.

    Order: ids, 1-D positions, aux hidden, next ids, sample indices, sequence
    lengths, block table, slots, rejected counts. The destinations are separate
    persistent buffers allocated by the full-graph adapter; they must not alias
    another source tensor. Source pointer values may change on every call.
    """
    reason = _layout_reason(sources, destinations)
    if reason is not None:
        COUNTS["fallback_" + reason] += 1
        return False
    hidden_elements = sources[2].numel()
    table_elements = sources[6].numel()
    _stage_kernel[(triton.cdiv(max(hidden_elements, table_elements), 1024),)](
        *sources,
        *destinations,
        HIDDEN_ELEMENTS=hidden_elements,
        TABLE_ELEMENTS=table_elements,
        BLOCK=1024,
        num_warps=4,
    )
    COUNTS["fast_calls"] += 1
    return True


def _make_wrapper(original, position_row):
    @wraps(original)
    def stage(self, ids, positions, hidden, next_ids, indices, cad, rejected):
        sources = (
            ids,
            position_row(self.proposer, positions),
            hidden,
            next_ids,
            indices,
            cad.seq_lens,
            cad.block_table_tensor,
            cad.slot_mapping,
            rejected,
        )
        destinations = (
            self.target_ids,
            self.target_positions,
            self.target_hidden,
            self.next_ids,
            self.indices,
            self.target_lengths,
            self.target_blocks,
            self.target_slots,
            self.rejected,
        )
        if not stage_tensors(sources, destinations):
            return original(
                self, ids, positions, hidden, next_ids, indices, cad, rejected
            )
        return None

    stage._vllm019_fused_draft_staging = True
    return stage


def install():
    """Explicitly enable this candidate after the full/bucket graph adapter."""
    from vllm019_draft_full_graph import _FullDraftGraph, _position_row

    original = _FullDraftGraph._stage
    if not getattr(original, "_vllm019_fused_draft_staging", False):
        _FullDraftGraph._stage = _make_wrapper(original, _position_row)


def metadata():
    return {
        "counts": dict(COUNTS),
        "scope": "Q5/B1 full-draft input staging only; nine copies in one kernel",
        "arithmetic": "integer and raw uint16 copies; no floating-point conversion",
        "aux_hidden_elements": 15360,
        "block_table": "entire row, with exact source column count",
    }
