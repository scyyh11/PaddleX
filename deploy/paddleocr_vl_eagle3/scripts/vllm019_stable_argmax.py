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

"""Explicit FP32 stable argmax; import installs no hooks.

Scope is contiguous CUDA FP32 [Q, 103424], Q in 1..7. Unsupported inputs raise;
there is no native fallback. The two kernels implement PyTorch GreaterOrNan:
NaN wins, ties (including two NaNs and signed zeros) choose the lowest token id.
Stage one has 26 blocks of 4096 elements per row; stage two reduces 26 pairs.

Plan buffers are persistent and overwritten on each call. Calls sharing a plan
must be ordered on one stream, with synchronization before a stream change.
Keep the plan, captured inputs, and external output alive until every graph
using them is retired. A returned default output aliases plan-owned storage.
No GPU allocation or scalar read occurs inside __call__.
"""

import torch
import triton
import triton.language as tl

VOCAB = 103424
BLOCK = 4096
PARTITIONS = (VOCAB + BLOCK - 1) // BLOCK
INVALID_INDEX = 2147483647


@triton.jit
def _greater_or_nan(a, ia, b, ib):
    a_nan = a != a
    b_nan = b != b
    # When only b is NaN, both ordered comparisons below are false.
    choose_a = tl.where(
        a_nan,
        (~b_nan) | (ia < ib),
        (a > b) | ((a == b) & (ia < ib)),
    )
    return tl.where(choose_a, a, b), tl.where(choose_a, ia, ib)


@triton.jit
def _partial_argmax(
    logits_ptr,
    values_ptr,
    indices_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NPARTS: tl.constexpr,
):
    part = tl.program_id(0)
    row = tl.program_id(1)
    indices = part * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = indices < VOCAB_SIZE
    values = tl.load(
        logits_ptr + row * VOCAB_SIZE + indices, valid, other=-float("inf")
    )
    indices = tl.where(valid, indices, 2147483647)
    best_value, best_index = tl.reduce((values, indices), 0, _greater_or_nan)
    tl.store(values_ptr + row * NPARTS + part, best_value)
    tl.store(indices_ptr + row * NPARTS + part, best_index)


@triton.jit
def _finish_argmax(
    values_ptr,
    indices_ptr,
    output_ptr,
    NPARTS: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    parts = tl.arange(0, MERGE_BLOCK)
    valid = parts < NPARTS
    values = tl.load(values_ptr + row * NPARTS + parts, valid, other=-float("inf"))
    indices = tl.load(indices_ptr + row * NPARTS + parts, valid, other=2147483647)
    _, best_index = tl.reduce((values, indices), 0, _greater_or_nan)
    tl.store(output_ptr + row, best_index.to(tl.int64))


class StableArgmax:
    """Allocate one reusable plan before graph capture, then launch two kernels."""

    def __init__(self, queries, device="cuda"):
        if type(queries) is not int or not 1 <= queries <= 7:
            raise ValueError("queries must be an integer in 1..7")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("StableArgmax requires a CUDA device")
        with torch.cuda.device(device):
            self.partial_values = torch.empty(
                (queries, PARTITIONS), dtype=torch.float32, device=device
            )
            self.partial_indices = torch.empty(
                (queries, PARTITIONS), dtype=torch.int32, device=device
            )
            self.output = torch.empty(queries, dtype=torch.int64, device=device)
        self.queries = queries
        self.device = self.output.device

    def __call__(self, logits, out=None):
        if (
            not isinstance(logits, torch.Tensor)
            or logits.device != self.device
            or logits.dtype != torch.float32
            or tuple(logits.shape) != (self.queries, VOCAB)
            or not logits.is_contiguous()
        ):
            raise ValueError(
                "logits must be contiguous CUDA float32 [Q, 103424] on the plan device"
            )
        out = self.output if out is None else out
        if (
            not isinstance(out, torch.Tensor)
            or out.device != self.device
            or out.dtype != torch.int64
            or tuple(out.shape) != (self.queries,)
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous CUDA int64 [Q] on the plan device")
        if any(
            out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
            for tensor in (logits, self.partial_values, self.partial_indices)
        ):
            raise ValueError("output must not alias logits or scratch storage")
        with torch.cuda.device(self.device):
            _partial_argmax[(PARTITIONS, self.queries)](
                logits,
                self.partial_values,
                self.partial_indices,
                VOCAB_SIZE=VOCAB,
                BLOCK_SIZE=BLOCK,
                NPARTS=PARTITIONS,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _finish_argmax[(self.queries,)](
                self.partial_values,
                self.partial_indices,
                out,
                NPARTS=PARTITIONS,
                MERGE_BLOCK=32,
                num_warps=1,
                enable_fp_fusion=False,
            )
            stream = torch.cuda.current_stream(self.device)
            for tensor in (logits, self.partial_values, self.partial_indices, out):
                tensor.record_stream(stream)
        return out

    def metadata(self):
        return {
            "queries": self.queries,
            "vocab": VOCAB,
            "block": BLOCK,
            "partitions": PARTITIONS,
            "stage1_warps": 4,
            "stage2_warps": 1,
            "workspace_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in (self.partial_values, self.partial_indices, self.output)
            ),
            "semantics": "FP32 GreaterOrNan, lowest global index on ties, output int64",
            "hooks_installed": False,
        }
