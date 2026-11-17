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

"""Optional integer-only metadata fusion for the vLLM 0.19 AR1 adapter.

This module installs no hooks. Its caller must preserve the adapter's existing
eligibility checks and native FA2 arguments, including its AR1 split count.
The function does not cache outputs: every invocation reads current graph input
buffers, including padding and zero-length rows. Importing it initializes no GPU.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ar1_metadata_kernel(
    cu_ptr,
    lengths_ptr,
    source_table_ptr,
    endpoints_ptr,
    tables_ptr,
    CU_STRIDE: tl.constexpr,
    TABLE_COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(cu_ptr)
    end = tl.load(cu_ptr + CU_STRIDE)
    length = tl.load(lengths_ptr)
    valid = (row >= start) & (row < end) & (length > 0)
    endpoint = tl.maximum(length - end + row + 1, 0)
    tl.store(endpoints_ptr + row, tl.where(valid, endpoint, 0))
    columns = tl.arange(0, BLOCK)
    table = tl.load(source_table_ptr + columns, columns < TABLE_COLUMNS, other=0)
    tl.store(
        tables_ptr + row * TABLE_COLUMNS + columns,
        tl.where(valid, table, 0),
        columns < TABLE_COLUMNS,
    )


def build_ar1_metadata(cu, lengths, block_table, count):
    """Return the exact int32 endpoints/tables used by `_batched_attention`.

    `cu` is [start, end], `lengths` contains one current KV length, and
    `block_table` has one contiguous row. `count` includes graph padding.
    Only the original small-query profile (2..5 rows) is supported.
    """
    if not 2 <= count <= 5:
        raise ValueError("AR1 metadata fusion requires 2..5 query rows")
    if (
        cu.shape != (2,)
        or lengths.shape != (1,)
        or block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.shape[1] < 1
        or block_table.stride(-1) != 1
        or cu.dtype != torch.int32
        or lengths.dtype != torch.int32
        or block_table.dtype != torch.int32
        or not cu.is_cuda
        or cu.device != lengths.device
        or cu.device != block_table.device
    ):
        raise ValueError("Expected CUDA int32 single-request AR1 metadata")
    columns = block_table.shape[1]
    endpoints = torch.empty((count,), dtype=torch.int32, device=cu.device)
    tables = torch.empty((count, columns), dtype=torch.int32, device=cu.device)
    _ar1_metadata_kernel[(count,)](
        cu,
        lengths,
        block_table,
        endpoints,
        tables,
        CU_STRIDE=cu.stride(0),
        TABLE_COLUMNS=columns,
        BLOCK=triton.next_power_of_2(columns),
        num_warps=4,
    )
    return endpoints, tables
