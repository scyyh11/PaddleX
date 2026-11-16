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

"""Workspace experiment: FA2 speculative queries as independent decode rows.

Each row shares the original paged KV and receives its own causal endpoint.
Force 32 splits, matching observed single-request FA2 autoregressive graphs at
max_model_len=4096. This is not a general numerical-equivalence guarantee.
"""
import torch
import triton
import triton.language as tl

_BUFFERS = {}
_COUNTS = {"row_path_python_calls": 0, "original_path_python_calls": 0}


@triton.jit
def _metadata(
    CU,
    LENS,
    BT,
    NEWLENS,
    NEWBT,
    NREQ: tl.constexpr,
    COLS: tl.constexpr,
    BTSTRIDE: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
):
    row = tl.program_id(0)
    reqs = tl.arange(0, RS)
    ends = tl.load(CU + reqs + 1, reqs < NREQ, 0x7FFFFFFF)
    idx = tl.sum((ends <= row).to(tl.int32), 0)
    idx = tl.minimum(idx, NREQ - 1)
    start = tl.load(CU + idx)
    end = tl.load(CU + idx + 1)
    length = tl.load(LENS + idx)
    valid = (row >= start) & (row < end) & (length > 0)
    endpoint = length - end + row + 1
    tl.store(NEWLENS + row, tl.where(valid, endpoint, 0))
    cols = tl.arange(0, CS)
    page = tl.load(BT + idx * BTSTRIDE + cols, cols < COLS, 0)
    tl.store(NEWBT + row * COLS + cols, tl.where(valid, page, 0), cols < COLS)


def install():
    import vllm.v1.attention.backends.flash_attn as module

    if getattr(module, "_ocr_round3_rows", False):
        return
    original = module.FA4DenseAttentionKernel.kernel

    def kernel(*args, **kw):
        q = kw.get("q")
        bt = kw.get("block_table")
        cu = kw.get("cu_seqlens_q")
        eligible = (
            not args
            and q is not None
            and bt is not None
            and cu is not None
            and kw.get("fa_version") == 2
            and kw.get("causal") is True
            and 1 < kw.get("max_seqlen_q", 0) <= 9
            and q.shape[0] <= 80
            and q.shape[1:] == (16, 128)
            and kw["k"].shape[-2:] == (2, 128)
            and kw.get("max_seqlen_k", 0) <= 4096
            and kw.get("alibi_slopes") is None
            and kw.get("softcap", 0) == 0
            and kw.get("window_size") in (None, (-1, -1), [-1, -1])
        )
        if not eligible:
            _COUNTS["original_path_python_calls"] += 1
            return original(*args, **kw)
        nq = q.shape[0]
        nreq = cu.numel() - 1
        cols = bt.shape[1]
        key = (nq, cols, q.device.index)
        if key not in _BUFFERS:
            _BUFFERS[key] = (
                torch.arange(nq + 1, device=q.device, dtype=torch.int32),
                torch.zeros(nq + 1, device=q.device, dtype=torch.int32),
                torch.empty(nq, device=q.device, dtype=torch.int32),
                torch.empty((nq, cols), device=q.device, dtype=torch.int32),
            )
        rowcu, dummy, rowlens, rowbt = _BUFFERS[key]
        _metadata[(nq,)](
            cu,
            kw["seqused_k"],
            bt,
            rowlens,
            rowbt,
            nreq,
            cols,
            bt.stride(0),
            triton.next_power_of_2(nreq),
            triton.next_power_of_2(cols),
            num_warps=4,
        )
        _COUNTS["row_path_python_calls"] += 1
        return torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            kw["k"],
            kw["v"],
            kw.get("out"),
            rowcu,
            dummy,
            rowlens,
            None,
            rowbt,
            None,
            1,
            kw["max_seqlen_k"],
            0.0,
            kw["softmax_scale"],
            False,
            True,
            -1,
            -1,
            0.0,
            False,
            32,
            None,
        )[0]

    module.FA4DenseAttentionKernel.kernel = staticmethod(kernel)
    module._ocr_round3_rows = True


install()

# Install the target auxiliary-state contract before vLLM traces the model.
import native_dflash_target  # noqa: E402,F401
