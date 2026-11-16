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

"""vLLM 0.19 target attention compatibility using native FA2 AR1 rows.

Small single-request verification blocks follow the native one-query FA2 path
and split heuristic. On the validated A100 80GB/108-SM device these independent
rows share one native FA2 call with the original AR1 split count; other devices
use separate native calls. Unmatched attention inputs retain the upstream path.
Target linear layers are unchanged; token equality is not guaranteed for all inputs.
"""
import importlib.metadata
import math
import struct

import torch

_ROW_CU = {}
_BATCH_CU = {}
_DEVICE_SUPPORT = {}
COUNTS = {"row_calls": 0, "row_batch_calls": 0, "fallback_calls": 0}


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def ar1_num_splits(max_seqlen_k, sm_count):
    """Reproduce pinned FA2's AR1 GQA split heuristic for 16 Q / 2 KV heads.

    vLLM 0.19 pins flash-attention commit 29210221863736a08f71a866459e368ad1ac4a95.
    Its csrc/flash_attn/flash_api.cpp:262-324 uses FP32 occupancy estimates.
    AR1 transposes the 8 GQA groups into query positions, leaving 2 KV heads,
    one 64-query M block and one request. Head dim 128 gives 128-key blocks.
    """
    n_blocks = (int(max_seqlen_k) + 127) // 128
    n_sms = int(sm_count) * 2
    workers = 2
    if n_blocks < 1 or n_sms < 1:
        raise ValueError("Positive KV bound and SM count required")
    if workers >= _f32(_f32(0.8) * n_sms):
        return 1
    efficiencies = []
    for splits in range(1, min(128, n_sms, n_blocks) + 1):
        if splits > 1 and (n_blocks + splits - 1) // splits == (
            n_blocks + splits - 2
        ) // (splits - 1):
            continue
        waves = _f32(_f32(workers * splits) / n_sms)
        efficiency = _f32(waves / math.ceil(waves))
        efficiencies.append((splits, efficiency))
    threshold = 0.85 * max(value for _, value in efficiencies)
    return next(splits for splits, value in efficiencies if value >= threshold)


def _supports_row_batch_device(q):
    if not q.is_cuda:
        return False
    if q.device not in _DEVICE_SUPPORT:
        props = torch.cuda.get_device_properties(q.device)
        _DEVICE_SUPPORT[q.device] = (
            props.major == 8
            and props.minor == 0
            and props.multi_processor_count == 108
            and "A100" in props.name
            and props.total_memory >= 79 * 1024**3
        )
    return _DEVICE_SUPPORT[q.device]


def _batched_attention(kw, output, kernel=None):
    q, cu = kw["q"], kw["cu_seqlens_q"]
    count = q.shape[0]
    key = (count, q.device, cu.dtype)
    if key not in _BATCH_CU:
        _BATCH_CU[key] = (
            torch.arange(count + 1, device=q.device, dtype=cu.dtype),
            torch.zeros(count + 1, device=q.device, dtype=cu.dtype),
        )
    row_cu, dummy_cu = _BATCH_CU[key]
    rows = row_cu[:-1]
    start, end, lengths = cu[:1], cu[1:2], kw["seqused_k"]
    valid = (rows >= start) & (rows < end) & (lengths > 0)
    endpoints = (lengths - end + rows + 1).clamp_min(0)
    endpoints = torch.where(valid, endpoints, 0)
    tables = torch.where(valid[:, None], kw["block_table"].expand(count, -1), 0)
    splits = ar1_num_splits(kw["max_seqlen_k"], 108)
    scale = kw.get("softmax_scale")
    if scale is None:
        scale = q.shape[-1] ** (-0.5)
    if kernel is None:
        kernel = torch.ops._vllm_fa2_C.varlen_fwd
    # The public Python wrapper rejects explicit splits >1, but its pinned
    # native operator supports them (flash_api.cpp set_params_splitkv). Preserve
    # the exact 0.19 native signature, checked at installation below.
    result = kernel(
        q,
        kw["k"],
        kw["v"],
        output,
        row_cu,
        dummy_cu,
        endpoints,
        None,
        tables,
        None,
        1,
        kw["max_seqlen_k"],
        0.0,
        scale,
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        splits,
        None,
    )[0]
    COUNTS["row_batch_calls"] += 1
    key = f"row_batch_split_{splits}_calls"
    COUNTS[key] = COUNTS.get(key, 0) + 1
    return result


def _verify_fa2_schema():
    expected = (
        "q k v out cu_seqlens_q cu_seqlens_k seqused_k leftpad_k block_table "
        "alibi_slopes max_seqlen_q max_seqlen_k p_dropout softmax_scale "
        "zero_tensors is_causal window_size_left window_size_right softcap "
        "return_softmax num_splits gen"
    ).split()
    schema = torch.ops._vllm_fa2_C.varlen_fwd.default._schema
    if [argument.name for argument in schema.arguments] != expected:
        raise RuntimeError("Unexpected vLLM 0.19 FA2 native operator signature")


def _eligible(args, kw):
    q, k, v = kw.get("q"), kw.get("k"), kw.get("v")
    cu, lengths, tables = (
        kw.get("cu_seqlens_q"),
        kw.get("seqused_k"),
        kw.get("block_table"),
    )
    return (
        not args
        and q is not None
        and k is not None
        and v is not None
        and cu is not None
        and lengths is not None
        and tables is not None
        and q.ndim == 3
        and 2 <= q.shape[0] <= 5
        and q.shape[1:] == (16, 128)
        and k.ndim == v.ndim == 4
        and k.shape[-2:] == v.shape[-2:] == (2, 128)
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and cu.shape == (2,)
        and lengths.shape == (1,)
        and tables.ndim == 2
        and tables.shape[0] == 1
        and tables.stride(-1) == 1
        and kw.get("max_seqlen_q") == q.shape[0]
        and 0 < kw.get("max_seqlen_k", 0) <= 12288
        and kw.get("fa_version") == 2
        and kw.get("causal") is True
        and kw.get("num_splits", 0) == 0
        and kw.get("dropout_p", 0.0) == 0.0
        and kw.get("softcap", 0.0) == 0.0
        and kw.get("window_size") in (None, (-1, -1), [-1, -1])
        and kw.get("alibi_slopes") is None
        and kw.get("cu_seqlens_k") is None
        and kw.get("q_v") is None
        and kw.get("scheduler_metadata") is None
        and kw.get("s_aux") is None
        and kw.get("cp_world_size", 1) == 1
        and kw.get("cp_tot_seqused_k") is None
        and not kw.get("return_softmax_lse", False)
        and not kw.get("return_attn_probs", False)
    )


def _make_wrapper(original):
    def attention(*args, **kw):
        if not _eligible(args, kw):
            COUNTS["fallback_calls"] += 1
            return original(*args, **kw)
        q, cu = kw["q"], kw["cu_seqlens_q"]
        output = kw.get("out")
        if output is None:
            output = torch.empty_like(q)
        if (
            _supports_row_batch_device(q)
            and q.stride(-1) == kw["k"].stride(-1) == kw["v"].stride(-1) == 1
            and cu.dtype
            == kw["seqused_k"].dtype
            == kw["block_table"].dtype
            == torch.int32
        ):
            return _batched_attention(kw, output)
        key = (q.device, cu.dtype)
        if key not in _ROW_CU:
            # arange can be captured without a host-to-device tensor copy.
            _ROW_CU[key] = torch.arange(2, device=q.device, dtype=cu.dtype)
        row_cu = _ROW_CU[key]
        start, end = cu[:1], cu[1:2]
        lengths = kw["seqused_k"]
        for row in range(q.shape[0]):
            valid = (row >= start) & (row < end) & (lengths > 0)
            endpoint = (lengths - end + row + 1).clamp_min(0)
            endpoint = torch.where(valid, endpoint, 0)
            row_kw = dict(
                kw,
                q=q[row : row + 1],
                out=output[row : row + 1],
                cu_seqlens_q=row_cu,
                max_seqlen_q=1,
                seqused_k=endpoint,
            )
            original(**row_kw)
        COUNTS["row_calls"] += 1
        return output

    return attention


def install():
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("AR1 attention adapter requires vLLM 0.19.0")
    import vllm.v1.attention.backends.flash_attn as module

    _verify_fa2_schema()
    if getattr(module, "_paddleocr_vllm019_ar1_attention", False):
        return
    module.flash_attn_varlen_func = _make_wrapper(module.flash_attn_varlen_func)
    module._paddleocr_vllm019_ar1_attention = True
