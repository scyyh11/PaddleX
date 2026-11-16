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

"""Native local-argmax path for the unchanged complete private INT8 head.

Two kernels retain the original BF16 rounding/logit scaling, then reduce ties
by absolute target token ID. No full target-sized logits or task RPC is needed.
Enable only with vLLM's native use_local_argmax_reduction greedy fixed-K mode.
"""
import torch
import triton
import triton.language as tl
import round3_eagle_int8_head as old
import round3_eagle_fused_int8_head as full

CALLS = {}


@triton.jit
def _gemv_top_blocks(
    X,
    W,
    S,
    MAP,
    VALUES,
    IDS,
    O: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BO: tl.constexpr,
    BLOCKS: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    block = tl.program_id(0)
    q = tl.program_id(1)
    row = block * BO + tl.arange(0, BO)
    k = tl.arange(0, BK)
    valid = row < O
    x = tl.load(X + q * K + k, k < K, 0).to(tl.float32)
    w = tl.load(
        W + row[:, None] * K + k[None, :], valid[:, None] & (k[None, :] < K), 0
    ).to(tl.float32)
    scales = tl.load(S + row, valid, 0)
    acc = tl.sum(w * x[None, :], 1) * scales
    value = acc.to(tl.bfloat16).to(tl.float32) * LOGIT_SCALE
    value = tl.where(valid, value, float("-inf"))
    target = tl.load(MAP + row, valid, 2147483647).to(tl.int32)
    nan = valid & (value != value)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(nan, float("inf"), value), 0)
    selected = valid & tl.where(has_nan, nan, value == maximum)
    token = tl.min(tl.where(selected, target, 2147483647), 0)
    maximum = tl.where(has_nan, float("nan"), maximum)
    tl.store(VALUES + q * BLOCKS + block, maximum)
    tl.store(IDS + q * BLOCKS + block, token)


@triton.jit
def _reduce_top_blocks(VALUES, IDS, OUT, BLOCKS: tl.constexpr, B: tl.constexpr):
    q = tl.program_id(0)
    index = tl.arange(0, B)
    valid = index < BLOCKS
    values = tl.load(VALUES + q * BLOCKS + index, valid, float("-inf"))
    ids = tl.load(IDS + q * BLOCKS + index, valid, 2147483647)
    nan = valid & (values != values)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(nan, float("inf"), values), 0)
    selected = valid & tl.where(has_nan, nan, values == maximum)
    token = tl.min(tl.where(selected, ids, 2147483647), 0)
    # If every private logit is -inf, the untouched full-vocab scatter is also
    # entirely -inf; torch.argmax then chooses target ID0 even if it is unmapped.
    token = tl.where((~has_nan) & (maximum == float("-inf")), 0, token)
    tl.store(OUT + q, token.to(tl.int64))


def top_tokens(x, plan, logit_scale=1.0):
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != 1024:
        raise ValueError("Direct INT8 top1 requires BF16 [queries,1024]")
    x = x.contiguous()
    blocks = triton.cdiv(plan["private_vocab_size"], 8)
    values = torch.empty((x.shape[0], blocks), device=x.device, dtype=torch.float32)
    ids = torch.empty_like(values, dtype=torch.int32)
    output = torch.empty((x.shape[0],), device=x.device, dtype=torch.int64)
    _gemv_top_blocks[(blocks, x.shape[0])](
        x,
        plan["weight"],
        plan["scales"],
        plan["mapping"],
        values,
        ids,
        plan["private_vocab_size"],
        x.shape[1],
        triton.next_power_of_2(x.shape[1]),
        8,
        blocks,
        float(logit_scale),
        num_warps=4,
    )
    _reduce_top_blocks[(x.shape[0],)](
        values, ids, output, blocks, triton.next_power_of_2(blocks), num_warps=4
    )
    return output


@torch.library.custom_op("round3_eagle_fused_int8::top1", mutates_args=())
def fused_top1(
    x: torch.Tensor, weight: torch.Tensor, logit_scale: float
) -> torch.Tensor:
    CALLS[x.shape[0]] = CALLS.get(x.shape[0], 0) + 1
    return top_tokens(x, full.PLANS[weight.data_ptr()], logit_scale)


@fused_top1.register_fake
def fake(x, weight, logit_scale):
    return torch.empty((x.shape[0],), device=x.device, dtype=torch.int64)


def install():
    full.install()
    from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

    cls = Eagle3LlamaForCausalLM
    if getattr(cls, "_round3_full_private_int8_top1", False):
        return

    def get_top_tokens(self, hidden_states):
        weight = self.lm_head.weight
        if weight.data_ptr() not in full.PLANS or hidden_states.dtype != torch.bfloat16:
            return old.compute_logits(self, hidden_states).argmax(-1)
        return fused_top1(
            hidden_states, weight, float(getattr(self.config, "logit_scale", 1.0))
        )

    cls.get_top_tokens = get_top_tokens
    cls._round3_full_private_int8_top1 = True


def metadata():
    return {
        **full.metadata(),
        "top1_dispatches": dict(CALLS),
        "gpu_ops": "INT8 GEMV + block max/absolute-ID tie reduction; second global block reduction",
        "tie_policy": "Smallest absolute target ID among maxima; NaNs have torch.argmax priority; all -inf returns target0",
        "native_flag": "use_local_argmax_reduction=True; greedy fixed-K only; no acceptance estimator/probabilistic sampling",
        "intermediate_bytes_per_query": [
            (p["private_vocab_size"] + 7) // 8 * 8 for p in full.PLANS.values()
        ],
    }
