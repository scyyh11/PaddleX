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

"""Fuse the complete private INT8 draft head into fill + GEMV/scatter.

Reuses the original INT8 weight/scales without pruning any private token.
The BF16 GEMV output rounding occurs before FP32 logit_scale multiplication,
matching round3_eagle_int8_head.compute_logits. Target layers are untouched.
"""
import torch
import triton
import triton.language as tl
import round3_eagle_int8_head as original_int8

PLANS = {}
CALLS = {}


def prepare_plan(weight, offsets, private_vocab_size, target_vocab_size):
    n = int(private_vocab_size)
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.shape[1] != 1024:
        raise ValueError("Requires the original BF16 private head with hidden_size1024")
    if not 0 < n <= 40000 or weight.shape[0] < n:
        raise ValueError("Invalid private vocabulary size")
    if (
        offsets.ndim != 1
        or len(offsets) != n
        or offsets.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Native d2t must contain exactly one integer offset per private token"
        )
    key = weight.data_ptr()
    if key not in original_int8._PLANS:
        raise RuntimeError(
            "Original INT8 load/prepare must run before fused head preparation"
        )
    quantized, scales = original_int8._PLANS[key]
    mapping = torch.arange(n, device=weight.device, dtype=torch.int64) + offsets.to(
        device=weight.device, dtype=torch.int64
    )
    if len(mapping.unique()) != n or not bool(
        ((mapping >= 0) & (mapping < target_vocab_size)).all()
    ):
        raise ValueError(
            "Private-to-target map must be injective and within target vocabulary"
        )
    if quantized.shape != weight.shape or scales.shape != (weight.shape[0],):
        raise ValueError("Original quantized plan shape mismatch")
    return dict(
        weight=quantized,
        scales=scales,
        mapping=mapping.contiguous(),
        private_vocab_size=n,
        target_vocab_size=int(target_vocab_size),
        original_padded_head_rows=weight.shape[0],
    )


@triton.jit
def _gemv_scatter(
    X,
    W,
    S,
    MAP,
    Y,
    O: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BO: tl.constexpr,
    VOCAB: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row = tl.program_id(0) * BO + tl.arange(0, BO)
    k = tl.arange(0, BK)
    q = tl.program_id(1)
    x = tl.load(X + q * K + k, k < K, 0).to(tl.float32)
    w = tl.load(
        W + row[:, None] * K + k[None, :], (row[:, None] < O) & (k[None, :] < K), 0
    ).to(tl.float32)
    scale = tl.load(S + row, row < O, 0)
    acc = tl.sum(w * x[None, :], 1) * scale
    # The original GEMV stores to an input-dtype BF16 tensor. Preserve this
    # rounding boundary before converting to the full-vocabulary FP32 result.
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    value = rounded * LOGIT_SCALE
    target = tl.load(MAP + row, row < O, 0)
    tl.store(Y + q * VOCAB + target, value, row < O)


def full_logits(x, plan, logit_scale=1.0):
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != 1024:
        raise ValueError("Fused path requires BF16 [queries,1024] input")
    x = x.contiguous()
    output = torch.full(
        (x.shape[0], plan["target_vocab_size"]),
        float("-inf"),
        device=x.device,
        dtype=torch.float32,
    )
    _gemv_scatter[(triton.cdiv(plan["private_vocab_size"], 8), x.shape[0])](
        x,
        plan["weight"],
        plan["scales"],
        plan["mapping"],
        output,
        plan["private_vocab_size"],
        x.shape[1],
        triton.next_power_of_2(x.shape[1]),
        8,
        plan["target_vocab_size"],
        float(logit_scale),
        num_warps=4,
    )
    return output


@torch.library.custom_op("round3_eagle_fused_int8::head", mutates_args=())
def fused_head(
    x: torch.Tensor, weight: torch.Tensor, target_vocab_size: int, logit_scale: float
) -> torch.Tensor:
    plan = PLANS[weight.data_ptr()]
    if plan["target_vocab_size"] != target_vocab_size:
        raise ValueError("Target vocabulary changed after preparation")
    CALLS[x.shape[0]] = CALLS.get(x.shape[0], 0) + 1
    return full_logits(x, plan, logit_scale)


@fused_head.register_fake
def fake(x, weight, target_vocab_size, logit_scale):
    return torch.empty(
        (x.shape[0], target_vocab_size), device=x.device, dtype=torch.float32
    )


def install():
    from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

    cls = Eagle3LlamaForCausalLM
    if getattr(cls, "_round3_fused_complete_int8_head", False):
        return
    old_load, old_compute = cls.load_weights, cls.compute_logits

    def load_weights(self, weights):
        result = old_load(self, weights)
        w = self.lm_head.weight
        if (
            self.draft_id_to_target_id is not None
            and self.config.draft_vocab_size <= 40000
            and w.is_cuda
            and w.dtype == torch.bfloat16
            and w.shape[1] == 1024
        ):
            PLANS[w.data_ptr()] = prepare_plan(
                w,
                self.draft_id_to_target_id,
                self.config.draft_vocab_size,
                self.config.vocab_size,
            )
        return result

    def compute_logits(self, hidden_states):
        weight = self.lm_head.weight
        if weight.data_ptr() not in PLANS or hidden_states.dtype != torch.bfloat16:
            return old_compute(self, hidden_states)
        return fused_head(
            hidden_states,
            weight,
            self.config.vocab_size,
            float(getattr(self.config, "logit_scale", 1.0)),
        )

    cls.load_weights, cls.compute_logits = load_weights, compute_logits
    cls._round3_fused_complete_int8_head = True


def metadata():
    return dict(
        plans=len(PLANS),
        private_vocab_sizes=[p["private_vocab_size"] for p in PLANS.values()],
        target_vocab_sizes=[p["target_vocab_size"] for p in PLANS.values()],
        dispatches=dict(CALLS),
        dispatch_scope="Python eager/init/graph-capture calls, not replay count",
        scope="Complete private INT8 EAGLE head only; original quantized weights/scales reused by reference; target untouched.",
        arithmetic="FP32 dot*row_scale -> BF16 rounding -> FP32 conversion*logit_scale -> absolute target scatter",
        vocabulary_pruning=False,
        per_request_rpc=False,
        gpu_ops="FP32 target-size -inf fill plus one GEMV/scatter kernel",
    )
