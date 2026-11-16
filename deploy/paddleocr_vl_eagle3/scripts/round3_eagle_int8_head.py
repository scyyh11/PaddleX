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

"""Quantize only the draft's private LM head; target remains original BF16.

This approximates draft proposals and must be evaluated for acceptance/speed.
The full target verifier still determines every emitted token.
"""
import torch
import triton
import triton.language as tl
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

_PLANS = {}
_CALLS = {}


@triton.jit
def _gemv(
    X, W, S, Y, O: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BO: tl.constexpr
):
    row = tl.program_id(0) * BO + tl.arange(0, BO)
    k = tl.arange(0, BK)
    q = tl.program_id(1)
    x = tl.load(X + q * K + k, k < K, 0).to(tl.float32)
    w = tl.load(
        W + row[:, None] * K + k[None, :], (row[:, None] < O) & (k[None, :] < K), 0
    ).to(tl.float32)
    scale = tl.load(S + row, row < O, 0)
    y = tl.sum(w * x[None, :], 1) * scale
    tl.store(Y + q * O + row, y, row < O)


def quantize(weight):
    scale = weight.float().abs().amax(-1).clamp_min(1e-12) / 127
    w = (weight.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return w.contiguous(), scale.contiguous()


def gemv(x, w, scale):
    y = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
    _gemv[(triton.cdiv(w.shape[0], 8), x.shape[0])](
        x,
        w,
        scale,
        y,
        w.shape[0],
        w.shape[1],
        triton.next_power_of_2(w.shape[1]),
        8,
        num_warps=4,
    )
    return y


@torch.library.custom_op("round3_eagle_int8::head", mutates_args=())
def quantized_head(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    key = w.data_ptr()
    if key not in _PLANS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Prepare private INT8 draft head before CUDA graph capture"
            )
        _PLANS[key] = quantize(w)
    _CALLS[x.shape[0]] = _CALLS.get(x.shape[0], 0) + 1
    return gemv(x.contiguous(), *_PLANS[key])


@quantized_head.register_fake
def fake(x, w):
    return x.new_empty((x.shape[0], w.shape[0]))


_original = Eagle3LlamaForCausalLM.compute_logits
_load = Eagle3LlamaForCausalLM.load_weights


def load_weights(self, weights):
    result = _load(self, weights)
    if self.draft_id_to_target_id is not None and self.config.draft_vocab_size <= 40000:
        w = self.lm_head.weight
        if w.is_cuda:
            _PLANS[w.data_ptr()] = quantize(w)
    return result


Eagle3LlamaForCausalLM.load_weights = load_weights


def compute_logits(self, hidden_states):
    # Target sharing would violate the scope of this optimization. Require a
    # compact, private draft head with a native d2t map and the common TP1 path.
    if self.draft_id_to_target_id is None or self.config.draft_vocab_size > 40000:
        return _original(self, hidden_states)
    assert self.lm_head.weight.shape[1] == 1024
    logits = quantized_head(hidden_states, self.lm_head.weight)[
        :, : self.config.draft_vocab_size
    ].float()
    logits = logits * getattr(self.config, "logit_scale", 1.0)
    mapping = (
        torch.arange(self.config.draft_vocab_size, device=logits.device)
        + self.draft_id_to_target_id
    )
    output = logits.new_full((logits.shape[0], self.config.vocab_size), float("-inf"))
    output[:, mapping] = logits
    return output


Eagle3LlamaForCausalLM.compute_logits = compute_logits
