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

"""Fused PaddleOCR-VL vision RoPE for validated vLLM runtimes."""

from .....utils.deps import is_vllm_performance_version

if not is_vllm_performance_version():
    raise ImportError("PaddleX vision RoPE requires vLLM 0.33.0.")
else:
    import torch
    import triton
    import triton.language as tl
    from vllm.model_executor.models.paddleocr_vl import SiglipAttention


@triton.jit
def _rotate_qk(
    qkv,
    cos,
    sin,
    size,
    stride,
    HALF: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    s = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    j = tl.arange(0, BLOCK_J)[None, :]
    head, dim = j // HALF, j % HALF
    mask = (s < size) & (j < HEADS * HALF)
    c = tl.load(cos + s * HALF + dim, mask, other=0.0).to(tl.float32)
    sn = tl.load(sin + s * HALF + dim, mask, other=0.0).to(tl.float32)
    for part in tl.static_range(2):
        p = qkv + s * stride + part * HEADS * DIM + head * DIM + dim
        x = tl.load(p, mask, other=0.0).to(tl.float32)
        y = tl.load(p + HALF, mask, other=0.0).to(tl.float32)
        tl.store(p, x * c - y * sn, mask)
        tl.store(p + HALF, y * c + x * sn, mask)


class _PaddleXSiglipAttention(SiglipAttention):
    def forward(self, hidden_states, *, cu_seqlens, rotary_pos_emb, max_seqlen):
        if not isinstance(rotary_pos_emb, tuple):
            return super().forward(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
            )

        cos, sin = rotary_pos_emb
        x, _ = self.qkv_proj(hidden_states.transpose(0, 1))
        seq = x.shape[0]
        nh = self.num_attention_heads_per_partition
        hd = self.hidden_size_per_attention_head
        expect = (seq, hd // 2)
        if x.stride(-1) == 1 and cos.shape == sin.shape == expect:
            _rotate_qk[(triton.cdiv(seq, 8),)](
                x,
                cos,
                sin,
                seq,
                x.stride(0),
                HALF=hd // 2,
                HEADS=nh,
                DIM=hd,
                BLOCK_S=8,
                BLOCK_J=triton.next_power_of_2(nh * hd // 2),
                num_warps=8,
                enable_fp_fusion=False,
            )
            q, k, v = (t.transpose(0, 1) for t in self.split_qkv(x))
        else:
            q, k, v = (t.transpose(0, 1) for t in self.split_qkv(x))
            q, k = self.apply_rotary_emb(torch.cat((q, k), dim=0), cos, sin).chunk(2)

        context = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        output, _ = self.out_proj(context.flatten(-2))
        return output
