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

"""Trainable mirror of vLLM's native Llama EAGLE3 feature-conditioned drafter.

Features at t and the known token at t+1 predict token t+2. Attention positions
remain t, matching TargetDependentARSpeculator's input shift. This is a new
locally trained model; no official pretrained EAGLE weights are claimed.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (y * self.weight.float()).to(x.dtype)


def rope(x, positions, theta=500000.0):
    dim = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = positions.float()[..., None] * inv_freq
    cos, sin = angles.cos().to(x.dtype)[:, None], angles.sin().to(x.dtype)[:, None]
    first, second = x.chunk(2, dim=-1)
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


class MLP(nn.Module):
    def __init__(self, hidden=1024, intermediate=2048):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, input_size, heads=8, kv_heads=2, head_dim=128):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(input_size, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_size, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(input_size, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, 1024, bias=False)

    def forward(
        self, x, positions, cache=None, anchor_positions=None, base_length=None
    ):
        b, s, _ = x.shape

        def split(z, heads):
            return z.view(b, s, heads, self.head_dim).transpose(1, 2)

        q = rope(split(self.q_proj(x), self.heads), positions)
        k = rope(split(self.k_proj(x), self.kv_heads), positions)
        v = split(self.v_proj(x), self.kv_heads)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        # At inference the incremental query is one token, with no future KV.
        if cache is not None and s != 1 and anchor_positions is None:
            raise ValueError("Cached forward supports exactly one new token")
        mask = None
        if anchor_positions is not None:
            # Independent rollout stream per anchor: original ground-truth
            # context only through that anchor, plus its own recurrent KVs.
            # Computing all streams together must not reveal another stream.
            assert cache is not None and base_length is not None
            tail = k.shape[2] - base_length
            assert tail % s == 0
            base = (
                torch.arange(base_length, device=x.device)[None, None, :]
                <= anchor_positions[:, :, None]
            )
            own = (
                torch.arange(tail, device=x.device)[None, :] % s
                == torch.arange(s, device=x.device)[:, None]
            )
            mask = torch.cat((base, own[None].expand(b, -1, -1)), -1)[:, None]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=cache is None, enable_gqa=True
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1)), (k, v)


class Layer(nn.Module):
    def __init__(self, first):
        super().__init__()
        self.first = first
        self.self_attn = Attention(2048 if first else 1024)
        self.mlp = MLP()
        self.input_layernorm = RMSNorm(1024)
        self.hidden_norm = RMSNorm(1024)
        self.post_attention_layernorm = RMSNorm(1024)

    def forward(
        self,
        hidden,
        embeds,
        positions,
        cache=None,
        anchor_positions=None,
        base_length=None,
    ):
        if self.first:
            x = torch.cat((self.input_layernorm(embeds), self.hidden_norm(hidden)), -1)
        else:
            x = self.input_layernorm(hidden)
        attended, kv = self.self_attn(
            x, positions, cache, anchor_positions, base_length
        )
        hidden = hidden + attended
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, kv


class Eagle3(nn.Module):
    def __init__(self, layers=1, output_scale=1.0):
        super().__init__()
        self.fc = nn.Linear(3072, 1024, bias=False)
        self.layers = nn.ModuleList(Layer(i == 0) for i in range(layers))
        self.norm = RMSNorm(1024)
        self.apply(self._init)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(1024).repeat(1, 3) / 3)
            self.norm.weight.fill_(output_scale)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)

    def initialize(self, path):
        payload = torch.load(path, weights_only=True, map_location="cpu")
        state = payload["state_dict"]
        depth = max(int(k.split(".")[1]) for k in state if k.startswith("layers.")) + 1
        missing, unexpected = self.load_state_dict(state, strict=False)
        invalid = [
            k
            for k in missing
            if not (k.startswith("layers.") and int(k.split(".")[1]) >= depth)
        ]
        if invalid or unexpected:
            raise ValueError((invalid, unexpected))
        with torch.no_grad():
            for i in range(depth, len(self.layers)):
                self.layers[i].self_attn.o_proj.weight.zero_()
                self.layers[i].mlp.down_proj.weight.zero_()
        return dict(
            source_step=payload.get("step"),
            source_depth=depth,
            identity_added=list(range(depth, len(self.layers))),
        )

    def forward(
        self,
        features,
        input_ids,
        embed,
        positions=None,
        caches=None,
        already_combined=False,
        anchor_positions=None,
        base_length=None,
    ):
        hidden = features if already_combined else self.fc(features.flatten(-2))
        b, s, _ = hidden.shape
        if positions is None:
            positions = torch.arange(s, device=hidden.device)[None].expand(b, -1)
        embedded = F.embedding(input_ids, embed)
        new_caches = []
        for i, layer in enumerate(self.layers):
            hidden, kv = layer(
                hidden,
                embedded,
                positions,
                None if caches is None else caches[i],
                anchor_positions,
                base_length,
            )
            new_caches.append(kv)
        return self.norm(hidden), hidden, new_caches

    def export(self, directory, metadata=None):
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = dict(
            architectures=["Eagle3LlamaForCausalLM"],
            model_type="llama",
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=len(self.layers),
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=128,
            hidden_act="silu",
            vocab_size=103424,
            draft_vocab_size=103424,
            max_position_embeddings=4096,
            rope_theta=500000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            torch_dtype="bfloat16",
            attention_bias=False,
            mlp_bias=False,
            use_aux_hidden_state=True,
            has_own_lm_head=False,
            has_own_embed_tokens=False,
            eagle_aux_hidden_state_layer_ids=[4, 10, 16],
            target_hidden_size=1024,
            num_aux_hidden_states=3,
            norm_output=False,
            norm_before_residual=False,
            eagle_config={
                "use_aux_hidden_state": True,
                "eagle_aux_hidden_state_layer_ids": [4, 10, 16],
            },
        )
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        state = {
            k: v.detach().cpu().to(torch.bfloat16).contiguous()
            for k, v in self.state_dict().items()
        }
        save_file(state, directory / "model.safetensors")
        if metadata is not None:
            (directory / "training_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
