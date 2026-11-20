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

"""Shared vision position and sequence preparation for the two vLLM adapters.

Based on vLLM's Apache-2.0 PaddleOCR-VL implementation (vLLM contributors).
"""


def encode_vision_layers(
    encoder,
    inputs_embeds,
    cu_seqlens,
    image_grid_thw,
    height_position_ids,
    width_position_ids,
    copy_cu_seqlens,
    *,
    reuse_trig=False,
):
    import torch
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    device = inputs_embeds.device
    flatten_image_grid_thw = encoder.flatten_list(image_grid_thw)
    if width_position_ids is None or height_position_ids is None:
        split_hids = list()
        split_wids = list()
        for t, h, w in flatten_image_grid_thw:
            image_pids = torch.arange(t * h * w, device=device) % (h * w)
            sample_hids = image_pids // w
            sample_wids = image_pids % w
            split_hids.append(sample_hids)
            split_wids.append(sample_wids)
        width_position_ids = torch.concat(split_wids, dim=0)
        height_position_ids = torch.concat(split_hids, dim=0)
    pids = torch.stack(
        [height_position_ids, width_position_ids],
        dim=-1,
    )
    # The ids are built from the grids above, so `h`/`w` bound them
    # and the table size is known on the host.
    max_grid_size = max(max(h, w) for _, h, w in flatten_image_grid_thw)
    rope_emb_max_grid = encoder.rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rope_emb_max_grid[pids].flatten(1)
    if cu_seqlens is None:
        raise ValueError("cu_seqlens cannot be None for SiglipEncoder.")
    if not isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
    # Computed before the H2D copy: the attention wrappers call `.item()`
    # on it, which would sync on every layer for a device tensor.
    max_seqlen = None
    if encoder.attn_backend in {
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
        AttentionBackendEnum.TRITON_ATTN,
    }:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    if cu_seqlens.is_cpu:
        cu_seqlens = copy_cu_seqlens(cu_seqlens, device)
    else:
        cu_seqlens = cu_seqlens.to(device=device, non_blocking=True)
    if reuse_trig:
        # Reuse the same FP32 phase tensors within this forward only.
        rotary_pos_emb = (
            rotary_pos_emb.cos().contiguous(),
            rotary_pos_emb.sin().contiguous(),
        )
    hidden_states = inputs_embeds
    for encoder_layer in encoder.layers:
        hidden_states = encoder_layer(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
        )
    return hidden_states
