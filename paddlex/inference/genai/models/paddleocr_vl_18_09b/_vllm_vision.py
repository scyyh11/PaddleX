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

"""Optional PaddleOCR-VL vision path, shared across vLLM runtimes.

Based on vLLM's Apache-2.0 PaddleOCR-VL implementation (vLLM contributors).
The native processor, projection, attention and decoder are retained.
"""

import inspect
import math
from itertools import accumulate

from .....utils import logging
from .....utils.deps import is_dep_available, is_vllm_performance_version

if not all(map(is_dep_available, ("torch", "vllm"))):
    raise ImportError("PaddleX vision requires Torch and vLLM.")
else:
    import torch
    from vllm.model_executor.models.paddleocr_vl import (
        PaddleOCRVLDummyInputsBuilder,
        PaddleOCRVLForConditionalGeneration,
        PaddleOCRVLMultiModalProcessor,
        PaddleOCRVLProcessingInfo,
        SiglipEncoder,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY


def _accepts_keywords(fn, *names):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return all(
        name in params
        and params[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for name in names
    )


def _vision_support(vllm_config, visual, encoder):
    parallel = vllm_config.parallel_config
    if (
        torch.version.cuda is None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (8, 0)
        or vllm_config.model_config.dtype != torch.bfloat16
        or vllm_config.quant_config is not None
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or getattr(parallel, "enable_expert_parallel", False)
        or getattr(getattr(encoder, "attn_backend", None), "name", None) != "FLASH_ATTN"
    ):
        return (
            "requires one SM80 GPU, BF16, unquantized weights and vision FlashAttention"
        )
    if not isinstance(encoder, SiglipEncoder) or not (
        _accepts_keywords(
            visual.forward,
            "pixel_values",
            "image_grid_thw",
            "position_ids",
            "interpolate_pos_encoding",
            "cu_seqlens",
        )
        and _accepts_keywords(
            encoder.forward,
            "inputs_embeds",
            "cu_seqlens",
            "image_grid_thw",
            "height_position_ids",
            "width_position_ids",
        )
        and getattr(encoder, "layers", None)
        and callable(getattr(encoder, "flatten_list", None))
        and callable(getattr(encoder, "rotary_pos_emb", None))
        and all(
            _accepts_keywords(
                layer.forward, "cu_seqlens", "rotary_pos_emb", "max_seqlen"
            )
            for layer in encoder.layers
        )
    ):
        return "incompatible vLLM vision interface"
    return None


def _pack_pixels(pixel_values, lengths):
    # Native processors may supply a flat patch tensor, a [B,P,C,H,W]
    # tensor of equal-length images, or a sequence of variable-length tensors.
    if isinstance(pixel_values, torch.Tensor):
        if pixel_values.ndim == 4:
            return pixel_values if pixel_values.shape[0] == sum(lengths) else None
        if (
            pixel_values.ndim == 5
            and pixel_values.shape[0] == len(lengths)
            and all(n == pixel_values.shape[1] for n in lengths)
        ):
            return pixel_values.flatten(0, 1)
        return None
    if not isinstance(pixel_values, (tuple, list)) or len(pixel_values) != len(lengths):
        return None
    if any(
        not isinstance(pixels, torch.Tensor)
        or pixels.ndim != 4
        or pixels.shape[0] != n
        or pixels.shape[1:] != pixel_values[0].shape[1:]
        or pixels.device != pixel_values[0].device
        or pixels.dtype != pixel_values[0].dtype
        for pixels, n in zip(pixel_values, lengths)
    ):
        return None
    return pixel_values[0] if len(pixel_values) == 1 else torch.cat(pixel_values)


def _encode_vision_layers(
    encoder, inputs_embeds, cu_seqlens, image_grid_thw, *, reuse_trig
):
    device = inputs_embeds.device
    grids = encoder.flatten_list(image_grid_thw)
    split_hids, split_wids = [], []
    for t, h, w in grids:
        image_pids = torch.arange(t * h * w, device=device) % (h * w)
        split_hids.append(image_pids // w)
        split_wids.append(image_pids % w)
    height_position_ids = torch.cat(split_hids)
    width_position_ids = torch.cat(split_wids)
    pids = torch.stack(
        [height_position_ids, width_position_ids],
        dim=-1,
    )
    # The ids are built from the grids above, so `h`/`w` bound them
    # and the table size is known on the host.
    max_grid_size = max(max(h, w) for _, h, w in grids)
    rope_emb_max_grid = encoder.rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rope_emb_max_grid[pids].flatten(1)
    if cu_seqlens is None:
        raise ValueError("cu_seqlens cannot be None for SiglipEncoder.")
    if not isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
    # Computed before the H2D copy: the attention wrappers call `.item()`
    # on it, which would sync on every layer for a device tensor.
    max_seqlen = None
    if getattr(encoder.attn_backend, "name", None) == "FLASH_ATTN":
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    if cu_seqlens.is_cpu and device.type == "cuda" and not cu_seqlens.is_pinned():
        cu_seqlens = cu_seqlens.pin_memory()
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


class _PaddleXSiglipEncoder(SiglipEncoder):
    def forward(
        self,
        inputs_embeds,
        cu_seqlens=None,
        image_grid_thw=None,
        height_position_ids=None,
        width_position_ids=None,
    ):
        if (
            inputs_embeds.dtype != torch.bfloat16
            or inputs_embeds.ndim != 3
            or inputs_embeds.shape[0] != 1
            or not inputs_embeds.is_cuda
            or image_grid_thw is None
            or height_position_ids is not None
            or width_position_ids is not None
            or torch.compiler.is_compiling()
        ):
            return super().forward(
                inputs_embeds,
                cu_seqlens,
                image_grid_thw,
                height_position_ids,
                width_position_ids,
            )
        return _encode_vision_layers(
            self,
            inputs_embeds,
            cu_seqlens,
            image_grid_thw,
            reuse_trig=self._paddlex_vision_rope,
        )


@MULTIMODAL_REGISTRY.register_processor(
    PaddleOCRVLMultiModalProcessor,
    info=PaddleOCRVLProcessingInfo,
    dummy_inputs=PaddleOCRVLDummyInputsBuilder,
)
class PaddleXPaddleOCRVLForConditionalGeneration(PaddleOCRVLForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hf_config = vllm_config.model_config.hf_config
        # The serving profile always sets both. Internal tests may enable one.
        batching = getattr(hf_config, "paddlex_vision_batching", False)
        rope = getattr(hf_config, "paddlex_vision_rope", False)
        self._paddlex_vision_batching = False
        if not (batching or rope):
            return
        if rope and not is_vllm_performance_version():
            raise ValueError("paddlex-vision-rope requires vLLM 0.33.0.")
        encoder = getattr(getattr(self.visual, "vision_model", None), "encoder", None)
        reason = _vision_support(vllm_config, self.visual, encoder)
        if reason:
            if rope:
                feature = "vision performance profile" if batching else "vision RoPE"
                raise ValueError(f"PaddleX {feature} unavailable: {reason}.")
            logging.warning(
                f"PaddleX vision batching unavailable ({reason}); using native vLLM vision."
            )
            return

        if rope:
            from ._vision_rope import _PaddleXSiglipAttention

            for layer in encoder.layers:
                layer.self_attn.__class__ = _PaddleXSiglipAttention
            logging.info("PaddleX FP32 vision RoPE enabled (vLLM 0.33.0, SM80 BF16)")
        # Bind only this instance. Its parameters and native loaders stay intact.
        encoder._paddlex_vision_rope = rope
        encoder.__class__ = _PaddleXSiglipEncoder
        self._paddlex_vision_batching = batching
        if batching:
            logging.info("PaddleX vision batching enabled (SM80 BF16)")

    def _process_image_input(self, image_input):
        if not self._paddlex_vision_batching:
            return super()._process_image_input(image_input)
        grid_tensor = image_input.image_grid_thw
        if (
            not isinstance(grid_tensor, torch.Tensor)
            or grid_tensor.ndim != 2
            or grid_tensor.shape[1] != 3
            or grid_tensor.is_floating_point()
            or grid_tensor.is_complex()
            or grid_tensor.dtype == torch.bool
        ):
            return super()._process_image_input(image_input)
        grids = [tuple(grid) for grid in grid_tensor.tolist()]
        if not grids or any(min(grid) <= 0 for grid in grids):
            return super()._process_image_input(image_input)
        lengths = [math.prod(grid) for grid in grids]
        if sum(lengths) >= 2**31:
            return super()._process_image_input(image_input)
        pixels = _pack_pixels(image_input.pixel_values, lengths)
        if pixels is None:
            return super()._process_image_input(image_input)

        pixels = pixels.to(dtype=self.visual.dtype)
        pin = pixels.is_cuda
        position_ids = torch.cat(
            [
                torch.arange(n, device="cpu") % (h * w)
                for n, (_, h, w) in zip(lengths, grids)
            ],
            out=torch.empty(
                sum(lengths), dtype=torch.int64, device="cpu", pin_memory=pin
            ),
        ).to(pixels.device, non_blocking=True)
        # A separate FlashAttention varlen segment for each image.
        cu_seqlens = torch.tensor(
            [0, *accumulate(lengths)], dtype=torch.int32, device="cpu", pin_memory=pin
        )
        vision = self.visual(
            pixel_values=pixels,
            image_grid_thw=grids,
            position_ids=position_ids,
            interpolate_pos_encoding=True,
            cu_seqlens=cu_seqlens,
        ).squeeze(0)
        return self.mlp_AR(vision.split(lengths), grid_tensor)
