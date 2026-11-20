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

"""Opt-in vision batching for vLLM 0.19.0's per-image execution path."""

import math

import numpy as np

from .....utils import logging
from .....utils.deps import is_vllm_legacy_version
from ._vllm_vision import encode_vision_layers

if not is_vllm_legacy_version():
    raise ImportError("This PaddleX adapter requires vLLM 0.19.0.")
else:
    import torch
    from vllm.model_executor.models.paddleocr_vl import (
        AttentionBackendEnum,
        PaddleOCRVLDummyInputsBuilder,
        PaddleOCRVLForConditionalGeneration,
        PaddleOCRVLMultiModalProcessor,
        PaddleOCRVLProcessingInfo,
        SiglipEncoder,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY


def _copy_cu_seqlens(tensor, device):
    return tensor.to(device=device, non_blocking=True)


class _BatchedSiglipEncoder(SiglipEncoder):
    def forward(
        self,
        inputs_embeds,
        cu_seqlens=None,
        image_grid_thw=None,
        height_position_ids=None,
        width_position_ids=None,
    ):
        return encode_vision_layers(
            self,
            inputs_embeds,
            cu_seqlens,
            image_grid_thw,
            height_position_ids,
            width_position_ids,
            _copy_cu_seqlens,
        )


@MULTIMODAL_REGISTRY.register_processor(
    PaddleOCRVLMultiModalProcessor,
    info=PaddleOCRVLProcessingInfo,
    dummy_inputs=PaddleOCRVLDummyInputsBuilder,
)
class PaddleXBatchedPaddleOCRVLForConditionalGeneration(
    PaddleOCRVLForConditionalGeneration
):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        parallel = vllm_config.parallel_config
        encoder = self.visual.vision_model.encoder
        # Keep the backport opt-in and within the tested CUDA path. In
        # particular, do not route V100 through the multi-image FlashAttention path.
        if (
            torch.version.cuda is None
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability() != (8, 0)
            or vllm_config.model_config.dtype != torch.bfloat16
            or vllm_config.quant_config is not None
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or encoder.attn_backend != AttentionBackendEnum.FLASH_ATTN
        ):
            raise ValueError(
                "PaddleX vision batching on vLLM 0.19.0 requires one SM80 GPU, "
                "BF16, unquantized weights and FlashAttention for vision."
            )
        encoder.__class__ = _BatchedSiglipEncoder
        logging.info("PaddleX vision batching enabled (vLLM 0.19.0, SM80 BF16)")

    def _process_image_input(self, image_input):
        grids = [tuple(grid) for grid in image_input.image_grid_thw.tolist()]
        lengths = [math.prod(grid) for grid in grids]
        pixels = image_input.pixel_values
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.reshape(-1, *pixels.shape[-3:])
        else:
            pixels = torch.cat([p.reshape(-1, *p.shape[-3:]) for p in pixels])
        pixels = pixels.type(self.visual.dtype)
        position_ids = torch.cat(
            [torch.arange(n) % (h * w) for n, (_, h, w) in zip(lengths, grids)]
        ).to(pixels.device, non_blocking=True)
        # Separate varlen segments prevent attention between distinct images.
        cu_seqlens = torch.from_numpy(np.cumsum([0, *lengths], dtype=np.int32))
        vision = self.visual(
            pixel_values=pixels,
            image_grid_thw=grids,
            position_ids=position_ids,
            interpolate_pos_encoding=True,
            cu_seqlens=cu_seqlens,
        ).squeeze(0)
        return self.mlp_AR(vision.split(lengths), image_input.image_grid_thw)
