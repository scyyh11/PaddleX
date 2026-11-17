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

"""Opt-in batched vision bridge for the unmodified native vLLM 0.19 target.

PaddleX develop_zjy's legacy model contains a batched image encoder, but its
registration is skipped once vLLM provides the integrated model. The legacy
class uses older multimodal/logits APIs. This bridge keeps the current native
model and processor, batches its existing variable-length vision encoder, then
splits its output before the unchanged native per-image projector.

Both benchmark modes must use the same setting. A class or vision entry point
already customized by another plugin takes precedence and is left untouched.
"""

import ast
import hashlib
import inspect
import math
import textwrap

NATIVE_MODULE = "vllm.model_executor.models.paddleocr_vl"
NATIVE_CLASS = "PaddleOCRVLForConditionalGeneration"
# Normalized AST hashes of these methods in upstream vLLM v0.19.0. These are
# dispatch guards, not dependency/build pins: a vendor/plugin customization is
# preserved instead of being overwritten. Whitespace and comments are ignored.
NATIVE_METHOD_HASHES = {
    "__init__": "09f3784e707df424ae8c22619254a4fe07c99c4408a488ac577dd8bb5b0d9601",
    "_process_image_input": "44ef34677af3aac30882ac488dd243f6ca7b87435dbd95430545254010d4dbf9",
    "encode_image": "4ce254d7244276df113453e23c98921e5fb65cdf4d74fd09294d987b7e37064f",
    "embed_multimodal": "9133c18a8f0d6dd28fe5c3277ad0b8edbc750fcc582163ca5b40001e67081312",
}


def _method_ast_hash(method):
    if method is None or hasattr(method, "__wrapped__"):
        return None
    if getattr(method, "__module__", None) != NATIVE_MODULE:
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    except (TypeError, OSError, SyntaxError):
        return None
    if len(tree.body) != 1:
        return None
    return hashlib.sha256(
        ast.dump(tree.body[0], include_attributes=False).encode()
    ).hexdigest()


def process_image_input_batch(self, image_input):
    """Run one native varlen vision call and preserve native projector inputs."""
    import torch

    pixel_values = image_input.pixel_values
    grids = image_input.image_grid_thw
    if not isinstance(grids, torch.Tensor):
        grids = torch.stack(tuple(grids))
    if grids.ndim != 2 or grids.shape[1] != 3 or len(pixel_values) != len(grids):
        raise ValueError("Image pixels and [N, 3] grids must have the same image count")
    if len(grids) == 0:
        return ()

    # One transfer of all grid metadata replaces the original per-image syncs.
    grid_tuples = [tuple(grid) for grid in grids.tolist()]
    lengths = [math.prod(grid) for grid in grid_tuples]
    pixels = [pixel.reshape(-1, *pixel.shape[-3:]) for pixel in pixel_values]
    if any(
        length <= 0 or len(pixel) != length for pixel, length in zip(pixels, lengths)
    ):
        raise ValueError("Each image grid must match its positive patch count")
    packed_pixels = torch.cat(pixels, dim=0).type(self.visual.dtype)

    image_positions = torch.cat(
        [
            torch.arange(length) % (grid[1] * grid[2])
            for grid, length in zip(grid_tuples, lengths)
        ]
    ).to(packed_pixels.device)
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    cu_seqlens = torch.tensor(
        cumulative, dtype=torch.int32, device=packed_pixels.device
    )

    vision_outputs = self.visual(
        pixel_values=packed_pixels,
        image_grid_thw=grid_tuples,
        position_ids=image_positions,
        interpolate_pos_encoding=True,
        cu_seqlens=cu_seqlens,
    )
    # Native 0.19 returns [1, total_patches, vision_hidden]. Its projector's
    # list branch performs each image's 2x2 spatial merge and preserves output
    # boundaries. Do not use the projector's differently shaped tensor branch.
    per_image = vision_outputs.squeeze(0).split(lengths, dim=0)
    return self.mlp_AR(per_image, grids)


def adapt_target_class(target_cls):
    if target_cls.__dict__.get("_paddlex_hps_batch_vision", False):
        return dict(applied=True, reason="native batch vision bridge already installed")
    if (
        target_cls.__module__ != NATIVE_MODULE
        or target_cls.__qualname__ != NATIVE_CLASS
    ):
        return dict(applied=False, reason="custom registered model class preserved")
    for name, expected in NATIVE_METHOD_HASHES.items():
        if _method_ast_hash(getattr(target_cls, name, None)) != expected:
            return dict(
                applied=False, reason=f"custom or unrecognized native {name} preserved"
            )
    target_cls._process_image_input = process_image_input_batch
    target_cls._paddlex_hps_batch_vision = True
    return dict(applied=True, reason="native variable-length vision encoder batched")
