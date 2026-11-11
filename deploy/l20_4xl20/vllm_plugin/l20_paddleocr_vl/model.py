"""Upstream vLLM PaddleOCR-VL with the two optimizations from the live PaddleX backend.

1. PaddleX #5229: encode all images of a scheduler step in one vision call
   instead of one call per image (attention stays per image via cu_seqlens).
2. PaddleX 878b044: take max_seqlen and the RoPE table size from host-side grid
   sizes instead of GPU reductions followed by .item() in every encoder layer.

Unlike PaddleX on vLLM 0.10.2, vLLM 0.19 profiles startup memory with up to 104
maximum-size images in one step; encoding them in a single call leaves no memory
for the KV cache. Images are therefore grouped into vision calls of at most
L20_PADDLEOCR_VL_MAX_PATCHES patches (default 32768, about 6 maximum-size images).
"""
import os

import numpy as np
import torch

from vllm.model_executor.models import paddleocr_vl as upstream
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MAX_PATCHES = int(os.environ.get("L20_PADDLEOCR_VL_MAX_PATCHES", "32768"))
# Copy host-built position ids / cu_seqlens through pinned memory without blocking.
# A pageable .to(device) waits for all queued GPU work (the previous step), which
# stalls the engine loop; py-spy showed ~26% of EngineCore time on that line.
ASYNC_H2D = os.environ.get("L20_PADDLEOCR_VL_ASYNC_H2D") == "1"


# Optional GPU timing of vision encoding (CUDA events, no host sync on the hot path).
TIMING = os.environ.get("L20_PADDLEOCR_VL_TIMING") == "1"


class _VisionTimer:
    def __init__(self, interval_s=10.0):
        import collections
        import time

        self._time = time
        self._pending = collections.deque()
        self._interval = interval_s
        self._window_start = time.monotonic()
        self._gpu_ms = 0.0
        self._calls = 0
        self._images = 0
        self._patches = 0

    def start(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, start, images, patches):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending.append((start, end, images, patches))
        while self._pending and self._pending[0][1].query():
            s, e, n, p = self._pending.popleft()
            self._gpu_ms += s.elapsed_time(e)
            self._calls += 1
            self._images += n
            self._patches += p
        now = self._time.monotonic()
        if now - self._window_start >= self._interval:
            wall_ms = (now - self._window_start) * 1000
            print(f"[l20_paddleocr_vl] vision_gpu_share={self._gpu_ms / wall_ms:.3f} "
                  f"vision_gpu_ms={self._gpu_ms:.0f} wall_ms={wall_ms:.0f} calls={self._calls} "
                  f"images={self._images} patches={self._patches}", flush=True)
            self._window_start, self._gpu_ms, self._calls, self._images, self._patches = now, 0.0, 0, 0, 0


_TIMER = _VisionTimer() if TIMING else None


def _to_device(tensor, device):
    if ASYNC_H2D:
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)

_FA_BACKENDS = {
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.ROCM_AITER_FA,
    AttentionBackendEnum.TRITON_ATTN,
}


def _encoder_forward(
    self,
    inputs_embeds,
    cu_seqlens=None,
    image_grid_thw=None,
    height_position_ids=None,
    width_position_ids=None,
):
    device = inputs_embeds.device
    flatten_image_grid_thw = self.flatten_list(image_grid_thw)

    max_grid_size = None
    if width_position_ids is None or height_position_ids is None:
        split_hids = list()
        split_wids = list()
        for t, h, w in flatten_image_grid_thw:
            image_pids = torch.arange(t * h * w, device=device) % (h * w)
            split_hids.append(image_pids // w)
            split_wids.append(image_pids % w)
        width_position_ids = torch.concat(split_wids, dim=0)
        height_position_ids = torch.concat(split_hids, dim=0)
        # Equals pids.max() + 1 for these ids, computed without a GPU reduction.
        max_grid_size = max(max(h, w) for _, h, w in flatten_image_grid_thw)

    pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
    if max_grid_size is None:
        max_grid_size = pids.max() + 1
    rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)
    rotary_pos_emb = rope_emb_max_grid[pids].flatten(1)

    if cu_seqlens is None:
        raise ValueError("cu_seqlens cannot be None for SiglipEncoder.")
    if not isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    else:
        cu_seqlens = cu_seqlens.to(device=device)

    max_seqlen = None
    if self.attn_backend in _FA_BACKENDS:
        # CPU tensor: the attention wrapper calls .item() per layer, which is
        # then a host read instead of a device synchronization.
        max_seqlen = torch.tensor(
            max(t * h * w for t, h, w in flatten_image_grid_thw), dtype=torch.int32
        )

    hidden_states = inputs_embeds
    for encoder_layer in self.layers:
        hidden_states = encoder_layer(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
        )
    return hidden_states


upstream.SiglipEncoder.forward = _encoder_forward

# Optional torch.compile of each ViT encoder layer (vLLM's mm-encoder compile path,
# as used for Qwen2.5-VL vision blocks). Needs L20_PADDLEOCR_VL_COMPILE_VIT=1 and
# --compilation-config '{"compile_mm_encoder": true}'.
if os.environ.get("L20_PADDLEOCR_VL_COMPILE_VIT") == "1":
    from vllm.compilation.decorators import (
        should_torch_compile_mm_encoder,
        support_torch_compile,
    )

    @support_torch_compile(
        dynamic_arg_dims={"hidden_states": 1, "cu_seqlens": 0, "rotary_pos_emb": 0},
        enable_if=should_torch_compile_mm_encoder,
        is_encoder=True,
    )
    class CompiledSiglipEncoderLayer(upstream.SiglipEncoderLayer):
        pass

    # SiglipEncoder.__init__ looks this name up at construction time.
    upstream.SiglipEncoderLayer = CompiledSiglipEncoderLayer


class BatchedPaddleOCRVLForConditionalGeneration(
    upstream.PaddleOCRVLForConditionalGeneration
):
    def _process_image_input(self, image_input):
        pixel_values = image_input.pixel_values
        image_grid_thw = image_input.image_grid_thw
        pvs = [pv.reshape(-1, *pv.shape[-3:]) for pv in pixel_values]
        if not pvs:
            return []
        if isinstance(image_grid_thw, torch.Tensor):
            grid_rows = image_grid_thw.reshape(-1, 3).tolist()
        else:
            grid_rows = torch.cat([g.reshape(-1, 3) for g in image_grid_thw]).tolist()
        # One host transfer for all grid sizes of this step.
        grids = [tuple(int(x) for x in row) for row in grid_rows]

        numels = [int(np.prod(g)) for g in grids]
        timing_start = _TIMER.start() if _TIMER else None
        groups = []
        for i, numel in enumerate(numels):
            if groups and groups[-1][1] + numel <= MAX_PATCHES:
                groups[-1][0].append(i)
                groups[-1][1] += numel
            else:
                groups.append([[i], numel])

        features = []
        for indices, _ in groups:
            features.extend(
                self._encode_group(
                    [pvs[i] for i in indices],
                    [grids[i] for i in indices],
                    [numels[i] for i in indices],
                )
            )
        image_embeds = self.mlp_AR(features, grids)
        if _TIMER:
            _TIMER.stop(timing_start, len(grids), sum(numels))
        return image_embeds

    def _encode_group(self, pvs, grids, numels):
        pixel = torch.cat(pvs, dim=0).type(self.visual.dtype)
        cu_seqlens = [0]
        for numel in numels:
            cu_seqlens.append(cu_seqlens[-1] + numel)
        position_ids = _to_device(
            torch.concat(
                [torch.arange(n) % (h * w) for n, (_, h, w) in zip(numels, grids)]
            ),
            pixel.device,
        )
        cu_seqlens = _to_device(torch.tensor(cu_seqlens, dtype=torch.int32), pixel.device)

        vision_outputs = self.visual(
            pixel_values=pixel,
            image_grid_thw=grids,
            position_ids=position_ids,
            interpolate_pos_encoding=True,
            cu_seqlens=cu_seqlens,
        )
        return torch.split(vision_outputs.squeeze(0), numels, dim=0)
