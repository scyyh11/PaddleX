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

"""Whole-page PP-DocLayout detection, EAGLE3 region recognition and PaddleX page assembly."""
from __future__ import annotations

import asyncio
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import warnings

import httpx
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
PAGE_IMAGE_LIMIT = 32 * 1024 * 1024
PAGE_PIXEL_LIMIT = 160_000_000
PAGE_RECOVERY_SECONDS = 120
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
LOG = logging.getLogger(__name__)
LAYOUT_MODELS = ("PP-DocLayoutV3",)
PAGE_PIPELINES = {"1.6": "PaddleOCR-VL-1.6"}
PAGE_VLM_URLS = {"1.6": "http://127.0.0.1:18780/v1"}
V16_TARGET = (
    Path(os.getenv("OCR_EAGLE_TARGET", str(ROOT / "models/PaddleOCR-VL-1.6")))
    .expanduser()
    .resolve()
)
V16_DRAFT = (
    Path(
        os.getenv(
            "OCR_EAGLE_DRAFT", str(ROOT / "models/PaddleOCR-VL-1.6-EAGLE3-K4-dense")
        )
    )
    .expanduser()
    .resolve()
)
TEXT_LABELS = {
    "abstract",
    "algorithm",
    "aside_text",
    "content",
    "doc_title",
    "figure_title",
    "footer",
    "footnote",
    "formula_number",
    "header",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "text",
    "vertical_text",
    "vision_footnote",
}
LABELS = {
    **dict.fromkeys(TEXT_LABELS, "text"),
    **dict.fromkeys(("display_formula", "inline_formula"), "formula"),
    **dict.fromkeys(("image", "header_image", "footer_image"), "figure"),
    "table": "table",
    "chart": "chart",
    "seal": "seal",
}


def configure_layout(config, layout_model="PP-DocLayoutV3", cpu_threads=4):
    """Configure the page detector while preserving the selected recognition pipeline."""
    from paddlex.inference.pipelines import load_pipeline_config

    if layout_model not in LAYOUT_MODELS:
        raise ValueError(f"Unsupported layout model: {layout_model}")
    modules = config["SubModules"]
    layout = modules["LayoutDetection"]
    if layout["model_name"] != layout_model:
        # V3's official pipeline uses one confidence threshold for all classes.
        # Take only its detector settings so recognition keeps its original preset.
        preset = (
            "PaddleOCR-VL-1.5" if layout_model == "PP-DocLayoutV3" else "PaddleOCR-VL"
        )
        layout = load_pipeline_config(preset)["SubModules"]["LayoutDetection"]
        if layout["model_name"] != layout_model:
            raise ValueError(
                f"Expected {layout_model} in installed {preset} layout preset"
            )
        modules["LayoutDetection"] = layout
    layout_dir = (
        Path(
            os.getenv(
                "OCR_LAYOUT_MODEL_DIR",
                str(ROOT / "models/hps_layout" / f"{layout_model}_infer"),
            )
        )
        .expanduser()
        .resolve()
    )
    layout.update(
        batch_size=1,
        model_dir=str(layout_dir),
        engine_config={"run_mode": "paddle", "cpu_threads": cpu_threads},
    )


def _read_image(data):
    import cv2
    import numpy as np

    if len(data) > PAGE_IMAGE_LIMIT:
        raise ValueError("Page image exceeds 32 MiB")
    # Pillow warns at ~89 MP and rejects above ~179 MP. Valid benchmark
    # pages reach 145.5 MP. Enforce our explicit lower hard bound before
    # decoding, including when the process promotes warnings to errors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        source = Image.open(
            BytesIO(data), formats=("PNG", "JPEG", "WEBP", "BMP", "TIFF")
        )
    with source:
        # Pillow reports an already oriented size for TIFF. The TIFF width
        # and length tags still describe the original raster used for boxes.
        size = (
            (source.tag_v2[256], source.tag_v2[257])
            if source.format == "TIFF"
            else source.size
        )
        if size[0] * size[1] > PAGE_PIXEL_LIMIT or getattr(source, "n_frames", 1) != 1:
            raise ValueError("Expected a single page image up to 160 megapixels")
        orientation = source.getexif().get(274, 1)
        if orientation not in range(1, 9):
            raise ValueError("Invalid EXIF orientation")
        if orientation == 1:
            # Same BGR decode used by the official PaddleX image reader.
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            upright = ImageOps.exif_transpose(source).convert("RGB")
            image = cv2.cvtColor(np.asarray(upright), cv2.COLOR_RGB2BGR)
    expected = size[::-1] if orientation >= 5 else size
    if image is None or image.shape[:2][::-1] != expected:
        raise ValueError("Could not decode page with the expected dimensions")
    return image, size, orientation


def _source_bbox(box, size, orientation):
    # Predictions are on the upright image. Report edge coordinates on the input raster.
    x1, y1, x2, y2 = box
    width, height = size
    if orientation >= 5:
        x1, y1, x2, y2 = y1, x1, y2, x2
    if orientation in (2, 3, 7, 8):
        x1, x2 = width - x2, width - x1
    if orientation in (3, 4, 6, 7):
        y1, y2 = height - y2, height - y1
    return [x1, y1, x2, y2]


def _platform_result(result, filename, size, orientation):
    doc = result.json["res"]
    expected = size[::-1] if orientation >= 5 else size
    if (doc["width"], doc["height"]) != expected or not doc["model_settings"][
        "use_layout_detection"
    ]:
        raise ValueError("Unexpected layout pipeline output dimensions or settings")
    blocks = []
    # PaddleX uses V3 polygons when cropping. The platform takes source-raster
    # boxes; keep the assembled list in its official reading order, including
    # headers, footers and figures for which PaddleX can leave block_order unset.
    for index, block in enumerate(doc["parsing_res_list"]):
        blocks.append(
            {
                "block_label": LABELS[block["block_label"]],
                "block_content": block["block_content"],
                "block_bbox": _source_bbox(block["block_bbox"], size, orientation),
                "block_id": index,
                "block_order": index,
                "group_id": block["group_id"],
            }
        )
    return {
        "md": result.markdown["markdown_texts"],
        "json": {
            "input_path": filename,
            "width": size[0],
            "height": size[1],
            "model_settings": doc["model_settings"],
            "parsing_res_list": blocks,
        },
    }


class PageBackendUnavailable(RuntimeError):
    """The active task cannot continue until a compatible local backend is ready."""


class PageCropClient(httpx.AsyncClient):
    """Preserve PaddleX's exact crop and prompt when retrying a length-limited answer."""

    completion_limit = 1024
    incomplete = None
    length_retries = 0

    async def send(self, request, **kwargs):
        response = await super().send(request, **kwargs)
        if request.url.path != "/v1/chat/completions" or not response.is_success:
            return response
        await response.aread()
        (choice,) = response.json()["choices"]
        reason = choice.get("finish_reason")
        if reason == "length" and self.completion_limit > 1024:
            payload = json.loads(await request.aread())
            fields = [
                f for f in ("max_completion_tokens", "max_tokens") if f in payload
            ]
            for field in fields or ["max_completion_tokens"]:
                payload[field] = self.completion_limit
            # Keep exactly the caller's request headers and timeouts, with
            # a fresh Content-Length for the one changed numeric field.
            headers = request.headers.copy()
            headers.pop("content-length", None)
            retry = self.build_request(
                request.method,
                request.url,
                headers=headers,
                json=payload,
                extensions=request.extensions.copy(),
            )
            await response.aclose()
            self.length_retries += 1
            response = await super().send(retry, **kwargs)
            if not response.is_success:
                return response
            await response.aread()
            (choice,) = response.json()["choices"]
            reason = choice.get("finish_reason")
        if reason != "stop":
            self.incomplete = f"Incomplete VLM crop (finish_reason={reason!r})"
        return response


def _retryable_crop_error(exc):
    from openai import APIConnectionError

    status = getattr(getattr(exc, "response", None), "status_code", None)
    return (
        isinstance(exc, (httpx.TransportError, APIConnectionError))
        or status in RETRYABLE_STATUS
    )


class PageRecognizer:
    """One resident layout predictor and one serial connection to the EAGLE3 VLM."""

    def __init__(
        self,
        vlm_url=None,
        device="gpu:0",
        layout_model="PP-DocLayoutV3",
        vlm_version="1.6",
    ):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddlex import create_pipeline
        from paddlex.inference.pipelines import load_pipeline_config

        if vlm_version not in PAGE_PIPELINES:
            raise ValueError(f"Unsupported page VLM version: {vlm_version}")
        self.vlm_version = vlm_version
        config = load_pipeline_config(PAGE_PIPELINES[vlm_version])
        recognition = config["SubModules"]["VLRecognition"]
        self.wire_model = recognition["model_name"]
        url = urlsplit(vlm_url or PAGE_VLM_URLS[vlm_version])
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or url.username is not None
            or url.query
            or url.fragment
            or url.path.rstrip("/") not in ("", "/v1")
        ):
            raise ValueError("Expected the local EAGLE3 HPS HTTP endpoint")
        self.origin = f"http://{url.netloc}"
        self._profile = None
        self._suspended_instance = None
        self._verified_model_instance = None
        self._backend_failure_pages = 0
        self.instance_id = self.wait_for_backend()
        config.update(batch_size=1, use_queues=False)
        configure_layout(config, layout_model)
        self._http = http = PageCropClient(trust_env=False)
        recognition.update(
            batch_size=1,
            genai_config={
                "backend": "vllm-server",
                "server_url": self.origin + "/v1",
                "max_concurrency": 1,
                "client_kwargs": {
                    "model_name": self.wire_model,
                    "max_retries": 0,
                    "http_client": http,
                },
            },
        )
        try:
            self.pipeline = create_pipeline(config=config, device=device)
        except Exception:
            asyncio.run(http.aclose())
            raise

    def _ready_instance(self):
        with httpx.Client(trust_env=False, timeout=5) as client:
            response = client.get(self.origin + "/health/ready")
            response.raise_for_status()
            health = response.json()
        if not isinstance(health, dict):
            raise PageBackendUnavailable("Invalid EAGLE3 readiness response")
        mode = health.get("mode")
        if mode == "eagle3-recovery":
            evidence = health.get("speculative_decoding")
            worker = evidence.get("worker") if isinstance(evidence, dict) else None
            if (
                not isinstance(worker, dict)
                or worker.get("method") != "eagle3"
                or worker.get("target") != "PaddleOCRVLForConditionalGeneration"
            ):
                raise PageBackendUnavailable(
                    "Recovery backend must attest to the PaddleOCR-VL EAGLE3 worker"
                )
        if (
            health.get("status") != "ready"
            or mode not in ("accelerated", "eagle3-recovery")
            or not isinstance(health.get("instance_id"), str)
            or not health["instance_id"]
        ):
            raise PageBackendUnavailable(
                "EAGLE3 VLM is unavailable or in the wrong mode"
            )
        if self.vlm_version == "1.6":
            if (
                mode != "eagle3-recovery"
                or health.get("model_path") != str(V16_TARGET)
                or health.get("draft_model_path") != str(V16_DRAFT)
                or worker.get("checkpoint") != str(V16_DRAFT)
                or worker.get("k") != 4
            ):
                raise PageBackendUnavailable(
                    "Expected the VL-1.6 target and its frozen EAGLE3 K4 dense draft"
                )
            if self._verified_model_instance != health["instance_id"]:
                with httpx.Client(trust_env=False, timeout=5) as client:
                    models = client.get(self.origin + "/v1/models")
                    models.raise_for_status()
                    entries = models.json()["data"]
                if not isinstance(entries, list) or [
                    row.get("id") for row in entries
                ] != [self.wire_model]:
                    raise PageBackendUnavailable(
                        f"Local backend must serve exactly {self.wire_model}"
                    )
                self._verified_model_instance = health["instance_id"]
        # Ignore telemetry and server capacity fields when a fresh process
        # resumes a task. Its target, drafter and arithmetic must still match.
        profile = (
            mode,
            (
                {
                    key: worker.get(key)
                    for key in (
                        "method",
                        "speculator",
                        "target",
                        "draft",
                        "checkpoint",
                        "draft_head_dtype",
                        "k",
                        "aux_layers",
                        "eager",
                        "async_scheduling",
                        "cudagraph_mode",
                        "draft_cudagraph_mode",
                        "capture_sizes",
                        "draft_head_execution",
                        "target_attention",
                        "runtime_profile",
                        "target_projection",
                        "attention_metadata",
                        "draft_execution",
                        "target_sampler",
                        "spec_metadata",
                        "max_model_len",
                    )
                }
                if mode == "eagle3-recovery"
                else {}
            ),
        )
        if self._profile is not None and self._profile != profile:
            raise PageBackendUnavailable(
                "Restored EAGLE3 backend has a different model or execution profile"
            )
        limit = health.get("max_completion_tokens", 1024)
        if type(limit) is not int or limit < 1024:
            raise PageBackendUnavailable(
                "Backend advertised an invalid completion limit"
            )
        self._profile = profile
        self._completion_limit = min(limit, 8192)
        return health["instance_id"]

    def wait_for_backend(self):
        deadline = time.monotonic() + PAGE_RECOVERY_SECONDS
        while True:
            try:
                instance = self._ready_instance()
                if instance != self._suspended_instance:
                    self._suspended_instance = None
                    return instance
                detail = "backend repeatedly failed crop requests; waiting for a replacement instance"
            except httpx.TransportError as exc:
                detail = type(exc).__name__
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in RETRYABLE_STATUS:
                    raise PageBackendUnavailable(
                        f"Readiness returned HTTP {exc.response.status_code}"
                    ) from exc
                detail = f"HTTP {exc.response.status_code}"
            except (ValueError, KeyError, TypeError) as exc:
                raise PageBackendUnavailable(
                    "Invalid EAGLE3 readiness response"
                ) from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PageBackendUnavailable(
                    f"EAGLE3 not ready within {PAGE_RECOVERY_SECONDS}s ({detail})"
                )
            time.sleep(min(2, remaining))

    def infer_single_image(self, image_bytes: bytes, filename: str) -> dict:
        self.instance_id = self.wait_for_backend()
        image, size, orientation = _read_image(image_bytes)
        for attempt in range(2):
            self._http.completion_limit = self._completion_limit
            self._http.incomplete = None
            self._http.length_retries = 0
            try:
                (result,) = self.pipeline.predict(
                    image,
                    max_new_tokens=1024,
                    temperature=0,
                    min_pixels=112896,
                    max_pixels=1003520,
                )
            except Exception as exc:
                if not _retryable_crop_error(exc):
                    raise
                failure = type(exc).__name__
            else:
                if self._http.incomplete:
                    raise ValueError(self._http.incomplete)
                if self.wait_for_backend() == self.instance_id:
                    self._backend_failure_pages = 0
                    if self._http.length_retries:
                        LOG.warning(
                            "Completed %s after extending %s truncated crops",
                            filename,
                            self._http.length_retries,
                        )
                    return _platform_result(result, filename, size, orientation)
                failure = "EAGLE3 instance changed during the page"
            restored = self.wait_for_backend()
            if attempt == 0:
                LOG.warning("Retrying full page %s after %s", filename, failure)
                self.instance_id = restored
                time.sleep(1)
        self._backend_failure_pages += 1
        if self._backend_failure_pages >= 3:
            self._suspended_instance = self.instance_id
            self._backend_failure_pages = 0
            raise PageBackendUnavailable(
                "Three consecutive pages exhausted EAGLE3 backend retries"
            )
        raise RuntimeError(f"Page exhausted EAGLE3 backend retries ({failure})")

    def close(self):
        self.pipeline.close()
