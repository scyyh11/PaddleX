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

"""Document-level post-processing orchestrator for multi-page PaddleOCR-VL results."""

import concurrent.futures
import importlib
import json
import os
import pathlib
import re
import threading
import time
import warnings
from collections.abc import Mapping

import numpy as np

from .....utils import logging
from . import features, models
from .config import DocPostprocessConfig
from .result import DocPostprocessResult

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
THREADS_ENV_VAR = "PADDLEOCR_DOC_POSTPROCESS_THREADS"
_IMAGE_REFERENCE = re.compile(r"page(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.jpg")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def _parse_device(device):
    """Return ``('cuda', index)`` or ``('cpu', None)``."""
    if device is None or device == "auto":
        try:
            cupy = importlib.import_module("cupy")
            if cupy.cuda.runtime.getDeviceCount() > 0:
                return "cuda", 0
        except Exception:
            pass
        return "cpu", None
    name, _, index = str(device).lower().partition(":")
    if name == "cpu":
        return "cpu", None
    if name in ("gpu", "cuda"):
        return "cuda", int(index or 0)
    raise ValueError(f"Unsupported device for document post-processing: {device!r}")


def _default_threads():
    if os.environ.get(THREADS_ENV_VAR):
        return max(1, int(os.environ[THREADS_ENV_VAR]))
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(4, available))


class _GpuContext:
    """Activate CUDA device ``index`` while the engine runs (no-op on CPU)."""

    def __init__(self, device, index):
        self.context = None
        if device == "cuda":
            cupy = importlib.import_module("cupy")
            self.context = cupy.cuda.Device(index or 0)

    def __enter__(self):
        if self.context is not None:
            self.context.__enter__()

    def __exit__(self, *exc):
        if self.context is not None:
            self.context.__exit__(*exc)


def _is_paddlex_result(value):
    return hasattr(value, "_to_json") and hasattr(value, "json")


def _normalized_box(bbox, width, height):
    return tuple(
        max(0, min(1000, round(float(v) / size * 1000)))
        for v, size in zip(bbox, [width, height, width, height])
    )


def _block_crops(result, width, height):
    """Collect ``{normalized_box: BGR_crop}`` from image blocks of a page result."""
    crops = {}
    for block in result.get("parsing_res_list") or []:
        image = getattr(block, "image", None)
        if not isinstance(image, Mapping) or image.get("img") is None:
            continue
        array = np.asarray(image["img"])
        if array.ndim == 3 and array.shape[2] >= 3:
            array = np.ascontiguousarray(array[:, :, 2::-1])
        crops[_normalized_box(block.bbox, width, height)] = array
    return crops


def _page_input(result):
    """Normalize one page result into ``(plain_dict, page_image_or_None, crops)``."""
    image = None
    crops = {}
    if _is_paddlex_result(result):
        settings = result.get("model_settings") or {}
        if settings.get("format_block_content") and isinstance(result, dict):
            unformatted = type(result)(
                {
                    **result,
                    "model_settings": dict(settings, format_block_content=False),
                }
            )
            data = unformatted.json["res"]
        else:
            data = result.json["res"]
        preprocessed = result.get("doc_preprocessor_res")
        if preprocessed is not None:
            try:
                image = preprocessed["output_img"]
            except (KeyError, TypeError):
                image = None
        crops = _block_crops(result, data["width"], data["height"])
    elif isinstance(result, Mapping):
        data = result.get("res", result)
    else:
        raise TypeError(f"Unsupported page result type: {type(result).__name__}")
    return json.loads(json.dumps(data)), image, crops


class _PageImages:
    """Lazy grayscale cache of page images for a single document."""

    def __init__(self, sources):
        self.sources = sources
        self.gray = {}

    def color(self, index):
        source = self.sources[index]
        if isinstance(source, (str, os.PathLike)):
            import cv2

            return cv2.imread(str(source), cv2.IMREAD_COLOR)
        return source

    def __call__(self, index):
        if index not in self.gray:
            source = self.sources[index]
            if source is None:
                gray = None
            elif isinstance(source, (str, os.PathLike)):
                import cv2

                gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            else:
                gray = features.to_gray(source)
            self.gray[index] = gray
        return self.gray[index]


def _image_source(page, image):
    if image is not None:
        return image
    path = page.get("input_path")
    if isinstance(path, str) and pathlib.Path(path).suffix.lower() in _IMAGE_SUFFIXES:
        if pathlib.Path(path).is_file():
            return path
    return None


def default_doc_name(pages):
    for page in pages:
        path = page.get("input_path")
        if isinstance(path, str) and path:
            return pathlib.Path(path).stem
    return "document"


class DocPostprocessor:
    """Multi-page document post-processor for PaddleOCR-VL results.

    Each instance manages its own model store, configuration, and thread pool
    without mutating process-global environment variables or garbage collector state.
    Document calls (``process`` / ``process_pages``) on a single instance are serialized
    by an internal lock while ``num_threads`` parallelizes intra-document feature
    extraction; create separate instances for concurrent multi-document throughput.
    """

    def __init__(
        self,
        models_dir=None,
        device="cpu",
        num_threads=None,
        config=None,
    ):
        self.models_dir = models.find_models_dir(models_dir)
        self.device, self.gpu_index = _parse_device(device)
        self.num_threads = (
            _default_threads() if num_threads is None else max(1, int(num_threads))
        )
        self.config = config if config is not None else DocPostprocessConfig()
        self._lock = threading.RLock()
        self._executor = None
        self._model_store = None
        self._default_fold = 0
        self._engines = {}

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _feature_executor(self):
        if self.num_threads <= 1:
            return None
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_threads, thread_name_prefix="doc-postprocess"
            )
        return self._executor

    def available(self):
        """True when the model bundle at ``self.models_dir`` is complete."""
        return not models.problems(self.models_dir)

    def warmup(self):
        """Validate models and warm up the engine."""
        self._ensure_engine()

    def _ensure_store(self):
        if self._model_store is None:
            models.check(self.models_dir)
            settings = models.read_bundle(self.models_dir)
            self._default_fold = settings.get("default_fold", 0)
            self._model_store = models.ModelStore(self.models_dir, device=self.device)
        return self._model_store

    def _build_engine(self, device, index):
        from .engine import runtime_context, unified

        store = self._ensure_store()
        if device == "cuda" and store.device != "cuda":
            store = models.ModelStore(self.models_dir, device="cuda")
            self._model_store = store

        engine = self._engines.get(device)
        if engine is not None:
            return engine

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names"
            )
            store_tok = runtime_context.model_store.set(store)
            fold_tok = runtime_context.default_fold.set(self._default_fold)
            cfg_tok = runtime_context.config.set(self.config)
            try:
                engine = unified.DocumentPostprocessor(device=device)
                started = time.perf_counter()
                with _GpuContext(device, index):
                    engine.warmup()
            finally:
                runtime_context.config.reset(cfg_tok)
                runtime_context.default_fold.reset(fold_tok)
                runtime_context.model_store.reset(store_tok)

        logging.info(
            "Document post-processing engine ready on %s (%.1f s).",
            device if index is None else f"{device}:{index}",
            time.perf_counter() - started,
        )
        self._engines[device] = engine
        return engine

    def _ensure_engine(self):
        with self._lock:
            self._ensure_store()
            if self.device == "cuda":
                try:
                    return self._build_engine("cuda", self.gpu_index), "cuda"
                except Exception as error:
                    logging.warning(
                        "Document post-processing cannot use the GPU (%s); falling back to CPU.",
                        error,
                    )
                    self.device, self.gpu_index = "cpu", None
                    self._model_store = models.ModelStore(self.models_dir, device="cpu")
            return self._build_engine("cpu", None), "cpu"

    @staticmethod
    def measure_page(page, gray):
        """Measure heading typography features for one page and grayscale image."""
        return features.measure_page_headings(page, gray)

    def __call__(self, results, *, doc_name=None, images=None):
        return self.process(results, doc_name=doc_name, images=images)

    def process(self, results, *, doc_name=None, images=None):
        """Post-process the page results of one document in page order."""
        started = time.perf_counter()
        results = list(results)
        if images is not None and len(images) != len(results):
            raise ValueError("`images` must have one entry per page result")
        pages, sources, crops = [], [], []
        for i, result in enumerate(results):
            page, image, block_crops = _page_input(result)
            if images is not None and images[i] is not None:
                image = images[i]
            pages.append(page)
            sources.append(_image_source(page, image))
            crops.append(block_crops)
        prepared_ms = (time.perf_counter() - started) * 1000
        return self.process_pages(
            pages,
            sources,
            doc_name=doc_name,
            block_crops=crops,
            prepared_ms=prepared_ms,
        )

    def process_pages(
        self,
        pages,
        images,
        *,
        doc_name=None,
        block_crops=None,
        heading_measurements=None,
        prepared_ms=0.0,
    ):
        """Post-process plain page dicts with aligned page images or paths."""
        from .engine import runtime_context

        if len(images) != len(pages):
            raise ValueError("`images` must have one entry per page")
        if heading_measurements is not None and len(heading_measurements) != len(pages):
            raise ValueError("`heading_measurements` must have one entry per page")
        if not pages:
            raise ValueError("no pages to post-process")

        started = time.perf_counter()
        display_name = doc_name or default_doc_name(pages)
        doc_id = _SAFE_NAME.sub("_", display_name or "document")[:200] or "document"
        page_images = _PageImages(images)
        if all(source is None for source in images):
            logging.warning(
                "No page images for %s: document post-processing runs without visual features.",
                display_name,
            )

        with self._lock:
            engine, device = self._ensure_engine()
            feature_started = time.perf_counter()
            executor = self._feature_executor()
            measurements = features.Measurements()
            heading_visual = {
                "features": features.heading_features(
                    pages, page_images, measurements, executor, heading_measurements
                )
            }
            heading_feature_ms = (time.perf_counter() - feature_started) * 1000
            text_visual, label_visual = {}, {}

            def measure_prepared(prepared):
                features.line_features(
                    prepared,
                    page_images,
                    text_visual,
                    label_visual,
                    measurements,
                    executor,
                )

            store_tok = runtime_context.model_store.set(self._model_store)
            fold_tok = runtime_context.default_fold.set(self._default_fold)
            cfg_tok = runtime_context.config.set(self.config)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="X does not have valid feature names"
                    )
                    with _GpuContext(device, self.gpu_index):
                        output = engine.process(
                            doc_id,
                            pages=pages,
                            heading_visual=heading_visual,
                            text_visual=text_visual,
                            label_visual=label_visual,
                            prepared_hook=measure_prepared,
                        )
            finally:
                runtime_context.config.reset(cfg_tok)
                runtime_context.default_fold.reset(fold_tok)
                runtime_context.model_store.reset(store_tok)

        for page, source in zip(output["pages"], pages):
            for key in ("input_path", "page_index", "page_count"):
                if key in source:
                    page[key] = source[key]
        markdown = output["markdown"]
        crops = _image_crops(markdown, len(pages), page_images, block_crops)
        timings = dict(output["timings"])
        timings["input_ms"] = prepared_ms
        timings["heading_features_ms"] = heading_feature_ms
        timings["document_ms"] = prepared_ms + (time.perf_counter() - started) * 1000
        return DocPostprocessResult(
            doc_name=display_name,
            input_path=pages[0].get("input_path"),
            page_count=len(pages),
            markdown_text=markdown,
            pages=output["pages"],
            images=crops,
            events=output["events"],
            timings=timings,
            device=device,
        )


def _image_crops(markdown, page_count, page_images, block_crops=None):
    """Extract BGR image crops for ``page<n>_<x1>_<y1>_<x2>_<y2>.jpg`` references."""
    crops = {}
    for match in _IMAGE_REFERENCE.finditer(markdown):
        name = match.group(0)
        if name in crops:
            continue
        pi = int(match.group(1)) - 1
        if not 0 <= pi < page_count:
            continue
        x1, y1, x2, y2 = (int(v) for v in match.groups()[1:])
        if block_crops and block_crops[pi] and (x1, y1, x2, y2) in block_crops[pi]:
            crops[name] = block_crops[pi][(x1, y1, x2, y2)]
            continue
        image = page_images.color(pi)
        if image is None or not hasattr(image, "shape") or image.ndim < 2:
            continue
        h, w = image.shape[:2]
        box = (
            int(round(x1 / 1000 * w)),
            int(round(y1 / 1000 * h)),
            int(round(x2 / 1000 * w)),
            int(round(y2 / 1000 * h)),
        )
        crop = image[box[1] : box[3], box[0] : box[2]]
        if crop.size:
            crops[name] = np.ascontiguousarray(
                crop if crop.ndim == 3 else np.dstack([crop] * 3)
            )
    return crops
