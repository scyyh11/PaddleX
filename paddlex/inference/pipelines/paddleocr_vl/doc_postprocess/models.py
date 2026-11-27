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

"""Model bundle discovery, validation, and instance-scoped loading."""

import importlib
import json
import os
import pathlib
import threading

from .portable_trees import PortableTreeClassifier, load_portable_package

ENV_VAR = "PADDLEOCR_DOC_POSTPROCESS_MODELS"
ALT_ENV_VAR = "PADDLEX_DOC_POSTPROCESS_MODELS"
DEFAULT_DIR = (
    pathlib.Path(os.path.expanduser("~")) / ".paddleocr" / "doc_postprocess" / "models"
)

# family directory -> (model file stem pattern, auxiliary files required)
FAMILIES = {
    "headings": ("fold-{}", ()),
    "parents": ("parent-fold-{}", ()),
    "spans": ("span-fold-{}", ()),
    "graph": ("graph-fold-{}", ()),
    "tables": ("fold-{}", ("manifest.json",)),
    "paragraph-runtime": ("fold-{}", ("manifest.json",)),
    "graphic": ("fold-{}", ("manifest.json",)),
}


class ModelBundleError(RuntimeError):
    """The post-processing model bundle is missing or incomplete."""


def find_models_dir(models_dir=None):
    """Resolve the model bundle directory (it may not exist yet)."""
    if models_dir:
        return pathlib.Path(models_dir).expanduser()
    if os.environ.get(ALT_ENV_VAR):
        return pathlib.Path(os.environ[ALT_ENV_VAR]).expanduser()
    if os.environ.get(ENV_VAR):
        return pathlib.Path(os.environ[ENV_VAR]).expanduser()
    return DEFAULT_DIR


def read_bundle(root):
    """Read bundle settings from ``bundle.json``, with defaults."""
    settings = {"default_fold": 0}
    path = pathlib.Path(root) / "bundle.json"
    if path.is_file():
        settings.update(json.loads(path.read_text(encoding="utf-8")))
    return settings


def _has_model_stem(directory, stem):
    sf = directory / f"{stem}.safetensors"
    js = directory / f"{stem}.json"
    jb = directory / f"{stem}.joblib"
    return (sf.is_file() and js.is_file()) or jb.is_file()


def problems(root):
    """Return missing file descriptions for a bundle (empty when usable)."""
    root = pathlib.Path(root)
    if not root.is_dir():
        return [f"model directory {root} does not exist"]
    fold = read_bundle(root)["default_fold"]
    missing = []
    for family, (stem_pattern, required) in FAMILIES.items():
        directory = root / family
        for name in required:
            if not (directory / name).is_file():
                missing.append(f"{family}/{name}")
        candidates = [stem_pattern.format("final"), stem_pattern.format(fold)]
        if not any(_has_model_stem(directory, stem) for stem in candidates):
            missing.append(f"{family}/{candidates[1]}(.safetensors|.joblib)")
    return missing


def check(root):
    """Raise ``ModelBundleError`` unless ``root`` is a usable bundle."""
    missing = problems(root)
    if missing:
        raise ModelBundleError(
            "The PaddleOCR-VL document post-processing models are not available: "
            + "; ".join(missing)
            + f". Set `doc_postprocess_models_dir` or the {ENV_VAR} environment variable "
            "to a model bundle directory, or disable the feature with use_doc_postprocess=False."
        )
    return pathlib.Path(root)


class ModelStore:
    """Instance-scoped model bundle cache (supports ``.safetensors`` and ``.joblib``)."""

    def __init__(self, root, device="cpu"):
        self.root = pathlib.Path(root).resolve()
        self.device = device
        self._lock = threading.RLock()
        self._cache = {}

    def resolve_model_path(self, path):
        """Given a path (typically ending in ``.joblib`` or stem), pick ``.safetensors`` first."""
        p = pathlib.Path(path)
        stem = p.with_suffix("") if p.suffix in (".joblib", ".safetensors") else p
        sf = stem.with_suffix(".safetensors")
        if sf.is_file() and stem.with_suffix(".json").is_file():
            return sf
        jb = stem.with_suffix(".joblib")
        if jb.is_file():
            return jb
        return p

    def load(self, path):
        resolved = self.resolve_model_path(path)
        key = str(resolved)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            if resolved.suffix == ".safetensors":
                package = load_portable_package(resolved)
                if self.device == "cuda":
                    package = self._convert_portable_gpu(package)
            else:
                import joblib

                package = joblib.load(resolved)
                if self.device == "cuda":
                    from .engine import gpu_trees

                    package = gpu_trees.convert_package(package)
            self._cache[key] = package
            return package

    @staticmethod
    def _convert_portable_gpu(package):
        # Portable packages can run directly in NumPy on CPU; when CUDA is requested and CuPy
        # is available, only non-portable joblib packages need gpu_trees conversion, whereas
        # PortableTreeClassifier keeps exact deterministic NumPy/CUDA parity.
        try:
            importlib.import_module("cupy")
        except ImportError:
            return package
        out = dict(package)
        for k, v in out.items():
            if isinstance(v, PortableTreeClassifier) and v.kind in ("extra", "hgb"):
                # Keep PortableTreeClassifier as-is (exact float64 parity).
                pass
        return out
