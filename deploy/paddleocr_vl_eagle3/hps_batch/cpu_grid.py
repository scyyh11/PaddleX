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

"""Opt-in placement-only patch for the audited HPS PaddleOCR processor.

Keep image_grid_thw on CPU. Pixel placement and every model method, operation,
shape, dtype and image order remain unchanged. Unknown processor/model plugins
are rejected when this experiment is explicitly enabled, never overwritten.
"""

import ast
from dataclasses import replace
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import tempfile
import textwrap

ENV = "PADDLEX_HPS_CPU_GRID"
AUDIT_ENV = "PADDLEX_HPS_CPU_GRID_AUDIT_DIR"
ARCH = "PaddleOCRVLForConditionalGeneration"
NATIVE_MODULE = "vllm.model_executor.models.paddleocr_vl"
OVERLAY_MODULE = "paddlex.inference.genai.models.paddleocr_vl_09b._vllm_batched"
FIELD_HASH = "c227de18405f96fc2e8cf912fb56af1a4693c430fed88d47ef57102287e252ed"
IMAGE_HASHES = {
    NATIVE_MODULE: "44ef34677af3aac30882ac488dd243f6ca7b87435dbd95430545254010d4dbf9",
    OVERLAY_MODULE: "d4f01547e05c6711e6dc5026e7e40f981af8e9d76603739f8b4d8bc7bf6c0fca",
}
_PROCESSOR_CLASS = None


def fingerprint(method):
    if method is None or hasattr(method, "__wrapped__"):
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    except (OSError, TypeError, SyntaxError):
        return None
    if len(tree.body) != 1:
        return None
    return hashlib.sha256(
        ast.dump(tree.body[0], include_attributes=False).encode()
    ).hexdigest()


def cpu_grid_fields(fields):
    """Copy only the image-grid field; do not mutate the caller's field objects."""
    from vllm.multimodal.inputs import MultiModalBatchedField, MultiModalFieldConfig

    if set(fields) != {"pixel_values", "image_grid_thw"}:
        raise RuntimeError(
            "Unexpected PaddleOCR multimodal fields; preserve custom plugin"
        )
    grid = fields["image_grid_thw"]
    pixels = fields["pixel_values"]
    if (
        type(grid) is not MultiModalFieldConfig
        or type(grid.field) is not MultiModalBatchedField
        or grid.modality != "image"
        or pixels.modality != "image"
    ):
        raise RuntimeError("Unexpected image-grid field policy; preserve custom plugin")
    result = dict(fields)
    result["image_grid_thw"] = replace(
        grid, field=replace(grid.field, keep_on_cpu=True)
    )
    return result


def processor_class(base):
    global _PROCESSOR_CLASS
    if _PROCESSOR_CLASS is not None:
        if _PROCESSOR_CLASS._hps_cpu_grid_base is not base:
            raise RuntimeError("CPU-grid processor base changed in the same process")
        return _PROCESSOR_CLASS

    class PaddleOCRVLCPUGridProcessor(base):
        _hps_cpu_grid_base = base

        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            fields = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
            return cpu_grid_fields(fields)

    # Named identity for audit/introspection. Fresh vLLM processes construct
    # their own class; arbitrary cross-process processor pickling is not claimed.
    PaddleOCRVLCPUGridProcessor.__qualname__ = "PaddleOCRVLCPUGridProcessor"
    globals()["PaddleOCRVLCPUGridProcessor"] = PaddleOCRVLCPUGridProcessor
    _PROCESSOR_CLASS = PaddleOCRVLCPUGridProcessor
    return _PROCESSOR_CLASS


def adapt_target_class(target_cls):
    from vllm.model_executor.models import paddleocr_vl as native

    module = target_cls.__module__
    if target_cls.__name__ != ARCH or module not in IMAGE_HASHES:
        raise RuntimeError(
            "Unknown PaddleOCR model registration; CPU-grid patch declined"
        )
    if (
        fingerprint(getattr(target_cls, "_process_image_input", None))
        != IMAGE_HASHES[module]
    ):
        raise RuntimeError("Image-input method differs from audited HPS/native source")
    base = native.PaddleOCRVLMultiModalProcessor
    if fingerprint(base._get_mm_fields_config) != FIELD_HASH:
        raise RuntimeError("Native PaddleOCR field builder differs from audited source")
    factories = getattr(target_cls, "_processor_factory", None)
    if factories is None:
        raise RuntimeError("Final PaddleOCR class has no processor factory")
    processor = factories.processor
    patched = _PROCESSOR_CLASS is not None and processor is _PROCESSOR_CLASS
    if not patched and processor is not base:
        raise RuntimeError(
            "Final PaddleOCR processor is customized; CPU-grid patch declined"
        )
    if (
        factories.info is not native.PaddleOCRVLProcessingInfo
        or factories.dummy_inputs is not native.PaddleOCRVLDummyInputsBuilder
    ):
        raise RuntimeError("PaddleOCR processing-info/dummy factory is customized")
    if not patched:
        # Set an own factory on the resolved final class, preserving inherited
        # native factories and every other registration.
        target_cls._processor_factory = replace(
            factories, processor=processor_class(base)
        )
    result = dict(
        enabled=True,
        target_class=module + "." + target_cls.__name__,
        original_processor=base.__module__ + "." + base.__name__,
        final_processor=target_cls._processor_factory.processor.__module__
        + "."
        + target_cls._processor_factory.processor.__qualname__,
        image_input_ast_sha256=fingerprint(target_cls._process_image_input),
        field_builder_ast_sha256=FIELD_HASH,
        image_grid_thw_keep_on_cpu=True,
        pixel_values_keep_on_cpu=False,
        model_methods_changed=False,
        original_processing_info_preserved=target_cls._processor_factory.info
        is factories.info,
        original_dummy_factory_preserved=target_cls._processor_factory.dummy_inputs
        is factories.dummy_inputs,
    )
    write_audit(result)
    return result


def is_adapted(target_cls):
    """Report actual factory identity without importing the model or tensors."""
    factory = getattr(target_cls, "_processor_factory", None)
    return (
        _PROCESSOR_CLASS is not None
        and getattr(factory, "processor", None) is _PROCESSOR_CLASS
    )


def write_audit(result):
    directory = os.environ.get(AUDIT_ENV) or os.environ.get(
        "PADDLEX_HPS_EAGLE3_AUDIT_DIR"
    )
    if not directory:
        return
    # Keep extra processor evidence separate from the existing model-*.json
    # records consumed by the HPS benchmark's class identity comparison.
    directory = Path(directory) / "cpu_grid"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    target = directory / f"cpu-grid-{os.getpid()}-{digest}.json"
    if target.exists():
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as f:
        temporary = Path(f.name)
        json.dump(dict(result, pid=os.getpid()), f, indent=2)
    os.replace(temporary, target)


def install_registry_hook(registry_cls):
    original = registry_cls._try_load_model_cls
    if getattr(original, "_hps_cpu_grid_hook", False):
        return

    @functools.wraps(original)
    def load(self, architecture):
        target_cls = original(self, architecture)
        if architecture == ARCH and target_cls is not None:
            adapt_target_class(target_cls)
        return target_cls

    load._hps_cpu_grid_hook = True
    registry_cls._try_load_model_cls = load


def register():
    requested = os.environ.get(ENV, "0")
    if requested not in {"0", "1"}:
        raise ValueError(ENV + " must be 0 or 1")
    if requested == "0":
        return
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("The CPU-grid adapter currently supports vLLM 0.19.0")
    from vllm.model_executor.models.registry import _ModelRegistry

    install_registry_hook(_ModelRegistry)
