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

"""Opt-in EAGLE3 model interface for the existing HPS/vLLM batch service.

The vLLM general-plugin entry point runs in API, model-inspection and worker
processes. Resolve the actual registered target class lazily, after the other
plugins have registered or patched it. Preserve its forward, image encoder,
multimodal processor and arithmetic; do not replace the registry entry or change
the scheduler, request admission, CUDA kernels, or VLLM_PLUGINS. An independent
opt-in bridge can batch the unmodified native vision encoder in both modes.

Importing this module has no side effects and does not import torch/vLLM. The
plugin is enabled by PADDLEX_HPS_EAGLE3_ENABLED=1. A baseline may request only
class-identity auditing, without altering the model/configuration.
"""

import functools
import hashlib
import importlib.metadata
import inspect
import json
import marshal
import os
from pathlib import Path
import tempfile

ENABLED_ENV = "PADDLEX_HPS_EAGLE3_ENABLED"
AUDIT_DIR_ENV = "PADDLEX_HPS_EAGLE3_AUDIT_DIR"
BATCH_VISION_ENV = "PADDLEX_HPS_BATCH_VISION"
TARGET_ARCHITECTURE = "PaddleOCRVLForConditionalGeneration"
AUX_LAYERS = (4, 10, 16)
PLUGIN_NAME = "paddlex_eagle3_hps"
_INSTALLED = False
_ENABLE_MODEL_ADAPTER = False
_ENABLE_BATCH_VISION = False
_AUDIT_DIR = None


def _decoder_for(model):
    # Native vLLM wraps ErnieForCausalLM in language_model; older PaddleX
    # registrations inherit it directly and expose the bare decoder as .model.
    inner = getattr(model, "model", None)
    if callable(getattr(inner, "_set_aux_hidden_state_layers", None)):
        return inner
    language = getattr(model, "language_model", None)
    inner = getattr(language, "model", language)
    if callable(getattr(inner, "_set_aux_hidden_state_layers", None)):
        return inner
    raise TypeError("PaddleOCR language decoder does not implement EagleModelMixin")


def _is_language_wrapper(model):
    decoder = getattr(model, "model", None)
    return (
        decoder is not None
        and callable(getattr(decoder, "_set_aux_hidden_state_layers", None))
        and (hasattr(decoder, "embed_tokens") or hasattr(decoder, "embedding"))
        and hasattr(model, "lm_head")
    )


def _image_token_alias(config):
    image_id = getattr(config, "image_token_id", None)
    if type(image_id) is not int or image_id != 100295:
        raise ValueError(
            "Image token ID does not match the trained draft's target layout"
        )
    if getattr(config, "image_token_index", image_id) != image_id:
        raise ValueError("image_token_index differs from image_token_id")
    config.image_token_index = image_id


def adapt_target_class(target_cls):
    """Add the auxiliary-state contract to the class selected by existing plugins."""
    if target_cls.__dict__.get("_paddlex_hps_eagle3_adapter", False):
        return target_cls
    if getattr(target_cls, "_ocr_native_dflash_adapter", False):
        raise RuntimeError("Do not combine HPS EAGLE3 with the B1 target adapter")
    original_init = target_cls.__init__
    original_get_language = getattr(target_cls, "get_language_model", None)

    @functools.wraps(original_init)
    def init(self, *, vllm_config, prefix=""):
        hf = vllm_config.model_config.hf_config
        _image_token_alias(hf)
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        if getattr(self, "config", hf) is not hf:
            _image_token_alias(self.config)

    def set_aux(self, layers):
        layers = tuple(layers)
        if layers != AUX_LAYERS:
            raise ValueError("This trained draft requires auxiliary layers (4, 10, 16)")
        _decoder_for(self)._set_aux_hidden_state_layers(layers)

    def default_aux(self):
        return AUX_LAYERS

    def get_language(self):
        if original_get_language is not None:
            try:
                language = original_get_language(self)
            except NotImplementedError:
                language = None
            if _is_language_wrapper(language):
                return language
        # A PaddleX target that directly inherits ErnieForCausalLM must return
        # itself, not the bare decoder, for native proposer weight sharing.
        if _is_language_wrapper(self):
            return self
        language = getattr(self, "language_model", None)
        if _is_language_wrapper(language):
            return language
        raise TypeError("Cannot find the PaddleOCR language model and output head")

    target_cls.__init__ = init
    target_cls.supports_eagle3 = True
    if not hasattr(target_cls, "has_own_lm_head"):
        target_cls.has_own_lm_head = False
    if not hasattr(target_cls, "has_own_embed_tokens"):
        target_cls.has_own_embed_tokens = False
    target_cls.set_aux_hidden_state_layers = set_aux
    target_cls.get_eagle3_default_aux_hidden_state_layers = default_aux
    target_cls.get_language_model = get_language
    target_cls._paddlex_hps_eagle3_adapter = True
    return target_cls


def _function_identity(function):
    if function is None:
        return None
    try:
        source = inspect.getsource(function).encode()
    except (TypeError, OSError):
        source = None
    code = getattr(function, "__code__", None)
    return dict(
        module=getattr(function, "__module__", None),
        qualname=getattr(function, "__qualname__", None),
        source_sha256=hashlib.sha256(source).hexdigest() if source else None,
        code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest() if code else None,
    )


def collect_class_identity(target_cls):
    """Aggregate class/dispatch identity only; no tensors or document content."""
    from . import cpu_grid

    return dict(
        module=target_cls.__module__,
        qualname=target_cls.__qualname__,
        adapter_installed=bool(
            target_cls.__dict__.get("_paddlex_hps_eagle3_adapter", False)
        ),
        batch_vision_bridge_installed=bool(
            target_cls.__dict__.get("_paddlex_hps_batch_vision", False)
        ),
        cpu_grid_enabled=cpu_grid.is_adapted(target_cls),
        methods={
            name: _function_identity(getattr(target_cls, name, None))
            for name in (
                "forward",
                "compute_logits",
                "encode_image",
                "get_multimodal_embeddings",
                "_process_image_input",
                "embed_multimodal",
                "get_language_model",
            )
        },
    )


def _write_audit(target_cls, architecture, batch_vision=None):
    if _AUDIT_DIR is None:
        return
    identity = collect_class_identity(target_cls)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    payload = dict(
        pid=os.getpid(),
        architecture=architecture,
        eagle3_enabled=_ENABLE_MODEL_ADAPTER,
        batch_vision=batch_vision,
        adapter=metadata(),
        identity=identity,
    )
    directory = Path(_AUDIT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"model-{os.getpid()}-{digest[:16]}.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.close()
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def install_registry_hook(registry_cls):
    original = registry_cls._try_load_model_cls
    if getattr(original, "_paddlex_hps_eagle3", False):
        return

    @functools.wraps(original)
    def load(self, architecture):
        target_cls = original(self, architecture)
        if architecture == TARGET_ARCHITECTURE and target_cls is not None:
            batch_vision = None
            if _ENABLE_BATCH_VISION:
                from . import native_batch_vision

                batch_vision = native_batch_vision.adapt_target_class(target_cls)
            if _ENABLE_MODEL_ADAPTER:
                adapt_target_class(target_cls)
            _write_audit(target_cls, architecture, batch_vision)
        return target_cls

    load._paddlex_hps_eagle3 = True
    registry_cls._try_load_model_cls = load


def install(*, enabled=True, audit_dir=None, batch_vision=False):
    """Install in the current process; spawned vLLM processes use register()."""
    global _INSTALLED, _ENABLE_MODEL_ADAPTER, _AUDIT_DIR, _ENABLE_BATCH_VISION
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("The HPS EAGLE3 adapter currently supports vLLM 0.19.0")
    if _ENABLE_MODEL_ADAPTER and not enabled:
        raise RuntimeError("A process with EAGLE3 enabled cannot become a baseline")
    if _ENABLE_BATCH_VISION and not batch_vision:
        raise RuntimeError(
            "Batch vision cannot be disabled after installation in a process"
        )
    if enabled:
        from . import eagle3_config, eagle3_positions

        eagle3_config.install()
        eagle3_positions.install()
    from vllm.model_executor.models.registry import _ModelRegistry

    _ENABLE_MODEL_ADAPTER = bool(enabled)
    _ENABLE_BATCH_VISION = bool(batch_vision)
    _AUDIT_DIR = audit_dir
    install_registry_hook(_ModelRegistry)
    _INSTALLED = True
    return metadata()


def metadata():
    return dict(
        installed=_INSTALLED,
        enabled=_ENABLE_MODEL_ADAPTER,
        auxiliary_layers=list(AUX_LAYERS),
        scheduling="native vLLM continuous batching",
        arithmetic="native target and EAGLE3 draft operators",
        eagle3_changes_target_forward_or_vision=False,
        batch_vision_requested=_ENABLE_BATCH_VISION,
        draft_position_contract="absolute token ordinal (matching draft training)",
        class_audit_enabled=_AUDIT_DIR is not None,
    )


def register():
    """vllm.general_plugins entry point; preserve all other plugin selection."""
    from . import cpu_grid

    requested = os.environ.get(ENABLED_ENV, "0")
    if requested not in ("0", "1"):
        raise ValueError(f"{ENABLED_ENV} must be 0 or 1")
    audit_dir = os.environ.get(AUDIT_DIR_ENV) or None
    vision = os.environ.get(BATCH_VISION_ENV, "0")
    if vision not in ("0", "1"):
        raise ValueError(f"{BATCH_VISION_ENV} must be 0 or 1")
    if vision == "1" and os.environ.get(cpu_grid.ENV, "0") == "1":
        raise ValueError(
            "CPU-grid has not been validated with the native batch vision bridge"
        )
    # Install the inner resolution hook first so class audits below observe the
    # final processor. Reuse our entry point to preserve explicit allowlists.
    cpu_grid.register()
    if requested == "1" or audit_dir is not None or vision == "1":
        install(
            enabled=requested == "1", audit_dir=audit_dir, batch_vision=vision == "1"
        )
