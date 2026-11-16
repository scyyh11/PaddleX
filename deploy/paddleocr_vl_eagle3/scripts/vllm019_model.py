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

"""Narrow PaddleOCR-VL EAGLE3 compatibility adapter for vLLM 0.19.0.

Install before EngineArgs creates its configuration and before worker model
loading. No installed vLLM file or checkpoint is modified. The config subclass
is module-level so vLLM can pickle it into a spawned engine process.
"""

import copy
import functools
import importlib.metadata

from pydantic import model_validator
from pydantic.dataclasses import rebuild_dataclass
from vllm.config.speculative import SpeculativeConfig as _SpeculativeConfig
from vllm.config.utils import config

TARGET_ARCHITECTURE = "PaddleOCRVLForConditionalGeneration"
DRAFT_ARCHITECTURE = "Eagle3LlamaForCausalLM"
AUX_LAYERS = (4, 10, 16)
MAX_MODEL_LEN = 12288
_ORIGINAL_VERIFY_ARGS = _SpeculativeConfig._verify_args


def _is_paddle_target(target):
    return (
        target is not None
        and getattr(target.hf_config, "model_type", None) == "paddleocr_vl"
    )


def _validate_profile(spec):
    target = spec.target_model_config
    hf = target.hf_config
    expected = {
        "architectures": [TARGET_ARCHITECTURE],
        "hidden_size": 1024,
        "num_hidden_layers": 18,
        "vocab_size": 103424,
        "image_token_id": 100295,
    }
    if any(getattr(hf, key, None) != value for key, value in expected.items()):
        raise ValueError("The vLLM 0.19 adapter requires the VL-1.6 target structure")
    if spec.num_speculative_tokens != 4 or spec.parallel_drafting:
        raise ValueError("The vLLM 0.19 adapter supports sequential EAGLE3 K=4 only")
    parallel = spec.target_parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("The vLLM 0.19 adapter supports a single GPU only")
    if not 0 < target.max_model_len <= MAX_MODEL_LEN:
        raise ValueError(f"Target context must be in [1, {MAX_MODEL_LEN}]")
    draft = spec.draft_model_config.hf_config
    expected_draft = {
        "architectures": [DRAFT_ARCHITECTURE],
        "hidden_size": 1024,
        "num_hidden_layers": 1,
        "vocab_size": 103424,
        "draft_vocab_size": 40000,
        "has_own_lm_head": True,
        "has_own_embed_tokens": False,
    }
    if any(getattr(draft, key, None) != value for key, value in expected_draft.items()):
        raise ValueError("The vLLM 0.19 adapter requires the dense VL-1.6 EAGLE3 draft")
    if tuple(getattr(draft, "eagle_aux_hidden_state_layer_ids", ())) != AUX_LAYERS:
        raise ValueError("The EAGLE3 draft must capture target layers (4, 10, 16)")


@config
class PaddleOCRSpeculativeConfig(_SpeculativeConfig):
    """Extend the 0.19 EAGLE3 allowlist for the verified Paddle model pair."""

    def __post_init__(self):
        super().__post_init__()
        if self.method == "eagle3" and _is_paddle_target(self.target_model_config):
            _validate_profile(self)
            # 0.19 otherwise limits this checkpoint to its training-time 4096
            # positions. Only the draft's in-memory config is extended, before
            # worker/model/RoPE construction. The target config stays unchanged.
            draft = self.draft_model_config
            length = self.target_model_config.max_model_len
            draft.hf_config.max_position_embeddings = max(
                draft.hf_config.max_position_embeddings, length
            )
            draft.hf_text_config.max_position_embeddings = (
                draft.hf_config.max_position_embeddings
            )
            draft.max_model_len = (
                length if self.max_model_len is None else self.max_model_len
            )

    @model_validator(mode="after")
    def _verify_args(self):
        return _verify_args_with_paddle(self)


def _verify_args_with_paddle(spec):
    if spec.method != "eagle3" or not _is_paddle_target(spec.target_model_config):
        return _ORIGINAL_VERIFY_ARGS(spec)
    _validate_profile(spec)
    # The upstream validator's only target-dependent EAGLE3 check is the
    # hardcoded architecture allowlist. Validate everything else on a shallow
    # view, without changing the live target or pretending it is a Llama.
    validation_view = copy.copy(spec)
    validation_view.target_model_config = None
    _ORIGINAL_VERIFY_ARGS(validation_view)
    return spec


def _install_nested_config_validator():
    from vllm.config import VllmConfig

    if getattr(_SpeculativeConfig, "_paddleocr_vllm019_validator", False):
        return
    # VllmConfig re-runs its annotated base class's after-validator even for
    # our subclass. Rebuild the two Pydantic schemas with the same narrowly
    # scoped wrapper. Merely assigning the Python method would leave the
    # already-compiled validator unchanged.
    decorators = _SpeculativeConfig.__pydantic_decorators__.model_validators
    info = decorators.get("_verify_args")
    if info is None or info.func is not _ORIGINAL_VERIFY_ARGS:
        raise RuntimeError("Unexpected vLLM 0.19 speculative validator layout")
    _SpeculativeConfig._verify_args = _verify_args_with_paddle
    info.func = _verify_args_with_paddle
    rebuild_dataclass(_SpeculativeConfig, force=True)
    rebuild_dataclass(VllmConfig, force=True)
    _SpeculativeConfig._paddleocr_vllm019_validator = True


def _install_config_factory(engine_args_cls):
    original = engine_args_cls.create_speculative_config
    if getattr(original, "_paddleocr_vllm019", False):
        return

    @functools.wraps(original)
    def create(self, target_model_config, target_parallel_config):
        options = self.speculative_config
        if (
            options is not None
            and options.get("method") == "eagle3"
            and _is_paddle_target(target_model_config)
        ):
            # Match upstream's propagation of the two generated config fields.
            options.update(
                target_model_config=target_model_config,
                target_parallel_config=target_parallel_config,
            )
            return PaddleOCRSpeculativeConfig(**options)
        return original(self, target_model_config, target_parallel_config)

    create._paddleocr_vllm019 = True
    engine_args_cls.create_speculative_config = create


def _install_image_token_alias(target_cls):
    original = target_cls.__init__
    if getattr(original, "_paddleocr_vllm019", False):
        return

    @functools.wraps(original)
    def init(self, *, vllm_config, prefix=""):
        hf = vllm_config.model_config.hf_config
        image_id = hf.image_token_id
        existing = getattr(hf, "image_token_index", image_id)
        if existing != image_id:
            raise ValueError("PaddleOCR image_token_index differs from image_token_id")
        # EagleProposer.load_model uses image_token_index for multimodal models
        # not in its explicit list. This is an alias on this target config only.
        hf.image_token_index = image_id
        original(self, vllm_config=vllm_config, prefix=prefix)

    init._paddleocr_vllm019 = True
    target_cls.__init__ = init


def install():
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("vllm019_model is restricted to vLLM 0.19.0")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.models.paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration,
    )
    import native_dflash_target

    native_dflash_target.install()
    _install_nested_config_validator()
    _install_config_factory(EngineArgs)
    _install_image_token_alias(PaddleOCRVLForConditionalGeneration)
