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

"""Model-structure compatibility for native vLLM 0.19 EAGLE3 batching.

This module is separate from the lightweight plugin entry point so its Pydantic
subclass has a stable import path when a vLLM configuration is sent to a spawned
process. It does not select CUDA kernels, alter target arithmetic, configure
scheduling, or pin a CUDA/Pydantic patch version.
"""

import copy
import functools

from pydantic import model_validator
from pydantic.dataclasses import rebuild_dataclass
from vllm.config.speculative import SpeculativeConfig as _SpeculativeConfig
from vllm.config.utils import config

TARGET_ARCHITECTURE = "PaddleOCRVLForConditionalGeneration"
DRAFT_ARCHITECTURE = "Eagle3LlamaForCausalLM"
AUX_LAYERS = (4, 10, 16)
_ORIGINAL_VERIFY_ARGS = _SpeculativeConfig._verify_args


def is_paddle_target(target):
    return (
        target is not None
        and getattr(target.hf_config, "model_type", None) == "paddleocr_vl"
    )


def validate_profile(spec):
    """Validate the trained draft's structure without imposing a batch size.

    Checkpoint identity and the ground-truth benchmark belong to deployment
    preflight. Matching dimensions alone is not evidence of accuracy or support
    for another target-model release.
    """
    target = spec.target_model_config
    hf = target.hf_config
    text = getattr(target, "hf_text_config", hf)
    expected_text = dict(hidden_size=1024, num_hidden_layers=18, vocab_size=103424)
    if (
        getattr(hf, "architectures", None) != [TARGET_ARCHITECTURE]
        or getattr(hf, "image_token_id", None) != 100295
        or any(
            getattr(text, key, None) != value for key, value in expected_text.items()
        )
    ):
        raise ValueError("Target structure does not match the trained EAGLE3 draft")
    if (
        type(spec.num_speculative_tokens) is not int
        or spec.num_speculative_tokens < 1
        or spec.parallel_drafting
    ):
        raise ValueError("This draft requires sequential EAGLE3 with positive K")
    parallel = spec.target_parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("This model-pair adapter currently requires TP=PP=1")
    # Native ModelConfig already validates the effective target context and
    # RoPE scaling. Do not carry over the former B1 experiment's 12288 cap.
    if type(target.max_model_len) is not int or target.max_model_len <= 0:
        raise ValueError("Target context must be a positive integer")
    draft = spec.draft_model_config.hf_config
    expected_draft = dict(
        architectures=[DRAFT_ARCHITECTURE],
        hidden_size=1024,
        num_hidden_layers=1,
        vocab_size=103424,
        draft_vocab_size=40000,
        has_own_lm_head=True,
        has_own_embed_tokens=False,
    )
    if any(getattr(draft, key, None) != value for key, value in expected_draft.items()):
        raise ValueError(
            "This adapter requires the supported one-layer, 40000-token EAGLE3 draft structure"
        )
    if tuple(getattr(draft, "eagle_aux_hidden_state_layer_ids", ())) != AUX_LAYERS:
        raise ValueError("The draft must consume auxiliary layers (4, 10, 16)")


def extend_draft_context(spec):
    """Extend only the in-memory draft RoPE/context to the requested target span."""
    draft = spec.draft_model_config
    length = spec.target_model_config.max_model_len
    draft.hf_config.max_position_embeddings = max(
        draft.hf_config.max_position_embeddings, length
    )
    draft.hf_text_config.max_position_embeddings = (
        draft.hf_config.max_position_embeddings
    )
    draft.max_model_len = length if spec.max_model_len is None else spec.max_model_len


def verify_args_with_paddle(spec):
    if spec.method != "eagle3" or not is_paddle_target(spec.target_model_config):
        return _ORIGINAL_VERIFY_ARGS(spec)
    validate_profile(spec)
    # In 0.19.0 the sole target-dependent EAGLE3 check in this validator is
    # its model-type allowlist. Keep all other upstream checks on a shallow
    # validation view; never rename/mutate the real target model type.
    view = copy.copy(spec)
    view.target_model_config = None
    _ORIGINAL_VERIFY_ARGS(view)
    return spec


@config
class PaddleOCRSpeculativeConfig(_SpeculativeConfig):
    """Picklable native config with a narrow PaddleOCR EAGLE3 extension."""

    def __post_init__(self):
        super().__post_init__()
        if self.method == "eagle3" and is_paddle_target(self.target_model_config):
            validate_profile(self)
            extend_draft_context(self)

    @model_validator(mode="after")
    def _verify_args(self):
        return verify_args_with_paddle(self)


def install_config_factory(engine_args_cls):
    original = engine_args_cls.create_speculative_config
    if getattr(original, "_paddlex_hps_eagle3", False):
        return

    @functools.wraps(original)
    def create(self, target_model_config, target_parallel_config):
        options = self.speculative_config
        if (
            options is not None
            and options.get("method") == "eagle3"
            and is_paddle_target(target_model_config)
        ):
            options = dict(options)
            options.update(
                target_model_config=target_model_config,
                target_parallel_config=target_parallel_config,
            )
            return PaddleOCRSpeculativeConfig(**options)
        return original(self, target_model_config, target_parallel_config)

    create._paddlex_hps_eagle3 = True
    engine_args_cls.create_speculative_config = create


def install():
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs

    if getattr(_SpeculativeConfig, "_paddlex_hps_eagle3_validator", False):
        install_config_factory(EngineArgs)
        return
    if getattr(_SpeculativeConfig, "_paddleocr_vllm019_validator", False):
        raise RuntimeError("Do not combine the HPS batch adapter with the B1 adapter")
    info = _SpeculativeConfig.__pydantic_decorators__.model_validators.get(
        "_verify_args"
    )
    if (
        info is None
        or info.func is not _ORIGINAL_VERIFY_ARGS
        or _ORIGINAL_VERIFY_ARGS.__module__ != "vllm.config.speculative"
    ):
        raise RuntimeError(
            "A different plugin changed the speculative config validator"
        )
    _SpeculativeConfig._verify_args = verify_args_with_paddle
    info.func = verify_args_with_paddle
    rebuild_dataclass(_SpeculativeConfig, force=True)
    rebuild_dataclass(VllmConfig, force=True)
    _SpeculativeConfig._paddlex_hps_eagle3_validator = True
    install_config_factory(EngineArgs)
