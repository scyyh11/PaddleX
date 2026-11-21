# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import sys

from ....utils import logging
from ....utils.deps import (
    is_genai_engine_plugin_available,
    is_vllm_legacy_version,
    is_vllm_performance_version,
    require_genai_engine_plugin,
)
from ..configs.utils import (
    backend_config_to_args,
    set_config_defaults,
    update_backend_config,
)
from ..models import ALL_MODEL_INFO, get_model_components, is_integrated_model_available

_VISION_MODEL = "PaddleOCR-VL-1.8-0.9B"
_VISION_ARCH = "PaddleXPaddleOCRVLForConditionalGeneration"
_VISION_PERFORMANCE_OPTION = "paddlex-vision-performance"
_MODEL_MODULE = "paddlex.inference.genai.models.paddleocr_vl_18_09b._vllm_vision"


def register_models():
    from vllm import ModelRegistry

    if is_genai_engine_plugin_available("vllm-server"):
        # Registration is lazy. The stock architecture remains untouched;
        # only the validated RoPE runtime registers the opt-in alias.
        if is_vllm_performance_version():
            ModelRegistry.register_model(
                _VISION_ARCH, f"{_MODEL_MODULE}:{_VISION_ARCH}"
            )
        for model_name in ALL_MODEL_INFO:
            if (
                not is_integrated_model_available(model_name, "vllm")
                and model_name not in ModelRegistry.get_supported_archs()
            ):
                net_cls, _ = get_model_components(model_name, "vllm")
                ModelRegistry.register_model(net_cls.__name__, net_cls)


def run_vllm_server(host, port, model_name, model_dir, config, chat_template_path):
    require_genai_engine_plugin("vllm-server")

    config = {k.replace("_", "-"): v for k, v in config.items()}
    legacy, modern = is_vllm_legacy_version(), is_vllm_performance_version()
    if {"paddlex-vision-batching", "paddlex-vision-rope"} & config.keys():
        raise ValueError(
            "The paddlex-vision-batching and paddlex-vision-rope options have been "
            f"removed. Remove both keys. Use {_VISION_PERFORMANCE_OPTION}: true "
            "only to enable fused RoPE on the supported A100 profile. "
            "Vision batching is left to the backend."
        )
    vision_performance = config.pop(_VISION_PERFORMANCE_OPTION, False)
    if not isinstance(vision_performance, bool):
        raise ValueError(f"{_VISION_PERFORMANCE_OPTION} must be a YAML boolean.")
    if vision_performance:
        if model_name != _VISION_MODEL or not modern:
            raise ValueError(
                f"{_VISION_PERFORMANCE_OPTION} requires PaddleOCR-VL 1.8 and vLLM 0.33.0."
            )
        if "hf-overrides" in config:
            raise ValueError(
                f"{_VISION_PERFORMANCE_OPTION} cannot be combined with hf-overrides."
            )
        config["hf-overrides"] = json.dumps({"architectures": [_VISION_ARCH]})
    if legacy and "prefill-schedule-interval" in config:
        raise ValueError("prefill-schedule-interval requires vLLM 0.33.0.")

    set_config_defaults(config, {"served-model-name": model_name})

    if chat_template_path:
        set_config_defaults(config, {"chat-template": str(chat_template_path)})

    update_backend_config(
        config,
        {
            "model": model_dir,
            "host": host,
            "port": port,
        },
    )

    import torch

    if vision_performance and torch.version.cuda is None:
        raise ValueError(f"{_VISION_PERFORMANCE_OPTION} requires NVIDIA CUDA.")

    if torch.version.hip is not None and torch.version.cuda is None:
        # For DCU
        if "api-server-count" in config:
            logging.warning(
                "Key 'api-server-count' will be popped as it is not supported"
            )
            config.pop("api-server-count")

    if (legacy or modern) and torch.version.cuda is not None:
        # Use the real serve command; run_server() alone starts one frontend.
        # Preserve explicitly disabled BooleanOptionalAction options in this path.
        disabled = [
            f"no-{k}" for k in ("enable-prefix-caching",) if config.get(k) is False
        ]
        config = {k: v for k, v in config.items() if v is not False}
        config.update(dict.fromkeys(disabled, True))
        engine_args = backend_config_to_args(config)
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            *engine_args,
        ]
        os.execv(sys.executable, command)
        return

    import uvloop
    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        cli_env_setup,
        make_arg_parser,
        run_server,
        validate_parsed_serve_args,
    )

    cli_env_setup()
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(backend_config_to_args(config))
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
