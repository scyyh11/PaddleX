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
    is_vllm_performance_version,
    require_genai_engine_plugin,
)
from ..configs.utils import (
    backend_config_to_args,
    set_config_defaults,
    update_backend_config,
)
from ..models import ALL_MODEL_INFO, get_model_components, is_integrated_model_available

_PERFORMANCE_MODEL = "PaddleOCR-VL-1.8-0.9B"
_PERFORMANCE_ARCH = "PaddleXPaddleOCRVLForConditionalGeneration"


def register_models():
    from vllm import ModelRegistry

    if is_genai_engine_plugin_available("vllm-server"):
        if is_vllm_performance_version():
            # The stock architecture stays registered. Only explicit opt-in
            # selects the optimized implementation, in the API and engine workers.
            ModelRegistry.register_model(
                _PERFORMANCE_ARCH,
                "paddlex.inference.genai.models.paddleocr_vl_18_09b._vllm033:"
                + _PERFORMANCE_ARCH,
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

    config = dict(config)
    enable_rope = config.pop("paddlex-pvl-vision-rope", False)
    if not isinstance(enable_rope, bool):
        raise ValueError("paddlex-pvl-vision-rope must be a YAML boolean.")
    if enable_rope:
        if not is_vllm_performance_version() or model_name != _PERFORMANCE_MODEL:
            raise ValueError(
                "paddlex-pvl-vision-rope requires PaddleOCR-VL 1.8 and vLLM 0.33.0."
            )
        if "hf-overrides" in config or "hf_overrides" in config:
            raise ValueError(
                "paddlex-pvl-vision-rope cannot be combined with hf-overrides."
            )
        config["hf-overrides"] = json.dumps({"architectures": [_PERFORMANCE_ARCH]})

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

    if torch.version.hip is not None and torch.version.cuda is None:
        # For DCU
        if "api-server-count" in config:
            logging.warning(
                "Key 'api-server-count' will be popped as it is not supported"
            )
            config.pop("api-server-count")

    if is_vllm_performance_version():
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
