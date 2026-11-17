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

"""Native 0.19 CLI/plugin ordering test; no model loading or CUDA initialization.

The ordinary CPU suite remains usable without the heavy server dependencies.
Run this file in the existing vLLM image to validate native startup integration.
"""

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hps_batch import launch

try:
    SUPPORTED_VLLM = importlib.metadata.version("vllm") == "0.19.0"
except importlib.metadata.PackageNotFoundError:
    SUPPORTED_VLLM = False


@unittest.skipUnless(SUPPORTED_VLLM, "requires the existing vLLM 0.19.0 environment")
class NativePluginOrderingTests(unittest.TestCase):
    def test_native_api_dispatch_respects_configured_count_without_starting_gpu(self):
        code = r"""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("unexpected CUDA init")):
    from vllm.entrypoints.cli import serve
    from hps_batch.native_api_dispatch import dispatch_native_serve
    def arguments(count):
        return SimpleNamespace(model_tag=None, headless=False, api_server_count=count, data_parallel_external_lb=False, data_parallel_rank=None, data_parallel_hybrid_lb=False, data_parallel_start_rank=None, data_parallel_size=1)
    with patch.object(serve, "run_multi_api_server") as multi, patch.object(serve, "run_server", new=Mock(return_value="single-coroutine")) as single, patch.object(serve.uvloop, "run") as loop:
        args = arguments(4)
        dispatch_native_serve(args)
        multi.assert_called_once_with(args)
        single.assert_not_called()
        loop.assert_not_called()
        args = arguments(1)
        dispatch_native_serve(args)
        single.assert_called_once_with(args)
        loop.assert_called_once_with("single-coroutine")
print("native-api-dispatch-ok")
"""
        environment = launch.configure_plugin_environment(
            dict(os.environ, VLLM_PLUGINS=launch.PLUGIN_NAME), enabled=False
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native-api-dispatch-ok", result.stdout)

    def test_real_cli_installs_plugin_before_speculative_config_creation(self):
        code = r"""
import json
import os
from types import SimpleNamespace
from unittest.mock import patch
import torch

# Fail if this CPU configuration test tries to initialize CUDA.
with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("unexpected CUDA init")):
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from hps_batch import eagle3_adapter

    assert not eagle3_adapter._INSTALLED
    assert not getattr(EngineArgs.create_speculative_config, "_paddlex_hps_eagle3", False)
    # Both PaddleX.run_vllm_server and native ServeSubcommand use this parser.
    parser = make_arg_parser(FlexibleArgumentParser())
    enabled = os.environ["PADDLEX_HPS_EAGLE3_ENABLED"] == "1"
    assert eagle3_adapter._INSTALLED
    assert eagle3_adapter._ENABLE_MODEL_ADAPTER == enabled
    assert bool(getattr(EngineArgs.create_speculative_config, "_paddlex_hps_eagle3", False)) == enabled
    values = ["--model", "/not-loaded-target", "--max-num-seqs", "128", "--max-num-batched-tokens", "131072"]
    if enabled:
        values += ["--speculative-config", json.dumps({"method": "eagle3", "model": "/not-loaded-draft", "num_speculative_tokens": 4})]
    engine_args = AsyncEngineArgs.from_cli_args(parser.parse_args(values))
    assert engine_args.max_num_seqs == 128
    assert engine_args.max_num_batched_tokens == 131072
    if enabled:
        from hps_batch import eagle3_config
        target = SimpleNamespace(hf_config=SimpleNamespace(model_type="paddleocr_vl"))
        parallel = object()
        sentinel = object()
        with patch.object(eagle3_config, "PaddleOCRSpeculativeConfig", return_value=sentinel) as factory:
            assert engine_args.create_speculative_config(target, parallel) is sentinel
            assert factory.call_args.kwargs["target_model_config"] is target
            assert factory.call_args.kwargs["target_parallel_config"] is parallel
            assert factory.call_args.kwargs["method"] == "eagle3"
    else:
        assert engine_args.speculative_config is None
        assert engine_args.create_speculative_config(None, None) is None
print("native-cli-plugin-ordering-ok")
"""
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as folder:
                # Isolate the plugin under test from optional installed plugins;
                # launcher policy-preservation is tested separately.
                environment = launch.configure_plugin_environment(
                    dict(os.environ, VLLM_PLUGINS=launch.PLUGIN_NAME),
                    enabled=enabled,
                )
                environment[launch.AUDIT_DIR_ENV] = folder
                environment["HF_HUB_OFFLINE"] = "0"
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("native-cli-plugin-ordering-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
