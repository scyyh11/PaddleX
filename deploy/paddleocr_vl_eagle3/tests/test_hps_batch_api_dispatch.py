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

import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hps_batch.native_api_dispatch import adapt_backend


def run_vllm_server():
    return "unsupported backend shape"


class NativeAPIDispatchTests(unittest.TestCase):
    def load_original(self):
        root = Path(__file__).resolve().parents[3]
        path = root / "paddlex/inference/genai/backends/vllm.py"
        tree = ast.parse(path.read_text())
        node = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "run_vllm_server"
        )

        def defaults(config, values):
            for key, value in values.items():
                config.setdefault(key, value)

        namespace = {
            "__name__": "paddlex.inference.genai.backends.vllm",
            "require_genai_engine_plugin": Mock(),
            "set_config_defaults": defaults,
            "update_backend_config": lambda config, values: config.update(values),
            "backend_config_to_args": lambda config: copy.deepcopy(config),
        }
        # Compile the original function unchanged, without importing PaddleX's
        # heavy package initialization. All original call sites remain intact.
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace["run_vllm_server"]

    def test_real_paddlex_source_preserves_config_and_routes_native_count(self):
        original = self.load_original()
        original_code = original.__code__
        dispatcher = Mock()
        patched = adapt_backend(original, dispatch=dispatcher)
        self.assertIs(original.__code__, original_code)
        self.assertIs(adapt_backend(patched), patched)
        single = Mock(return_value="single-coroutine")
        loop = Mock()

        class Parser:
            def parse_args(self, config):
                return SimpleNamespace(
                    **{key.replace("-", "_"): value for key, value in config.items()}
                )

        api = SimpleNamespace(
            FlexibleArgumentParser=Parser,
            cli_env_setup=Mock(),
            make_arg_parser=lambda parser: parser,
            run_server=single,
            validate_parsed_serve_args=Mock(),
        )
        fake_modules = {
            "uvloop": SimpleNamespace(run=loop),
            "vllm.entrypoints.openai.api_server": api,
            "torch": SimpleNamespace(version=SimpleNamespace(hip=None, cuda="12.9")),
        }
        base = {
            "api-server-count": 4,
            "max-num-seqs": 128,
            "max-model-len": 16384,
            "dtype": "bfloat16",
        }
        with patch.dict(sys.modules, fake_modules):
            old_config, fixed_config = copy.deepcopy(base), copy.deepcopy(base)
            original(
                "0.0.0.0", 8080, "target-name", "/target", old_config, "/chat.jinja"
            )
            single.assert_called_once()
            loop.assert_called_once_with("single-coroutine")
            self.assertEqual(single.call_args.args[0].api_server_count, 4)
            single.reset_mock()
            loop.reset_mock()
            patched(
                "0.0.0.0", 8080, "target-name", "/target", fixed_config, "/chat.jinja"
            )
        dispatcher.assert_called_once()
        single.assert_not_called()
        loop.assert_not_called()
        self.assertEqual(old_config, fixed_config)
        parsed = vars(dispatcher.call_args.args[0])
        self.assertEqual(parsed["api_server_count"], 4)
        self.assertEqual(parsed["max_num_seqs"], 128)
        self.assertEqual(parsed["served_model_name"], "target-name")
        self.assertEqual(parsed["chat_template"], "/chat.jinja")
        self.assertFalse(
            patched._paddlex_hps_dispatch_audit["api_server_count_overridden"]
        )

    def test_unknown_source_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            adapt_backend(run_vllm_server)


if __name__ == "__main__":
    unittest.main()
