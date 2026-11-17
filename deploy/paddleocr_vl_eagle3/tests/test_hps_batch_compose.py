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

"""Exercise original shell command → console shim → native CLI with CPU fakes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hps_batch import compose_bridge


class ComposeBridgeTests(unittest.TestCase):
    def test_original_args_and_environment_are_preserved(self):
        original = [
            "genai_server",
            "--model_name",
            "model with spaces",
            "--backend",
            "vllm",
        ]
        environment = {
            "EAGLE3_MODE": "baseline",
            "EAGLE3_AUDIT_OUTPUT": "/private/audit.json",
        }
        args = compose_bridge.launcher_arguments("paddleocr", original, environment)
        self.assertEqual(args[args.index("--") + 1 :], original)
        self.assertEqual(args[args.index("--entrypoint") + 1], "paddleocr")
        configured = compose_bridge.prepare_environment(
            {
                "PATH": os.environ["PATH"],
                "PADDLE_PDX_MODEL_SOURCE": "bos",
                "VLLM_PLUGINS": "custom",
            }
        )
        self.assertEqual(configured["PADDLE_PDX_MODEL_SOURCE"], "bos")
        self.assertEqual(configured["VLLM_PLUGINS"], "custom")
        self.assertTrue(configured["PATH"].endswith(os.pathsep + os.environ["PATH"]))

    def test_inherited_shell_reaches_original_cli_and_only_adds_speculation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # These minimal modules stand in for the existing image's CLIs;
            # launch's plugin discovery and the complete exec/shim path are real.
            for name in (
                "paddlex",
                "paddlex/inference",
                "paddlex/inference/genai",
                "paddlex/inference/genai/backends",
                "paddleocr",
            ):
                directory = root / name
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "__init__.py").write_text("")
            (root / "paddlex/inference/genai/server.py").write_text(
                """
import argparse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser
def run_genai_server(args):
    from .backends import vllm
    return vllm.run_vllm_server(args.host, args.port, args.model_name, "/target", {"max-num-seqs": 128, "api-server-count": 4}, None)
"""
            )
            (root / "paddlex/inference/genai/backends/vllm.py").write_text(
                """
import json, os, sys
def run_vllm_server(host, port, model_name, model_dir, config, template):
    with open(os.environ["TEST_CAPTURE"], "w") as output:
        json.dump(dict(host=host, port=port, model=model_name, config=config, argv=sys.argv, source=os.environ["PADDLE_PDX_MODEL_SOURCE"]), output)
"""
            )
            (root / "paddleocr/_cli.py").write_text(
                """
import sys
def main():
    from paddlex.inference.genai import server
    assert sys.argv[1] == "genai_server"
    server.run_genai_server(server.get_arg_parser().parse_args(sys.argv[2:]))
"""
            )
            entrypoint = root / "entrypoint.sh"
            entrypoint.write_text(
                '#!/bin/sh\nset -eu\nexec paddleocr genai_server --model_name "$TEST_MODEL_NAME" --host 0.0.0.0 --port 8080 --backend vllm\n'
            )
            draft = root / "draft"
            draft.mkdir()
            captures = []
            for mode in ("baseline", "eagle3"):
                capture = root / (mode + "-capture.json")
                audit = root / (mode + "-audit.json")
                environment = dict(
                    os.environ,
                    PYTHONPATH=str(root),
                    PATH=str(Path(sys.executable).parent)
                    + os.pathsep
                    + os.environ["PATH"],
                    EAGLE3_MODE=mode,
                    EAGLE3_AUDIT_OUTPUT=str(audit),
                    EAGLE3_DRAFT_MODEL=str(draft),
                    EAGLE3_K="4",
                    PADDLEX_HPS_BATCH_VISION="0",
                    PADDLE_PDX_MODEL_SOURCE="bos",
                    TEST_MODEL_NAME="PaddleOCR with spaces",
                    TEST_CAPTURE=str(capture),
                    VLLM_PLUGINS="custom_vision",
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        str(compose_bridge.HERE / "compose_bridge.py"),
                        "--",
                        "/bin/sh",
                        str(entrypoint),
                    ],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                data = json.loads(capture.read_text())
                audit_data = json.loads(audit.read_text())
                self.assertEqual(
                    data["argv"],
                    [
                        "paddleocr",
                        "genai_server",
                        "--model_name",
                        "PaddleOCR with spaces",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "8080",
                        "--backend",
                        "vllm",
                    ],
                )
                self.assertEqual(data["source"], "bos")
                self.assertEqual(
                    audit_data["plugin_allowlist"], "custom_vision,paddlex_eagle3_hps"
                )
                self.assertEqual(audit_data["status"], "entering_native_server")
                captures.append(data)
            spec = captures[1]["config"].pop("speculative-config")
            self.assertEqual(json.loads(spec)["method"], "eagle3")
            self.assertEqual(captures[0], captures[1])

    def test_non_server_paddleocr_commands_fall_back_to_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "paddleocr"
            original.write_text(
                f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            original.chmod(0o755)
            environment = dict(
                os.environ,
                PATH=str(root)
                + os.pathsep
                + str(Path(sys.executable).parent)
                + os.pathsep
                + os.environ["PATH"],
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(compose_bridge.HERE / "compose_bridge.py"),
                    "--",
                    "paddleocr",
                    "--version",
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout), ["--version"])


if __name__ == "__main__":
    unittest.main()
