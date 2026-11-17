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

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hps_batch import launch


class PluginEnvironmentTests(unittest.TestCase):
    def test_unset_policy_preserves_native_plugin_discovery(self):
        original = {
            "PYTHONPATH": "/existing",
            "VLLM_USE_V2_MODEL_RUNNER": "custom",
            "CUDA_VISIBLE_DEVICES": "3",
        }
        configured = launch.configure_plugin_environment(original, enabled=True)
        self.assertNotIn("VLLM_PLUGINS", configured)
        self.assertNotIn(launch.ENABLE_ENV, original)
        self.assertEqual(configured[launch.ENABLE_ENV], "1")
        self.assertEqual(configured[launch.VISION_ENV], "0")
        self.assertEqual(configured["VLLM_USE_V2_MODEL_RUNNER"], "custom")
        self.assertEqual(configured["CUDA_VISIBLE_DEVICES"], "3")
        self.assertTrue(configured["PYTHONPATH"].endswith(os.pathsep + "/existing"))

    def test_existing_paddlex_and_custom_plugins_are_preserved(self):
        original = {
            "VLLM_PLUGINS": "register_paddlex_genai_models,custom_vision",
            launch.ENABLE_ENV: "1",
        }
        once = launch.configure_plugin_environment(original, enabled=False)
        twice = launch.configure_plugin_environment(once, enabled=False)
        self.assertEqual(once, twice)
        self.assertEqual(
            once["VLLM_PLUGINS"],
            "register_paddlex_genai_models,custom_vision,paddlex_eagle3_hps",
        )
        self.assertEqual(once[launch.ENABLE_ENV], "0")

    def test_disabled_plugin_policy_is_not_silently_overridden(self):
        for policy in ("", " , "):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                launch.configure_plugin_environment(
                    {"VLLM_PLUGINS": policy}, enabled=True
                )

    def test_vision_bridge_can_be_identical_in_both_groups(self):
        for enabled in (False, True):
            configured = launch.configure_plugin_environment(
                {}, enabled=enabled, batch_vision=True
            )
            self.assertEqual(configured[launch.VISION_ENV], "1")

    def test_cpu_grid_is_independent_and_preserves_explicit_plugin_policy(self):
        original = {"VLLM_PLUGINS": "register_paddlex_genai_models,custom_vision"}
        for enabled in (False, True):
            configured = launch.configure_plugin_environment(
                original, enabled=enabled, cpu_grid=True
            )
            self.assertEqual(configured[launch.CPU_GRID_ENV], "1")
            self.assertEqual(configured[launch.VISION_ENV], "0")
            self.assertEqual(
                configured["VLLM_PLUGINS"],
                "register_paddlex_genai_models,custom_vision,paddlex_eagle3_hps",
            )
        self.assertNotIn(launch.CPU_GRID_ENV, original)

    def test_mounted_entrypoint_is_discovered_in_fresh_process(self):
        environment = launch.configure_plugin_environment(os.environ, enabled=False)
        code = "import importlib.metadata,json; print(json.dumps([(x.name,x.value) for x in importlib.metadata.entry_points(group='vllm.general_plugins') if x.name=='paddlex_eagle3_hps']))"
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            [["paddlex_eagle3_hps", "hps_batch.eagle3_adapter:register"]],
        )

    def test_dry_run_works_without_importing_server_frameworks(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "audit.json"
            code = """
import sys
from hps_batch import launch
launch.main(["--mode", "baseline", "--audit-output", sys.argv[1], "--dry-run", "--", "--model_name", "PaddleOCR-VL-1.8-0.9B", "--backend", "vllm"])
assert not any(name in sys.modules for name in ("torch", "vllm", "paddlex", "paddleocr"))
"""
            environment = launch.configure_plugin_environment(os.environ, enabled=False)
            result = subprocess.run(
                [sys.executable, "-S", "-c", code, str(output)],
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(result.stdout)["status"], "planned")
            audit = json.loads(output.read_text())
            self.assertTrue(
                any(
                    row["name"] == launch.PLUGIN_NAME
                    for row in audit["discovered_general_plugins"]
                )
            )


class NativeConfigurationTests(unittest.TestCase):
    def test_only_speculation_differs_without_modifying_input(self):
        base = {
            "max-num-seqs": 256,
            "max-num-batched-tokens": 131072,
            "api-server-count": 4,
            "max-model-len": 16384,
            "dtype": "bfloat16",
            "gpu-memory-utilization": 0.5,
        }
        snapshot = dict(base)
        spec = {"method": "eagle3", "model": "/draft", "num_speculative_tokens": 4}
        baseline = launch.merge_speculative_config(base, None)
        eagle = launch.merge_speculative_config(base, spec)
        self.assertEqual(base, snapshot)
        self.assertEqual(baseline, base)
        self.assertEqual(
            {k: v for k, v in eagle.items() if k != "speculative-config"}, base
        )
        self.assertEqual(json.loads(eagle["speculative-config"]), spec)

    def test_existing_speculation_is_rejected_in_both_modes(self):
        for field in ("speculative-config", "speculative_config"):
            for spec in (None, {"method": "eagle3"}):
                with self.subTest(field=field, spec=spec), self.assertRaises(
                    ValueError
                ):
                    launch.merge_speculative_config({field: {}}, spec)

    def test_native_cli_preserves_all_base_arguments(self):
        base = ["serve", "/target", "--max-num-seqs", "128", "--api-server-count=4"]
        spec = {"method": "eagle3", "model": "/draft", "num_speculative_tokens": 4}
        self.assertEqual(launch.native_vllm_argv(base, None), base)
        eagle = launch.native_vllm_argv(base, spec)
        self.assertEqual(eagle[:-2], base)
        self.assertEqual(json.loads(eagle[-1]), spec)
        with self.assertRaises(ValueError):
            launch.native_vllm_argv([*base, "--speculative-config={}"], spec)

    def test_native_yaml_cannot_hide_speculation_and_is_auditable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(json.dumps({"speculative-config": {"method": "eagle3"}}))
            args = ["serve", "/target", "--config", str(path)]
            self.assertEqual(launch.server_file_arguments(args), [path])
            # JSON is a YAML subset. Keep the stdlib test independent of PyYAML.
            with patch.dict(
                sys.modules, {"yaml": SimpleNamespace(safe_load=json.loads)}
            ):
                with self.assertRaises(ValueError):
                    launch.validate_native_configuration(args)
                path.write_text(json.dumps({"max-num-seqs": 128}))
                launch.validate_native_configuration(args)

    def test_secret_audit_redaction_preserves_nonsecret_config(self):
        self.assertEqual(
            launch.redact_argv(
                [
                    "--api-key",
                    "sensitive",
                    "--hf-token=other",
                    "--model",
                    "public-model",
                ]
            ),
            [
                "--api-key",
                "<redacted>",
                "--hf-token=<redacted>",
                "--model",
                "public-model",
            ],
        )
        self.assertEqual(
            launch.redact({"api_key": "secret", "model": "visible"}),
            {"api_key": "<redacted>", "model": "visible"},
        )

    def test_private_audit_file_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.json"
            launch.write_audit(path, {"api_key": "private", "value": 3})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                json.loads(path.read_text()), {"api_key": "<redacted>", "value": 3}
            )


if __name__ == "__main__":
    unittest.main()
