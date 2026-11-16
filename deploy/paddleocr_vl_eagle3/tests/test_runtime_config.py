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

"""Configuration/export checks that do not construct a CUDA engine."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runtime_config as runtime
from serve_hps_vlm import parse_args
from serve_ocr import build_engine_config, create_app, verify_model_file


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.target = self.root / "target"
        self.draft = self.root / "draft"
        self.target.mkdir()
        self.draft.mkdir()

    def args(self, *extra):
        return parse_args(
            ["--model", str(self.target), "--draft-model", str(self.draft), *extra]
        )

    def test_default_profile_matches_deployed_engine(self):
        args = self.args()
        config, _, _ = build_engine_config(args)
        self.assertEqual(args.model_version, "1.6")
        self.assertEqual(args.mode, "eagle3-recovery")
        self.assertEqual(args.recovery_decode_graphs, "int8")
        self.assertEqual(config["max_model_len"], 12288)
        self.assertEqual(config["dtype"], "bfloat16")
        self.assertEqual(config["tensor_parallel_size"], 1)
        self.assertEqual(config["speculative_config"]["num_speculative_tokens"], 4)
        self.assertTrue(config["speculative_config"]["use_local_argmax_reduction"])
        self.assertEqual(config["async_scheduling"], runtime.is_vllm019())
        self.assertFalse(config["enable_prefix_caching"])
        self.assertEqual(
            config["compilation_config"]["cudagraph_mode"],
            "FULL_AND_PIECEWISE" if runtime.is_vllm019() else "FULL_DECODE_ONLY",
        )
        self.assertEqual(config["compilation_config"]["cudagraph_capture_sizes"], [5])

    def test_baseline_does_not_load_worker_or_require_draft(self):
        self.draft.rmdir()
        config, _, _ = build_engine_config(
            self.args("--mode", "baseline", "--hps-max-completion-tokens", "8192")
        )
        self.assertEqual(config["max_model_len"], 12288)
        self.assertNotIn("worker_extension_cls", config)
        self.assertNotIn("speculative_config", config)

    def test_wrong_model_profile_k_and_missing_directory_fail(self):
        with self.assertRaises(ValueError):
            create_app(argparse.Namespace(mode="eagle3-recovery", model_version="1.0"))
        args = self.args()
        args.k = 3
        with self.assertRaises(ValueError):
            build_engine_config(args)
        self.target.rmdir()
        with self.assertRaises(FileNotFoundError):
            build_engine_config(self.args())

    def test_model_identity_rejects_wrong_or_missing_bytes(self):
        artifact = self.target / "model.safetensors"
        artifact.write_bytes(b"fixture")
        verify_model_file(artifact, hashlib.sha256(b"fixture").hexdigest())
        with self.assertRaises(RuntimeError):
            verify_model_file(artifact, hashlib.sha256(b"other").hexdigest())
        with self.assertRaises(FileNotFoundError):
            verify_model_file(self.target / "missing", "0" * 64)

    def test_version_gate_and_self_contained_cache_identity(self):
        (self.root / "scripts").mkdir()
        for name in runtime.RUNTIME_SOURCES:
            (self.root / "scripts" / name).write_text("# exported runtime source\n")
        args = self.args()
        config, worker_args, _ = build_engine_config(args)
        versions = dict(runtime.PINNED_VERSIONS)
        with patch.object(runtime, "ROOT", self.root), patch.dict(
            os.environ, {}, clear=True
        ), patch.object(
            runtime.importlib.metadata,
            "version",
            side_effect=lambda name: versions[name],
        ):
            first = runtime.configure_environment(worker_args, config)
            self.assertFalse((self.root / "results").exists())
            (self.root / "scripts/serve_ocr.py").write_text(
                "# changed exported source\n"
            )
            second = runtime.configure_environment(worker_args, config)
            self.assertNotEqual(
                first["cache_identity"]["fingerprint"],
                second["cache_identity"]["fingerprint"],
            )
            versions["torch"] = "0.0.0"
            with self.assertRaisesRegex(RuntimeError, "Unvalidated runtime versions"):
                runtime.configure_environment(worker_args, config)


if __name__ == "__main__":
    unittest.main()
