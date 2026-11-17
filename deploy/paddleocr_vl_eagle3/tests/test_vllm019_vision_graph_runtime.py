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

"""CPU-only checks for explicit vision graph admission and worker propagation."""
from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import profile_vllm019
import runtime_config as runtime
import serve_hps_vlm
import serve_ocr
import supervise_hps_vlm


class VisionGraphConfigurationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target, self.draft = self.root / "target", self.root / "draft"
        self.target.mkdir()
        self.draft.mkdir()
        self.versions = dict(runtime.PINNED_VERSIONS_019)
        version_patch = patch.object(
            runtime.importlib.metadata,
            "version",
            side_effect=lambda name: self.versions[name],
        )
        version_patch.start()
        self.addCleanup(version_patch.stop)

    def args(self, *extra):
        return serve_hps_vlm.parse_args(
            ["--model", str(self.target), "--draft-model", str(self.draft), *extra]
        )

    def test_default_is_disabled_and_both_validated_contexts_opt_in(self):
        self.assertFalse(self.args().vllm019_vision_graphs)
        for cap, bound in ((1024, 4096), (8192, 12288)):
            with self.subTest(cap=cap):
                args = self.args(
                    "--vllm019-vision-graphs", "--hps-max-completion-tokens", str(cap)
                )
                config, worker_args, _ = serve_ocr.build_engine_config(args)
                self.assertTrue(worker_args.vllm019_vision_graphs)
                self.assertEqual(config["max_model_len"], bound)
                self.assertEqual(config["max_num_seqs"], 1)
                self.assertEqual(
                    config["speculative_config"]["num_speculative_tokens"], 4
                )
                # The extension flag must never leak into AsyncEngineArgs.
                self.assertNotIn("vllm019_vision_graphs", config)

    def test_cli_rejects_baseline_bf16_and_untested_context(self):
        for extra in (
            ["--mode", "baseline"],
            ["--recovery-decode-graphs", "bf16"],
            ["--hps-max-completion-tokens", "4096"],
        ):
            with self.subTest(extra=extra), redirect_stderr(
                StringIO()
            ), self.assertRaises(SystemExit):
                self.args("--vllm019-vision-graphs", *extra)

    def test_other_runtime_rejected_before_environment_changes(self):
        self.versions.clear()
        self.versions.update(runtime.PINNED_VERSIONS)
        with self.assertRaisesRegex(ValueError, "requires VL-1.6 on vLLM 0.19"):
            serve_ocr.build_engine_config(self.args("--vllm019-vision-graphs"))

    def test_programmatic_profile_changes_cannot_bypass_admission(self):
        args = self.args("--vllm019-vision-graphs")
        config, _, _ = serve_ocr.build_engine_config(args)
        changes = (
            ("max_model_len", 8192),
            ("max_num_seqs", 2),
            ("tensor_parallel_size", 2),
            ("dtype", "float16"),
            ("async_scheduling", False),
            ("enforce_eager", True),
            ("worker_extension_cls", "hps_vllm019_worker.Vllm019EagleWorker"),
            ("speculative_config.method", "eagle"),
            ("speculative_config.num_speculative_tokens", 6),
            ("speculative_config.use_local_argmax_reduction", False),
            ("compilation_config.mode", 0),
            ("compilation_config.cudagraph_mode", "NONE"),
            ("compilation_config.cudagraph_capture_sizes", [1, 5]),
        )
        for name, value in changes:
            changed = deepcopy(config)
            if "." in name:
                section, key = name.split(".")
                changed[section][key] = value
            else:
                changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.validate_vision_graph_configuration(
                    args, changed, version="0.19.0"
                )
        for name, value in (
            ("mode", "baseline"),
            ("model_version", "1.0"),
            ("recovery_decode_graphs", "bf16"),
        ):
            changed_args = deepcopy(args)
            setattr(changed_args, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.validate_vision_graph_configuration(
                    changed_args, config, version="0.19.0"
                )

    def test_environment_revalidates_normalized_worker_config(self):
        config, worker_args, _ = serve_ocr.build_engine_config(
            self.args("--vllm019-vision-graphs")
        )
        self.assertEqual(worker_args.mode, "accelerated")
        config["max_num_seqs"] = 2
        with patch.dict(
            os.environ, {runtime.VISION_GRAPHS_ENV: "0"}
        ), self.assertRaises(ValueError):
            runtime.configure_environment(worker_args, config)

    def test_environment_resets_inherited_flag_in_both_runtimes(self):
        for versions in (runtime.PINNED_VERSIONS_019, runtime.PINNED_VERSIONS):
            self.versions.clear()
            self.versions.update(versions)
            _, worker_args, identity = serve_ocr.build_engine_config(self.args())
            with self.subTest(version=versions["vllm"]), patch.dict(
                os.environ, {runtime.VISION_GRAPHS_ENV: "1"}, clear=True
            ):
                runtime.configure_environment(worker_args, identity)
                self.assertEqual(os.environ[runtime.VISION_GRAPHS_ENV], "0")

    def test_enabled_flag_policy_and_source_participate_in_cache_identity(self):
        config, args, identity = serve_ocr.build_engine_config(self.args())
        with patch.dict(os.environ, {}, clear=True), patch.dict(
            sys.modules, {"vllm019_vision_graph": None}
        ):
            disabled = runtime.configure_environment(args, identity)["cache_identity"]
            args.vllm019_vision_graphs = True
            enabled = runtime.configure_environment(args, identity)["cache_identity"]
            self.assertEqual(os.environ[runtime.VISION_GRAPHS_ENV], "1")
            self.assertNotEqual(disabled["fingerprint"], enabled["fingerprint"])
            self.assertEqual(
                disabled["basis"]["engine_config"], enabled["basis"]["engine_config"]
            )
            self.assertEqual(
                enabled["basis"]["vision_graphs"],
                dict(enabled=True, **runtime.VISION_GRAPHS_POLICY),
            )
            # Leave headroom for DocLayoutV3 on the same GPU after warming
            # different crop shapes; the earlier 1.5 GiB cache exhausted it.
            self.assertEqual(
                enabled["basis"]["vision_graphs"]["max_cached_bytes"], 1024**3
            )
            self.assertEqual(
                enabled["basis"]["vision_graphs"]["minimum_free_bytes"],
                1536 * 1024**2,
            )
            source = runtime.ROOT / "scripts/vllm019_vision_graph.py"
            self.assertEqual(
                enabled["basis"]["current_dependency_sha256"][source.name],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            original_read = Path.read_bytes

            def changed_read(path):
                data = original_read(path)
                return (
                    data + b"\n# changed graph implementation\n"
                    if path == source
                    else data
                )

            with patch.object(Path, "read_bytes", changed_read):
                changed = runtime.configure_environment(args, identity)[
                    "cache_identity"
                ]
            self.assertNotEqual(enabled["fingerprint"], changed["fingerprint"])
            self.assertNotIn("vision_graphs", config)

    def supervisor_argv(self, *extra):
        return [
            "supervise_hps_vlm.py",
            "--run-dir",
            str(self.root / "run"),
            "--gpu",
            "0",
            "--port",
            "18780",
            "--mode",
            "eagle3-recovery",
            *extra,
        ]

    def test_supervisor_forwards_only_explicit_opt_in(self):
        for enabled in (False, True):
            argv = self.supervisor_argv(
                *(["--vllm019-vision-graphs"] if enabled else [])
            )
            with self.subTest(enabled=enabled), patch.object(
                sys, "argv", argv
            ), patch.object(supervise_hps_vlm, "supervise") as supervise, patch.object(
                supervise_hps_vlm.signal, "signal"
            ):
                supervise_hps_vlm.main()
                command = supervise.call_args.args[1]
                self.assertEqual("--vllm019-vision-graphs" in command, enabled)
                forwarded = serve_hps_vlm.parse_args(command[3:])
                self.assertEqual(forwarded.vllm019_vision_graphs, enabled)

    def test_supervisor_rejects_invalid_combinations_before_spawning(self):
        for extra in (
            ["--mode", "baseline"],
            ["--recovery-decode-graphs", "bf16"],
            ["--hps-max-completion-tokens", "4096"],
        ):
            argv = self.supervisor_argv("--vllm019-vision-graphs", *extra)
            with self.subTest(extra=extra), patch.object(
                sys, "argv", argv
            ), patch.object(
                supervise_hps_vlm, "supervise"
            ) as supervise, redirect_stderr(
                StringIO()
            ), self.assertRaises(
                SystemExit
            ):
                supervise_hps_vlm.main()
            supervise.assert_not_called()

    def test_profiler_preserves_the_explicit_service_flag_and_validation(self):
        for enabled in (False, True):
            args = profile_vllm019.parse_args(
                [
                    "--manifest",
                    str(self.root / "manifest.jsonl"),
                    "--output",
                    str(self.root / "profile.json"),
                    "--model",
                    str(self.target),
                    "--draft-model",
                    str(self.draft),
                    *(["--vllm019-vision-graphs"] if enabled else []),
                ]
            )
            argv = profile_vllm019.service_argv(args)
            with self.subTest(enabled=enabled):
                self.assertEqual("--vllm019-vision-graphs" in argv, enabled)
                service_args = serve_hps_vlm.parse_args(argv)
                config, worker_args, _ = serve_ocr.build_engine_config(service_args)
                self.assertEqual(worker_args.vllm019_vision_graphs, enabled)
                self.assertEqual(config["max_model_len"], 12288)


if __name__ == "__main__":
    unittest.main()
