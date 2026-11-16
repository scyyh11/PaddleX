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

"""CPU contracts for explicit artifact admission and pre-capture plan ownership."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import vllm019_packed_attention as packed
import runtime_config as runtime
from serve_hps_vlm import parse_args
from serve_ocr import build_engine_config
import supervise_hps_vlm


class ArtifactAndConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "bundle"
        shutil.copytree(ROOT / "kernels/fa2_sm80", self.bundle)
        self.build = packed._load_source("_packed_test_build", self.bundle / "build.py")
        self.library = self.root / self.build.LIBRARY_NAME
        self.library.write_bytes(b"CPU fixture: never dlopen")
        self.manifest = {
            "schema_version": 1,
            "profile": "paddleocr-vl-fa2-sm80-v3-finite",
            "gpu_arch": "sm80",
            "nvcc_version": "12.8.93",
            "torch_headers_version": "2.10.0",
            "cuda_source_sha256": self.build.EXPECTED_SOURCE_HASHES,
            "patched_header_sha256": self.build.PATCHED_KERNEL_SHA256,
            "library_name": self.library.name,
            "library_sha256": hashlib.sha256(self.library.read_bytes()).hexdigest(),
            "source_pins": {
                name: {key: value[key] for key in ("commit", "sha256")}
                for name, value in self.build.SOURCES.items()
            },
        }
        self.write_manifest()
        self.target, self.draft = self.root / "target", self.root / "draft"
        self.target.mkdir()
        self.draft.mkdir()
        self.versions = dict(runtime.PINNED_VERSIONS_019)
        versions = patch.object(
            runtime.importlib.metadata,
            "version",
            side_effect=lambda name: self.versions[name],
        )
        versions.start()
        self.addCleanup(versions.stop)

    def write_manifest(self):
        self.library.with_name("build_manifest.json").write_text(
            json.dumps(self.manifest)
        )

    def args(self, *extra):
        return parse_args(
            [
                "--model",
                str(self.target),
                "--draft-model",
                str(self.draft),
                "--vllm019-packed-attention-library",
                str(self.library),
                *extra,
            ]
        )

    def test_admission_hashes_manifest_and_library(self):
        result = packed.verify_artifact(self.library, bundle_root=self.bundle)
        self.assertEqual(result["library"], str(self.library.resolve()))
        self.assertEqual(result["library_sha256"], self.manifest["library_sha256"])
        self.assertEqual(
            result["manifest_sha256"],
            self.build.sha256(self.library.with_name("build_manifest.json")),
        )

    def test_changed_source_and_binary_are_rejected(self):
        source = self.bundle / "src/packed_mask.h"
        before = source.read_bytes()
        source.write_bytes(before + b"\n// changed")
        with self.assertRaisesRegex(ValueError, "source was changed"):
            packed.verify_artifact(self.library, bundle_root=self.bundle)
        source.write_bytes(before)
        self.library.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            packed.verify_artifact(self.library, bundle_root=self.bundle)

    def test_unvalidated_manifest_fields_are_rejected(self):
        original = copy.deepcopy(self.manifest)
        for key, value in (
            ("profile", "other"),
            ("gpu_arch", "sm90"),
            ("nvcc_version", "13.0.0"),
            ("cuda_source_sha256", {}),
            ("patched_header_sha256", "changed"),
            ("source_pins", {}),
            ("torch_headers_version", "2.13.0"),
            ("library_name", "other.so"),
        ):
            self.manifest = copy.deepcopy(original)
            self.manifest[key] = value
            self.write_manifest()
            with self.subTest(key=key), self.assertRaises(ValueError):
                packed.verify_artifact(self.library, bundle_root=self.bundle)

    def test_cli_requires_recovery_int8_and_supported_bound(self):
        for extra in (
            ("--mode", "baseline"),
            ("--recovery-decode-graphs", "bf16"),
            ("--hps-max-completion-tokens", "4096"),
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                self.args(*extra)
        for cap, bound in ((1024, 4096), (8192, 12288)):
            config, _, _ = build_engine_config(
                self.args("--hps-max-completion-tokens", str(cap))
            )
            self.assertEqual(config["max_model_len"], bound)
            self.assertEqual(config["max_num_seqs"], 1)

    def test_032_and_programmatic_unsupported_config_are_rejected(self):
        args = self.args()
        config, _, _ = build_engine_config(args)
        for changes in (
            {"max_model_len": 8192},
            {"max_num_seqs": 2},
            {"tensor_parallel_size": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                packed.validate_configuration(
                    args, dict(config, **changes), version="0.19.0"
                )
        self.versions["vllm"] = "0.32.0"
        with self.assertRaisesRegex(ValueError, "requires vLLM 0.19"):
            build_engine_config(args)

    def test_environment_propagates_only_validated_explicit_artifact(self):
        config, args, identity = build_engine_config(self.args())
        with patch.dict(os.environ, {}, clear=True):
            result = runtime.configure_environment(args, identity)
            self.assertEqual(os.environ[packed.LIBRARY_ENV], str(self.library))
            basis = result["cache_identity"]["basis"]
            self.assertEqual(
                basis["packed_attention"]["library_sha256"],
                self.manifest["library_sha256"],
            )
            self.assertTrue(
                set(packed.SOURCE_NAMES).issubset(basis["current_dependency_sha256"])
            )
            self.manifest["build_note"] = "different manifest bytes"
            self.write_manifest()
            changed = runtime.configure_environment(args, identity)
            self.assertNotEqual(
                result["cache_identity"]["fingerprint"],
                changed["cache_identity"]["fingerprint"],
            )
            args.vllm019_packed_attention_library = None
            disabled = runtime.configure_environment(args, identity)
            self.assertNotIn(packed.LIBRARY_ENV, os.environ)
            self.assertNotIn("packed_attention", disabled["cache_identity"]["basis"])

    def test_supervisor_forwards_explicit_library_and_default_omits_it(self):
        base = [
            "supervise_hps_vlm.py",
            "--run-dir",
            str(self.root / "run"),
            "--gpu",
            "0",
            "--port",
            "18780",
            "--mode",
            "eagle3-recovery",
        ]
        for supplied in (False, True):
            argv = base + (
                ["--vllm019-packed-attention-library", str(self.library)]
                if supplied
                else []
            )
            with self.subTest(supplied=supplied), patch.object(
                sys, "argv", argv
            ), patch.object(supervise_hps_vlm, "supervise") as supervise, patch.object(
                supervise_hps_vlm.signal, "signal"
            ):
                supervise_hps_vlm.main()
                command = supervise.call_args.args[1]
                if supplied:
                    index = command.index("--vllm019-packed-attention-library")
                    self.assertEqual(command[index + 1], str(self.library.resolve()))
                else:
                    self.assertNotIn("--vllm019-packed-attention-library", command)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        for name, value in (
            ("PLANS", {}),
            ("COUNTS", packed.Counter()),
            ("_ARTIFACT", {"library": "validated.so"}),
        ):
            item = patch.object(packed, name, value)
            item.start()
            self.addCleanup(item.stop)

    def tensor(self, shape, dtype=None, pointer=4096):
        tensor = Mock(spec=self.torch.Tensor)
        tensor.shape, tensor.ndim = shape, len(shape)
        tensor.dtype = dtype or self.torch.bfloat16
        tensor.device, tensor.is_cuda = self.torch.device("cuda:0"), True
        strides, product = [], 1
        for dim in reversed(shape):
            strides.insert(0, product)
            product *= dim
        tensor.stride.side_effect = lambda axis=None: (
            tuple(strides) if axis is None else strides[axis]
        )
        tensor.data_ptr.return_value = pointer
        tensor.untyped_storage.return_value.data_ptr.return_value = pointer
        return tensor

    def inputs(self):
        q = self.tensor((5, 16, 128))
        kw = dict(
            q=q,
            k=self.tensor((256, 16, 2, 128), pointer=8192),
            v=self.tensor((256, 16, 2, 128), pointer=12288),
            cu_seqlens_q=self.tensor((2,), self.torch.int32),
            seqused_k=self.tensor((1,), self.torch.int32),
            block_table=self.tensor((1, 768), self.torch.int32),
            max_seqlen_k=12288,
        )
        return kw, self.tensor((5, 16, 128), pointer=16384)

    def test_current_layout_is_admitted_and_bad_layout_falls_back(self):
        kw, out = self.inputs()
        self.assertIsNone(packed._fallback_reason(kw, out, None))
        for key, value, reason in (
            ("max_seqlen_k", 8192, "context_bound"),
            ("q", self.tensor((6, 16, 128)), "qkv_layout"),
            ("k", self.tensor((256, 32, 2, 128)), "qkv_layout"),
            ("block_table", self.tensor((1, 767), self.torch.int32), "metadata_layout"),
            ("cu_seqlens_q", self.tensor((2,), self.torch.int64), "metadata_layout"),
            ("softmax_scale", 1.0, "softmax_scale"),
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    packed._fallback_reason(dict(kw, **{key: value}), out, None), reason
                )
        self.assertEqual(packed._fallback_reason(kw, kw["q"], None), "output_alias")

    def test_missing_plan_calls_original_without_constructing(self):
        kw, out = self.inputs()
        original = Mock(return_value=out)
        with patch.object(
            packed, "_PLAN_TYPE", side_effect=AssertionError("No call-time constructor")
        ):
            self.assertIs(packed._make_attention(original)(kw, out), out)
        original.assert_called_once_with(kw, out, None)
        self.assertEqual(packed.COUNTS["fallback_unprepared_plan"], 1)
        self.assertEqual(packed.COUNTS["packed_calls"], 0)

    def test_prepared_plan_preserves_original_inputs_and_external_output(self):
        kw, out = self.inputs()
        original, plan = Mock(), Mock(return_value=out)
        key = packed._plan_key(5, 12288, 768, kw["q"].device)
        packed.PLANS[key] = plan
        self.assertIs(packed._make_attention(original)(kw, out), out)
        original.assert_not_called()
        plan.assert_called_once_with(
            kw["q"],
            kw["k"],
            kw["v"],
            kw["cu_seqlens_q"],
            kw["seqused_k"],
            kw["block_table"],
            out=out,
            softmax_scale=None,
        )
        self.assertEqual(packed.COUNTS["packed_calls"], 1)

    def test_explicit_kernel_is_passed_to_original(self):
        kw, out = self.inputs()
        original, kernel = Mock(return_value=out), object()
        packed._make_attention(original)(kw, out, kernel)
        original.assert_called_once_with(kw, out, kernel)
        self.assertEqual(packed.COUNTS["fallback_explicit_kernel"], 1)

    def runner(self):
        table = SimpleNamespace(shape=(1, 769), device=self.torch.device("cuda:0"))
        return SimpleNamespace(
            vllm_config=SimpleNamespace(
                speculative_config=SimpleNamespace(
                    method="eagle3", num_speculative_tokens=4
                ),
                model_config=SimpleNamespace(max_model_len=12288),
                scheduler_config=SimpleNamespace(max_num_seqs=1),
            ),
            input_batch=SimpleNamespace(
                block_table=[SimpleNamespace(block_table=SimpleNamespace(gpu=table))]
            ),
        )

    def test_capture_prepares_from_actual_table_before_original_and_only_once(self):
        runner, events = self.runner(), []
        plan = object()
        factory = Mock(side_effect=lambda *a, **kw: (events.append("prepare"), plan)[1])
        original = Mock(side_effect=lambda *a, **kw: (events.append("capture"), 123)[1])
        wrapper = packed._make_capture(original)
        with patch.object(packed, "_PLAN_TYPE", factory):
            self.assertEqual(wrapper(runner, "arg", option=True), 123)
            self.assertEqual(wrapper(runner), 123)
        self.assertEqual(events, ["prepare", "capture", "capture"])
        factory.assert_called_once_with(
            5,
            12288,
            769,
            device=self.torch.device("cuda:0"),
            library_path="validated.so",
        )
        self.assertIs(next(iter(packed.PLANS.values())), plan)
        original.assert_any_call(runner, "arg", option=True)

    def test_invalid_runner_never_constructs_or_captures(self):
        for path, value in (
            ("max_num_seqs", 2),
            ("num_speculative_tokens", 6),
            ("max_model_len", 8192),
        ):
            runner = self.runner()
            for config in vars(runner.vllm_config).values():
                if hasattr(config, path):
                    setattr(config, path, value)
            original, factory = Mock(), Mock()
            with self.subTest(path=path), patch.object(
                packed, "_PLAN_TYPE", factory
            ), self.assertRaises(RuntimeError):
                packed._make_capture(original)(runner)
            original.assert_not_called()
            factory.assert_not_called()

    def test_disabled_install_does_not_import_or_validate_artifact(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            packed, "verify_artifact"
        ) as verify:
            self.assertFalse(packed.install()["enabled"])
        verify.assert_not_called()

    def test_install_wraps_attention_and_capture_once_and_keeps_owners(self):
        attention = ModuleType("vllm019_ar1_attention")
        attention._batched_attention = Mock()
        original_attention = attention._batched_attention

        class Runner:
            capture_model = Mock()

        original_capture = Runner.capture_model
        runner_module = ModuleType("vllm.v1.worker.gpu_model_runner")
        runner_module.GPUModelRunner = Runner
        modules = {
            name: ModuleType(name) for name in ("vllm", "vllm.v1", "vllm.v1.worker")
        }
        modules.update(
            {
                "vllm.v1.worker.gpu_model_runner": runner_module,
                "vllm019_ar1_attention": attention,
            }
        )
        adapter = SimpleNamespace(PackedFA2=Mock())
        artifact = {
            "library": "validated.so",
            "library_sha256": "a",
            "manifest_sha256": "b",
        }
        with patch.dict(sys.modules, modules), patch.object(
            packed, "_INSTALLATION", None
        ), patch.object(
            packed, "verify_artifact", return_value=artifact
        ) as verify, patch.object(
            packed, "_load_source", return_value=adapter
        ) as load, patch.object(
            packed.importlib.metadata, "version", return_value="0.19.0"
        ), patch.object(
            packed, "_PLAN_TYPE", None
        ):
            self.assertTrue(packed.install("validated.so")["enabled"])
            self.assertIs(packed._INSTALLATION[0], adapter)
            self.assertIs(packed._INSTALLATION[3], original_attention)
            self.assertIs(packed._INSTALLATION[4], original_capture)
            wrapped_attention, wrapped_capture = (
                attention._batched_attention,
                Runner.capture_model,
            )
            packed.install("validated.so")
            self.assertIs(attention._batched_attention, wrapped_attention)
            self.assertIs(Runner.capture_model, wrapped_capture)
            load.assert_called_once()
            verify.return_value = dict(artifact, library_sha256="different")
            with self.assertRaisesRegex(RuntimeError, "while graphs are alive"):
                packed.install("changed.so")


if __name__ == "__main__":
    unittest.main()
