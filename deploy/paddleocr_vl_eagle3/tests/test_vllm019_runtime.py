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

"""Exercise the 0.19 host API without importing vLLM or allocating a GPU."""
from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runtime_config as runtime
import serve_ocr
from serve_hps_vlm import parse_args
from vllm019_packed_attention import LIBRARY_ENV, SOURCE_NAMES as PACKED_SOURCES


class Vllm019ConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
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
        return parse_args(
            ["--model", str(self.target), "--draft-model", str(self.draft), *extra]
        )

    def test_019_uses_supported_engine_and_speculative_options(self):
        args = self.args()
        for graph_mode, expected_worker in (
            ("int8", "hps_vllm019_int8_worker.Vllm019Int8Worker"),
            ("bf16", "hps_vllm019_worker.Vllm019EagleWorker"),
            (None, "hps_vllm019_worker.Vllm019EagleWorker"),
        ):
            with self.subTest(graph_mode=graph_mode):
                args.recovery_decode_graphs = graph_mode
                config, _, _ = serve_ocr.build_engine_config(args)
                self.assertNotIn("per_request_spec_decode_metrics", config)
                spec = config["speculative_config"]
                self.assertNotIn("draft_sample_method", spec)
                self.assertNotIn("enable_adaptive_verification", spec)
                self.assertEqual(spec["method"], "eagle3")
                self.assertEqual(spec["num_speculative_tokens"], 4)
                self.assertEqual(config["worker_extension_cls"], expected_worker)
                self.assertEqual(config["async_scheduling"], graph_mode == "int8")
                if graph_mode == "int8":
                    self.assertEqual(config["max_num_seqs"], 1)
                self.assertEqual(config["dtype"], "bfloat16")
                self.assertEqual(
                    spec["use_local_argmax_reduction"], graph_mode == "int8"
                )
                self.assertEqual(
                    config["compilation_config"]["cudagraph_mode"],
                    "FULL_AND_PIECEWISE" if graph_mode else "NONE",
                )

    def test_baseline_needs_no_draft_or_speculative_worker(self):
        self.draft.rmdir()
        config, _, _ = serve_ocr.build_engine_config(self.args("--mode", "baseline"))
        self.assertNotIn("per_request_spec_decode_metrics", config)
        self.assertNotIn("speculative_config", config)
        self.assertNotIn("worker_extension_cls", config)

    def test_environment_selects_v1_runner_and_hashes_019_sources(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in (
            runtime.RUNTIME_SOURCES
            + (
                "vllm019_model.py",
                "hps_vllm019_worker.py",
                "hps_vllm019_int8_worker.py",
                "vllm019_ar1_attention.py",
                "vllm019_draft_loop.py",
                "vllm019_draft_full_graph.py",
                "vllm019_draft_graph_buckets.py",
                "vllm019_draft_staging.py",
                "vllm019_attention_metadata.py",
                "vllm019_greedy_sampler.py",
                "vllm019_spec_metadata.py",
                "vllm019_image_lut.py",
                "vllm019_locked_lt.py",
                "vllm019_target_linear.py",
                "vllm019_stable_argmax.py",
                "vllm019_graph_argmax.py",
                "vllm019_sampler_graph.py",
                "vllm019_vision_scalar.py",
                "vllm019_vision_graph.py",
            )
            + PACKED_SOURCES
        ):
            path = scripts / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n")
        config, args, _ = serve_ocr.build_engine_config(self.args())
        with patch.object(runtime, "ROOT", self.root), patch.dict(
            os.environ,
            {"VLLM_USE_V2_MODEL_RUNNER": "1", LIBRARY_ENV: "stale.so"},
            clear=True,
        ):
            first = runtime.configure_environment(args, config)
            self.assertEqual(os.environ["VLLM_USE_V2_MODEL_RUNNER"], "0")
            self.assertEqual(os.environ["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
            self.assertNotIn(LIBRARY_ENV, os.environ)
            basis = first["cache_identity"]["basis"]
            self.assertEqual(basis["runtime_package_versions"], self.versions)
            previous = first
            for name in (
                "vllm019_model.py",
                "vllm019_locked_lt.py",
                "vllm019_target_linear.py",
                "vllm019_vision_graph.py",
                *PACKED_SOURCES,
            ):
                with self.subTest(source=name):
                    self.assertIn(name, basis["current_dependency_sha256"])
                    (scripts / name).write_text("# changed adapter\n")
                    changed = runtime.configure_environment(args, config)
                    self.assertNotEqual(
                        previous["cache_identity"]["fingerprint"],
                        changed["cache_identity"]["fingerprint"],
                    )
                    previous = changed
            self.versions["torch"] = "2.13.0+cu130"
            with self.assertRaisesRegex(RuntimeError, "Unvalidated runtime versions"):
                runtime.configure_environment(args, config)

    def test_unported_vllm_version_is_rejected(self):
        self.versions["vllm"] = "0.19.1"
        with self.assertRaisesRegex(RuntimeError, "Unsupported vLLM runtime"):
            serve_ocr.build_engine_config(self.args())


class EngineDeadError(Exception):
    pass


class LegacyEngine:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []

    async def generate(self, model_input, sampling, request_id):
        self.calls.append((model_input, sampling, request_id))
        if self.failure is not None:
            raise self.failure
        # vLLM 0.19 CompletionOutput has no spec_decode_metrics attribute.
        output = SimpleNamespace(
            text="recognized", token_ids=[17, 18], finish_reason="length"
        )
        yield SimpleNamespace(
            finished=True, outputs=[output], prompt_token_ids=[3, 4, 5]
        )

    async def check_health(self):
        return None


class Vllm019RecognitionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        version_patch = patch.object(serve_ocr, "is_vllm019", return_value=True)
        version_patch.start()
        self.addCleanup(version_patch.stop)
        exceptions = ModuleType("vllm.v1.engine.exceptions")
        exceptions.EngineDeadError = EngineDeadError
        module_patch = patch.dict(
            sys.modules, {"vllm.v1.engine.exceptions": exceptions}
        )
        module_patch.start()
        self.addCleanup(module_patch.stop)
        with BytesIO() as stream:
            with Image.new("RGB", (28, 28), "white") as picture:
                picture.save(stream, format="PNG")
            self.pixels = stream.getvalue()

    def resident(self, failure=None):
        return serve_ocr.Resident(
            engine=LegacyEngine(failure),
            sampling=SimpleNamespace(temperature=0, max_tokens=1024),
            prompts={"ocr": "OCR:"},
            processor=None,
            mode="eagle3-recovery",
            queue_size=8,
            recovery_worker={"method": "eagle3"},
            request_metrics_available=False,
        )

    async def test_legacy_output_preserves_tokens_and_finish_without_metrics(self):
        resident = self.resident()
        result = await resident.recognize(
            "ocr", self.pixels, "request", mm_processor_kwargs={"max_pixels": 784}
        )
        self.assertEqual(result["token_ids"], [17, 18])
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(resident.completed, 1)
        model_input, _, request_id = resident.engine.calls[0]
        self.assertEqual(model_input["mm_processor_kwargs"], {"max_pixels": 784})
        self.assertEqual(request_id, "request")

    async def test_validation_and_engine_failures_remain_distinct(self):
        for failure, status in (
            (ValueError("prompt too long"), 422),
            (EngineDeadError("engine failed"), 503),
        ):
            with self.subTest(status=status):
                resident = self.resident(failure)
                with self.assertRaises(HTTPException) as caught:
                    await resident.recognize("ocr", self.pixels, "request")
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(resident.completed, 0)
        resident = self.resident(RuntimeError("unexpected execution error"))
        with self.assertRaisesRegex(RuntimeError, "unexpected execution error"):
            await resident.recognize("ocr", self.pixels, "request")

    async def test_readiness_reports_missing_request_metrics_as_unavailable(self):
        app = serve_ocr.create_app(
            SimpleNamespace(mode="eagle3-recovery", model_version="1.6")
        )
        app.state.ocr = self.resident()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://asgi"
        ) as client:
            response = await client.get("/health/ready")
        self.assertEqual(response.status_code, 200, response.text)
        evidence = response.json()["speculative_decoding"]
        self.assertFalse(evidence["request_metrics_available"])
        for key in ("verification_steps", "draft_tokens", "accepted_draft_tokens"):
            self.assertIsNone(evidence[key])
        self.assertEqual(evidence["worker"]["method"], "eagle3")


if __name__ == "__main__":
    unittest.main()
