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

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hps_batch import benchmark


def response(text="content", pages=1):
    return {
        "logId": "transport-id",
        "errorCode": 0,
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {"parsing_res_list": [{"block_content": text}]},
                    "markdown": {"text": text, "images": {"url": "ephemeral"}},
                }
                for _ in range(pages)
            ]
        },
    }


def recording(mode="baseline", wall=100):
    source = {
        "id": "private-id",
        "file_sha256": "a" * 64,
        "file_type": 1,
        "page_count": 1,
    }
    raw = response()
    row = dict(
        source,
        success=True,
        attempts=[
            {"http_status": 200, "error": None, "elapsed_ms": 9, "response": raw}
        ],
        outputs=benchmark.page_outputs(raw, 1),
        request_latency_ms=10,
        client_queue_ms=0,
        latency_including_client_queue_ms=10,
    )
    params = {"temperature": 0, "maxNewTokens": 8192}
    return {
        "format_version": 1,
        "status": "complete",
        "mode": mode,
        "protocol": {
            "repeats": 2,
            "warmup_runs": 0,
            "max_retries": 0,
            "expected_pages": 1,
            "concurrency": 16,
            "request_params": params,
            "request_params_sha256": benchmark.digest(params),
        },
        "inputs": [source],
        "input_identity_sha256": benchmark.digest([source]),
        "deployment_identity": {"shared_backend_config": {"max-num-seqs": 128}},
        "worker_class_audits": [
            {
                "eagle3_enabled": mode == "eagle3",
                "identity": {
                    "module": "test_target",
                    "qualname": "Target",
                    "adapter_installed": mode == "eagle3",
                    "methods": {
                        "forward": {"source_sha256": "same"},
                        "get_language_model": {"source_sha256": mode},
                    },
                },
            }
        ],
        "warmups": [],
        "trials": [
            {"wall_ms": wall, "records": [copy.deepcopy(row)]} for _ in range(2)
        ],
    }


class PairingTests(unittest.TestCase):
    def test_cpu_grid_requires_actual_processor_and_identical_placement(self):
        base, eagle = recording(), recording("eagle3", 40)
        eagle["deployment_identity"]["cpu_grid_enabled"] = True
        with self.assertRaises(ValueError):
            benchmark.worker_computation_identity(eagle)
        eagle["worker_class_audits"][0]["identity"]["cpu_grid_enabled"] = True
        with self.assertRaises(ValueError):
            benchmark.compare_data(base, eagle)
        base["deployment_identity"]["cpu_grid_enabled"] = True
        base["worker_class_audits"][0]["identity"]["cpu_grid_enabled"] = True
        self.assertTrue(benchmark.compare_data(base, eagle)["all_outputs_exact"])
        base["deployment_identity"]["cpu_grid_enabled"] = False
        with self.assertRaises(ValueError):
            benchmark.worker_computation_identity(base)

    def test_exact_output_and_speedup_report_contains_no_private_data(self):
        base, eagle = recording(), recording("eagle3", 40)
        # Transport identifiers and image URLs are not recognition content.
        for trial in eagle["trials"]:
            trial["records"][0]["attempts"][0]["response"]["logId"] = "different"
        report = benchmark.compare_data(base, eagle)
        self.assertTrue(report["twofold_and_exact_demonstrated"])
        self.assertEqual(report["median_wall_speedup"], 2.5)
        serialized = json.dumps(report)
        self.assertNotIn("private-id", serialized)
        self.assertNotIn("block_content", serialized)
        self.assertNotIn("a" * 64, serialized)
        self.assertEqual(report["exact_pages"], 2)

    def test_whitespace_and_layout_changes_fail_accuracy_gate(self):
        for field in ("markdown", "prunedResult"):
            eagle = recording("eagle3", 40)
            row = eagle["trials"][0]["records"][0]
            page = row["attempts"][0]["response"]["result"]["layoutParsingResults"][0]
            if field == "markdown":
                page["markdown"]["text"] += " "
            else:
                page["prunedResult"]["parsing_res_list"][0]["block_content"] = "changed"
            row["outputs"] = benchmark.page_outputs(row["attempts"][0]["response"], 1)
            report = benchmark.compare_data(recording(), eagle)
            self.assertFalse(report["all_outputs_exact"])
            self.assertFalse(report["twofold_and_exact_demonstrated"])

    def test_changed_inputs_concurrency_or_backend_cannot_be_compared(self):
        for key in ("inputs", "protocol", "deployment_identity"):
            eagle = recording("eagle3", 40)
            if key == "inputs":
                eagle["inputs"][0]["id"] = "changed"
                eagle["input_identity_sha256"] = benchmark.digest(eagle["inputs"])
            elif key == "protocol":
                eagle["protocol"]["concurrency"] = 1
            else:
                eagle["deployment_identity"]["shared_backend_config"][
                    "max-num-seqs"
                ] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark.compare_data(recording(), eagle)

    def test_fast_failed_run_never_counts_as_speedup(self):
        eagle = recording("eagle3", 1)
        eagle["status"] = "failed"
        row = eagle["trials"][0]["records"][0]
        row.update(success=False, outputs=None)
        row["attempts"] = [
            {
                "http_status": 429,
                "error": "http_error",
                "elapsed_ms": 1,
                "response": {"errorCode": 429},
            }
        ]
        report = benchmark.compare_data(recording(), eagle)
        self.assertIsNone(report["median_wall_speedup"])
        self.assertFalse(report["twofold_and_exact_demonstrated"])
        self.assertEqual(report["eagle3"]["http_429_attempts"], 1)

    def test_missing_results_and_forged_canonical_outputs_rejected(self):
        eagle = recording("eagle3", 40)
        eagle["trials"].pop()
        with self.assertRaises(ValueError):
            benchmark.compare_data(recording(), eagle)
        eagle = recording("eagle3", 40)
        eagle["trials"][0]["records"][0]["outputs"][0]["markdown"] = "forged"
        with self.assertRaises(ValueError):
            benchmark.compare_data(recording(), eagle)

    def test_unstable_baseline_is_not_an_accuracy_proof(self):
        base = recording()
        row = base["trials"][1]["records"][0]
        row["attempts"][0]["response"] = response("other")
        row["outputs"] = benchmark.page_outputs(row["attempts"][0]["response"], 1)
        report = benchmark.compare_data(base, recording("eagle3", 40))
        self.assertFalse(report["baseline_outputs_stable_across_repeats"])
        self.assertFalse(report["all_outputs_exact"])

    def test_invalid_request_timing_cannot_enter_aggregate(self):
        eagle = recording("eagle3", 40)
        eagle["trials"][0]["records"][0]["request_latency_ms"] = float("nan")
        with self.assertRaises(ValueError):
            benchmark.compare_data(recording(), eagle)

    def test_missing_wrong_mode_or_changed_worker_implementation_rejected(self):
        for issue in ("missing", "wrong_mode", "changed_vision"):
            eagle = recording("eagle3", 40)
            if issue == "missing":
                eagle["worker_class_audits"] = []
            elif issue == "wrong_mode":
                eagle["worker_class_audits"][0]["identity"]["adapter_installed"] = False
            else:
                eagle["worker_class_audits"][0]["identity"]["methods"]["forward"] = {
                    "source_sha256": "different"
                }
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                benchmark.compare_data(recording(), eagle)

    def test_manifest_preserves_pdf_request_grouping_and_page_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "example.pdf").write_bytes(b"pdf-fixture")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "document",
                        "file": "example.pdf",
                        "file_type": 0,
                        "page_count": 100,
                    }
                )
                + "\n"
            )
            identities, bodies = benchmark.prepare_inputs(
                manifest, {"maxNewTokens": 8192}, 100
            )
            self.assertEqual(len(bodies), 1)
            self.assertEqual(identities[0]["page_count"], 100)
            self.assertEqual(json.loads(bodies[0])["fileType"], 0)
            with self.assertRaises(ValueError):
                benchmark.prepare_inputs(manifest, {}, 99)
            with self.assertRaises(ValueError):
                benchmark.prepare_inputs(manifest, {"file": "override"}, 100)


class ConcurrentHTTPTests(unittest.TestCase):
    def test_client_uses_requested_concurrency_and_preserves_result_order(self):
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.035)
                data = json.dumps(response()).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                with lock:
                    state["active"] -= 1

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            inputs = [{"id": str(i), "page_count": 1} for i in range(12)]
            args = SimpleNamespace(
                url=f"http://127.0.0.1:{server.server_port}/layout-parsing",
                concurrency=4,
                timeout=2,
                max_retries=0,
                retry_delay=0,
            )
            result = benchmark.run_trial(inputs, [b"{}"] * len(inputs), args, {})
            self.assertGreaterEqual(state["peak"], 2)
            self.assertLessEqual(state["peak"], 4)
            self.assertEqual(
                [row["id"] for row in result["records"]], [str(i) for i in range(12)]
            )
            self.assertTrue(all(row["success"] for row in result["records"]))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
