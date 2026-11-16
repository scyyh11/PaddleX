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

"""CPU-only regression checks for long crops and page-level outage handling."""
import asyncio
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_eagle3_client as cli
import benchmark_eagle3_page as page
from benchmark_eagle3_client import process_task, register_and_run


class PageRecoveryTests(TestCase):
    def check_length_retry(self, reasons, cap):
        inputs = []

        async def reply(request):
            payload = json.loads(await request.aread())
            inputs.append(payload)
            reason = reasons[len(inputs) - 1]
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": reason, "message": {"content": reason}}
                    ]
                },
            )

        async def run():
            async with page.PageCropClient(
                transport=httpx.MockTransport(reply)
            ) as client:
                client.completion_limit = cap
                response = await client.post(
                    "http://127.0.0.1:9999/v1/chat/completions",
                    json={
                        "max_completion_tokens": 1024,
                        "messages": [
                            {"role": "user", "content": "exact original payload"}
                        ],
                    },
                )
                return response.json()["choices"][0]["finish_reason"], client.incomplete

        reason, error = asyncio.run(run())
        self.assertEqual(
            [r["max_completion_tokens"] for r in inputs], [1024, cap][: len(reasons)]
        )
        self.assertTrue(all(r["messages"] == inputs[0]["messages"] for r in inputs))
        self.assertEqual(reason, reasons[-1])
        self.assertEqual(bool(error), reason != "stop")

    def test_length_retry(self):
        for reasons, cap in [
            (["length", "stop"], 8192),
            (["length", "length"], 8192),
            (["stop"], 8192),
            (["length"], 1024),
        ]:
            self.check_length_retry(reasons, cap)

    def recognizer(self, instances, results):
        parser = page.PageRecognizer.__new__(page.PageRecognizer)
        parser._http = Mock()
        parser._completion_limit = 1024
        parser._backend_failure_pages = 0
        parser._suspended_instance = None
        parser.wait_for_backend = Mock(side_effect=instances)
        parser.pipeline = Mock()
        parser.pipeline.predict.side_effect = results
        return parser

    @patch.object(page.time, "sleep")
    @patch.object(page, "_read_image", return_value=(object(), (10, 10), 1))
    @patch.object(page, "_platform_result", side_effect=lambda result, *args: result)
    def test_page_recovery(self, convert, read, sleep):
        parser = self.recognizer(["A", "B", "B", "B"], [["partial-A"], ["complete-B"]])
        self.assertEqual(parser.infer_single_image(b"image", "page"), "complete-B")
        self.assertEqual(parser.pipeline.predict.call_count, 2)
        parser = self.recognizer(
            ["A", "B", "B"], [httpx.ConnectError("reset"), ["restored"]]
        )
        self.assertEqual(parser.infer_single_image(b"image", "page"), "restored")
        parser = self.recognizer(["A"], [ValueError("malformed output")])
        with self.assertRaisesRegex(ValueError, "malformed"):
            parser.infer_single_image(b"image", "page")
        self.assertEqual(parser.pipeline.predict.call_count, 1)
        parser = self.recognizer(
            ["A", page.PageBackendUnavailable("offline")], [httpx.ConnectError("reset")]
        )
        with self.assertRaises(page.PageBackendUnavailable):
            parser.infer_single_image(b"image", "page")
        self.assertEqual(parser.pipeline.predict.call_count, 1)

    def test_expired_readiness(self):
        parser = page.PageRecognizer.__new__(page.PageRecognizer)
        parser._ready_instance = Mock(side_effect=httpx.ConnectError("offline"))
        with patch.object(page, "PAGE_RECOVERY_SECONDS", 0), self.assertRaises(
            page.PageBackendUnavailable
        ):
            parser.wait_for_backend()

    def test_pin_profile_across_instance_replacement(self):
        parser = page.PageRecognizer.__new__(page.PageRecognizer)
        parser._profile = None
        parser.vlm_version = "1.6"
        parser._verified_model_instance = None
        parser.wire_model = "PaddleOCR-VL-1.6-0.9B"
        parser.origin = "http://127.0.0.1:9999"
        worker = dict(
            method="eagle3",
            target="PaddleOCRVLForConditionalGeneration",
            checkpoint=str(page.V16_DRAFT),
            k=4,
        )

        def response(instance, mode, worker):
            data = dict(
                instance_id=instance,
                status="ready",
                mode=mode,
                model_path=str(page.V16_TARGET),
                draft_model_path=str(page.V16_DRAFT),
                speculative_decoding={"worker": worker},
            )
            return Mock(json=Mock(return_value=data))

        models = Mock(json=Mock(return_value={"data": [{"id": parser.wire_model}]}))
        with patch.object(page.httpx, "Client") as client:
            client.return_value.__enter__.return_value.get.side_effect = [
                response("a", "eagle3-recovery", worker),
                models,
                response("b", "eagle3-recovery", worker),
                models,
                response(
                    "c", "eagle3-recovery", dict(worker, target_attention="different")
                ),
                models,
                response("d", "baseline", {}),
            ]
            self.assertEqual(parser._ready_instance(), "a")
            self.assertEqual(parser._ready_instance(), "b")
            with self.assertRaises(page.PageBackendUnavailable):
                parser._ready_instance()
            with self.assertRaises(page.PageBackendUnavailable):
                parser._ready_instance()

    def test_v16_rejects_wrong_target_draft_or_advertised_model(self):
        parser = page.PageRecognizer.__new__(page.PageRecognizer)
        parser._profile = parser._verified_model_instance = None
        parser.vlm_version = "1.6"
        parser.wire_model = "PaddleOCR-VL-1.6-0.9B"
        parser.origin = "http://127.0.0.1:18780"
        worker = dict(
            method="eagle3",
            target="PaddleOCRVLForConditionalGeneration",
            checkpoint=str(page.V16_DRAFT),
            k=4,
        )
        health = dict(
            status="ready",
            mode="eagle3-recovery",
            instance_id="v16",
            model_path=str(page.V16_TARGET),
            draft_model_path=str(page.V16_DRAFT),
            speculative_decoding={"worker": worker},
        )
        models = {"data": [{"id": parser.wire_model}]}
        inputs = [
            dict(health, model_path="v1"),
            dict(health, draft_model_path="old"),
            health,
            {"data": [{"id": "PaddleOCR-VL-0.9B"}]},
            health,
            models,
            health,
        ]
        with patch.object(page.httpx, "Client") as client:
            client.return_value.__enter__.return_value.get.side_effect = [
                Mock(json=Mock(return_value=d)) for d in inputs
            ]
            for _ in range(3):
                with self.assertRaises(page.PageBackendUnavailable):
                    parser._ready_instance()
            self.assertEqual(parser._ready_instance(), "v16")
            self.assertEqual(parser._ready_instance(), "v16")

    def test_backend_outage_suspends_task_before_failed_results(self):
        ws = Mock(send=AsyncMock())
        worker = Mock()
        worker.select_task.return_value = ("page", "test")
        worker.download_and_infer.side_effect = page.PageBackendUnavailable("offline")
        message = {"experiment_id": "offline-test", "files": {"p1": "/p1", "p2": "/p2"}}
        with ThreadPoolExecutor(max_workers=1) as executor, self.assertRaises(
            page.PageBackendUnavailable
        ):
            asyncio.run(process_task(ws, message, worker, "fake-fixture", executor))
        self.assertEqual(worker.download_and_infer.call_count, 1)
        self.assertEqual(
            {json.loads(c.args[0])["type"] for c in ws.send.await_args_list}, {"log"}
        )

    def test_re_registration_waits_for_backend(self):
        args = SimpleNamespace(
            server_url="http://127.0.0.1:9999", model_name="test-model"
        )
        ws = Mock(
            send=AsyncMock(), recv=AsyncMock(return_value='{"type":"registered"}')
        )
        connection = MagicMock(uri="ws://127.0.0.1:9999/api/client/ws")
        connection.__aenter__.return_value = ws
        worker = SimpleNamespace(models={"page": Mock()})
        worker.models["page"].wait_for_backend.side_effect = [
            page.PageBackendUnavailable("offline"),
            "restored",
        ]
        with patch(
            "benchmark_eagle3_client.connect", return_value=connection
        ) as connect, patch(
            "benchmark_eagle3_client.serve_connection",
            AsyncMock(side_effect=[page.PageBackendUnavailable(), None]),
        ), patch.object(
            asyncio, "sleep", AsyncMock()
        ), ThreadPoolExecutor(
            max_workers=1
        ) as executor:
            asyncio.run(register_and_run(args, worker, "fake-fixture", executor))
        self.assertEqual(connect.call_count, 2)
        self.assertEqual(worker.models["page"].wait_for_backend.call_count, 2)


class PageCliTests(TestCase):
    def test_registration_requires_explicit_platform_origin(self):
        with patch.object(cli, "DEFAULT_SERVER", None), patch.object(
            sys, "argv", ["client"]
        ), patch.object(cli, "run", AsyncMock()) as run, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli.main()
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()

    def test_local_page_check_does_not_require_platform_configuration(self):
        with patch.object(cli, "DEFAULT_SERVER", None), patch.object(
            sys, "argv", ["client", "--self-test-image", "fixture.png"]
        ), patch.object(cli, "run", AsyncMock()) as run:
            self.assertEqual(cli.main(), 0)
        args = run.call_args.args[0]
        self.assertIsNone(args.server_url)
        self.assertEqual(args.pipeline, "page")
        self.assertEqual(args.vlm_version, "1.6")
        self.assertEqual(args.layout_model, "PP-DocLayoutV3")


if __name__ == "__main__":
    main()
