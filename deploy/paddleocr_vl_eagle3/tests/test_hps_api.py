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

"""CPU-only HTTP protocol checks with a stubbed recognition backend."""
import base64
import copy
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_hps_vlm import create_hps_app


class Sampling:
    max_tokens = 1024
    skip_special_tokens = True

    def clone(self):
        return copy.copy(self)


class StubResident:
    mode = "eagle3-recovery"
    admitted = 0
    queue_size = 1
    max_completion_tokens = 8192
    sampling = Sampling()

    def __init__(self):
        self.calls = []
        self.reason = "stop"

    async def run_queued(self, task, pixels, sampling, options, pixel_limit):
        self.calls.append((task, pixels, sampling, options, pixel_limit))
        return dict(
            request_id="fixture",
            text="ok",
            finish_reason=self.reason,
            prompt_tokens=3,
            output_tokens=2,
            token_ids=[11, 2],
            queue_ms=0.0,
            generate_ms=1.0,
        )


class HpsApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = create_hps_app(
            SimpleNamespace(mode="eagle3-recovery", model_version="1.6")
        )
        self.resident = self.app.state.ocr = StubResident()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://asgi"
        )
        self.addAsyncCleanup(self.client.aclose)
        buf = BytesIO()
        Image.new("RGB", (28, 28), "red").save(buf, format="PNG")
        self.pixels = buf.getvalue()
        self.payload = {
            "model": "PaddleOCR-VL-1.6-0.9B",
            "max_completion_tokens": 8192,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "OCR:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(self.pixels).decode()
                            },
                        },
                    ],
                }
            ],
        }

    async def test_model_id_exact_pixels_and_sampling(self):
        response = await self.client.post("/v1/chat/completions", json=self.payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["ocr_details"]["token_ids"], [11, 2])
        task, pixels, sampling, _, _ = self.resident.calls[0]
        self.assertEqual(
            (task, pixels, sampling.max_tokens), ("ocr", self.pixels, 8192)
        )
        self.assertEqual(self.resident.sampling.max_tokens, 1024)
        models = (await self.client.get("/v1/models")).json()
        self.assertEqual(models["data"][0]["id"], "PaddleOCR-VL-1.6-0.9B")

    async def test_invalid_model_limit_and_sampling_are_rejected(self):
        for field, value in [
            ("model", "PaddleOCR-VL-0.9B"),
            ("max_completion_tokens", 8193),
            ("max_completion_tokens", True),
            ("temperature", 0.5),
            ("repetition_penalty", 1.1),
            ("stream", True),
        ]:
            response = await self.client.post(
                "/v1/chat/completions", json={**self.payload, field: value}
            )
            self.assertEqual(response.status_code, 422, (field, response.text))
        self.assertFalse(self.resident.calls)

    async def test_length_is_preserved_and_admission_is_bounded(self):
        self.resident.reason = "length"
        response = await self.client.post("/v1/chat/completions", json=self.payload)
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")
        self.assertEqual(self.resident.admitted, 0)
        self.resident.admitted = 2
        response = await self.client.post("/v1/chat/completions", json=self.payload)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(len(self.resident.calls), 1)


if __name__ == "__main__":
    unittest.main()
