#!/usr/bin/env python3
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

"""Register the VL-1.6 + DocLayoutV3 full-page recognizer with a document benchmark."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import requests
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from benchmark_eagle3_page import (
    LAYOUT_MODELS,
    PAGE_IMAGE_LIMIT,
    PAGE_PIPELINES,
    PAGE_VLM_URLS,
    PageBackendUnavailable,
)


DEFAULT_SERVER = os.getenv("OCR_BENCH_SERVER_URL", "").strip() or None
PAGE_TYPES = ("e2e", "pip")


def load_token() -> str:
    token = os.getenv("OCR_BENCH_API_TOKEN", "").strip()
    token_file = os.getenv("OCR_BENCH_TOKEN_FILE")
    if not token and token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token or "..." in token or "…" in token or any(c.isspace() for c in token):
        raise ValueError(
            "Set a complete token in OCR_BENCH_API_TOKEN or OCR_BENCH_TOKEN_FILE."
        )
    return token


def server_origin(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "--server-url must contain only the HTTP(S) scheme, host and optional port"
        )
    return f"{parts.scheme}://{parts.netloc}"


def kind_name(value: str, pipeline: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Invalid benchmark experiment type")
    kind = value.strip().lower().removesuffix("_experiment")
    if pipeline == "page" and kind in PAGE_TYPES:
        return "page"
    return kind


class Worker:
    """Initialize, run and close the recognizer on a single executor thread."""

    def __init__(self, args):
        self.url = args.vlm_url
        self.origin = args.server_url
        self.pipeline = args.pipeline
        self.device = args.device
        self.layout_model = args.layout_model
        self.vlm_version = args.vlm_version
        self.default_task = "page"
        self.image_limit = PAGE_IMAGE_LIMIT
        self.models = {}
        self.timing_stream = None
        if args.timing_log is not None:
            args.timing_log.parent.mkdir(parents=True, exist_ok=True)
            self.timing_stream = args.timing_log.open(
                "a", encoding="utf-8", buffering=1
            )
        self.http = requests.Session()
        self.http.trust_env = False

    def model(self, task):
        if task not in self.models:
            self.supported_kind(task)
            from benchmark_eagle3_page import PageRecognizer

            self.models[task] = PageRecognizer(
                vlm_url=self.url,
                device=self.device,
                layout_model=self.layout_model,
                vlm_version=self.vlm_version,
            )
        return self.models[task]

    def supported_kind(self, kind):
        supported = ("page",)
        if kind not in supported:
            raise ValueError(f"Unsupported {self.pipeline} benchmark type: {kind[:64]}")
        return kind

    def metadata(self, collection, identifier, token):
        identifier = str(identifier)
        if not identifier or identifier in (".", ".."):
            raise ValueError("Invalid platform record ID")
        endpoint = f"{self.origin}/api/{collection}/{quote(identifier, safe='')}"
        # Credentials are scoped to registration and these same-origin metadata reads.
        with self.http.get(
            endpoint, headers={"X-API-Token": token}, timeout=30, allow_redirects=False
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"{collection} metadata returned HTTP {response.status_code}"
                )
            return response.json()

    def select_task(self, message, token):
        generic = {"client", "custom"}
        explicit = {
            kind_name(message[key], self.pipeline)
            for key in ("experiment_type", "task_type")
            if key in message
        }
        explicit -= generic
        if len(explicit) > 1:
            raise ValueError("Conflicting experiment type fields")
        if explicit:
            return self.supported_kind(explicit.pop()), "task message"

        experiment = self.metadata("experiments", message["experiment_id"], token)
        kind = kind_name(experiment["type"], self.pipeline)
        if kind not in generic:
            return self.supported_kind(kind), "experiment metadata"

        declared = set()
        if experiment.get("dataset_id") is not None:
            dataset = self.metadata("datasets", experiment["dataset_id"], token)
            values = dataset.get("supported_infer_types")
            if values is None:
                values = []
            if not isinstance(values, list):
                raise ValueError("Invalid supported_infer_types in dataset metadata")
            declared = {kind_name(value, self.pipeline) for value in values} - generic
            if len(declared) == 1:
                return self.supported_kind(declared.pop()), "dataset metadata"

        if not self.default_task:
            raise ValueError("Generic experiment has no full-page task.")
        if declared and self.default_task not in declared:
            raise ValueError(
                "Configured client task is incompatible with dataset metadata"
            )
        return self.default_task, "configured client task (generic experiment)"

    def download_image(self, url):
        full_url = urljoin(self.origin + "/", url)
        parts = urlsplit(full_url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
        ):
            raise ValueError("Invalid image download URL")
        try:
            with self.http.get(full_url, stream=True, timeout=(10, 120)) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Image download returned HTTP {response.status_code}"
                    )
                image = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    if len(image) + len(chunk) > self.image_limit:
                        raise ValueError(
                            f"Input image exceeds {self.image_limit // (1024 * 1024)} MiB"
                        )
                    image.extend(chunk)
        except requests.RequestException as exc:
            # Signed download URLs may contain credentials. Do not echo them into platform logs.
            raise RuntimeError(
                f"Image download failed ({type(exc).__name__})"
            ) from None
        return bytes(image)

    def download_and_infer(self, task, filename, url, experiment_id=None):
        started = time.perf_counter()
        inference_started = None
        status = "failed"
        try:
            image = self.download_image(url)
            inference_started = time.perf_counter()
            result = self.model(task).infer_single_image(image, filename)
            status = "success"
            return result
        finally:
            if self.timing_stream is not None:
                finished = time.perf_counter()
                record = {
                    "experiment_id": experiment_id,
                    "filename": filename,
                    "status": status,
                    "download_s": (
                        inference_started if inference_started is not None else finished
                    )
                    - started,
                    "inference_s": (
                        finished - inference_started
                        if inference_started is not None
                        else None
                    ),
                    "total_s": finished - started,
                }
                try:
                    self.timing_stream.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                except OSError:
                    logging.exception("Failed to append local timing for %s", filename)

    def close(self):
        if self.timing_stream is not None:
            self.timing_stream.close()
        for model in self.models.values():
            model.close()
        self.http.close()


async def send(ws, kind, **fields):
    await ws.send(
        json.dumps({"type": kind, **fields}, ensure_ascii=False, allow_nan=False)
    )


async def process_task(ws, message, worker, token, executor):
    loop = asyncio.get_running_loop()
    experiment_id = message["experiment_id"]
    files = message["files"]
    if not isinstance(files, dict):
        raise ValueError("infer_task.files must be an object")
    try:
        task, source = await loop.run_in_executor(
            executor, worker.select_task, message, token
        )
        task_error = None
    except Exception as exc:
        task_error = exc

    if task_error is None:
        await send(ws, "log", message=f"识别任务 {task} ({source})")
    total = len(files)
    outcomes = Counter()
    print(f"Received experiment {experiment_id} with {total} images", flush=True)
    for number, (filename, url) in enumerate(files.items(), 1):
        try:
            if task_error is not None:
                raise ValueError(f"Cannot determine experiment task: {task_error}")
            await send(ws, "log", message=f"推理中 → {filename}")
            answer = await loop.run_in_executor(
                executor, worker.download_and_infer, task, filename, url, experiment_id
            )
            if not isinstance(answer.get("md"), str) or not isinstance(
                answer.get("json"), dict
            ):
                raise ValueError("Inference returned an invalid benchmark result")
            result = {"status": "success", "md": answer["md"], "json": answer["json"]}
        except ConnectionClosed:
            raise
        except PageBackendUnavailable as exc:
            detail = str(exc).replace(token, "[redacted]")[:400]
            await send(
                ws,
                "log",
                message=f"Backend unavailable at {number}/{total}; suspending client: {detail}",
            )
            print(
                f"Suspended experiment {experiment_id} at {number}/{total}; reported {dict(outcomes)} ({detail})",
                file=sys.stderr,
                flush=True,
            )
            # No failed placeholder or task_done. Once the backend recovers,
            # a fresh registration can receive work from the platform again.
            raise
        except Exception as exc:
            # The protocol requires failed samples to carry only an error, never a placeholder answer.
            detail = str(exc).replace(token, "[redacted]")[:600]
            result = {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": detail},
            }
        await send(
            ws,
            "infer_result",
            experiment_id=experiment_id,
            result={"filename": filename, **result},
        )
        outcomes[
            (
                result["status"]
                if result["status"] == "success"
                else result["error"]["type"]
            )
        ] += 1
        await send(
            ws, "log", message=f"[{number}/{total}] {filename}: {result['status']}"
        )
    await send(ws, "task_done", experiment_id=experiment_id)
    print(
        f"Reported {total}/{total} images for {experiment_id}: {dict(outcomes)}",
        flush=True,
    )


async def serve_connection(ws, worker, token, executor):
    # Receive application pings even during long experiments. Results and task_done stay ordered.
    jobs = asyncio.Queue()

    async def consume():
        while True:
            await process_task(ws, await jobs.get(), worker, token, executor)

    consumer = asyncio.create_task(consume())
    receiver = asyncio.create_task(ws.recv())
    try:
        while True:
            done, _ = await asyncio.wait(
                (consumer, receiver), timeout=30, return_when=asyncio.FIRST_COMPLETED
            )
            if consumer in done:
                consumer.result()
            if receiver not in done:
                await send(ws, "ping")
                continue
            message = json.loads(receiver.result())
            kind = message.get("type")
            if kind == "disconnect":
                return
            if kind == "ping":
                await send(ws, "pong")
            elif kind == "infer_task":
                jobs.put_nowait(message)
            receiver = asyncio.create_task(ws.recv())
    finally:
        for child in (receiver, consumer):
            child.cancel()
        await asyncio.gather(receiver, consumer, return_exceptions=True)


async def register_and_run(args, worker, token, executor):
    origin = urlsplit(args.server_url)
    ws_url = urlunsplit(
        (
            "wss" if origin.scheme == "https" else "ws",
            origin.netloc,
            "/api/client/ws",
            "",
            "",
        )
    )
    while True:
        connection = connect(
            ws_url, proxy=None, max_size=100 * 1024 * 1024, open_timeout=15
        )
        try:
            async with connection as ws:
                if connection.uri != ws_url:
                    raise RuntimeError("Platform redirected the WebSocket registration")
                await send(
                    ws,
                    "register",
                    token=token,
                    model_name=args.model_name,
                    server_url=ws_url,
                )
                response = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if response.get("type") != "registered":
                    raise RuntimeError(
                        "Platform rejected the registration. Check the API token and model name."
                    )
                print(
                    f"Registered {args.model_name}, client_id={str(response.get('client_id')).replace(token, '[redacted]')}",
                    flush=True,
                )
                await serve_connection(ws, worker, token, executor)
                return
        except InvalidStatus as exc:
            status = exc.response.status_code
            if status < 500:
                raise RuntimeError(
                    f"Platform WebSocket handshake returned HTTP {status}"
                ) from None
        except (OSError, ConnectionClosed):
            pass
        except PageBackendUnavailable:
            # Stay off the platform through an outage. A healthy service
            # with the original execution profile can take over after restart.
            loop = asyncio.get_running_loop()
            model = worker.models["page"]
            while True:
                await asyncio.sleep(5)
                try:
                    await loop.run_in_executor(executor, model.wait_for_backend)
                    break
                except PageBackendUnavailable as exc:
                    print(f"Waiting to re-register: {exc}", file=sys.stderr, flush=True)
            print(
                "Page backend ready; re-registering client.",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            "Platform connection interrupted. Reconnecting in 5 seconds.",
            file=sys.stderr,
            flush=True,
        )
        await asyncio.sleep(5)


async def run(args):
    token = None if args.self_test_image else load_token()
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = Worker(args)
        try:
            model = await loop.run_in_executor(
                executor, worker.model, worker.default_task or "ocr"
            )
            if args.self_test_image:
                path = args.self_test_image
                result = await loop.run_in_executor(
                    executor, model.infer_single_image, path.read_bytes(), path.name
                )
                text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + "\n", encoding="utf-8")
                else:
                    print(text)
                return
            await register_and_run(args, worker, token, executor)
        finally:
            await loop.run_in_executor(executor, worker.close)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        type=server_origin,
        default=DEFAULT_SERVER,
        help="Benchmark origin; alternatively set OCR_BENCH_SERVER_URL. Required for registration.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("page",),
        default="page",
        help="Official PaddleX full-page layout and assembly pipeline",
    )
    parser.add_argument(
        "--vlm-version",
        choices=PAGE_PIPELINES,
        help="PaddleOCR-VL page pipeline and recognition model version (default: 1.6)",
    )
    parser.add_argument(
        "--model-name", help="Override the default registration name for this pipeline"
    )
    parser.add_argument(
        "--vlm-url", help="Override the default local VLM URL for this pipeline"
    )
    parser.add_argument(
        "--device", default="gpu:0", help="PaddleX layout device for page mode"
    )
    parser.add_argument(
        "--layout-model",
        choices=LAYOUT_MODELS,
        help="Page layout detector (default: PP-DocLayoutV3)",
    )
    parser.add_argument("--self-test-image", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Write the local image result to this JSON file"
    )
    parser.add_argument(
        "--timing-log",
        type=Path,
        help="Append download and inference wall time per dispatched image to a local JSONL file",
    )
    args = parser.parse_args()
    if not args.self_test_image and not args.server_url:
        parser.error("Registration requires --server-url or OCR_BENCH_SERVER_URL")
    args.vlm_version = args.vlm_version or "1.6"
    if args.pipeline == "page" and args.layout_model is None:
        args.layout_model = "PP-DocLayoutV3"
    if args.model_name is None:
        args.model_name = "PaddleOCR-VL-1.6-EAGLE3"
        if args.pipeline == "page":
            layout_name = (
                "DocLayoutV3" if args.layout_model == "PP-DocLayoutV3" else "Layout"
            )
            args.model_name = f"PaddleOCR-VL-{args.vlm_version}-EAGLE3-{layout_name}"
    if args.vlm_url is None:
        args.vlm_url = PAGE_VLM_URLS[args.vlm_version]
    if args.output and not args.self_test_image:
        parser.error("--output requires --self-test-image")
    logging.getLogger("websockets.client").setLevel(logging.WARNING)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
