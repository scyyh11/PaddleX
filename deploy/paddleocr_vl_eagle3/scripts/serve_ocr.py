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

"""Loopback HTTP service for single-image recognition with a resident PaddleOCR-VL engine."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
from pathlib import Path
import time
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from python_multipart.exceptions import MultipartParseError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from runtime_config import (
    DEFAULT_DRAFT,
    PROMPTS,
    configure_environment,
    engine_config,
    is_vllm019,
    request_for,
)

BODY_LIMIT = 8 * 1024 * 1024
PIXEL_LIMIT = 16_000_000
RECOVERY_MODE = "eagle3-recovery"
RECOVERY_EXTENSION = "hps_native_eagle_worker.NativeEagleWorker"


async def read_upload(request: Request) -> tuple[str, bytes]:
    if (
        request.headers.get("content-type", "").split(";")[0].strip().lower()
        != "multipart/form-data"
    ):
        raise HTTPException(415, "Send a multipart form with file and task fields")
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError as exc:
        raise HTTPException(400, "Invalid Content-Length") from exc
    if length > BODY_LIMIT:
        raise HTTPException(413, "Request exceeds 8 MiB")
    # Bound actual bytes for chunked uploads too, before creating multipart tempfiles.
    body = bytearray()
    async with asyncio.timeout(30):
        async for chunk in request.stream():
            if len(body) + len(chunk) > BODY_LIMIT:
                raise HTTPException(413, "Request exceeds 8 MiB")
            body.extend(chunk)

    async def buffered_body():
        yield bytes(body)

    try:
        form = await MultiPartParser(
            request.headers,
            buffered_body(),
            max_files=1,
            max_fields=1,
            max_part_size=64,
        ).parse()
    except (MultiPartException, MultipartParseError) as exc:
        raise HTTPException(400, "Invalid multipart form") from exc
    try:
        task, upload = form.get("task"), form.get("file")
        if not isinstance(task, str) or task.lower() not in PROMPTS:
            raise HTTPException(422, "Choose OCR, table or formula")
        if not isinstance(upload, UploadFile):
            raise HTTPException(422, "One image file is required")
        return task.lower(), await upload.read()
    finally:
        await form.close()


def decode_image(data: bytes, pixel_limit: int = PIXEL_LIMIT) -> Image.Image:
    try:
        with Image.open(BytesIO(data), formats=("PNG", "JPEG", "WEBP")) as source:
            if source.width * source.height > pixel_limit:
                raise HTTPException(413, f"Image exceeds {pixel_limit:,} pixels")
            if max(source.size) > 200 * min(source.size):
                raise HTTPException(422, "Image aspect ratio exceeds 200:1")
            if getattr(source, "n_frames", 1) != 1:
                raise HTTPException(422, "Send a single-frame image")
            return source.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "Image exceeds the pixel limit") from exc
    except (UnidentifiedImageError, ValueError, OSError, SyntaxError) as exc:
        raise HTTPException(422, "Unable to decode PNG, JPEG or WebP image") from exc


def verify_model_file(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise RuntimeError(f"Unexpected model file: {path}")


def verify_accelerated_files(
    environment: dict, draft_model: Path = DEFAULT_DRAFT
) -> None:
    # The exported runtime computes its cache identity from its own source files.
    # Keep the experimentally frozen 1.6 draft identity independent of its path.
    checks = {
        "model.safetensors": "b7f5dc74168e594eea4c7d961c4c95e7f9c730adbd058f76d20ea48275a4157d",
        "config.json": "a460492a86f5b707cc66e1bd22b3e4423a4ac42b7b71768ab6ed3651cd08c625",
    }
    for name, expected in checks.items():
        verify_model_file(draft_model / name, expected)


@dataclass
class Resident:
    engine: Any
    sampling: Any
    prompts: dict[str, str]
    processor: Any
    mode: str
    queue_size: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    admitted: int = 0
    started: int = 0
    completed: int = 0
    recovery_worker: dict | None = None
    max_completion_tokens: int = 1024
    max_model_len: int = 4096
    draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    verification_steps: int = 0
    request_metrics_available: bool = True

    async def recognize(
        self,
        task: str,
        data: bytes,
        request_id: str,
        sampling=None,
        mm_processor_kwargs=None,
        pixel_limit=PIXEL_LIMIT,
    ) -> dict:
        from vllm.v1.engine.exceptions import EngineDeadError

        if is_vllm019():
            client_errors = (ValueError,)
            graceful_errors = ()
        else:
            from vllm.exceptions import GracefulHTTPError, VLLMClientError

            client_errors = (VLLMClientError,)
            graceful_errors = (GracefulHTTPError,)

        picture = await asyncio.to_thread(decode_image, data, pixel_limit)
        try:
            started = time.perf_counter()
            self.started += 1
            model_input = request_for(self.prompts[task], picture)
            if mm_processor_kwargs:
                model_input["mm_processor_kwargs"] = mm_processor_kwargs
            stream = self.engine.generate(
                model_input, sampling or self.sampling, request_id
            )
            result = None
            async with aclosing(stream):
                async for result in stream:
                    pass
            if result is None or not result.finished or len(result.outputs) != 1:
                raise RuntimeError("Incomplete inference result")
            output = result.outputs[0]
            if (
                self.recovery_worker is not None
                and getattr(output, "spec_decode_metrics", None) is not None
            ):
                metrics = output.spec_decode_metrics.to_dict()
                self.draft_tokens += metrics["num_draft_tokens"]
                self.accepted_draft_tokens += metrics["num_accepted_draft_tokens"]
                self.verification_steps += metrics["num_spec_steps"]
            self.completed += 1
            return dict(
                request_id=request_id,
                task=task,
                text=output.text,
                token_ids=list(output.token_ids),
                finish_reason=output.finish_reason,
                prompt_tokens=len(result.prompt_token_ids),
                output_tokens=len(output.token_ids),
                generate_ms=round(1000 * (time.perf_counter() - started), 3),
            )
        except asyncio.CancelledError:
            # generate() aborts its collector after registration. Also cover cancellation
            # during add_request(), before the generator has received the collector.
            # Raw-prompt rendering can still hold a reference to the in-memory RGB image
            # on a worker thread, so let that reference own its remaining lifetime.
            picture = None
            await self.engine.abort(request_id)
            raise
        except client_errors as exc:
            raise HTTPException(422, "Request exceeds the model input limits") from exc
        except graceful_errors as exc:
            raise HTTPException(
                int(exc.http_status), "Engine rejected the request"
            ) from exc
        except EngineDeadError as exc:
            raise HTTPException(503, "Inference engine unavailable") from exc
        finally:
            if picture is not None:
                picture.close()

    async def run_queued(
        self,
        task: str,
        data: bytes,
        sampling=None,
        mm_processor_kwargs=None,
        pixel_limit=PIXEL_LIMIT,
    ) -> dict:
        queued = time.perf_counter()
        try:
            async with asyncio.timeout(30):
                await self.lock.acquire()
        except TimeoutError as exc:
            raise HTTPException(503, "Queue wait timed out") from exc
        inference = asyncio.create_task(
            self.recognize(
                task, data, uuid.uuid4().hex, sampling, mm_processor_kwargs, pixel_limit
            )
        )
        try:
            queue_ms = round(1000 * (time.perf_counter() - queued), 3)
            timeout = max(120, (sampling or self.sampling).max_tokens / 10)
            async with asyncio.timeout(timeout):
                output = await asyncio.shield(inference)
            return {**output, "queue_ms": queue_ms}
        except TimeoutError as exc:
            raise HTTPException(504, "Inference timed out") from exc
        except ValueError as exc:
            raise HTTPException(
                422, "Image does not fit the model input limits"
            ) from exc
        finally:
            try:
                await cancel_and_join(inference)
            finally:
                self.lock.release()


async def wait_for_disconnect(request: Request) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def cancel_and_join(*tasks: asyncio.Task) -> None:
    # A timeout, disconnect and server shutdown may overlap. Deliver cancellation
    # once per child and let its cleanup (including the engine abort) finish.
    for task in tasks:
        if not task.done() and not task.cancelling():
            task.cancel()
    drained = asyncio.gather(*tasks, return_exceptions=True)
    interrupted = False
    while not drained.done():
        try:
            await asyncio.shield(drained)
        except asyncio.CancelledError:
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError


async def run_while_connected(
    request: Request,
    ocr: Resident,
    task: str,
    data: bytes,
    sampling=None,
    mm_processor_kwargs=None,
    pixel_limit=PIXEL_LIMIT,
) -> dict:
    # The body has been fully consumed. Only this listener now reads ASGI receive.
    disconnected = asyncio.create_task(wait_for_disconnect(request))
    work = asyncio.create_task(
        ocr.run_queued(task, data, sampling, mm_processor_kwargs, pixel_limit)
    )
    try:
        done, _ = await asyncio.wait(
            (disconnected, work), return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return work.result()
        disconnected.result()
        raise HTTPException(499, "Client disconnected")
    finally:
        await cancel_and_join(disconnected, work)


def build_engine_config(args):
    recovery = args.mode == RECOVERY_MODE
    decode_graphs = getattr(args, "recovery_decode_graphs", None)
    engine_args = args
    if recovery:
        worker_extension = {
            None: RECOVERY_EXTENSION,
            "bf16": "hps_decode_graph_eagle_worker.DecodeGraphEagleWorker",
            "int8": "hps_decode_graph_int8_worker.DecodeGraphInt8Worker",
        }[decode_graphs]
        if is_vllm019():
            worker_extension = (
                "hps_vllm019_int8_worker.Vllm019Int8Worker"
                if decode_graphs == "int8"
                else "hps_vllm019_worker.Vllm019EagleWorker"
            )
        engine_args = argparse.Namespace(
            **(
                vars(args)
                | {"mode": "accelerated", "worker_extension": worker_extension}
            )
        )
    config = engine_config(engine_args)
    max_output = getattr(args, "hps_max_completion_tokens", 1024)
    if max_output > 1024:
        # Full pixel budget gives at most 3600 visual tokens, plus the template.
        config["max_model_len"] = 4096 + max_output
    if recovery:
        # Match the compiled target arithmetic but avoid graph capture/replay
        # and async lookahead. The HTTP service admits one generation at a time.
        config.update(
            async_scheduling=False,
            compilation_config={"mode": 3, "cudagraph_mode": "NONE"},
        )
        config.pop("max_cudagraph_capture_size")
        config["speculative_config"].update(
            enforce_eager=False, use_local_argmax_reduction=False
        )
        if decode_graphs:
            config["compilation_config"].update(
                cudagraph_mode=(
                    "FULL_AND_PIECEWISE" if is_vllm019() else "FULL_DECODE_ONLY"
                ),
                cudagraph_capture_sizes=[args.k + 1],
            )
            config["speculative_config"]["use_local_argmax_reduction"] = (
                decode_graphs == "int8"
            )
    identity_config = config
    if decode_graphs:
        # The optional INT8 worker imports the same graph attention implementation
        # as its BF16 counterpart. Include that transitive input in both AOT keys.
        graph_worker = Path(__file__).with_name("hps_decode_graph_eagle_worker.py")
        identity_config = dict(
            config,
            graph_worker_sha256=hashlib.sha256(graph_worker.read_bytes()).hexdigest(),
        )
    return config, engine_args, identity_config


def create_app(args) -> FastAPI:
    if args.mode not in ("baseline", RECOVERY_MODE):
        raise ValueError(
            "The exported service supports baseline and eagle3-recovery modes"
        )
    if getattr(args, "model_version", "1.6") != "1.6":
        raise ValueError("The exported service targets PaddleOCR-VL-1.6")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        recovery = args.mode == RECOVERY_MODE
        max_output = getattr(args, "hps_max_completion_tokens", 1024)
        config, engine_args, identity_config = build_engine_config(args)
        environment = configure_environment(engine_args, identity_config)
        processor = None
        if getattr(args, "model_version", None) == "1.6":
            for filename, expected in {
                "model.safetensors": "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db",
                "tokenizer.json": "c8a215a59183d0d0781adc33bacd3ce6162716f7fd568fb30234a74d69803a7d",
                "chat_template.jinja": "2f27812dab7f333e471884e0c803d807f11953d5453140dfb1aaba234f872bc8",
                "preprocessor_config.json": "111872ab1e8bb7fd040ac5087bfced7ab8f011f02139b088cba294964c3b1d0e",
            }.items():
                verify_model_file(args.model / filename, expected)
        if engine_args.mode == "accelerated":
            verify_accelerated_files(environment, args.draft_model)
        # CUDA and vLLM imports follow device/cache configuration, including for spawned workers.
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.sampling_params import RequestOutputKind
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM

        if is_vllm019() and recovery:
            from vllm019_model import install

            install()

        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(**config), usage_context=UsageContext.LLM_CLASS
        )
        try:
            recovery_worker = None
            if recovery:
                (recovery_worker,) = await engine.collective_rpc(
                    "hps_eagle_metadata", timeout=60
                )
                if (
                    Path(recovery_worker["checkpoint"]).resolve()
                    != args.draft_model.resolve()
                ):
                    raise RuntimeError(
                        "Recovery worker loaded an unexpected EAGLE3 checkpoint"
                    )
            tokenizer = engine.get_tokenizer()
            prompts = {
                task: tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": instruction},
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for task, instruction in PROMPTS.items()
            }
            params = SamplingParams(
                temperature=0,
                max_tokens=1024,
                seed=0,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            app.state.ocr = Resident(
                engine,
                params,
                prompts,
                processor,
                args.mode,
                args.queue_size,
                recovery_worker=recovery_worker,
                max_completion_tokens=max_output,
                max_model_len=config["max_model_len"],
                request_metrics_available=not is_vllm019(),
            )
            yield
        finally:
            engine.shutdown()

    app = FastAPI(title="PaddleOCR-VL recognition", lifespan=lifespan)

    @app.get("/health/ready")
    async def ready():
        ocr = app.state.ocr
        try:
            await ocr.engine.check_health()
        except Exception as exc:
            raise HTTPException(503, "Inference engine unavailable") from exc
        evidence = {}
        if getattr(args, "model", None) is not None:
            evidence["model_path"] = str(args.model.resolve())
            evidence["draft_model_path"] = (
                str(args.draft_model.resolve()) if args.mode != "baseline" else None
            )
        if ocr.recovery_worker is not None:
            evidence["speculative_decoding"] = dict(
                worker=ocr.recovery_worker,
                request_metrics_available=ocr.request_metrics_available,
                verification_steps=(
                    ocr.verification_steps if ocr.request_metrics_available else None
                ),
                draft_tokens=(
                    ocr.draft_tokens if ocr.request_metrics_available else None
                ),
                accepted_draft_tokens=(
                    ocr.accepted_draft_tokens if ocr.request_metrics_available else None
                ),
            )
        return dict(
            status="ready",
            instance_id=ocr.instance_id,
            mode=ocr.mode,
            max_completion_tokens=ocr.max_completion_tokens,
            max_model_len=ocr.max_model_len,
            max_active_generations=1,
            max_waiting=ocr.queue_size,
            admitted=ocr.admitted,
            started=ocr.started,
            completed=ocr.completed,
            cpp_processor_calls=getattr(ocr.processor, "_round3_cpu_calls", 0),
            **evidence,
        )

    form_schema = {
        "type": "object",
        "required": ["file", "task"],
        "properties": {
            "file": {"type": "string", "format": "binary"},
            "task": {"type": "string", "enum": ["OCR", "table", "formula"]},
        },
    }

    @app.post(
        "/v1/ocr",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"multipart/form-data": {"schema": form_schema}},
            }
        },
    )
    async def recognize(request: Request):
        ocr = app.state.ocr
        # This app runs one event loop. Admission has no await between check and increment.
        if ocr.admitted >= 1 + ocr.queue_size:
            raise HTTPException(429, "OCR queue is full", headers={"Retry-After": "1"})
        ocr.admitted += 1
        try:
            try:
                task, data = await read_upload(request)
            except ClientDisconnect as exc:
                raise HTTPException(499, "Client disconnected during upload") from exc
            except TimeoutError as exc:
                raise HTTPException(408, "Upload timed out") from exc
            return await run_while_connected(request, ocr, task, data)
        finally:
            ocr.admitted -= 1

    return app


def main() -> None:
    # Both /v1/ocr and PaddleX-compatible chat routes share the same engine.
    from serve_hps_vlm import main as serve

    serve()


if __name__ == "__main__":
    main()
