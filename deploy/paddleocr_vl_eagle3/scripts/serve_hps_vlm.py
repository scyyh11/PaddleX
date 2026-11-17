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

"""PaddleX PaddleOCR-VL chat API backed by a versioned resident engine."""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import time
from typing import Annotated, Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.requests import ClientDisconnect

from pathlib import Path
from runtime_config import DEFAULT_DRAFT, DEFAULT_EXTENSION, DEFAULT_MODEL, PROMPTS
from serve_ocr import (
    BODY_LIMIT,
    PIXEL_LIMIT,
    RECOVERY_MODE,
    create_app,
    run_while_connected,
)

MODEL_NAME = "PaddleOCR-VL-1.6-0.9B"
V16_MODEL_NAME = MODEL_NAME
MODEL_PROFILES = {"1.6": (V16_MODEL_NAME, DEFAULT_MODEL, DEFAULT_DRAFT)}
HPS_BODY_LIMIT = 256 * 1024 * 1024
HPS_PIXEL_LIMIT = 160_000_000
HPS_TOKEN_LIMIT = 8192


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextPart(StrictModel):
    type: Literal["text"]
    text: str = Field(max_length=64)


class ImageURL(StrictModel):
    url: str
    detail: Literal["auto"] = "auto"


class ImagePart(StrictModel):
    type: Literal["image_url"]
    image_url: ImageURL


class Message(StrictModel):
    role: Literal["user"]
    content: list[Annotated[TextPart | ImagePart, Field(discriminator="type")]] = Field(
        min_length=2, max_length=2
    )


class PixelOptions(StrictModel):
    min_pixels: int = Field(ge=784, le=2822400, strict=True)
    max_pixels: int = Field(ge=784, le=2822400, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels must not exceed max_pixels")
        return self


class ChatRequest(StrictModel):
    model: Literal["PaddleOCR-VL-1.6-0.9B"]
    messages: list[Message] = Field(min_length=1, max_length=1)
    max_completion_tokens: int | None = Field(
        default=None, ge=1, le=HPS_TOKEN_LIMIT, strict=True
    )
    max_tokens: int | None = Field(default=None, ge=1, le=HPS_TOKEN_LIMIT, strict=True)
    temperature: Literal[0, 0.0] = 0
    top_p: Literal[1, 1.0] = 1
    repetition_penalty: Literal[1, 1.0] = 1
    n: Literal[1] = 1
    stream: Literal[False] = False
    seed: Literal[0] = 0
    skip_special_tokens: bool = Field(default=True, strict=True)
    mm_processor_kwargs: PixelOptions | None = None

    def recognition_input(self) -> tuple[str, bytes]:
        texts = [p.text for p in self.messages[0].content if isinstance(p, TextPart)]
        images = [
            p.image_url.url
            for p in self.messages[0].content
            if isinstance(p, ImagePart)
        ]
        tasks = {instruction: task for task, instruction in PROMPTS.items()}
        if len(texts) != 1 or len(images) != 1 or texts[0] not in tasks:
            raise HTTPException(
                422,
                "Send one image and OCR:, Table Recognition: or Formula Recognition:",
            )
        header, separator, payload = images[0].partition(",")
        if not separator or header not in {
            "data:image/jpeg;base64",
            "data:image/png;base64",
            "data:image/webp;base64",
        }:
            raise HTTPException(422, "Send an inline base64 PNG, JPEG or WebP image")
        if (
            self.max_tokens is not None
            and self.max_completion_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            raise HTTPException(422, "Token limit fields disagree")
        try:
            pixels = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(422, "Invalid base64 image") from exc
        return tasks[texts[0]], pixels


async def read_chat(request: Request, body_limit: int = BODY_LIMIT) -> ChatRequest:
    if (
        request.headers.get("content-type", "").split(";")[0].strip().lower()
        != "application/json"
    ):
        raise HTTPException(415, "Send application/json")
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError as exc:
        raise HTTPException(400, "Invalid Content-Length") from exc
    if length > body_limit:
        raise HTTPException(413, f"Request exceeds {body_limit // (1024 * 1024)} MiB")
    body = bytearray()
    async with asyncio.timeout(30):
        async for chunk in request.stream():
            if len(body) + len(chunk) > body_limit:
                raise HTTPException(
                    413, f"Request exceeds {body_limit // (1024 * 1024)} MiB"
                )
            body.extend(chunk)
    try:
        return ChatRequest.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(
            422, exc.errors(include_context=False, include_input=False)
        ) from exc


def create_hps_app(args):
    app = create_app(args)
    model_name = MODEL_PROFILES[getattr(args, "model_version", "1.6")][0]

    @app.get("/health")
    async def health():
        try:
            await app.state.ocr.engine.check_health()
        except Exception as exc:
            raise HTTPException(503, "Inference engine unavailable") from exc
        return {"status": "ready"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        ocr = app.state.ocr
        if ocr.admitted >= 1 + ocr.queue_size:
            raise HTTPException(429, "OCR queue is full", headers={"Retry-After": "1"})
        ocr.admitted += 1
        try:
            large_regions = (
                ocr.mode == RECOVERY_MODE or ocr.max_completion_tokens > 1024
            )
            try:
                chat = await read_chat(
                    request, HPS_BODY_LIMIT if large_regions else BODY_LIMIT
                )
            except ClientDisconnect as exc:
                raise HTTPException(499, "Client disconnected during upload") from exc
            except TimeoutError as exc:
                raise HTTPException(408, "Upload timed out") from exc
            if chat.model != model_name:
                raise HTTPException(422, f"This backend serves {model_name}")
            sampling = ocr.sampling.clone()
            sampling.max_tokens = chat.max_completion_tokens or chat.max_tokens or 1024
            if sampling.max_tokens > ocr.max_completion_tokens:
                raise HTTPException(
                    422,
                    f"Maximum completion tokens for this server is {ocr.max_completion_tokens}",
                )
            task, pixels = chat.recognition_input()
            sampling.skip_special_tokens = chat.skip_special_tokens
            mm_options = (
                chat.mm_processor_kwargs.model_dump()
                if chat.mm_processor_kwargs
                else None
            )
            result = await run_while_connected(
                request,
                ocr,
                task,
                pixels,
                sampling,
                mm_options,
                HPS_PIXEL_LIMIT if large_regions else PIXEL_LIMIT,
            )
            return {
                "id": "chatcmpl-" + result["request_id"],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": result["finish_reason"],
                    }
                ],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["output_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["output_tokens"],
                },
                "ocr_details": {
                    key: result[key] for key in ("token_ids", "queue_ms", "generate_ms")
                },
            }
        finally:
            ocr.admitted -= 1

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18780)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model-version", choices=tuple(MODEL_PROFILES), default="1.6")
    parser.add_argument(
        "--mode", choices=("baseline", RECOVERY_MODE), default=RECOVERY_MODE
    )
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument(
        "--hps-max-completion-tokens",
        type=int,
        choices=(1024, 4096, HPS_TOKEN_LIMIT),
        help="HPS wire request cap; recovery defaults to 8192, other modes to 1024",
    )
    parser.add_argument(
        "--recovery-decode-graphs",
        choices=("bf16", "int8"),
        help="Native decode CUDA graphs; EAGLE3 defaults to the validated INT8 draft head",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--vllm019-packed-attention-library",
        type=Path,
        help="Opt in to the built finite-QKV A100 FA2 library (vLLM 0.19 recovery/int8 only)",
    )
    parser.add_argument(
        "--vllm019-vision-graphs",
        action="store_true",
        help="Opt in to exact-shape short vision graphs (vLLM 0.19 recovery/int8 only)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--kv-cache-memory-mib", type=int, default=4096)
    args = parser.parse_args(argv)
    if (
        not args.gpu
        or "," in args.gpu
        or not 1 <= args.port <= 65535
        or not 0 <= args.queue_size <= 32
    ):
        parser.error("Choose one GPU, a valid port, and 0–32 waiting clients")
    args.worker_extension = DEFAULT_EXTENSION
    if not 0 < args.gpu_memory_utilization <= 1 or args.kv_cache_memory_mib <= 0:
        parser.error(
            "Choose a GPU memory fraction in (0, 1] and a positive KV cache size"
        )
    if args.hps_max_completion_tokens is None:
        args.hps_max_completion_tokens = (
            HPS_TOKEN_LIMIT if args.mode == RECOVERY_MODE else 1024
        )
    if args.recovery_decode_graphs and args.mode != RECOVERY_MODE:
        parser.error("--recovery-decode-graphs requires --mode eagle3-recovery")
    if args.mode == RECOVERY_MODE and args.recovery_decode_graphs is None:
        args.recovery_decode_graphs = "int8"
    if args.vllm019_packed_attention_library is not None and (
        args.mode != RECOVERY_MODE
        or args.recovery_decode_graphs != "int8"
        or args.hps_max_completion_tokens not in (1024, HPS_TOKEN_LIMIT)
    ):
        parser.error(
            "--vllm019-packed-attention-library requires recovery/int8 and context 4096 or 12288"
        )
    if args.vllm019_vision_graphs and (
        args.mode != RECOVERY_MODE
        or args.recovery_decode_graphs != "int8"
        or args.hps_max_completion_tokens not in (1024, HPS_TOKEN_LIMIT)
    ):
        parser.error(
            "--vllm019-vision-graphs requires recovery/int8 and context 4096 or 12288"
        )
    args.k, args.seed = 4, 0
    return args


def main():
    args = parse_args()
    import uvicorn

    uvicorn.run(create_hps_app(args), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
