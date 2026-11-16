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

"""Record sequential OCR requests, then compare baseline and EAGLE outputs.

Use separate record invocations so the two services can share one GPU. Warmup
is excluded. Artifacts contain token IDs; stdout never prints document text or
input paths. No benchmark-platform registration or other external API is used.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import time

import httpx
from PIL import Image

MODEL = "PaddleOCR-VL-1.6-0.9B"
PROMPTS = {
    "ocr": "OCR:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}


class BenchmarkError(Exception):
    """A diagnostic safe to print without input paths or response contents."""


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_cases(manifest):
    try:
        lines = manifest.read_text().splitlines()
    except OSError as exc:
        raise BenchmarkError("Cannot read the input manifest") from exc
    cases, payloads, identifiers = [], [], set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            identifier, task, image_path = row["id"], row["task"], row["image"]
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Invalid ID")
            if identifier in identifiers or task not in PROMPTS:
                raise ValueError("Duplicate ID or unsupported task")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError("Missing image")
            path = Path(image_path).expanduser()
            if not path.is_absolute():
                path = manifest.parent / path
            data = path.read_bytes()
            with Image.open(BytesIO(data)) as image:
                mime = Image.MIME.get(image.format)
                if not mime or not mime.startswith("image/"):
                    raise ValueError("Unsupported image")
                image.verify()
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise BenchmarkError(
                f"Invalid manifest entry or unreadable image at line {line_number}"
            ) from exc
        identifiers.add(identifier)
        cases.append(
            dict(
                id=identifier, task=task, image_sha256=hashlib.sha256(data).hexdigest()
            )
        )
        payloads.append(f"data:{mime};base64," + base64.b64encode(data).decode())
    if not cases:
        raise BenchmarkError("Manifest has no input cases")
    return cases, payloads


def infer(client, url, case, data_uri, max_tokens):
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPTS[case["task"]]},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }
    started = time.perf_counter()
    try:
        response = client.post(url, json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise BenchmarkError(
            f"Inference failed: HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise BenchmarkError(
            f"Inference transport failed: {type(exc).__name__}"
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    try:
        body = response.json()
        if body["model"] != MODEL or len(body["choices"]) != 1:
            raise ValueError("Wrong model or number of completions")
        tokens = body["ocr_details"]["token_ids"]
        finish = body["choices"][0]["finish_reason"]
        prompt_tokens = body["usage"]["prompt_tokens"]
        generate_ms = body["ocr_details"]["generate_ms"]
        if not isinstance(tokens, list) or not all(type(t) is int for t in tokens):
            raise ValueError("Invalid token IDs")
        if any(t < 0 for t in tokens) or finish not in ("stop", "length"):
            raise ValueError("Invalid completion")
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("Invalid prompt count")
        if body["usage"]["completion_tokens"] != len(tokens):
            raise ValueError("Token count differs from token IDs")
        if (
            not isinstance(generate_ms, (int, float))
            or not math.isfinite(generate_ms)
            or generate_ms < 0
        ):
            raise ValueError("Invalid generation timing")
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkError(
            "Inference response is missing valid benchmark fields"
        ) from exc
    return dict(
        token_ids=tokens,
        finish_reason=finish,
        prompt_tokens=prompt_tokens,
        generate_ms=generate_ms,
        latency_ms=latency_ms,
    )


def record(args):
    result = dict(
        format_version=1,
        status="running",
        model=MODEL,
        max_tokens=args.max_tokens,
        repeats=args.repeats,
        warmup_requests=1,
        started_at=datetime.now(timezone.utc).isoformat(),
        cases=[],
        records=[],
    )
    try:
        cases, payloads = load_cases(args.manifest)
        result["cases"] = cases
        url = args.url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/chat/completions"
        with httpx.Client(
            timeout=httpx.Timeout(600, connect=10), trust_env=False
        ) as client:
            infer(client, url, cases[0], payloads[0], args.max_tokens)
            started = time.perf_counter()
            for repeat in range(args.repeats):
                for case, data_uri in zip(cases, payloads, strict=True):
                    output = infer(client, url, case, data_uri, args.max_tokens)
                    result["records"].append(dict(**case, repeat=repeat, **output))
                    print(
                        json.dumps(
                            dict(
                                completed=len(result["records"]),
                                total=len(cases) * args.repeats,
                                latency_ms=round(output["latency_ms"], 3),
                            )
                        ),
                        flush=True,
                    )
            result["wall_ms"] = (time.perf_counter() - started) * 1000
        result["status"] = "complete"
        result["latency_sum_ms"] = sum(r["latency_ms"] for r in result["records"])
        save(args.output, result)
        return 0
    except BenchmarkError as exc:
        result.update(status="failed", error=str(exc))
        save(args.output, result)
        raise


def load_record(path):
    try:
        result = json.loads(path.read_text())
        if result["format_version"] != 1 or result["status"] != "complete":
            raise ValueError("Incomplete run")
        if (
            result["model"] != MODEL
            or type(result["max_tokens"]) is not int
            or result["max_tokens"] < 1
            or result["warmup_requests"] != 1
        ):
            raise ValueError("Missing or invalid inference settings")
        cases, repeats, records = result["cases"], result["repeats"], result["records"]
        if not cases or type(repeats) is not int or repeats < 1:
            raise ValueError("Missing cases/repeats")
        identifiers = set()
        for case in cases:
            if (
                not isinstance(case["id"], str)
                or not case["id"]
                or case["id"] in identifiers
                or case["task"] not in PROMPTS
                or not isinstance(case["image_sha256"], str)
                or len(case["image_sha256"]) != 64
            ):
                raise ValueError("Invalid input identities")
            identifiers.add(case["id"])
        expected = [(case, repeat) for repeat in range(repeats) for case in cases]
        if len(records) != len(expected):
            raise ValueError("Missing requests")
        for row, (case, repeat) in zip(records, expected, strict=True):
            if any(row[key] != case[key] for key in ("id", "task", "image_sha256")):
                raise ValueError("Request identity differs")
            if row["repeat"] != repeat or row["finish_reason"] not in (
                "stop",
                "length",
            ):
                raise ValueError("Invalid request order/completion")
            if not isinstance(row["token_ids"], list) or not all(
                type(token) is int and token >= 0 for token in row["token_ids"]
            ):
                raise ValueError("Missing token IDs")
            for key in ("latency_ms", "generate_ms", "prompt_tokens"):
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError("Invalid timing or prompt count")
        if not math.isfinite(result["wall_ms"]) or result["wall_ms"] <= 0:
            raise ValueError("Invalid wall timing")
        return result
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise BenchmarkError(
            "Comparison input is incomplete, failed, or malformed"
        ) from exc


def summarize_pairs(pairs):
    pairs = list(pairs)
    baseline_ms = sum(left["latency_ms"] for left, _ in pairs)
    eagle_ms = sum(right["latency_ms"] for _, right in pairs)
    return dict(
        total=len(pairs),
        exact_tokens=sum(a["token_ids"] == b["token_ids"] for a, b in pairs),
        exact_finish=sum(a["finish_reason"] == b["finish_reason"] for a, b in pairs),
        exact_prompt_tokens=sum(
            a["prompt_tokens"] == b["prompt_tokens"] for a, b in pairs
        ),
        exact_tokens_and_finish=sum(
            a["token_ids"] == b["token_ids"]
            and a["finish_reason"] == b["finish_reason"]
            for a, b in pairs
        ),
        baseline_latency_ms=baseline_ms,
        eagle_latency_ms=eagle_ms,
        latency_speedup=baseline_ms / eagle_ms if eagle_ms > 0 else None,
    )


def compare(args):
    baseline, eagle = load_record(args.baseline), load_record(args.eagle)
    for key in ("model", "cases", "max_tokens", "repeats", "warmup_requests"):
        if baseline[key] != eagle[key]:
            raise BenchmarkError(f"Comparison input settings differ: {key}")
    pairs = list(zip(baseline["records"], eagle["records"], strict=True))
    summary = summarize_pairs(pairs)
    exact = all(
        summary[key] == summary["total"]
        for key in ("exact_tokens", "exact_finish", "exact_prompt_tokens")
    )
    result = dict(
        status="complete" if exact else "mismatch",
        all_exact=exact,
        **summary,
        baseline_wall_ms=baseline["wall_ms"],
        eagle_wall_ms=eagle["wall_ms"],
        wall_speedup=baseline["wall_ms"] / eagle["wall_ms"],
        per_task={
            task: summarize_pairs((a, b) for a, b in pairs if a["task"] == task)
            for task in PROMPTS
            if any(a["task"] == task for a, _ in pairs)
        },
    )
    save(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if exact else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser(
        "record", help="Warm up once, then record OCR requests"
    )
    capture.add_argument(
        "--url", required=True, help="Local VLM base URL, optionally ending /v1"
    )
    capture.add_argument("--manifest", required=True, type=Path)
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("--max-tokens", default=1024, type=int)
    capture.add_argument("--repeats", default=1, type=int)
    comparison = commands.add_parser(
        "compare", help="Compare two completed record artifacts"
    )
    comparison.add_argument("--baseline", required=True, type=Path)
    comparison.add_argument("--eagle", required=True, type=Path)
    comparison.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "record" and (args.max_tokens < 1 or args.repeats < 1):
        parser.error("--max-tokens and --repeats must be positive")
    try:
        return record(args) if args.command == "record" else compare(args)
    except BenchmarkError as exc:
        if args.command == "compare":
            save(args.output, dict(status="failed", error=str(exc)))
        print(json.dumps(dict(status="failed", error=str(exc))))
        return 2
    except OSError:
        print(
            json.dumps(dict(status="failed", error="Cannot write benchmark artifact"))
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
