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

"""Record the existing HPS HTTP pipeline and compare paired full-page runs.

This client does not construct or reconfigure a PaddleX pipeline. It preserves
the gateway's batching, queues, layout model, VLM client and postprocessing.
Manifest rows preserve request grouping: one PDF can contain many pages.
Raw response data stay in the private recording; compare emits aggregates only.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def save_private(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(path.read_text())


def prepare_inputs(manifest, request_params, expected_pages):
    if not isinstance(request_params, dict) or any(
        key in request_params for key in ("file", "fileType", "logId")
    ):
        raise ValueError("Request params must be an object without file/fileType/logId")
    inputs, bodies, identifiers = [], [], set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = row.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise ValueError("Every manifest row needs a unique nonempty id")
        identifiers.add(identifier)
        path = Path(row["file"])
        if not path.is_absolute():
            path = manifest.parent / path
        file_type = row.get("file_type")
        pages = row.get("page_count")
        if type(file_type) is not int or file_type not in (0, 1):
            raise ValueError("Manifest file_type must be 0 (PDF) or 1 (image)")
        if type(pages) is not int or pages < 1 or (file_type == 1 and pages != 1):
            raise ValueError("Declare page_count; an image has exactly one page")
        data = path.read_bytes()
        inputs.append(
            {
                "id": identifier,
                "file_sha256": hashlib.sha256(data).hexdigest(),
                "file_type": file_type,
                "page_count": pages,
            }
        )
        body = dict(
            request_params,
            file=base64.b64encode(data).decode("ascii"),
            fileType=file_type,
        )
        bodies.append(canonical_bytes(body))
    if not inputs or sum(item["page_count"] for item in inputs) != expected_pages:
        raise ValueError("Manifest page count does not match expected-pages")
    return inputs, bodies


def page_outputs(response, expected_pages):
    if not isinstance(response, dict) or response.get("errorCode", 0) != 0:
        raise ValueError("HPS response reports an error")
    pages = response.get("result", {}).get("layoutParsingResults")
    if not isinstance(pages, list) or len(pages) != expected_pages:
        raise ValueError("HPS returned a different number of pages")
    outputs = []
    for page in pages:
        pruned = page.get("prunedResult")
        markdown = page.get("markdown", {}).get("text")
        if not isinstance(pruned, dict) or not isinstance(markdown, str):
            raise ValueError("HPS response lacks structured output or Markdown text")
        outputs.append({"prunedResult": pruned, "markdown": markdown})
    return outputs


def request_one(
    url, body, identity, *, timeout, max_retries, retry_delay, submitted, headers
):
    started = time.perf_counter()
    attempts = []
    outputs = None
    for index in range(max_retries + 1):
        attempt_start = time.perf_counter()
        status = None
        response = None
        error = None
        retryable = False
        try:
            request = Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", **headers},
                method="POST",
            )
            with urlopen(request, timeout=timeout) as stream:
                status = stream.status
                raw = stream.read()
            response = json.loads(raw)
            if status != 200:
                error = "unexpected_http_status"
                retryable = status in (429, 500, 502, 503, 504)
            else:
                outputs = page_outputs(response, identity["page_count"])
        except HTTPError as exc:
            status = exc.code
            error = "http_error"
            retryable = status in (429, 500, 502, 503, 504)
            try:
                response = json.loads(exc.read())
            except (ValueError, OSError):
                response = None
        except (URLError, TimeoutError, OSError) as exc:
            error = type(exc).__name__
            retryable = True
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            error = "invalid_response:" + type(exc).__name__
        attempts.append(
            {
                "http_status": status,
                "error": error,
                "elapsed_ms": (time.perf_counter() - attempt_start) * 1000,
                "response": response,
            }
        )
        if outputs is not None or not retryable or index == max_retries:
            break
        time.sleep(retry_delay)
    finished = time.perf_counter()
    return {
        **identity,
        "success": outputs is not None,
        "client_queue_ms": (started - submitted) * 1000,
        "request_latency_ms": (finished - started) * 1000,
        "latency_including_client_queue_ms": (finished - submitted) * 1000,
        "attempts": attempts,
        "outputs": outputs,
    }


def run_trial(inputs, bodies, args, headers):
    # Pool startup/shutdown and all retries are included. Manifest reads and
    # base64 preparation are excluded symmetrically and reported in protocol.
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        jobs = [
            pool.submit(
                request_one,
                args.url,
                body,
                identity,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                submitted=started,
                headers=headers,
            )
            for identity, body in zip(inputs, bodies, strict=True)
        ]
        records = [job.result() for job in jobs]
    return {"wall_ms": (time.perf_counter() - started) * 1000, "records": records}


def shared_deployment_identity(audit):
    if audit.get("status") != "entering_native_server":
        raise ValueError(
            "Use the launch audit written when the native server was entered"
        )
    keys = (
        "entrypoint",
        "image_reference",
        "package_versions",
        "original_server_argv",
        "source_sha256",
        "target_model",
        "target_directory",
        "shared_backend_config",
        "chat_template_sha256",
        "plugin_allowlist",
        "discovered_general_plugins",
        "explicit_source_files",
        "batch_vision_enabled",
        "cpu_grid_enabled",
        "native_api_dispatch_enabled",
        "native_api_dispatch",
    )
    return {key: audit.get(key) for key in keys}


def worker_computation_identity(data):
    audits = data.get("worker_class_audits")
    if not audits:
        raise ValueError(
            "Missing actual target-class audit; wait for model startup and check the plugin"
        )
    expected = data["mode"] == "eagle3"
    cpu_grid = bool(data.get("deployment_identity", {}).get("cpu_grid_enabled", False))
    identities = set()
    for audit in audits:
        identity = audit["identity"]
        if identity.get("cpu_grid_enabled", False) is not cpu_grid:
            raise ValueError(
                "Actual CPU-grid processor differs from the launch configuration"
            )
        if (
            audit.get("eagle3_enabled") is not expected
            or identity.get("adapter_installed") is not expected
        ):
            raise ValueError(
                "Actual target adapter state differs from the declared benchmark mode"
            )
        methods = {
            name: value
            for name, value in identity["methods"].items()
            if name != "get_language_model"
        }
        identities.add(
            canonical_bytes(
                {
                    "module": identity["module"],
                    "qualname": identity["qualname"],
                    "methods": methods,
                    "batch_vision_bridge_installed": identity.get(
                        "batch_vision_bridge_installed", False
                    ),
                    "cpu_grid_enabled": identity.get("cpu_grid_enabled", False),
                }
            )
        )
    return sorted(identities)


def record(args):
    if args.output.exists():
        raise ValueError(
            "Use a new output path; existing recordings are never overwritten"
        )
    parsed = urlsplit(args.url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Use an HTTP(S) HPS endpoint without embedded credentials or query parameters"
        )
    params = read_json(args.request_params)
    inputs, bodies = prepare_inputs(args.manifest, params, args.expected_pages)
    deployment = read_json(args.deployment_audit)
    if deployment.get("mode") != args.mode:
        raise ValueError("Recording mode does not match the VLM deployment audit")
    identity = shared_deployment_identity(deployment)
    protocol = {
        "concurrency": args.concurrency,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
        "retry_delay_seconds": args.retry_delay,
        "expected_pages": args.expected_pages,
        "request_params": params,
        "request_params_sha256": digest(params),
        "pipeline_config_sha256": hashlib.sha256(
            args.pipeline_config.read_bytes()
        ).hexdigest(),
        "input_preparation_timed": False,
        "pool_lifecycle_and_retries_timed": True,
        "pipeline_configuration_modified_by_client": False,
        "transport": "HPS HTTP layout-parsing",
    }
    result = {
        "format_version": 1,
        "status": "running",
        "mode": args.mode,
        "url": args.url,
        "protocol": protocol,
        "inputs": inputs,
        "input_identity_sha256": digest(inputs),
        "deployment_identity": identity,
        "deployment_audit_sha256": hashlib.sha256(
            args.deployment_audit.read_bytes()
        ).hexdigest(),
        "worker_class_audits": [
            read_json(path)
            for path in sorted(
                args.deployment_audit.resolve().with_suffix(".workers").glob("*.json")
            )
        ],
        "warmups": [],
        "trials": [],
    }
    worker_computation_identity(result)
    save_private(args.output, result)
    headers = {}
    if args.bearer_token_env:
        token = os.environ.get(args.bearer_token_env)
        if not token:
            raise ValueError("The requested bearer-token environment variable is empty")
        headers["Authorization"] = "Bearer " + token
    for index in range(args.warmup_runs + args.repeats):
        trial = run_trial(inputs, bodies, args, headers)
        group = "warmups" if index < args.warmup_runs else "trials"
        result[group].append(trial)
        save_private(args.output, result)
        print(
            json.dumps(
                {
                    "phase": group,
                    "run": len(result[group]),
                    "wall_ms": round(trial["wall_ms"], 3),
                    "successful_requests": sum(r["success"] for r in trial["records"]),
                    "requests": len(inputs),
                }
            ),
            flush=True,
        )
        if group == "warmups" and not all(r["success"] for r in trial["records"]):
            result["status"] = "failed"
            save_private(args.output, result)
            return False
    result["status"] = (
        "complete"
        if all(row["success"] for trial in result["trials"] for row in trial["records"])
        else "failed"
    )
    save_private(args.output, result)
    return result["status"] == "complete"


def percentile(values, probability):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return (
        values[low] * (high - index) + values[high] * (index - low)
        if high != low
        else values[low]
    )


def validate_recording(data):
    if data.get("format_version") != 1 or data.get("status") not in (
        "complete",
        "failed",
    ):
        raise ValueError("Incomplete or unsupported benchmark recording")
    worker_computation_identity(data)
    inputs, protocol, trials = data["inputs"], data["protocol"], data["trials"]
    if (
        digest(inputs) != data["input_identity_sha256"]
        or digest(protocol["request_params"]) != protocol["request_params_sha256"]
    ):
        raise ValueError("Recording input/parameter identity is inconsistent")
    if (
        len(trials) != protocol["repeats"]
        or len(data["warmups"]) != protocol["warmup_runs"]
    ):
        raise ValueError("Recording has missing timed or warmup trials")
    for trial in data["warmups"] + trials:
        if (
            not math.isfinite(trial["wall_ms"])
            or trial["wall_ms"] <= 0
            or len(trial["records"]) != len(inputs)
        ):
            raise ValueError("Recording has missing requests or invalid timing")
        for source, row in zip(inputs, trial["records"], strict=True):
            if any(source[key] != row[key] for key in source):
                raise ValueError("Recording input order or identity differs")
            if (
                not row["attempts"]
                or len(row["attempts"]) > protocol["max_retries"] + 1
            ):
                raise ValueError("Missing or excessive request attempts")
            timings = [
                row[name]
                for name in (
                    "client_queue_ms",
                    "request_latency_ms",
                    "latency_including_client_queue_ms",
                )
            ]
            timings.extend(attempt["elapsed_ms"] for attempt in row["attempts"])
            if any(not math.isfinite(value) or value < 0 for value in timings):
                raise ValueError("Recording has invalid request/attempt timing")
            if type(row["success"]) is not bool:
                raise ValueError("Invalid success marker")
            if row["success"]:
                last = row["attempts"][-1]
                if (
                    last["http_status"] != 200
                    or last["error"] is not None
                    or page_outputs(last["response"], source["page_count"])
                    != row["outputs"]
                ):
                    raise ValueError(
                        "Recorded output is inconsistent with the raw response"
                    )
            elif row["outputs"] is not None:
                raise ValueError("Failed request cannot have an accepted output")
    if data["status"] == "complete" and any(
        not row["success"]
        for trial in data["warmups"] + trials
        for row in trial["records"]
    ):
        raise ValueError("Complete recording contains a failure")


def summarize(data):
    rows = [row for trial in data["trials"] for row in trial["records"]]
    walls = [trial["wall_ms"] for trial in data["trials"]]
    successful_pages = [
        sum(row["page_count"] for row in trial["records"] if row["success"])
        for trial in data["trials"]
    ]
    return {
        "wall_ms": walls,
        "median_wall_ms": statistics.median(walls),
        "successful_pages_per_second": [
            pages / (wall / 1000)
            for pages, wall in zip(successful_pages, walls, strict=True)
        ],
        "successful_requests": sum(row["success"] for row in rows),
        "failed_requests": sum(not row["success"] for row in rows),
        "extra_attempts": sum(len(row["attempts"]) - 1 for row in rows),
        "http_429_attempts": sum(
            a["http_status"] == 429 for row in rows for a in row["attempts"]
        ),
        "request_p50_ms": percentile([row["request_latency_ms"] for row in rows], 0.5),
        "request_p95_ms": percentile([row["request_latency_ms"] for row in rows], 0.95),
        "queue_inclusive_p95_ms": percentile(
            [row["latency_including_client_queue_ms"] for row in rows], 0.95
        ),
    }


def compare_data(baseline, eagle):
    validate_recording(baseline)
    validate_recording(eagle)
    if baseline["mode"] != "baseline" or eagle["mode"] != "eagle3":
        raise ValueError("Compare a baseline recording with an EAGLE3 recording")
    for key in ("inputs", "protocol", "deployment_identity"):
        if baseline[key] != eagle[key]:
            raise ValueError("Paired recordings differ in " + key)
    if worker_computation_identity(baseline) != worker_computation_identity(eagle):
        raise ValueError(
            "Paired recordings used different actual target/vision implementations"
        )
    exact_pages, compared_pages = 0, 0
    baseline_stable = True
    first = baseline["trials"][0]["records"]
    for left_trial, right_trial in zip(
        baseline["trials"], eagle["trials"], strict=True
    ):
        for reference, left, right in zip(
            first, left_trial["records"], right_trial["records"], strict=True
        ):
            if (
                left["success"]
                and reference["success"]
                and left["outputs"] != reference["outputs"]
            ):
                baseline_stable = False
            if left["success"] and right["success"]:
                for left_page, right_page in zip(
                    left["outputs"], right["outputs"], strict=True
                ):
                    compared_pages += 1
                    exact_pages += left_page == right_page
    b, e = summarize(baseline), summarize(eagle)
    expected = baseline["protocol"]["expected_pages"] * baseline["protocol"]["repeats"]
    complete = baseline["status"] == eagle["status"] == "complete"
    exact = complete and baseline_stable and compared_pages == exact_pages == expected
    # A faster incomplete run is not a speedup result.
    ratio = b["median_wall_ms"] / e["median_wall_ms"] if complete else None
    return {
        "format_version": 1,
        "scope": "Paired full HPS HTTP pipeline; not a direct comparison to any unrecorded external 37-second result",
        "expected_pages_per_run": baseline["protocol"]["expected_pages"],
        "requests_per_run": len(baseline["inputs"]),
        "unique_input_files": len({row["file_sha256"] for row in baseline["inputs"]}),
        "repeats": baseline["protocol"]["repeats"],
        "concurrency": baseline["protocol"]["concurrency"],
        "baseline": b,
        "eagle3": e,
        "baseline_outputs_stable_across_repeats": baseline_stable,
        "compared_pages": compared_pages,
        "exact_pages": exact_pages,
        "all_outputs_exact": exact,
        "median_wall_speedup": ratio,
        "twofold_and_exact_demonstrated": exact and ratio is not None and ratio >= 2,
        "accuracy_scope": "Exact page prunedResult and Markdown text, including whitespace; transport logId and visualization/image URLs are excluded. Raw responses remain in private recordings. Token-ID equality is not measured by this HPS API.",
        "warmup_runs": baseline["protocol"]["warmup_runs"],
        "input_preparation_timed": False,
        "crop_request_count_measured": False,
        "peak_gpu_memory_measured": False,
        "image_digest_verified": False,
        "worker_model_identity_matched": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recording = commands.add_parser("record")
    recording.add_argument("--url", required=True)
    recording.add_argument("--mode", choices=("baseline", "eagle3"), required=True)
    recording.add_argument("--manifest", type=Path, required=True)
    recording.add_argument("--request-params", type=Path, required=True)
    recording.add_argument("--deployment-audit", type=Path, required=True)
    recording.add_argument("--pipeline-config", type=Path, required=True)
    recording.add_argument("--expected-pages", type=int, required=True)
    recording.add_argument("--concurrency", type=int, required=True)
    recording.add_argument("--repeats", type=int, default=3)
    recording.add_argument("--warmup-runs", type=int, default=0)
    recording.add_argument("--timeout", type=float, default=600)
    recording.add_argument("--max-retries", type=int, default=0)
    recording.add_argument("--retry-delay", type=float, default=1)
    recording.add_argument("--bearer-token-env")
    recording.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", type=Path, required=True)
    comparison.add_argument("--eagle", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "record":
        if (
            not math.isfinite(args.timeout)
            or not math.isfinite(args.retry_delay)
            or min(args.expected_pages, args.concurrency, args.repeats, args.timeout)
            <= 0
            or args.warmup_runs < 0
            or not 0 <= args.max_retries <= 10
            or args.retry_delay < 0
        ):
            parser.error(
                "Use positive sizes/timeouts, nonnegative warmups/delay and 0..10 retries"
            )
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "record":
        if not record(args):
            raise SystemExit(1)
    else:
        if args.output.exists():
            raise ValueError("Use a new output path for the comparison")
        report = compare_data(read_json(args.baseline), read_json(args.eagle))
        save_private(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
