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

"""Offline production-config profile; writes aggregate timings, never OCR text."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--vllm019-packed-attention-library", type=Path)
    parser.add_argument("--vllm019-vision-graphs", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["ocr", "table"])
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--kv-cache-memory-mib", type=int, default=768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.07)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--event-pool-size", type=int, default=32768)
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--trace-limit", type=int, default=1)
    parser.add_argument("--trace-max-iterations", type=int, default=64)
    parser.add_argument("--trace-delay-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.per_task, args.max_tokens, args.repeats, args.trace_limit) < 1:
        parser.error(
            "Sample counts, token limit, repeats and trace limit must be positive"
        )
    return args


def service_argv(args):
    """Use the same explicit opt-ins and profile validation as the HTTP entry."""
    argv = [
        "--gpu",
        args.gpu,
        "--mode",
        "eagle3-recovery",
        "--recovery-decode-graphs",
        "int8",
        "--hps-max-completion-tokens",
        "8192",
        "--kv-cache-memory-mib",
        str(args.kv_cache_memory_mib),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    for key in (
        "model",
        "draft_model",
        "cache_root",
        "vllm019_packed_attention_library",
    ):
        if getattr(args, key) is not None:
            argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if args.vllm019_vision_graphs:
        argv.append("--vllm019-vision-graphs")
    return argv


def main():
    args = parse_args()
    from runtime_config import configure_environment, request_for
    from serve_hps_vlm import parse_args as service_args
    from serve_ocr import build_engine_config

    argv = service_argv(args)
    config, engine_args, identity_config = build_engine_config(service_args(argv))
    environment = configure_environment(engine_args, identity_config)
    trace_dir = (
        args.output.resolve().with_suffix("").with_name(args.output.stem + "_traces")
    )
    if not args.no_events:
        config["worker_extension_cls"] = "profile_vllm019_worker.ProfileVllm019Worker"
    if not args.no_trace:
        config["profiler_config"] = dict(
            profiler="torch",
            torch_profiler_dir=str(trace_dir),
            torch_profiler_with_stack=False,
            torch_profiler_record_shapes=False,
            torch_profiler_with_memory=False,
            torch_profiler_use_gzip=True,
            torch_profiler_dump_cuda_time_total=True,
            ignore_frontend=True,
            max_iterations=args.trace_max_iterations,
            delay_iterations=args.trace_delay_iterations,
        )
    from vllm019_model import install

    install()
    from vllm019_image_lut import install as install_image_lut

    install_image_lut(engine_args.model)
    from PIL import Image
    from vllm import LLM, SamplingParams

    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    selected = []
    for task in args.tasks:
        selected.extend([row for row in rows if row["task"] == task][: args.per_task])
    if not selected:
        raise ValueError("No selected manifest rows")
    # vLLM normalizes nested config dictionaries in place during construction.
    config_snapshot = json.loads(json.dumps(config))
    llm = LLM(**config)
    tokenizer = llm.get_tokenizer()
    prepared = []
    for row in selected:
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["prompt"]},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        with Image.open(row["image"]) as image:
            picture = image.convert("RGB")
        prepared.append((row["task"], prompt, picture))
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=0)

    def metric_snapshot():
        values = {}
        for metric in llm.get_metrics():
            if "spec_decode" in metric.name and hasattr(metric, "value"):
                values[metric.name] = values.get(metric.name, 0) + metric.value
        return values

    def generate(item):
        task, prompt, picture = item
        return llm.generate([request_for(prompt, picture)], sampling, use_tqdm=False)[0]

    for item in prepared:
        generate(item)
    print("WARMUP_DONE", flush=True)
    worker_audit_start = llm.collective_rpc("hps_eagle_metadata")
    out = dict(
        config=config_snapshot,
        cache_root=environment["cache_root"],
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        scope="Offline LLM path: PIL loading excluded; production engine configuration retained. CUDA-event intervals include CPU submission gaps; nested intervals are not additive.",
        perf_stats={},
        repeat_outputs_equal=True,
        worker_audit_start=worker_audit_start,
    )
    expected = {}
    frontend = defaultdict(float)
    for obj, name, key in (
        (llm.renderer, "render_cmpl", "renderer_ms"),
        (llm.llm_engine.input_processor, "process_inputs", "input_process_ms"),
    ):
        original = getattr(obj, name)

        def timed(*a, _original=original, _key=key, **kw):
            before = time.perf_counter()
            try:
                return _original(*a, **kw)
            finally:
                frontend[_key] += (time.perf_counter() - before) * 1000

        setattr(obj, name, timed)

    def run_mode(mode, repeats, items):
        summary = defaultdict(
            lambda: dict(
                requests=0,
                wall_ms=0.0,
                output_tokens=0,
                prompt_tokens=0,
                length_capped=0,
                renderer_ms=0.0,
                input_process_ms=0.0,
                stages={},
            )
        )
        before_metrics = metric_snapshot()
        for _ in range(repeats):
            for index, item in enumerate(items):
                event_mode = mode == "events"
                if not args.no_events and mode != "control":
                    llm.collective_rpc(
                        "stage_profile_reset", args=(event_mode, mode == "trace")
                    )
                frontend.clear()
                before = time.perf_counter()
                result = generate(item)
                wall = (time.perf_counter() - before) * 1000
                output = result.outputs[0]
                ids = tuple(output.token_ids)
                expected.setdefault(index, ids)
                out["repeat_outputs_equal"] &= ids == expected[index]
                stages = (
                    llm.collective_rpc("stage_profile_collect")[0] if event_mode else {}
                )
                for group in ("all", item[0]):
                    stats = summary[group]
                    stats["requests"] += 1
                    stats["wall_ms"] += wall
                    stats["output_tokens"] += len(ids)
                    stats["prompt_tokens"] += len(result.prompt_token_ids)
                    stats["length_capped"] += output.finish_reason == "length"
                    for key, value in frontend.items():
                        stats[key] += value
                    for stage, values in stages.items():
                        target = stats["stages"].setdefault(
                            stage, dict(calls=0, gpu_stream_ms=0.0, cpu_submit_ms=0.0)
                        )
                        for key, value in values.items():
                            target[key] += value
        after_metrics = metric_snapshot()
        delta = {
            key: value - before_metrics.get(key, 0)
            for key, value in after_metrics.items()
        }
        drafts = delta.get("vllm:spec_decode_num_drafts", 0)
        proposed = delta.get("vllm:spec_decode_num_draft_tokens", 0)
        accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0)
        out["perf_stats"][mode] = dict(
            groups=dict(summary),
            spec_decode_counters=delta,
            draft_acceptance_fraction=accepted / proposed if proposed else None,
            mean_advance=1 + accepted / drafts if drafts else None,
        )
        print(
            json.dumps(
                {
                    "mode": mode,
                    "all": summary["all"],
                    "draft_acceptance_fraction": (
                        accepted / proposed if proposed else None
                    ),
                }
            ),
            flush=True,
        )

    # Controls precede worker instrumentation so their main path has no wrappers.
    run_mode("control", args.repeats, prepared)
    if not args.no_events:
        out["stage_installation"] = llm.collective_rpc(
            "install_stage_profile", args=(args.event_pool_size,)
        )[0]
        run_mode("events", 1, prepared)
    if not args.no_trace:
        llm.start_profile("ocr_v019")
        try:
            run_mode("trace", 1, prepared[: args.trace_limit])
        finally:
            llm.stop_profile()
        out["trace_files"] = [str(path) for path in sorted(trace_dir.glob("*"))]
    out["worker_audit_end"] = llm.collective_rpc("hps_eagle_metadata")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "trace_files": out.get("trace_files", []),
                "repeat_outputs_equal": out["repeat_outputs_equal"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
