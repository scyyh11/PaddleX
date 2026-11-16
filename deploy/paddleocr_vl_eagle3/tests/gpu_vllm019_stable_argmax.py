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

"""Explicit GPU parity gate and graph/eager microbenchmark; never runs on import."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import sys

import torch
import triton

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from vllm019_stable_argmax import StableArgmax, VOCAB


CASES = (
    "normal",
    "quantized_ties",
    "random_bits",
    "finite_bits",
    "increasing",
    "decreasing",
    "all_zero",
    "all_negative_zero",
    "alternating_signed_zero",
    "all_negative_inf",
    "all_positive_inf",
    "tie_same_block",
    "tie_cross_block",
    "tie_tail",
    "last_element",
    "nan_after_positive_inf",
    "nan_before_positive_inf",
    "multiple_nan_cross_block",
    "multiple_nan_same_block",
    "all_nan",
    "nan_payloads",
    "subnormal_positive",
    "subnormal_negative",
    "mixed_extreme",
    "adjacent_ulp",
)


def fill_case(logits, name, step=0):
    """Synthetic cases only. Distinct row offsets exercise both dimensions."""
    device = logits.device
    q = logits.shape[0]
    rows = torch.arange(q, device=device)
    offsets = (rows * 13 + step * 7) % 97
    logits.fill_(-3.0)
    if name == "normal":
        logits.normal_()
    elif name == "quantized_ties":
        logits.copy_(
            torch.randint(-5, 6, logits.shape, device=device, dtype=torch.int32)
        )
    elif name in ("random_bits", "finite_bits"):
        bits = torch.randint(
            -(2**31), 2**31, logits.shape, device=device, dtype=torch.int64
        ).to(torch.int32)
        if name == "finite_bits":
            bits.bitwise_and_(
                -8388609
            )  # Clear one exponent bit, preserving sign/mantissa.
        logits.copy_(bits.view(torch.float32))
    elif name in ("increasing", "decreasing"):
        values = torch.arange(VOCAB, device=device, dtype=torch.float32)
        logits.copy_(values if name == "increasing" else -values)
    elif name == "all_zero":
        logits.zero_()
    elif name == "all_negative_zero":
        logits.fill_(-0.0)
    elif name == "alternating_signed_zero":
        logits.zero_()
        logits[:, ::2] = -0.0
    elif name == "all_negative_inf":
        logits.fill_(-float("inf"))
    elif name == "all_positive_inf":
        logits.fill_(float("inf"))
    elif name == "tie_same_block":
        logits[rows, 17 + offsets] = 11.0
        logits[rows, 4000 + offsets] = 11.0
    elif name == "tie_cross_block":
        logits[:, 4095] = 11.0
        logits[:, 4096] = 11.0
        logits[:, 8192] = 11.0
    elif name == "tie_tail":
        logits[rows, 102400 + offsets] = 11.0
        logits[:, -1] = 11.0
    elif name == "last_element":
        logits[:, -1] = float("inf")
    elif name == "nan_after_positive_inf":
        logits[:, 0] = float("inf")
        logits[rows, VOCAB - 1 - offsets] = float("nan")
    elif name == "nan_before_positive_inf":
        logits[rows, offsets] = float("nan")
        logits[:, -1] = float("inf")
    elif name == "multiple_nan_cross_block":
        logits[:, 4096] = float("nan")
        logits[:, -1] = float("nan")
        logits[:, 4095] = float("nan")
        logits[:, 0] = float("inf")
    elif name == "multiple_nan_same_block":
        logits[rows, 99 + offsets] = float("nan")
        logits[rows, 33 + offsets] = float("nan")
    elif name == "all_nan":
        logits.fill_(float("nan"))
    elif name == "nan_payloads":
        # Positive/negative quiet and signaling NaNs, copied without arithmetic.
        payloads = torch.tensor(
            [0x7FC00001, -4194303, 0x7F800001, -8388607],
            dtype=torch.int32,
            device=device,
        ).view(torch.float32)
        logits[:, [4095, 4096, 102400, 103423]] = payloads
        logits[:, 0] = float("inf")
    elif name == "subnormal_positive":
        logits.zero_()
        values = torch.tensor(
            [1, 2, 0x007FFFFF], device=device, dtype=torch.int32
        ).view(torch.float32)
        logits[:, [4095, 4096, 103423]] = values
    elif name == "subnormal_negative":
        logits.fill_(-float("inf"))
        values = torch.tensor(
            [-2147483647, -2147483646, -2139095041], device=device, dtype=torch.int32
        ).view(torch.float32)
        logits[:, [4095, 4096, 103423]] = values
    elif name == "mixed_extreme":
        values = torch.tensor(
            [0x7F7FFFFF, -8388609, 0x00800000, -2139095040, 0, -2147483648],
            device=device,
            dtype=torch.int32,
        ).view(torch.float32)
        logits[:, [0, 4095, 4096, 8192, 102400, 103423]] = values
    elif name == "adjacent_ulp":
        values = torch.tensor(
            [0x3F800000, 0x3F800001, 0x3F800002],
            device=device,
            dtype=torch.int32,
        ).view(torch.float32)
        logits[:, [4095, 4096, 103423]] = values
    else:
        raise ValueError(name)


def parity_gate(queries, random_rounds, graph_steps):
    records, plans = [], []
    for q in queries:
        plan = StableArgmax(q)
        plans.append(plan.metadata())
        logits = torch.empty((q, VOCAB), device="cuda", dtype=torch.float32)
        external_out = torch.empty(q, device="cuda", dtype=torch.int64)

        def check(name, phase, step, out):
            expected = torch.argmax(logits, dim=-1)
            exact = torch.equal(out, expected)
            record = {
                "q": q,
                "case": name,
                "phase": phase,
                "step": step,
                "exact": exact,
            }
            if not exact:
                record.update(actual=out.tolist(), expected=expected.tolist())
                print(json.dumps(record), flush=True)
            records.append(record)

        eager_cases = (
            list(CASES)
            + ["normal", "quantized_ties", "random_bits", "finite_bits"] * random_rounds
        )
        for step, name in enumerate(eager_cases):
            fill_case(logits, name, step)
            plan.partial_values.fill_(float("nan"))
            plan.partial_indices.fill_(2147483647)
            out = plan(logits, out=external_out if step % 2 else None)
            check(name, "eager", step, out)

        # Warm up and capture on the current stream; no hidden transient buffers.
        for _ in range(3):
            plan(logits, out=external_out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan(logits, out=external_out)
        replay_cases = (
            "normal",
            "multiple_nan_cross_block",
            "all_negative_inf",
            "all_nan",
            "tie_tail",
            "nan_after_positive_inf",
            "alternating_signed_zero",
            "last_element",
        )
        for step in range(graph_steps):
            name = replay_cases[step % len(replay_cases)]
            fill_case(logits, name, step + 1)
            plan.partial_values.fill_(float("nan"))
            plan.partial_indices.fill_(2147483647)
            external_out.fill_(-99)
            graph.replay()
            check(name, "graph", step, external_out)
        del graph
        print(
            json.dumps(
                {
                    "q": q,
                    "gate_cases": len(records),
                    "failures": sum(not x["exact"] for x in records),
                }
            ),
            flush=True,
        )
    return records, plans


def summarize(values):
    ordered = sorted(values)
    return {
        "median_us": statistics.median(ordered),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "samples_us": values,
    }


def benchmark(queries, iterations, samples):
    records = []
    for q in queries:
        plan = StableArgmax(q)
        logits = torch.randn((q, VOCAB), device="cuda", dtype=torch.float32)
        native_out = torch.empty(q, device="cuda", dtype=torch.int64)
        calls = {
            "torch_argmax": lambda: torch.argmax(logits, dim=-1, out=native_out),
            "stable_two_stage": lambda: plan(logits),
        }
        for name, call in calls.items():
            for _ in range(20):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(iterations):
                    call()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            graph_us, eager_gpu_us, eager_wall_us = [], [], []
            for _ in range(samples):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                graph_us.append(start.elapsed_time(end) * 1000 / iterations)
                start.record()
                wall = time.perf_counter()
                for _ in range(iterations):
                    call()
                end.record()
                end.synchronize()
                eager_wall_us.append((time.perf_counter() - wall) * 1e6 / iterations)
                eager_gpu_us.append(start.elapsed_time(end) * 1000 / iterations)
            row = {
                "q": q,
                "implementation": name,
                "iterations": iterations,
                "graph": summarize(graph_us),
                "eager_gpu_elapsed": summarize(eager_gpu_us),
                "eager_wall": summarize(eager_wall_us),
            }
            records.append(row)
            print(
                json.dumps(
                    {
                        "benchmark": {
                            key: value
                            for key, value in row.items()
                            if key in ("q", "implementation")
                        },
                        "graph_median_us": row["graph"]["median_us"],
                        "eager_wall_median_us": row["eager_wall"]["median_us"],
                    }
                ),
                flush=True,
            )
            del graph
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queries", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument("--random-rounds", type=int, default=8)
    parser.add_argument("--graph-steps", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    if any(q < 1 or q > 7 for q in args.queries):
        parser.error("queries must be in 1..7")
    if (
        args.random_rounds < 0
        or min(args.graph_steps, args.iterations, args.samples) < 1
    ):
        parser.error("counts must be positive; random-rounds may be zero")
    torch.set_num_threads(2)
    torch.manual_seed(190190)
    with torch.inference_mode():
        records, plans = parity_gate(args.queries, args.random_rounds, args.graph_steps)
        all_exact = all(row["exact"] for row in records)
        results = (
            []
            if not all_exact or args.skip_benchmark
            else benchmark(args.queries, args.iterations, args.samples)
        )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    source = SCRIPTS / "vllm019_stable_argmax.py"
    result = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": props.name,
            "sm_count": props.multi_processor_count,
        },
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "FP32 contiguous [Q,103424], Q1..7, GreaterOrNan semantics; explicit GPU validation",
        "cases": len(records),
        "exact_cases": sum(row["exact"] for row in records),
        "all_exact": all_exact,
        "plans": plans,
        "records": records,
        "benchmark": results,
        "benchmark_note": "Repeated fixed-input graph measures hot logits; eager GPU elapsed includes host launch starvation. Native uses torch.argmax(out=...) to exclude output allocation; candidate workspaces are preallocated.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(records),
                "all_exact": all_exact,
                "benchmarks": len(results),
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if all_exact else 1)


if __name__ == "__main__":
    main()
