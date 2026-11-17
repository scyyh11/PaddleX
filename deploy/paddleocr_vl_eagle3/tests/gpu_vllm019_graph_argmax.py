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

"""Explicit GraphArgmax GPU gate and eager/outer-graph benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

import torch
import triton

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from gpu_vllm019_stable_argmax import CASES, fill_case, summarize
from vllm019_graph_argmax import GraphArgmax, GraphArgmaxCache
from vllm019_stable_argmax import StableArgmax, VOCAB


def gate(queries, random_rounds):
    cache = GraphArgmaxCache()
    records = []

    def check(actual, expected, **label):
        exact = torch.equal(actual, expected)
        row = {**label, "exact": exact}
        if not exact:
            row.update(actual=actual.tolist(), expected=expected.tolist())
            print(json.dumps(row), flush=True)
        records.append(row)

    for q in queries:
        plan = cache.prepare(q)
        logits = torch.empty((q, VOCAB), device="cuda", dtype=torch.float32)
        expected_pointer = plan.output.data_ptr()
        cases = (
            list(CASES)
            + ["normal", "quantized_ties", "random_bits", "finite_bits"] * random_rounds
        )
        for step, name in enumerate(cases):
            fill_case(logits, name, step)
            # Poison every persistent scratch/output: each call must overwrite it.
            plan.plan.partial_values.fill_(float("nan"))
            plan.plan.partial_indices.fill_(2147483647)
            plan.output.fill_(-99)
            actual = cache(logits) if step % 2 else plan(logits)
            assert actual.data_ptr() == expected_pointer
            check(
                actual,
                logits.argmax(dim=-1),
                q=q,
                case=name,
                phase="changed_input",
                step=step,
            )

        fill_case(logits, "increasing")
        borrowed = cache(logits)
        independent = borrowed.clone()  # Immediate consumer with owned output.
        fill_case(logits, "decreasing")
        next_output = cache(logits)
        assert next_output is borrowed
        check(
            independent,
            torch.full((q,), VOCAB - 1, dtype=torch.int64, device="cuda"),
            q=q,
            case="retained_consumer_output",
            phase="ownership",
        )
        check(
            borrowed,
            torch.zeros(q, dtype=torch.int64, device="cuda"),
            q=q,
            case="borrowed_output_overwritten",
            phase="ownership",
        )

        # Changing input allocation must feed the static graph input each time.
        other = torch.randn_like(logits)
        check(
            cache(other),
            other.argmax(dim=-1),
            q=q,
            case="new_input_storage",
            phase="ownership",
        )
        assert cache.prepare(q) is plan
        print(
            json.dumps(
                {
                    "q": q,
                    "cases": len(records),
                    "failures": sum(not row["exact"] for row in records),
                }
            ),
            flush=True,
        )
    assert len(cache.plans) <= 7
    return records, cache.metadata()


def benchmark(queries, iterations, samples):
    records = []
    for q in queries:
        logits = torch.randn((q, VOCAB), device="cuda", dtype=torch.float32)
        native_out = torch.empty(q, device="cuda", dtype=torch.int64)
        stable = StableArgmax(q)
        graph_plan = GraphArgmax(q)
        cache = GraphArgmaxCache()
        cache.prepare(q)
        calls = {
            "torch_argmax": lambda: torch.argmax(logits, dim=-1, out=native_out),
            "stable_two_stage": lambda: stable(logits),
            "graph_argmax": lambda: graph_plan(logits),
            "graph_argmax_cache": lambda: cache(logits),
        }
        outer_graphs, graph_errors = {}, {}
        measurements = {
            name: {"graph": [], "eager_gpu_elapsed": [], "eager_wall": []}
            for name in calls
        }
        for name, call in calls.items():
            for _ in range(20):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    for _ in range(iterations):
                        call()
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                outer_graphs[name] = graph
            except RuntimeError as exc:
                # A nested graph launch may be unsupported by a runtime. Eager
                # use is still measured, and this limitation remains explicit.
                graph_errors[name] = str(exc)
                del graph
        names = list(calls)
        for sample in range(samples):
            # Rotate ordering to reduce systematic clock/order bias.
            order = names[sample % len(names) :] + names[: sample % len(names)]
            for name in order:
                call = calls[name]
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                if name in outer_graphs:
                    start.record()
                    outer_graphs[name].replay()
                    end.record()
                    end.synchronize()
                    measurements[name]["graph"].append(
                        start.elapsed_time(end) * 1000 / iterations
                    )
                start.record()
                wall = time.perf_counter()
                for _ in range(iterations):
                    call()
                end.record()
                end.synchronize()
                measurements[name]["eager_wall"].append(
                    (time.perf_counter() - wall) * 1e6 / iterations
                )
                measurements[name]["eager_gpu_elapsed"].append(
                    start.elapsed_time(end) * 1000 / iterations
                )
        for name in names:
            row = {
                "q": q,
                "implementation": name,
                "iterations": iterations,
                **{
                    kind: summarize(values) if values else None
                    for kind, values in measurements[name].items()
                },
                "outer_graph_error": graph_errors.get(name),
            }
            records.append(row)
            print(
                json.dumps(
                    {
                        "q": q,
                        "implementation": name,
                        "eager_wall_median_us": row["eager_wall"]["median_us"],
                        "graph_median_us": (
                            row["graph"]["median_us"] if row["graph"] else None
                        ),
                    }
                ),
                flush=True,
            )
        del outer_graphs
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queries", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument("--random-rounds", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    if any(q < 1 or q > 7 for q in args.queries):
        parser.error("queries must be in 1..7")
    if args.random_rounds < 0 or min(args.iterations, args.samples) < 1:
        parser.error("counts must be positive; random-rounds may be zero")
    torch.set_num_threads(2)
    torch.manual_seed(190190)
    with torch.inference_mode():
        records, cache_metadata = gate(args.queries, args.random_rounds)
        all_exact = all(row["exact"] for row in records)
        timings = (
            []
            if not all_exact or args.skip_benchmark
            else benchmark(args.queries, args.iterations, args.samples)
        )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    source = SCRIPTS / "vllm019_graph_argmax.py"
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": props.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": len(records),
        "exact_cases": sum(row["exact"] for row in records),
        "all_exact": all_exact,
        "records": records,
        "cache": cache_metadata,
        "benchmark": timings,
        "return_ownership": "Borrowed per-Q output; only immediate same-stream consumption is supported without a clone.",
        "benchmark_note": "Hot fixed logits; graph wrapper timings include input copy. Eager GPU elapsed includes host starvation. Native output and candidate buffers are preallocated; initialization is excluded. Benchmark order rotates across samples.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(records),
                "all_exact": all_exact,
                "benchmarks": len(timings),
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if all_exact else 1)


if __name__ == "__main__":
    main()
