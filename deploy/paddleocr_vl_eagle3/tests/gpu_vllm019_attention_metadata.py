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

"""Explicit GPU parity/optional graph timing; excluded from CPU discovery.

Run with the pinned vllm019 interpreter, optionally adding --benchmark.
This checks metadata only and does not claim end-to-end attention parity.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vllm019_attention_metadata import build_ar1_metadata


def reference_metadata(cu, lengths, block_table, rows):
    count = rows.shape[0]
    start, end = cu[:1], cu[1:2]
    valid = (rows >= start) & (rows < end) & (lengths > 0)
    endpoints = (lengths - end + rows + 1).clamp_min(0)
    endpoints = torch.where(valid, endpoints, 0)
    tables = torch.where(valid[:, None], block_table.expand(count, -1), 0)
    return endpoints, tables


def graph_time_us(function, iterations):
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = function()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    # Keep outputs alive through every replay.
    assert len(outputs) == 2
    return start.elapsed_time(end) * 1000 / iterations


@torch.inference_mode()
def main(args):
    checked = 0
    timings = []
    for bound in (4096, 12288):
        for columns in (bound // 16, bound // 16 - 1):
            for count in range(2, 6):
                cu = torch.empty(2, dtype=torch.int32, device="cuda")
                lengths = torch.empty(1, dtype=torch.int32, device="cuda")
                table = (torch.arange(columns, device="cuda", dtype=torch.int32) + 17)[
                    None, :
                ]
                rows = torch.arange(count, device="cuda", dtype=torch.int32)
                fused = lambda: build_ar1_metadata(cu, lengths, table, count)
                reference = lambda: reference_metadata(cu, lengths, table, rows)
                for start, end in (
                    (0, count),
                    (0, count - 1),
                    (1, count),
                    (0, 0),
                    (count, count),
                ):
                    cu.copy_(torch.tensor([start, end], dtype=torch.int32))
                    for length in (0, 1, 2, count - 1, 64, 3916, bound):
                        lengths.fill_(length)
                        expected = reference()
                        actual = fused()
                        assert all(
                            torch.equal(a, b) for a, b in zip(actual, expected)
                        ), (bound, columns, count, start, end, length)
                        checked += 1

                # Capture once, then mutate the same input buffers. A stale
                # pointer-keyed metadata cache must fail this test.
                cu.copy_(torch.tensor([0, count], dtype=torch.int32))
                lengths.fill_(bound)
                fused()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_outputs = fused()
                for end, length in (
                    (count, bound),
                    (count - 1, 3),
                    (0, 0),
                    (count, 64),
                ):
                    cu.copy_(torch.tensor([0, end], dtype=torch.int32))
                    lengths.fill_(length)
                    table.add_(1)
                    graph.replay()
                    expected = reference()
                    assert all(
                        torch.equal(a, b) for a, b in zip(graph_outputs, expected)
                    )
                    checked += 1

                if args.benchmark and columns == bound // 16 and count == 5:
                    cu.copy_(torch.tensor([0, count], dtype=torch.int32))
                    lengths.fill_(bound)
                    timings.append(
                        {
                            "bound": bound,
                            "queries": count,
                            "reference_graph_us": graph_time_us(
                                reference, args.iterations
                            ),
                            "fused_graph_us": graph_time_us(fused, args.iterations),
                        }
                    )

    result = {"metadata_cases": checked, "all_exact": True, "timings": timings}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    main(args)
