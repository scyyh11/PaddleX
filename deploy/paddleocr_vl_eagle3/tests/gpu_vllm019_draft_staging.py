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

"""Explicit GPU bitwise/replay test; select the GPU with CUDA_VISIBLE_DEVICES.

Example:
  CUDA_VISIBLE_DEVICES=1 python tests/gpu_vllm019_draft_staging.py --output /tmp/staging.json

This is a component test, not an end-to-end model accuracy or speedup claim.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_draft_staging as staging


def make_sources(seed, columns=768):
    bits = ((torch.arange(15360, dtype=torch.int32) * 40503 + seed * 251) & 65535).to(
        torch.int16
    )
    special = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7FFF, 0xFF81, 0x0080]
    bits[:8] = torch.tensor(special, dtype=torch.int32).to(torch.int16)
    sources = (
        torch.arange(5, dtype=torch.int32) + seed * 7,
        torch.tensor([-1, 0, 4, 4095, 12287], dtype=torch.int64) + seed,
        bits.view(torch.bfloat16).reshape(5, 3072),
        torch.tensor([seed * 11 - 1], dtype=torch.int32),
        torch.tensor([seed % 5], dtype=torch.int32),
        torch.tensor([12288 - seed], dtype=torch.int32),
        (torch.arange(columns, dtype=torch.int32) * 3 - seed).reshape(1, columns),
        torch.tensor([-1, 0, 100, 1024, 123456], dtype=torch.int64) + seed,
        torch.tensor([seed % 5], dtype=torch.int32),
    )
    return tuple(tensor.cuda() for tensor in sources)


def assert_bitwise(sources, destinations):
    for index, (source, destination) in enumerate(zip(sources, destinations)):
        if not torch.equal(source.view(torch.uint8), destination.view(torch.uint8)):
            raise AssertionError(f"Tensor {index} differs at the raw-byte level")


def native_copy(sources, destinations):
    for source, destination in zip(sources, destinations):
        destination.copy_(source)


def timings(operation, iterations):
    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return {
        "iterations": iterations,
        "mean_wall_us": (time.perf_counter() - wall) * 1e6 / iterations,
        "mean_gpu_stream_us": start.elapsed_time(end) * 1000 / iterations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This explicit component test requires a CUDA GPU")

    eager_checks = 0
    for columns in (1, 768, 1025, 16385):
        sources = make_sources(0, columns)
        destinations = tuple(torch.empty_like(tensor) for tensor in sources)
        for seed in range(4):
            # New allocations prove source addresses need not be stable for
            # the actual eager staging call made before each draft replay.
            sources = make_sources(seed, columns)
            if not staging.stage_tensors(sources, destinations):
                raise AssertionError("Expected the supported fused staging path")
            assert_bitwise(sources, destinations)
            eager_checks += 9

    sources = make_sources(0)
    destinations = tuple(torch.empty_like(tensor) for tensor in sources)
    staging.stage_tensors(sources, destinations)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        staging.stage_tensors(sources, destinations)
    replay_checks = 0
    for seed in range(16):
        changed = make_sources(seed + 10)
        native_copy(changed, sources)
        graph.replay()
        assert_bitwise(changed, destinations)
        replay_checks += 9

    results = {
        "passed": True,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "eager_tensor_bitwise_checks": eager_checks,
        "replay_tensor_bitwise_checks": replay_checks,
        "hidden_patterns": "raw BF16 uint16 patterns including signed zero, infinities, NaN payloads",
        "table_columns": [1, 768, 1025, 16385],
        "native_nine_copies": timings(
            lambda: native_copy(sources, destinations), args.iterations
        ),
        "fused_one_kernel": timings(
            lambda: staging.stage_tensors(sources, destinations), args.iterations
        ),
        "timing_scope": "Input tensors already on GPU; stream time includes host submission gaps. Component-only test.",
        "metadata": staging.metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
