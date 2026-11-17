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

"""Explicit GPU differential check of cached indices against the native runner."""

import argparse
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_spec_metadata as adapter


def time_us(function, iterations):
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


@torch.inference_mode()
def main(args):
    ids = torch.arange(16, device="cuda", dtype=torch.int32)
    runner = SimpleNamespace(
        num_spec_tokens=4,
        input_ids=SimpleNamespace(gpu=ids),
        device=ids.device,
        _arange_scratch=np.empty(32, dtype=np.int32),
        arange_np=np.arange(32, dtype=np.int32),
    )
    runner._get_cumsum_and_arange = MethodType(
        GPUModelRunner._get_cumsum_and_arange,
        runner,
    )
    original = GPUModelRunner._calc_spec_decode_metadata
    candidate = adapter._make_wrapper(original)
    drafts, scheduled = np.array([4], dtype=np.int32), np.array([5], dtype=np.int32)
    tensor_fields = (
        "draft_token_ids",
        "cu_num_draft_tokens",
        "cu_num_sampled_tokens",
        "target_logits_indices",
        "bonus_logits_indices",
        "logits_indices",
    )
    prior = None
    for step in range(64):
        ids.add_(13)
        expected = original(runner, drafts, scheduled)
        actual = candidate(runner, drafts, scheduled)
        assert actual.num_draft_tokens == expected.num_draft_tokens
        assert actual.max_spec_len == expected.max_spec_len
        for name in tensor_fields:
            assert torch.equal(getattr(actual, name), getattr(expected, name)), name
        if prior is not None:
            assert torch.equal(prior.draft_token_ids + 13, actual.draft_token_ids)
        prior = actual
    timings = {}
    if args.benchmark:
        timings = {
            "native_eager_us": time_us(
                lambda: original(runner, drafts, scheduled), args.iterations
            ),
            "cached_eager_us": time_us(
                lambda: candidate(runner, drafts, scheduled), args.iterations
            ),
        }
    result = {"gpu_cases": 64, "all_exact": True, "timings": timings}
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
