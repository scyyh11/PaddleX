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

"""Explicit graph-wrapped FP32 stable argmax; no import-time hooks.

Each Q1..7 plan owns fixed FP32 [Q,103424] input, StableArgmax workspaces, and a
two-kernel CUDA graph. A call validates metadata, copies the current logits,
replays the graph, and returns a BORROWED persistent int64 [Q] output.

The returned tensor is only for immediate consumption on the same CUDA stream,
for example by the greedy selector which creates its own independent output.
The next call for that Q overwrites previous returns. Clone explicitly if an
output must be retained. This is not a general allocating torch.argmax drop-in.

Calls sharing a plan/cache must be ordered on one stream; callers must establish
a completed handoff before changing streams, and keep source logits alive until
their copy finishes. Plans/inputs/outputs/graphs remain strongly owned by the
cache. There are no per-call record_stream loops, allocations, or scalar reads.
Replay is supported only in eager execution, outside an enclosing CUDA capture.
Prebuilding a plan does not make nested graph replay supported by this runtime.
Lazy initialization is allowed only outside capture and includes one-time
warmup/capture overhead. Use StableArgmax directly for an enclosing capture.
"""

import torch

from vllm019_stable_argmax import StableArgmax, VOCAB


def _validate_logits(logits, device, expected_queries=None):
    if (
        not isinstance(logits, torch.Tensor)
        or logits.device != device
        or logits.dtype != torch.float32
        or logits.ndim != 2
        or logits.shape[1] != VOCAB
        or not 1 <= logits.shape[0] <= 7
        or not logits.is_contiguous()
        or (expected_queries is not None and logits.shape[0] != expected_queries)
    ):
        raise ValueError(
            "logits must be contiguous CUDA float32 [Q,103424], Q1..7, on the plan device"
        )
    return logits.shape[0]


class GraphArgmax:
    """One fixed-Q graph for eager replay; output is borrowed until next call."""

    def __init__(self, queries, device="cuda"):
        if type(queries) is not int or not 1 <= queries <= 7:
            raise ValueError("queries must be an integer in 1..7")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GraphArgmax requires CUDA")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Construct GraphArgmax outside CUDA capture; replay is eager only"
                )
            self.queries = queries
            self.input = torch.zeros(
                (queries, VOCAB), dtype=torch.float32, device=device
            )
            self.device = self.input.device
            self.plan = StableArgmax(queries, device=self.device)
            self.output = self.plan.output
            self.graph = torch.cuda.CUDAGraph()
            self.capture_stream = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            self.capture_stream.wait_stream(current)
            with torch.cuda.stream(self.capture_stream):
                for _ in range(3):
                    self.plan(self.input)
            current.wait_stream(self.capture_stream)
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self.plan(self.input)
            current.wait_stream(self.capture_stream)

    def __call__(self, logits):
        _validate_logits(logits, self.device, self.queries)
        self.input.copy_(logits)
        self.graph.replay()
        return self.output

    def metadata(self):
        return {
            **self.plan.metadata(),
            "input_bytes": self.input.numel() * self.input.element_size(),
            "cuda_graphs": 1,
            "return_ownership": "borrowed persistent output; consume immediately on the same stream",
            "per_call": "metadata guard, input.copy_, graph.replay",
            "nested_capture_supported": False,
        }


class GraphArgmaxCache:
    """Single-device eager-replay cache with at most seven Q1..7 plans."""

    def __init__(self, device="cuda"):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GraphArgmaxCache requires CUDA")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.plans = {}

    def prepare(self, queries):
        """Create one Q plan outside capture, or return the existing plan."""
        if type(queries) is not int or not 1 <= queries <= 7:
            raise ValueError("queries must be an integer in 1..7")
        plan = self.plans.get(queries)
        if plan is None:
            plan = GraphArgmax(queries, device=self.device)
            self.plans[queries] = plan
        return plan

    def __call__(self, logits):
        queries = _validate_logits(logits, self.device)
        plan = self.plans.get(queries)
        if plan is None:
            plan = self.prepare(queries)
        # Avoid running the same fixed-shape metadata guard twice.
        plan.input.copy_(logits)
        plan.graph.replay()
        return plan.output

    def metadata(self):
        plans = [self.plans[q].metadata() for q in sorted(self.plans)]
        return {
            "device": str(self.device),
            "max_plans": 7,
            "cuda_graphs": len(plans),
            "input_bytes": sum(plan["input_bytes"] for plan in plans),
            "workspace_bytes": sum(plan["workspace_bytes"] for plan in plans),
            "plans": plans,
            "hooks_installed": False,
        }
