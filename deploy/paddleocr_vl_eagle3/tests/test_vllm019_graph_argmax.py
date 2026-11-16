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

"""CPU mocks verify graph ownership/ordering; they do not prove CUDA execution."""

from pathlib import Path
import sys

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vllm019_graph_argmax as candidate


class ContractTests(unittest.TestCase):
    def test_initialization_orders_warmup_capture_and_current_stream(self):
        events = []
        current = SimpleNamespace(
            wait_stream=lambda stream: events.append("current_wait")
        )
        capture = SimpleNamespace(
            wait_stream=lambda stream: events.append("capture_wait")
        )
        fixed_input = SimpleNamespace(device=torch.device("cuda:0"))
        fixed_output = object()
        stable = Mock(output=fixed_output)
        stable.side_effect = lambda tensor: events.append("kernel_pair")
        graph = Mock()

        @contextmanager
        def capture_context(*args, **kwargs):
            self.assertIs(kwargs["stream"], capture)
            events.append("capture_begin")
            yield
            events.append("capture_end")

        with (
            patch.object(candidate.torch.cuda, "device", return_value=nullcontext()),
            patch.object(
                candidate.torch.cuda, "is_current_stream_capturing", return_value=False
            ),
            patch.object(candidate.torch.cuda, "current_stream", return_value=current),
            patch.object(candidate.torch.cuda, "Stream", return_value=capture),
            patch.object(candidate.torch.cuda, "stream", return_value=nullcontext()),
            patch.object(candidate.torch.cuda, "graph", side_effect=capture_context),
            patch.object(candidate.torch.cuda, "CUDAGraph", return_value=graph),
            patch.object(candidate.torch, "zeros", return_value=fixed_input),
            patch.object(candidate, "StableArgmax", return_value=stable),
        ):
            plan = candidate.GraphArgmax(5)
        self.assertEqual(
            events,
            [
                "capture_wait",
                "kernel_pair",
                "kernel_pair",
                "kernel_pair",
                "current_wait",
                "capture_begin",
                "kernel_pair",
                "capture_end",
                "current_wait",
            ],
        )
        self.assertIs(plan.plan, stable)
        self.assertIs(plan.input, fixed_input)
        self.assertIs(plan.output, fixed_output)
        self.assertIs(plan.graph, graph)
        self.assertIs(plan.capture_stream, capture)

    def test_constructor_refuses_initialization_during_capture(self):
        with (
            patch.object(candidate.torch.cuda, "device", return_value=nullcontext()),
            patch.object(
                candidate.torch.cuda, "is_current_stream_capturing", return_value=True
            ),
            patch.object(candidate.torch, "zeros") as allocate,
        ):
            with self.assertRaisesRegex(RuntimeError, "outside CUDA capture"):
                candidate.GraphArgmax(5)
        allocate.assert_not_called()

    def test_call_only_copies_replays_and_returns_same_storage(self):
        events = []
        plan = candidate.GraphArgmax.__new__(candidate.GraphArgmax)
        plan.queries = 5
        plan.device = torch.device("cuda:0")
        plan.input = SimpleNamespace(
            copy_=lambda tensor: events.append(("copy", tensor))
        )
        plan.graph = SimpleNamespace(replay=lambda: events.append(("replay",)))
        plan.output = object()
        first, second = object(), object()
        with patch.object(candidate, "_validate_logits", return_value=5):
            self.assertIs(plan(first), plan.output)
            self.assertIs(plan(second), plan.output)
        self.assertEqual(
            events, [("copy", first), ("replay",), ("copy", second), ("replay",)]
        )

    def test_cache_owns_at_most_seven_fixed_plans(self):
        cache = candidate.GraphArgmaxCache("cuda:0")
        with patch.object(
            candidate, "GraphArgmax", side_effect=lambda q, device: object()
        ) as factory:
            for q in range(1, 8):
                plan = cache.prepare(q)
                self.assertIs(cache.prepare(q), plan)
            for invalid in (0, 8, True, 1.5):
                with self.assertRaises(ValueError):
                    cache.prepare(invalid)
        self.assertEqual(factory.call_count, 7)
        self.assertEqual(len(cache.plans), 7)

    def test_cache_hit_does_not_initialize_or_call_plan_wrapper(self):
        cache = candidate.GraphArgmaxCache("cuda:0")
        plan = SimpleNamespace(input=Mock(), graph=Mock(), output=object())
        cache.plans[5] = plan
        logits = object()
        with (
            patch.object(candidate, "_validate_logits", return_value=5),
            patch.object(cache, "prepare") as prepare,
        ):
            self.assertIs(cache(logits), plan.output)
        prepare.assert_not_called()
        plan.input.copy_.assert_called_once_with(logits)
        plan.graph.replay.assert_called_once_with()

    def test_unsupported_cpu_inputs_and_non_cuda_plans_raise(self):
        with self.assertRaises(ValueError):
            candidate._validate_logits(torch.empty((1, 103424)), torch.device("cuda:0"))
        for factory in (
            candidate.GraphArgmaxCache,
            lambda device: candidate.GraphArgmax(1, device),
        ):
            with self.assertRaises(ValueError):
                factory("cpu")
        for q in (0, 8, False, 1.0):
            with self.assertRaises(ValueError):
                candidate.GraphArgmax(q)


if __name__ == "__main__":
    unittest.main()
