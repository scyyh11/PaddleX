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

"""CPU contracts for safe draft context bucketing without GPU synchronization."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_draft_graph_buckets as buckets
import vllm019_draft_full_graph as full
import vllm019_draft_loop as tail


class DraftGraphBucketTests(unittest.TestCase):
    def test_bounds_cover_every_possible_start_plus_three_steps(self):
        observed = set()
        for length in range(1, 12289):
            bound = buckets.context_bound(length, 12288)
            observed.add(bound)
            self.assertGreaterEqual(bound, min(length + 3, 12288))
            self.assertLessEqual(bound, 12288)
        self.assertEqual(observed, set(buckets.BOUNDS))
        self.assertEqual(buckets.context_bound(125, 12288), 128)
        self.assertEqual(buckets.context_bound(126, 12288), 256)
        self.assertEqual(buckets.context_bound(4999, 5000), 5000)

    def test_gpu_or_cpu_tensor_bounds_never_trigger_implicit_scalar_reads(self):
        with self.assertRaises(TypeError):
            buckets.context_bound(torch.tensor(10), 12288)
        for length, maximum in ((0, 12288), (20, 10), (100, 32768)):
            with self.assertRaises(ValueError):
                buckets.context_bound(length, maximum)

    def test_revisit_bucket_reuses_plan_and_always_forwards_inputs(self):
        proposer = NS(max_model_len=12288)
        proposer._determine_batch_execution_and_padding = lambda count: (None, 5, None)
        cad = NS(max_seq_len=100)
        native = Mock(return_value="proposal")
        wrapper = buckets._make_wrapper(native)
        plans = []

        def make_plan(*args):
            plan = NS(bound=args[-1], captures=0, replays=0)
            plans.append(plan)
            return plan

        with (
            patch.object(tail, "_fallback_reason", return_value=None),
            patch.object(full, "_input_contract", return_value=None),
            patch.object(buckets, "_BoundedFullGraph", side_effect=make_plan),
        ):
            for length in (100, 200, 100):
                cad.max_seq_len = length
                self.assertEqual(
                    wrapper(proposer, 1, 2, 3, 4, 5, cad, 6, num_rejected_tokens_gpu=7),
                    "proposal",
                )
        self.assertEqual(len(plans), 2)
        self.assertIs(proposer._vllm019_draft_full_graph, plans[0])
        self.assertEqual(set(proposer._vllm019_draft_graph_buckets), {128, 256})
        self.assertEqual(native.call_count, 3)
        self.assertEqual(native.call_args.kwargs["num_rejected_tokens_gpu"], 7)

    def test_first_and_tail_bounds_change_without_changing_model_limit(self):
        proposer = NS(max_model_len=12288)
        gpu_lengths = object()

        def initialize(plan, *args):
            plan.proposer = proposer
            plan.cad = NS(max_seq_len=12288, seq_lens=gpu_lengths)
            plan.first_cad = NS(max_seq_len=12288, seq_lens=object())
            plan.attention = {"draft": NS(max_seq_len=12288, seq_lens=gpu_lengths)}
            plan.first_attention = {"draft": NS(max_seq_len=12288)}

        with patch.object(full._FullDraftGraph, "__init__", initialize):
            plan = buckets._BoundedFullGraph(proposer, 1, 2, 3, 4, 5, 6, 512)
        self.assertEqual(proposer.max_model_len, 12288)
        self.assertEqual(plan.cad.max_seq_len, 512)
        self.assertEqual(plan.first_cad.max_seq_len, 512)
        self.assertEqual(plan.attention["draft"].max_seq_len, 512)
        self.assertEqual(plan.first_attention["draft"].max_seq_len, 512)
        self.assertIs(plan.attention["draft"].seq_lens, gpu_lengths)


if __name__ == "__main__":
    unittest.main()
