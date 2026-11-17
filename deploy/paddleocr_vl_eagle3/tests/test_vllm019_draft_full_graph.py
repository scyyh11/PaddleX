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

"""CPU contracts for full-proposal staging, causality and fallback."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from test_vllm019_draft_loop import fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_draft_full_graph as full
import vllm019_draft_loop as tail


class FullDraftGraphTests(unittest.TestCase):
    def setUp(self):
        self.p, self.cad, self.runtime = fixture()
        self.p.hidden_size = 3
        self.p.vllm_config.model_config = NS(uses_mrope=False)
        self.p.uses_xdrope_dim = 3
        self.p.draft_uses_xdrope_dim = 0
        self.p.model.combine_hidden_states = lambda hidden: hidden[:, :3].float()
        self.cad.slot_mapping = torch.arange(5, dtype=torch.int64)
        self.ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        self.positions = torch.tensor(
            [[3, 4, 5, 6, 7], [30, 40, 50, 60, 70], [300, 400, 500, 600, 700]]
        )
        self.hidden = torch.full((5, 9), 11, dtype=torch.bfloat16)
        self.next_ids = torch.tensor([7], dtype=torch.int32)
        self.indices = torch.tensor([1], dtype=torch.int32)
        self.rejected = torch.tensor([3], dtype=torch.int32)

    def inputs(self):
        return self.ids, self.positions, self.hidden, self.next_ids, self.indices

    def args(self):
        return *self.inputs(), self.cad, self.rejected

    def plan(self):
        with patch.object(tail, "_runtime", return_value=self.runtime):
            return full._FullDraftGraph(self.p, self.cad, *self.inputs())

    def test_complete_proposal_keeps_first_selected_state_and_all_four_ids(self):
        plan = self.plan()
        plan._stage(*self.args())
        plan._run_full()
        self.assertEqual(plan.output.tolist(), [[22, 51, 13, 35]])
        self.assertEqual(
            self.p.model.seen[1:], [(22, 5, 24), (51, 6, 53), (13, 7, 112)]
        )
        self.assertEqual(plan.seq_lens.tolist(), [8])
        self.assertEqual(plan.first_cad.seq_lens.tolist(), [8])
        self.assertEqual(plan.first_cad.query_start_loc.tolist(), [0, 5])
        self.assertEqual(plan.cad.query_start_loc.tolist(), [0, 1])
        self.assertEqual(plan.target_positions.tolist(), [3, 4, 5, 6, 7])

    def test_indices_rejection_and_request_table_change_in_persistent_buffers(self):
        plan = self.plan()
        plan._stage(*self.args())
        plan._run_full()
        ids_ptr, index_ptr = plan.target_ids.data_ptr(), plan.indices.data_ptr()
        self.indices.fill_(3)
        self.rejected.fill_(1)
        self.cad.block_table_tensor = torch.tensor(
            [[7, 6, 5, 4, 3, 2]], dtype=torch.int32
        )
        plan._stage(*self.args())
        plan._run_full()
        self.assertEqual(plan.output.tolist(), [[24, 57, 27, 65]])
        self.assertEqual(plan.seq_lens.tolist(), [10])
        self.assertEqual(plan.slots.tolist(), [21, -1, -1, -1, -1])
        self.assertEqual(plan.indices.data_ptr(), index_ptr)
        self.assertEqual(plan.target_ids.data_ptr(), ids_ptr)

    def test_warm_capture_and_replay_always_restart_from_staged_target_state(self):
        plan = self.plan()
        graph = NS(replay=plan._run_full)
        stream = NS(wait_stream=lambda other: None, synchronize=lambda: None)
        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "Stream", return_value=stream),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(torch.cuda, "graph", return_value=nullcontext()),
            patch.object(torch.cuda, "CUDAGraph", return_value=graph),
        ):
            first_result = plan.run_full(*self.args())
            self.assertEqual(first_result.tolist(), [[22, 51, 13, 35]])
            self.assertEqual(self.p.model.seen[:4] * 4, self.p.model.seen)
            self.assertEqual(plan.captures, 1)
            self.assertEqual(plan.replays, 1)
            self.assertNotEqual(first_result.data_ptr(), plan.output.data_ptr())

    def test_contract_accepts_target_xdrope_and_rejects_non_q5_or_bad_aux_shape(self):
        self.assertIsNone(full._input_contract(self.p, *self.inputs(), self.cad))
        self.assertEqual(
            full._input_contract(
                self.p, *self.inputs(), self.cad.replace(max_query_len=4)
            ),
            "non_q5",
        )
        self.hidden = self.hidden[:, :3]
        self.assertEqual(
            full._input_contract(self.p, *self.inputs(), self.cad),
            "input_shape_or_dtype",
        )

    def test_native_target_mrope_projects_first_axis_and_executes_full_graph(self):
        self.p.uses_xdrope_dim = 0
        self.p.vllm_config.model_config.uses_mrope = True
        self.assertIsNone(full._input_contract(self.p, *self.inputs(), self.cad))
        plan = self.plan()
        plan._stage(*self.args())
        plan._run_full()
        self.assertEqual(plan.target_positions.tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(plan.output.tolist(), [[22, 51, 13, 35]])

    def test_mrope_requires_native_three_axes_and_explicit_padded_sample_index(self):
        self.p.uses_xdrope_dim = 0
        self.p.vllm_config.model_config.uses_mrope = True
        self.indices = None
        self.assertEqual(
            full._input_contract(self.p, *self.inputs(), self.cad),
            "missing_sample_indices",
        )
        self.indices = torch.tensor([1], dtype=torch.int32)
        for positions in (self.positions[0], self.positions[:2]):
            with self.subTest(shape=positions.shape):
                self.assertEqual(
                    full._input_contract(
                        self.p,
                        self.ids,
                        positions,
                        self.hidden,
                        self.next_ids,
                        self.indices,
                        self.cad,
                    ),
                    "position_shape",
                )

    def test_non_q5_forwards_every_argument_to_existing_tail_fallback(self):
        original = Mock(return_value="tail")
        wrapper = full._make_wrapper(original)
        self.cad.max_query_len = 4
        self.p._determine_batch_execution_and_padding = Mock(
            side_effect=AssertionError("No dispatch")
        )
        with patch.object(tail, "_fallback_reason", return_value=None):
            result = wrapper(
                self.p,
                *self.inputs(),
                self.cad,
                "sampling",
                mm_embed_inputs=([], "mask"),
                num_rejected_tokens_gpu=self.rejected,
                slot_mappings="slots",
            )
        self.assertEqual(result, "tail")
        self.assertEqual(
            original.call_args.args, (self.p, *self.inputs(), self.cad, "sampling")
        )
        self.assertEqual(original.call_args.kwargs["mm_embed_inputs"], ([], "mask"))
        self.assertIs(
            original.call_args.kwargs["num_rejected_tokens_gpu"], self.rejected
        )
        self.assertEqual(original.call_args.kwargs["slot_mappings"], "slots")

    def test_async_non_q5_keeps_full_graph_fallback(self):
        scheduler = self.p.vllm_config.scheduler_config
        scheduler.async_scheduling = True
        scheduler.max_num_seqs = 1
        self.p.hidden_size = 1024
        self.p.input_ids = NS(is_cuda=True)
        self.cad.max_query_len = 4
        self.cad.num_actual_tokens = 4
        self.p._determine_batch_execution_and_padding = Mock(
            side_effect=AssertionError("Non-Q5 must not dispatch a full graph")
        )
        original = Mock(return_value="tail")
        wrapper = full._make_wrapper(original)
        result = wrapper(
            self.p,
            *self.inputs(),
            self.cad,
            "sampling",
            num_rejected_tokens_gpu=self.rejected,
        )
        self.assertEqual(result, "tail")
        self.assertEqual(
            original.call_args.args, (self.p, *self.inputs(), self.cad, "sampling")
        )
        self.assertIs(
            original.call_args.kwargs["num_rejected_tokens_gpu"], self.rejected
        )


if __name__ == "__main__":
    unittest.main()
