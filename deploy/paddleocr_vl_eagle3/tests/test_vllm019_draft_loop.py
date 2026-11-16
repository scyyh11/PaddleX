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

"""CPU lifecycle/causality contracts, not CUDA performance or accuracy proof."""
from contextlib import nullcontext
from copy import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_draft_loop as loop


class Common(NS):
    def batch_size(self):
        return self.seq_lens.shape[0]

    def replace(self, **kwargs):
        other = copy(self)
        other.__dict__.update(kwargs)
        return other


class FlashAttentionMetadataBuilder:
    aot_schedule = False

    def build_for_drafting(self, common_attn_metadata, draft_index):
        return common_attn_metadata


class Eagle3LlamaForCausalLM:
    _round3_full_private_int8_top1 = True
    config = NS(num_hidden_layers=1)

    def __init__(self):
        self.seen = []

    def embed_input_ids(self, ids):
        return ids[:, None].float().expand(-1, 3)

    def __call__(self, input_ids, positions, hidden_states, inputs_embeds):
        ids = input_ids if input_ids is not None else inputs_embeds[:, 0]
        self.seen.append((int(ids[0]), int(positions[0]), int(hidden_states[0, 0])))
        last = hidden_states + ids[:, None] + positions[:, None]
        return last, last + 2


def update(**kw):
    pos = kw["positions_1d"] + 1
    exceeds = pos >= kw["max_model_len"]
    clamped = torch.where(exceeds, 0, pos)
    table = kw["block_table_tensor"]
    blocks = (clamped // kw["block_size"]).clamp_max(table.shape[1] - 1)
    slots = table[0, blocks] * kw["block_size"] + clamped % kw["block_size"]
    kw["out_slot_mapping"].fill_(-1)
    kw["out_slot_mapping"][:1].copy_(torch.where(exceeds, -1, slots))
    kw["seq_lens"].copy_(
        torch.where(exceeds, 1, kw["seq_lens"] + 1).clamp_max(kw["max_model_len"])
    )
    kw["out_clamped_positions"].copy_(clamped)


def fixture():
    builder = FlashAttentionMetadataBuilder()
    proposer = NS(
        method="eagle3",
        num_speculative_tokens=4,
        parallel_drafting=False,
        pass_hidden_states_to_model=True,
        needs_extra_input_slots=False,
        uses_mrope=False,
        uses_xdrope_dim=0,
        draft_uses_xdrope_dim=0,
        block_size=4,
        max_model_len=24,
        supports_mm_inputs=False,
        use_local_argmax_reduction=True,
        hidden_size=1024,
        dtype=torch.bfloat16,
        speculative_config=NS(enforce_eager=False),
        vllm_config=NS(
            scheduler_config=NS(async_scheduling=False, max_num_seqs=8),
            parallel_config=NS(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
            ),
        ),
        model=Eagle3LlamaForCausalLM(),
        input_ids=torch.zeros(5, dtype=torch.int32),
        positions=torch.zeros(5, dtype=torch.int64),
        hidden_states=torch.zeros(5, 3),
        inputs_embeds=torch.zeros(5, 3),
        _slot_mapping_buffer=torch.full((5,), -1, dtype=torch.int64),
        arange=torch.arange(6, dtype=torch.int32),
        _draft_attn_layer_names=["draft"],
        draft_attn_groups=[
            NS(get_metadata_builder=lambda: builder, layer_names=["draft"])
        ],
        _greedy_sample=lambda h: h[:, 0].long() % 97,
    )
    cad = Common(
        num_actual_tokens=5,
        max_query_len=5,
        max_seq_len=8,
        causal=True,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        block_table_tensor=torch.tensor([[2, 8, 1, 9, 3, 4]], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([8]),
        _num_computed_tokens_cpu=torch.tensor([3]),
    )
    runtime = NS(no_graph="NONE", context=lambda *a, **kw: nullcontext(), update=update)
    return proposer, cad, runtime


class DraftLoopTests(unittest.TestCase):
    def setUp(self):
        self.proposer, self.cad, self.runtime = fixture()

    def plan(self):
        with patch.object(loop, "_runtime", return_value=self.runtime):
            return loop._DraftLoop(self.proposer, self.cad, 5)

    def args(self):
        return (
            torch.tensor([7]),
            torch.tensor([4]),
            torch.full((1, 3), 11.0),
            self.cad,
            torch.tensor([3]),
        )

    def test_rejection_then_three_causal_steps_preserve_ids_and_shape(self):
        plan = self.plan()
        plan._reset(*self.args())
        plan._run_steps()
        self.assertEqual(plan.output.tolist(), [[7, 23, 54, 20]])
        self.assertEqual(
            self.proposer.model.seen, [(7, 5, 11), (23, 6, 25), (54, 7, 56)]
        )
        self.assertEqual(plan.seq_lens.tolist(), [8])
        self.assertEqual(plan.slots.tolist(), [35, -1, -1, -1, -1])
        self.assertEqual(self.cad.seq_lens.tolist(), [8])
        self.assertIs(plan.attention["draft"].seq_lens, plan.seq_lens)
        self.assertEqual(plan.cad.max_seq_len, 24)

    def test_tail_embedding_branch_keeps_same_recurrence(self):
        self.proposer.supports_mm_inputs = True
        plan = self.plan()
        plan._reset(*self.args())
        plan._run_steps()
        self.assertEqual(plan.output.tolist(), [[7, 23, 54, 20]])

    def test_reset_changes_request_block_table_without_changing_addresses(self):
        plan = self.plan()
        table_ptr, lengths_ptr = plan.block_table.data_ptr(), plan.seq_lens.data_ptr()
        plan._reset(*self.args())
        plan._run_steps()
        self.cad.block_table_tensor = torch.tensor(
            [[7, 6, 5, 4, 3, 2]], dtype=torch.int32
        )
        plan._reset(*self.args())
        plan._run_steps()
        self.assertEqual(plan.slots.tolist(), [27, -1, -1, -1, -1])
        self.assertEqual(plan.block_table.data_ptr(), table_ptr)
        self.assertEqual(plan.seq_lens.data_ptr(), lengths_ptr)
        self.assertEqual(plan.output.tolist(), [[7, 23, 54, 20]])

    def test_context_boundary_uses_original_clamp_and_padding_slots(self):
        plan = self.plan()
        args = list(self.args())
        args[1] = torch.tensor([22])
        self.cad.seq_lens.fill_(23)
        args[-1] = torch.tensor([0])
        plan._reset(*args)
        plan._run_steps()
        self.assertEqual([v[1] for v in self.proposer.model.seen], [23, 0, 1])
        self.assertEqual(plan.seq_lens.tolist(), [2])
        self.assertEqual(plan.slots.tolist(), [9, -1, -1, -1, -1])

    def test_cold_capture_resets_each_warmup_and_return_owns_storage(self):
        plan = self.plan()
        graph = NS(replay=plan._run_steps)
        stream = NS(wait_stream=lambda other: None, synchronize=lambda: None)
        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "Stream", return_value=stream),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(torch.cuda, "graph", return_value=nullcontext()),
            patch.object(torch.cuda, "CUDAGraph", return_value=graph),
        ):
            first_result = plan.run(*self.args())
            self.assertEqual(plan.captures, 1)
            self.assertEqual(plan.replays, 1)
            self.assertEqual(
                self.proposer.model.seen, [(7, 5, 11), (23, 6, 25), (54, 7, 56)] * 4
            )
            self.assertEqual(self.cad.max_seq_len, 11)
            self.assertIsNone(self.cad._seq_lens_cpu)
            self.assertEqual(self.cad.query_start_loc.tolist(), [0, 1])
            args = list(self.args())
            args[0] = torch.tensor([9])
            second_result = plan.run(*args)
        self.assertEqual(plan.captures, 1)
        self.assertEqual(plan.replays, 2)
        self.assertEqual(first_result.tolist(), [[7, 23, 54, 20]])
        self.assertNotEqual(second_result.tolist(), first_result.tolist())
        self.assertNotEqual(first_result.data_ptr(), plan.output.data_ptr())

    def test_prefill_and_batching_fallback_do_not_dispatch_or_mutate(self):
        original = Mock(return_value="native")
        wrapper = loop._make_wrapper(original)
        self.proposer._determine_batch_execution_and_padding = Mock(
            side_effect=AssertionError("No graph dispatch during fallback")
        )
        for rejected, changes in (
            (None, {}),
            (torch.tensor([0]), {"num_actual_tokens": 100}),
            (torch.tensor([0, 0]), {"seq_lens": torch.tensor([8, 8])}),
        ):
            with self.subTest(changes=changes):
                cad = self.cad.replace(**changes)
                output = wrapper(
                    self.proposer,
                    1,
                    2,
                    3,
                    4,
                    5,
                    cad,
                    6,
                    num_rejected_tokens_gpu=rejected,
                )
                self.assertEqual(output, "native")
                self.assertEqual(
                    original.call_args.args, (self.proposer, 1, 2, 3, 4, 5, cad, 6)
                )
                self.assertIs(
                    original.call_args.kwargs["num_rejected_tokens_gpu"], rejected
                )
        self.assertFalse(hasattr(self.proposer, "_vllm019_draft_loop"))

    def test_cpu_and_unsupported_head_are_rejected_without_cuda_calls(self):
        self.assertEqual(
            loop._fallback_reason(self.proposer, self.cad, torch.tensor([0]), None),
            "non_cuda",
        )
        self.assertEqual(
            loop._fallback_reason(
                self.proposer, self.cad, torch.tensor([0]), ([], object())
            ),
            "non_cuda",
        )
        self.assertEqual(
            loop._fallback_reason(
                self.proposer, self.cad, torch.tensor([0]), ([object()], object())
            ),
            "prefill_or_non_b1",
        )
        self.proposer.use_local_argmax_reduction = False
        self.assertEqual(
            loop._fallback_reason(self.proposer, self.cad, torch.tensor([0]), None),
            "draft_model_or_head",
        )

    def test_plan_shape_change_must_fallback(self):
        plan = self.plan()
        self.assertTrue(plan.matches(self.cad, 5))
        self.assertFalse(plan.matches(self.cad, 1))
        self.assertFalse(
            plan.matches(self.cad.replace(block_table_tensor=torch.zeros(1, 7)), 5)
        )

    def test_target_xdrope_with_text_only_draft_is_admitted(self):
        self.proposer.uses_xdrope_dim = 3
        reason = loop._fallback_reason(self.proposer, self.cad, torch.tensor([0]), None)
        self.assertEqual(reason, "non_cuda")
        self.proposer.draft_uses_xdrope_dim = 3
        reason = loop._fallback_reason(self.proposer, self.cad, torch.tensor([0]), None)
        self.assertEqual(reason, "positions_or_attention")

    def test_async_requires_scheduler_limit_one_and_actual_batch_one(self):
        scheduler = self.proposer.vllm_config.scheduler_config
        scheduler.async_scheduling = True
        scheduler.max_num_seqs = 1
        # Admission is CPU-only. A fake device marker avoids CUDA allocation.
        self.proposer.input_ids = NS(is_cuda=True)
        reason = loop._fallback_reason(self.proposer, self.cad, torch.tensor([0]), None)
        self.assertIsNone(reason)

        scheduler.max_num_seqs = 2
        reason = loop._fallback_reason(self.proposer, self.cad, torch.tensor([0]), None)
        self.assertEqual(reason, "scheduler_or_parallelism")

        scheduler.max_num_seqs = 1
        cad = self.cad.replace(seq_lens=torch.tensor([8, 8]))
        reason = loop._fallback_reason(self.proposer, cad, torch.tensor([0, 0]), None)
        self.assertEqual(reason, "prefill_or_non_b1")


if __name__ == "__main__":
    unittest.main()
