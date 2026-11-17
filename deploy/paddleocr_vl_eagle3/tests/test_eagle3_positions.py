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

"""Exercise the real native first-pass code on CPU, without loading model weights."""

import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hps_batch import eagle3_positions


def has_native_vllm019():
    try:
        return importlib.metadata.version("vllm") == "0.19.0"
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(has_native_vllm019(), "Native position tests require vLLM 0.19.0")
class NativePositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

        cls.torch = torch
        cls.proposer_cls = SpecDecodeBaseProposer
        eagle3_positions.install_proposer_hook(cls.proposer_cls)

    def proposer(self, capacity=512):
        torch = self.torch
        proposer = self.proposer_cls.__new__(self.proposer_cls)
        proposer.vllm_config = NS(
            model_config=NS(hf_config=NS(model_type="paddleocr_vl"), uses_mrope=True)
        )
        proposer.method = "eagle3"
        proposer.uses_mrope = False
        proposer.uses_xdrope_dim = 0
        proposer.draft_uses_xdrope_dim = 0
        proposer.needs_extra_input_slots = False
        proposer.input_ids = torch.zeros(capacity, dtype=torch.int32)
        proposer.positions = torch.zeros(capacity, dtype=torch.int64)
        proposer.hidden_states = torch.zeros(capacity, 4)
        proposer.arange = torch.arange(capacity, dtype=torch.int32)
        return proposer

    def exercise_batch(self, query_lengths, seq_lengths, rope_deltas, rejections=None):
        torch = self.torch
        proposer = self.proposer()
        query_start_loc = torch.tensor(
            [0] + list(torch.tensor(query_lengths).cumsum(0)), dtype=torch.int32
        )
        num_tokens = sum(query_lengths)
        native_expected = torch.cat(
            [
                torch.arange(end - length, end)
                for length, end in zip(query_lengths, seq_lengths)
            ]
        )
        rope = native_expected + torch.repeat_interleave(
            torch.tensor(rope_deltas), torch.tensor(query_lengths)
        )
        target_positions = rope[None, :].repeat(3, 1)
        target_original = target_positions.clone()
        original_inputs = torch.arange(1000, 1000 + num_tokens, dtype=torch.int32)
        next_tokens = torch.arange(2000, 2000 + len(query_lengths), dtype=torch.int32)
        hidden_states = torch.arange(num_tokens * 4, dtype=torch.float).view(
            num_tokens, 4
        )
        seq_lens = torch.tensor(seq_lengths, dtype=torch.int32)
        cad = NS(query_start_loc=query_start_loc, seq_lens=seq_lens)
        rejected = (
            torch.tensor(rejections, dtype=torch.int32)
            if rejections is not None
            else None
        )
        sample_indices = query_start_loc[1:] - 1
        if rejected is not None:
            sample_indices -= rejected
        result = proposer.set_inputs_first_pass(
            target_token_ids=original_inputs,
            next_token_ids=next_tokens,
            target_positions=target_positions,
            target_hidden_states=hidden_states,
            token_indices_to_sample=sample_indices if rejected is not None else None,
            cad=cad,
            num_rejected_tokens_gpu=rejected,
        )
        torch.testing.assert_close(proposer.positions[:num_tokens], native_expected)
        torch.testing.assert_close(target_positions, target_original)
        torch.testing.assert_close(proposer.hidden_states[:num_tokens], hidden_states)
        torch.testing.assert_close(result[1], sample_indices)
        torch.testing.assert_close(proposer.input_ids[sample_indices], next_tokens)
        torch.testing.assert_close(
            cad.seq_lens, torch.tensor(seq_lengths, dtype=torch.int32)
        )
        self.assertIs(result[2], cad)
        # The untouched native recurrent kernel now receives the correct KV
        # ordinal after its own removal of rejected-token padding.
        accepted_seq_lens = seq_lens if rejected is None else seq_lens - rejected
        torch.testing.assert_close(
            proposer.positions[sample_indices] + 1, accepted_seq_lens.to(torch.int64)
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_b8_mixed_prefill_decode_and_chunk_offsets(self):
        self.exercise_batch(
            query_lengths=[5, 1, 4, 2, 8, 3, 1, 6],
            seq_lengths=[260, 68, 36, 133, 400, 83, 777, 18],
            rope_deltas=[-240, -56, -24, -100, -320, -64, 0, 0],
        )

    def test_b8_padded_rejection_keeps_accepted_endpoint_and_prefix(self):
        self.exercise_batch(
            query_lengths=[5, 1, 4, 2, 8, 3, 1, 6],
            seq_lengths=[260, 68, 36, 133, 400, 83, 777, 18],
            rope_deltas=[-240, -56, -24, -100, -320, -64, 0, 0],
            rejections=[4, 0, 2, 1, 0, 2, 0, 3],
        )

    def test_first_full_image_prefill_matches_training_ordinals(self):
        self.exercise_batch([260], [260], [-240])

    def test_other_models_keep_native_mrope_to_1d_behavior(self):
        torch = self.torch
        proposer = self.proposer()
        proposer.vllm_config.model_config.hf_config.model_type = "qwen2_vl"
        target_positions = torch.tensor([[7, 8], [7, 8], [7, 8]])
        proposer.set_inputs_first_pass(
            torch.tensor([10, 11], dtype=torch.int32),
            torch.tensor([12], dtype=torch.int32),
            target_positions,
            torch.zeros(2, 4),
            None,
            NS(query_start_loc=torch.tensor([0, 2]), seq_lens=torch.tensor([20])),
            None,
        )
        torch.testing.assert_close(proposer.positions[:2], target_positions[0])

    def test_native_paddle_mrope_exposes_difference_from_kv_ordinals(self):
        torch = self.torch
        from vllm.model_executor.models.paddleocr_vl import (
            PaddleOCRVLForConditionalGeneration,
        )

        config = NS(
            image_token_id=100295,
            video_token_id=101307,
            vision_start_token_id=101305,
            vision_config=NS(spatial_merge_size=2, tokens_per_second=2),
        )
        for h, w, prompt_length, delta, native_next in (
            (32, 32, 260, -240, 20),
            (16, 16, 68, -56, 12),
            (8, 16, 36, -24, 12),
        ):
            with self.subTest(grid=(h, w)):
                tokens = [101305] + [100295] * (h * w // 4) + [101306, 101, 102]
                with patch(
                    "vllm.model_executor.models.paddleocr_vl.MultiModalFeatureSpec.gather_kwargs",
                    return_value={"image_grid_thw": [torch.tensor([1, h, w])]},
                ):
                    positions, observed_delta = (
                        PaddleOCRVLForConditionalGeneration.get_mrope_input_positions(
                            NS(config=config), tokens, []
                        )
                    )
                self.assertEqual(len(tokens), prompt_length)
                self.assertEqual(observed_delta, delta)
                self.assertEqual(int(positions[0, -1]) + 1, native_next)
                self.assertNotEqual(native_next, prompt_length)

    def test_extra_slot_path_is_not_silently_modified(self):
        proposer = self.proposer()
        proposer.needs_extra_input_slots = True
        with self.assertRaisesRegex(ValueError, "sequential EAGLE3"):
            proposer.set_inputs_first_pass(None, None, None, None, None, None, None)


if __name__ == "__main__":
    unittest.main()
