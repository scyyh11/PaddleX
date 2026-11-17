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

"""CPU layout/fallback contracts for lossless draft staging."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_draft_staging as staging


def tensors():
    return (
        torch.zeros(5, dtype=torch.int32),
        torch.zeros(5, dtype=torch.int64),
        torch.zeros(5, 3072, dtype=torch.bfloat16),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, 768, dtype=torch.int32),
        torch.zeros(5, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
    )


class DraftStagingTests(unittest.TestCase):
    def setUp(self):
        self.sources = tensors()
        self.destinations = tuple(torch.empty_like(t) for t in self.sources)

    def test_cpu_input_does_not_launch_cuda(self):
        with patch.object(staging, "_stage_kernel") as kernel:
            self.assertFalse(staging.stage_tensors(self.sources, self.destinations))
            kernel.assert_not_called()
        self.assertEqual(
            staging._layout_reason(self.sources, self.destinations), "non_cuda"
        )

    def test_shape_dtype_stride_and_complete_table_are_checked(self):
        for index, replacement, expected in (
            (0, torch.zeros(5, dtype=torch.int64), "dtype"),
            (2, torch.zeros(5, 1024, dtype=torch.bfloat16), "shape"),
            (2, torch.zeros(5, 6144, dtype=torch.bfloat16)[:, ::2], "strides"),
            (6, torch.zeros(2, 768, dtype=torch.int32), "block_table_shape"),
            (6, torch.zeros(1, 512, dtype=torch.int32), "destination_shape"),
        ):
            with self.subTest(index=index, expected=expected):
                source = list(self.sources)
                source[index] = replacement
                self.assertEqual(
                    staging._layout_reason(source, self.destinations), expected
                )

    def test_wrapper_passes_native_position_projection_and_all_nine_buffers(self):
        s, d = self.sources, self.destinations
        plan = NS(proposer=object())
        names = (
            "target_ids",
            "target_positions",
            "target_hidden",
            "next_ids",
            "indices",
            "target_lengths",
            "target_blocks",
            "target_slots",
            "rejected",
        )
        for name, tensor in zip(names, d):
            setattr(plan, name, tensor)
        original = Mock(return_value="native")
        row = Mock(return_value=s[1])
        wrapper = staging._make_wrapper(original, row)
        cad = NS(seq_lens=s[5], block_table_tensor=s[6], slot_mapping=s[7])
        positions = object()
        with patch.object(staging, "stage_tensors", return_value=True) as fused:
            self.assertIsNone(
                wrapper(plan, s[0], positions, s[2], s[3], s[4], cad, s[8])
            )
            self.assertTrue(all(a is b for a, b in zip(fused.call_args.args[0], s)))
            self.assertTrue(all(a is b for a, b in zip(fused.call_args.args[1], d)))
            original.assert_not_called()
        row.assert_called_once_with(plan.proposer, positions)
        with patch.object(staging, "stage_tensors", return_value=False):
            self.assertEqual(
                wrapper(plan, s[0], positions, s[2], s[3], s[4], cad, s[8]), "native"
            )
        self.assertEqual(
            original.call_args.args,
            (plan, s[0], positions, s[2], s[3], s[4], cad, s[8]),
        )


if __name__ == "__main__":
    unittest.main()
