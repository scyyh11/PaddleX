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

"""CPU contracts for the one-query FA2 compatibility adapter."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_ar1_attention as ar1


class Ar1AttentionTests(unittest.TestCase):
    def setUp(self):
        q = torch.zeros(5, 16, 128, dtype=torch.bfloat16)
        self.kw = dict(
            q=q,
            k=torch.zeros(2, 16, 2, 128, dtype=torch.bfloat16),
            v=torch.zeros(2, 16, 2, 128, dtype=torch.bfloat16),
            out=torch.empty_like(q),
            cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
            seqused_k=torch.tensor([10], dtype=torch.int32),
            block_table=torch.zeros(1, 4, dtype=torch.int32),
            max_seqlen_q=5,
            max_seqlen_k=12288,
            fa_version=2,
            causal=True,
            num_splits=0,
        )
        self.calls = []

        def native(**kw):
            self.calls.append(kw)
            kw["out"].fill_(int(kw["seqused_k"][0]))
            return kw["out"]

        self.attention = ar1._make_wrapper(native)

    def test_rows_preserve_causal_prefix_and_write_original_output(self):
        result = self.attention(**self.kw)
        self.assertIs(result, self.kw["out"])
        self.assertEqual(result[:, 0, 0].tolist(), [6, 7, 8, 9, 10])
        self.assertEqual(len(self.calls), 5)
        for row, call in enumerate(self.calls):
            self.assertEqual(call["max_seqlen_q"], 1)
            self.assertEqual(call["cu_seqlens_q"].tolist(), [0, 1])
            self.assertEqual(call["num_splits"], 0)
            self.assertEqual(call["max_seqlen_k"], 12288)
            self.assertIs(call["block_table"], self.kw["block_table"])
            self.assertEqual(call["q"].data_ptr(), self.kw["q"][row].data_ptr())
        self.assertEqual(self.kw["seqused_k"].tolist(), [10])
        self.assertEqual(self.kw["cu_seqlens_q"].tolist(), [0, 5])

    def test_padding_and_empty_causal_prefix_never_read_future_kv(self):
        self.kw["cu_seqlens_q"] = torch.tensor([0, 3], dtype=torch.int32)
        result = self.attention(**self.kw)
        self.assertEqual(result[:, 0, 0].tolist(), [8, 9, 10, 0, 0])
        self.kw["cu_seqlens_q"] = torch.tensor([0, 5], dtype=torch.int32)
        self.kw["seqused_k"] = torch.tensor([2], dtype=torch.int32)
        result = self.attention(**self.kw)
        self.assertEqual(result[:, 0, 0].tolist(), [0, 0, 0, 1, 2])

    def test_non_target_shape_or_nonstandard_attention_uses_original_call(self):
        for changes in (
            {"q": torch.zeros(5, 8, 128, dtype=torch.bfloat16)},
            {"causal": False},
            {"num_splits": 1},
            {"return_softmax_lse": True},
            {"block_table": torch.zeros(2, 4, dtype=torch.int32)},
        ):
            with self.subTest(changes=tuple(changes)):
                self.calls.clear()
                kw = self.kw | changes
                self.attention(**kw)
                self.assertEqual(len(self.calls), 1)
                self.assertIs(self.calls[0]["q"], kw["q"])
                self.assertIs(self.calls[0]["seqused_k"], kw["seqused_k"])

    def test_split_count_matches_pinned_ar1_gqa_heuristic(self):
        for bound, splits in ((128, 1), (129, 2), (3916, 31), (4096, 32), (12288, 96)):
            with self.subTest(bound=bound):
                self.assertEqual(ar1.ar1_num_splits(bound, 108), splits)
        for bound, sms in ((0, 108), (12288, 0)):
            with self.assertRaises(ValueError):
                ar1.ar1_num_splits(bound, sms)

    def test_batch_uses_one_native_call_with_separate_causal_endpoints(self):
        self.kw["cu_seqlens_q"] = torch.tensor([0, 3], dtype=torch.int32)
        self.kw["block_table"] = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        self.kw["softmax_scale"] = 0.0
        calls = []

        def native(*args):
            calls.append(args)
            args[3].copy_(args[6][:, None, None].expand_as(args[3]))
            return [args[3], None]

        result = ar1._batched_attention(self.kw, self.kw["out"], kernel=native)
        self.assertIs(result, self.kw["out"])
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(len(call), 22)
        self.assertEqual(call[4].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(call[6].tolist(), [8, 9, 10, 0, 0])
        self.assertEqual(call[8].tolist(), [[1, 2, 3, 4]] * 3 + [[0] * 4] * 2)
        self.assertEqual(call[10], 1)
        self.assertEqual(call[13], 0.0)
        self.assertEqual(call[20], 96)
        self.assertEqual(result[:, 0, 0].tolist(), [8, 9, 10, 0, 0])

    def test_batch_device_guard_rejects_unvalidated_hardware(self):
        q = SimpleNamespace(is_cuda=True, device="fixture")
        expected = dict(
            major=8,
            minor=0,
            multi_processor_count=108,
            name="NVIDIA A100 80GB PCIe",
            total_memory=80 * 1024**3,
        )
        for changes, allowed in (
            ({}, True),
            ({"total_memory": 40 * 1024**3}, False),
            ({"major": 9, "name": "NVIDIA H100"}, False),
            ({"multi_processor_count": 107}, False),
        ):
            with self.subTest(changes=changes), patch.dict(
                ar1._DEVICE_SUPPORT, {}, clear=True
            ), patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(**(expected | changes)),
            ):
                self.assertEqual(ar1._supports_row_batch_device(q), allowed)


if __name__ == "__main__":
    unittest.main()
