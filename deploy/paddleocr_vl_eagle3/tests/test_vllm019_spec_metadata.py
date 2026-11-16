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

"""CPU contracts for static B1/K4/Q5 speculative metadata."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_spec_metadata as adapter


class SpecMetadataTests(unittest.TestCase):
    def setUp(self):
        self.ids = SimpleNamespace(
            is_cuda=True,
            ndim=1,
            shape=(4096,),
            dtype=torch.int32,
            stride=lambda dim: 1,
        )

    def test_only_exact_b1_k4_q5_shape_is_admitted(self):
        cases = (
            (4, [4], [5], True),
            (3, [4], [5], False),
            (4, [3], [4], False),
            (4, [4], [6], False),
            (4, [0], [1245], False),
            (4, [4, 4], [5, 10], False),
        )
        for k, drafts, scheduled, allowed in cases:
            with self.subTest(k=k, drafts=drafts, scheduled=scheduled):
                self.assertEqual(
                    adapter.eligible(
                        k,
                        np.array(drafts, dtype=np.int32),
                        np.array(scheduled, dtype=np.int32),
                        self.ids,
                    ),
                    allowed,
                )
        self.assertFalse(adapter.eligible(4, [4], [5], self.ids))

    def test_tensor_contract_falls_back(self):
        for changes in (
            {"is_cuda": False},
            {"ndim": 2},
            {"shape": (4,)},
            {"dtype": torch.int64},
            {"stride": lambda dim: 2},
        ):
            ids = SimpleNamespace(**(vars(self.ids) | changes))
            self.assertFalse(
                adapter.eligible(
                    4,
                    np.array([4]),
                    np.array([5]),
                    ids,
                )
            )

    def test_mapping_matches_native_integer_formula(self):
        ids = torch.tensor([91, 7, 101, 18, 90, 111], dtype=torch.int32)
        result, _ = adapter.build_metadata(ids, metadata_cls=SimpleNamespace)
        expected = {
            "draft_token_ids": [7, 101, 18, 90],
            "cu_num_draft_tokens": [4],
            "cu_num_sampled_tokens": [5],
            "target_logits_indices": [0, 1, 2, 3],
            "bonus_logits_indices": [4],
            "logits_indices": [0, 1, 2, 3, 4],
        }
        self.assertEqual(result.num_draft_tokens, [4])
        for name, values in expected.items():
            self.assertEqual(getattr(result, name).tolist(), values)
            self.assertEqual(getattr(result, name).dtype, torch.int32)

    def test_inflight_metadata_owns_tokens_while_indices_are_reused(self):
        ids = torch.arange(8, dtype=torch.int32)
        first, cache = adapter.build_metadata(ids, metadata_cls=SimpleNamespace)
        ids.add_(100)
        second, next_cache = adapter.build_metadata(
            ids,
            cache,
            metadata_cls=SimpleNamespace,
        )
        self.assertEqual(first.draft_token_ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(second.draft_token_ids.tolist(), [101, 102, 103, 104])
        self.assertNotEqual(first.draft_token_ids.data_ptr(), ids.data_ptr())
        self.assertIs(cache, next_cache)
        self.assertIs(first.logits_indices, second.logits_indices)
        self.assertIsNot(first.num_draft_tokens, second.num_draft_tokens)

    def test_wrapper_preserves_original_fallback(self):
        calls = []

        def original(self, drafts, scheduled):
            calls.append((self, drafts, scheduled))
            return "original result"

        runner = SimpleNamespace(
            num_spec_tokens=4, input_ids=SimpleNamespace(gpu=self.ids)
        )
        drafts, scheduled = np.array([3]), np.array([4])
        result = adapter._make_wrapper(original)(runner, drafts, scheduled)
        self.assertEqual(result, "original result")
        self.assertIs(calls[0][1], drafts)
        self.assertIs(calls[0][2], scheduled)


if __name__ == "__main__":
    unittest.main()
