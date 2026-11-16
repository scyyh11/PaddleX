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

"""CPU contracts only: mocked graph launch and selector; no CUDA allocation."""
from pathlib import Path
import sys

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vllm019_sampler_graph as hook


class HookTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("_CACHE", None),
            ("_OWNER", None),
            ("_INSTALLATION", None),
            ("_DEVICE_SUPPORT", {}),
            ("COUNTS", hook.Counter()),
        ):
            item = patch.object(hook, name, value)
            item.start()
            self.addCleanup(item.stop)
        self.spec = SimpleNamespace(
            max_spec_len=4,
            num_draft_tokens=[4],
            draft_token_ids=object(),
            cu_num_draft_tokens=object(),
            target_logits_indices=object(),
            bonus_logits_indices=object(),
        )
        self.sm = object()
        self.module = SimpleNamespace(
            eligible=Mock(return_value=True), _select_greedy_tokens=Mock()
        )

    def logits(
        self, rows=5, cols=103424, dtype=torch.float32, contiguous=True, cuda=True
    ):
        tensor = Mock(spec=torch.Tensor)
        tensor.shape, tensor.ndim = (rows, cols), 2
        tensor.dtype, tensor.is_cuda = dtype, cuda
        tensor.device = torch.device("cuda:0")
        tensor.is_contiguous.return_value = contiguous
        return tensor

    def test_scope_guards_keep_supported_bf16_fp32_only(self):
        for dtype in (torch.bfloat16, torch.float32):
            for q in (1, 5, 7):
                self.assertIsNone(
                    hook._profile_reason(self.spec, self.logits(q, dtype=dtype))
                )
        for kwargs in (
            {"rows": 0},
            {"rows": 8},
            {"cols": 40000},
            {"dtype": torch.float16},
            {"contiguous": False},
            {"cuda": False},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(
                    hook._profile_reason(self.spec, self.logits(**kwargs)),
                    "logits_profile",
                )
        self.spec.num_draft_tokens = [2, 2]
        self.assertEqual(
            hook._profile_reason(self.spec, self.logits()), "batch_profile"
        )

    def test_original_eligible_runs_before_new_guards_and_fallback_is_unchanged(self):
        original = Mock(return_value=object())
        self.module.eligible.return_value = False
        sample = hook._make_try(original, self.module, SimpleNamespace)
        with patch.object(hook, "_profile_reason") as profile, patch.object(
            hook, "_get_cache"
        ) as cache:
            self.assertIs(
                sample(metadata=self.spec, logits=None, sampling_metadata=self.sm),
                original.return_value,
            )
        self.module.eligible.assert_called_once_with(self.spec, None, self.sm)
        original.assert_called_once_with(self.spec, None, self.sm)
        profile.assert_not_called()
        cache.assert_not_called()

    def test_unsupported_shape_or_cache_owner_calls_original_function(self):
        for shape_reason, cache_reason in (
            ("logits_profile", None),
            (None, "stream_or_thread"),
            (None, "capture"),
        ):
            original = Mock(return_value=object())
            logits = self.logits()
            with patch.object(
                hook, "_profile_reason", return_value=shape_reason
            ), patch.object(hook, "_get_cache", return_value=(None, cache_reason)):
                result = hook._make_try(original, self.module, SimpleNamespace)(
                    self.spec, logits, self.sm
                )
            self.assertIs(result, original.return_value)
            original.assert_called_once_with(self.spec, logits, self.sm)

    def test_borrowed_argmax_is_consumed_immediately_into_independent_output(self):
        events, returns = [], []
        borrowed = torch.empty(5, dtype=torch.int64)
        logits, cast = self.logits(dtype=torch.bfloat16), object()
        logits.to.side_effect = lambda dtype: (events.append("cast"), cast)[1]
        round_number = 0

        def replay(value):
            nonlocal round_number
            self.assertIs(value, cast)
            round_number += 1
            borrowed.fill_(round_number)
            events.append("replay")
            return borrowed

        cache = Mock(side_effect=replay)

        def allocate(shape, **kwargs):
            self.assertEqual(kwargs, {"device": logits.device, "dtype": torch.int32})
            events.append("allocate")
            return torch.zeros(shape, dtype=torch.int32)

        def select(argmax, draft, cu, target, bonus, output, **kwargs):
            self.assertIs(argmax, borrowed)
            self.assertEqual(events[-1], "replay")
            self.assertEqual(kwargs, {"OUTPUT_COLUMNS": 5, "BLOCK": 8, "num_warps": 4})
            self.assertIs(draft, self.spec.draft_token_ids)
            self.assertIs(cu, self.spec.cu_num_draft_tokens)
            self.assertIs(target, self.spec.target_logits_indices)
            self.assertIs(bonus, self.spec.bonus_logits_indices)
            output.copy_(argmax.to(torch.int32).unsqueeze(0))
            events.append("select")

        class Selector:
            def __getitem__(self, grid):
                self_grid = grid
                assert self_grid == (1,)
                return select

        self.module._select_greedy_tokens = Selector()
        original = Mock()
        sample = hook._make_try(original, self.module, SimpleNamespace)
        with patch.object(hook, "_get_cache", return_value=(cache, None)), patch.object(
            hook.torch, "empty", side_effect=allocate
        ):
            returns.append(sample(self.spec, logits, self.sm))
            returns.append(sample(self.spec, logits, self.sm))
        self.assertEqual(events, ["allocate", "cast", "replay", "select"] * 2)
        self.assertTrue(
            torch.equal(
                returns[0].sampled_token_ids, torch.ones((1, 5), dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(
                returns[1].sampled_token_ids, torch.full((1, 5), 2, dtype=torch.int32)
            )
        )
        self.assertNotEqual(
            returns[0].sampled_token_ids.data_ptr(),
            returns[1].sampled_token_ids.data_ptr(),
        )
        self.assertIsNone(returns[0].logprobs_tensors)
        original.assert_not_called()

    def test_lazy_prepare_only_outside_capture_and_cache_hit_reuses_plan(self):
        device = torch.device("cuda:0")
        cache = SimpleNamespace(plans={}, prepare=Mock())
        cache.prepare.side_effect = lambda q: cache.plans.update({q: object()})
        with patch.object(hook, "_runtime_supported", return_value=True), patch.object(
            hook, "_owner", return_value=(device, 7, 1)
        ), patch.object(
            hook.torch.cuda, "device", return_value=nullcontext()
        ), patch.object(
            hook.torch.cuda, "is_current_stream_capturing", return_value=True
        ) as capturing, patch.object(
            hook, "GraphArgmaxCache", return_value=cache
        ) as factory:
            self.assertEqual(hook._get_cache(device, 5), (None, "capture"))
            factory.assert_not_called()
            capturing.return_value = False
            self.assertEqual(hook._get_cache(device, 5), (cache, None))
            capturing.return_value = True
            self.assertEqual(hook._get_cache(device, 5), (None, "capture"))
            self.assertEqual(hook._get_cache(device, 1), (None, "capture"))
            capturing.return_value = False
            self.assertEqual(hook._get_cache(device, 5), (cache, None))
        factory.assert_called_once_with(device)
        cache.prepare.assert_called_once_with(5)

    def test_capture_with_prepared_plan_returns_original_without_replay(self):
        logits = self.logits()
        device = logits.device
        owner = (device, 7, 1)
        cache = Mock(plans={5: object()})
        hook._CACHE, hook._OWNER = cache, owner
        original = Mock(return_value=object())
        sample = hook._make_try(original, self.module, SimpleNamespace)
        with (
            patch.object(hook, "_runtime_supported", return_value=True),
            patch.object(hook, "_owner", return_value=owner),
            patch.object(hook.torch.cuda, "device", return_value=nullcontext()),
            patch.object(
                hook.torch.cuda, "is_current_stream_capturing", return_value=True
            ),
        ):
            result = sample(self.spec, logits, self.sm)
        self.assertIs(result, original.return_value)
        original.assert_called_once_with(self.spec, logits, self.sm)
        cache.assert_not_called()
        cache.prepare.assert_not_called()
        self.assertEqual(hook.COUNTS["fallback_capture"], 1)

    def test_wrong_stream_thread_and_runtime_never_reuse_cache(self):
        device = torch.device("cuda:0")
        hook._OWNER = (device, 7, 1)
        hook._CACHE = SimpleNamespace(plans={5: object()})
        with patch.object(hook, "_runtime_supported", return_value=True), patch.object(
            hook, "_owner"
        ) as owner:
            for changed in (
                (device, 8, 1),
                (device, 7, 2),
                (torch.device("cuda:1"), 7, 1),
            ):
                owner.return_value = changed
                self.assertEqual(hook._get_cache(device, 5), (None, "stream_or_thread"))
        with patch.object(hook, "_runtime_supported", return_value=False):
            self.assertEqual(hook._get_cache(device, 5), (None, "runtime_profile"))

    def test_install_retains_native_wrapper_guards_and_is_idempotent(self):
        def original(*args):
            return None

        module = SimpleNamespace(try_greedy_rejection_sample=original, install=Mock())
        outer_guard = object()
        module.forward_guard = outer_guard
        outputs = SimpleNamespace(SamplerOutput=SimpleNamespace)

        def imported(name):
            return {"vllm019_greedy_sampler": module, "vllm.v1.outputs": outputs}[name]

        with patch.object(
            hook.importlib.metadata, "version", return_value="0.19.0"
        ), patch.object(hook.importlib, "import_module", side_effect=imported):
            self.assertTrue(hook.install()["installed"])
            wrapped = module.try_greedy_rejection_sample
            self.assertIs(module.forward_guard, outer_guard)
            hook.install()
            self.assertIs(module.try_greedy_rejection_sample, wrapped)
            module.install.assert_called_once_with()
            self.assertIs(hook._INSTALLATION[1], original)
            module.try_greedy_rejection_sample = object()
            with self.assertRaisesRegex(RuntimeError, "replaced after installation"):
                hook.install()

    def test_install_rejects_other_runtime_without_importing_sampler(self):
        with patch.object(
            hook.importlib.metadata, "version", return_value="0.32.0"
        ), patch.object(hook.importlib, "import_module") as imported:
            with self.assertRaises(RuntimeError):
                hook.install()
        imported.assert_not_called()

    def test_prepare_requires_install_and_supported_queries(self):
        with self.assertRaisesRegex(RuntimeError, "install"):
            hook.prepare(device="cuda:0")
        hook._INSTALLATION = object()
        for query in (0, 8, True, 1.5):
            with self.subTest(query=query), self.assertRaises(ValueError):
                hook.prepare([query], device="cuda:0")


if __name__ == "__main__":
    unittest.main()
