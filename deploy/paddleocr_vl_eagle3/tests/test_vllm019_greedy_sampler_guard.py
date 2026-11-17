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

"""CPU-only contracts for preserving customized sampler behavior."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn.modules import module as nn_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_greedy_sampler as adapter


class NativeSampler(torch.nn.Module):
    def forward(self, value):
        return value


class NativeRejection(torch.nn.Module):
    def __init__(self, sampler=None):
        super().__init__()
        self.sampler = sampler if sampler is not None else NativeSampler()


class CustomSampler(NativeSampler):
    def forward(self, value):
        return "custom bonus token"


class CustomRejection(NativeRejection):
    pass


class GreedySamplerGuardTests(unittest.TestCase):
    def setUp(self):
        self.original = Mock(return_value="native path")
        self.wrapper = adapter._make_wrapper(
            self.original,
            NativeRejection,
            NativeSampler,
        )
        self.args = tuple(object() for _ in range(4))

    def assert_fallback(self, instance):
        with patch.object(adapter, "try_greedy_rejection_sample") as fast:
            self.assertEqual(self.wrapper(instance, *self.args), "native path")
            fast.assert_not_called()
        self.original.assert_called_once_with(instance, *self.args)
        self.original.reset_mock()

    def test_plain_native_types_use_existing_fast_path(self):
        instance = NativeRejection()
        with patch.object(
            adapter, "try_greedy_rejection_sample", return_value="fast"
        ) as fast:
            self.assertEqual(self.wrapper(instance, *self.args), "fast")
            fast.assert_called_once_with(self.args[0], self.args[2], self.args[3])
        self.original.assert_not_called()

    def test_subclasses_preserve_original_calls(self):
        self.assert_fallback(NativeRejection(CustomSampler()))
        self.assert_fallback(CustomRejection())

    def test_instance_forward_hooks_and_pre_hooks_preserve_original_calls(self):
        for nested in (False, True):
            for pre_hook in (False, True):
                instance = NativeRejection()
                module = instance.sampler if nested else instance
                register = (
                    module.register_forward_pre_hook
                    if pre_hook
                    else module.register_forward_hook
                )
                handle = register(lambda *args: None)
                try:
                    self.assert_fallback(instance)
                finally:
                    handle.remove()

    def test_global_forward_hooks_preserve_nested_sampler_hook_execution(self):
        for register in (
            nn_module.register_module_forward_hook,
            nn_module.register_module_forward_pre_hook,
        ):
            handle = register(lambda *args: None)
            try:
                self.assert_fallback(NativeRejection())
            finally:
                handle.remove()

    def test_instance_overrides_and_compiled_calls_preserve_original_calls(self):
        for name in (
            "forward",
            "sample",
            "greedy_sample",
            "apply_logits_processors",
            "apply_penalties",
            "_compiled_call_impl",
        ):
            instance = NativeRejection()
            setattr(instance.sampler, name, lambda *args: "custom sampler")
            self.assert_fallback(instance)
        instance = NativeRejection()
        instance.apply_logits_processors = lambda *args: "custom target processor"
        self.assert_fallback(instance)

    def test_sampling_guard_failure_still_forwards_every_original_argument(self):
        instance = NativeRejection()
        with patch.object(adapter, "try_greedy_rejection_sample", return_value=None):
            self.assertEqual(self.wrapper(instance, *self.args), "native path")
        self.original.assert_called_once_with(instance, *self.args)


if __name__ == "__main__":
    unittest.main()
