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

"""CPU contracts only. Fake source item counts are not GPU performance evidence."""

import unittest
from unittest.mock import patch
from functools import wraps
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_vision_scalar as candidate


class Source:
    def __init__(self, value):
        self.value = value
        self.reads = 0

    def item(self):
        self.reads += 1
        return self.value


class NativeMM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_backend = "flash"
        self._fa_version = 2
        self._forward_method = self.forward_cuda

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs):
        return self._forward_fa(*args, **kwargs)

    def _forward_fa(self, *args, **kwargs):
        return args


class NativeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_size = 1
        self.attn = NativeMM()

    def forward(self, *args, **kwargs):
        return args


class NativeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = NativeAttention()

    def forward(self, *args, **kwargs):
        return args


class NativeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1152, num_attention_heads=16)
        self.attn_backend = "flash"
        self.layers = torch.nn.ModuleList(NativeLayer() for _ in range(27))

    def forward(self, *args, **kwargs):
        return args


class ContractTests(unittest.TestCase):
    def setUp(self):
        for key in candidate.COUNTS:
            candidate.COUNTS[key] = 0
        self.calls = []

        def original(**kwargs):
            self.calls.append(kwargs.copy())
            return kwargs["max_seqlen"].item()

        self.attention = candidate._wrap_attention(original)
        self.qkv = object()
        self.cu = object()

    def invoke(self, source, cu=None):
        return self.attention(
            self.qkv,
            self.qkv,
            self.qkv,
            1,
            False,
            2,
            0.125,
            self.cu if cu is None else cu,
            source,
        )

    def scoped(self, body, eligible=True):
        return candidate._wrap_encoder(body, lambda _: eligible)

    @patch.object(candidate, "_attention_eligible", return_value=True)
    def test_once_per_native_forward_and_no_stale_length(self, _):
        source = Source(120)
        run = self.scoped(lambda self_: [self.invoke(source) for _ in range(27)])
        self.assertEqual(run(object()), [120] * 27)
        source.value = 240
        self.assertEqual(run(object()), [240] * 27)
        self.assertEqual(source.reads, 2)
        self.assertEqual(candidate.COUNTS["scalar_cache_hits"], 52)
        self.assertIsNone(candidate._ACTIVE.get())
        cpu_scalars = [kwargs["max_seqlen"] for kwargs in self.calls]
        self.assertIs(cpu_scalars[0], cpu_scalars[26])
        self.assertIsNot(cpu_scalars[0], cpu_scalars[27])
        self.assertTrue(
            all(t.device.type == "cpu" and t.dtype == torch.int32 for t in cpu_scalars)
        )
        self.assertTrue(
            all(
                kwargs["q"] is self.qkv
                and kwargs["cu_seqlens"] is self.cu
                and kwargs["scale"] == 0.125
                for kwargs in self.calls
            )
        )

    @patch.object(candidate, "_attention_eligible", return_value=True)
    def test_metadata_identity_change_falls_back(self, _):
        one, two = Source(10), Source(20)

        def body(self_):
            return self.invoke(one), self.invoke(two), self.invoke(one, object())

        self.assertEqual(self.scoped(body)(object()), (10, 20, 10))
        self.assertEqual(candidate.COUNTS["metadata_identity_fallbacks"], 2)
        self.assertIs(self.calls[1]["max_seqlen"], two)
        self.assertIs(self.calls[2]["max_seqlen"], one)

    @patch.object(candidate, "_attention_eligible", return_value=True)
    def test_nested_scope_restores_parent(self, _):
        one, two = Source(10), Source(20)
        inner = self.scoped(lambda self_: self.invoke(two))

        def body(self_):
            return self.invoke(one), inner(object()), self.invoke(one)

        self.assertEqual(self.scoped(body)(object()), (10, 20, 10))
        self.assertEqual((one.reads, two.reads), (1, 1))

    @patch.object(candidate, "_attention_eligible", return_value=True)
    def test_unsupported_nested_scope_masks_parent(self, _):
        source = Source(10)
        inner = self.scoped(lambda self_: self.invoke(source), eligible=False)

        def body(self_):
            return self.invoke(source), inner(object()), self.invoke(source)

        self.assertEqual(self.scoped(body)(object()), (10, 10, 10))
        self.assertEqual(source.reads, 2)
        self.assertIs(self.calls[1]["max_seqlen"], source)

    @patch.object(candidate, "_attention_eligible", return_value=True)
    def test_exception_clears_context(self, _):
        source = Source(10)

        def body(self_):
            self.invoke(source)
            raise RuntimeError("expected failure")

        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            self.scoped(body)(object())
        self.assertIsNone(candidate._ACTIVE.get())
        self.assertEqual(self.invoke(source), 10)
        self.assertIs(self.calls[-1]["max_seqlen"], source)

    @patch.object(candidate, "_attention_eligible", return_value=False)
    def test_unsupported_attention_keeps_native_argument(self, _):
        source = Source(10)
        self.assertEqual(self.scoped(lambda self_: self.invoke(source))(object()), 10)
        self.assertIs(self.calls[-1]["max_seqlen"], source)
        self.assertEqual(candidate.COUNTS["gpu_scalar_reads"], 0)

    def test_cpu_attention_is_not_eligible(self):
        q = torch.empty((1, 16, 16, 72), dtype=torch.bfloat16)
        self.assertFalse(
            candidate._attention_eligible(
                q,
                q,
                q,
                1,
                False,
                2,
                torch.tensor([0, 16], dtype=torch.int32),
                torch.tensor(16, dtype=torch.int32),
            )
        )

    def test_no_scope_means_no_cache(self):
        source = Source(10)
        self.assertEqual(self.invoke(source), 10)
        self.assertIs(self.calls[-1]["max_seqlen"], source)
        self.assertEqual(candidate.COUNTS["gpu_scalar_reads"], 0)


class NativeGuardTests(unittest.TestCase):
    def setUp(self):
        original_attention = lambda **kwargs: kwargs
        self.state = SimpleNamespace(
            encoder_type=NativeEncoder,
            layer_type=NativeLayer,
            attention_type=NativeAttention,
            mm_type=NativeMM,
            backend="flash",
            original_forward=NativeEncoder.forward,
            original_attention=original_attention,
            layer_forward=NativeLayer.forward,
            self_attention_forward=NativeAttention.forward,
            mm_forward=NativeMM.forward,
            mm_forward_cuda=NativeMM.forward_cuda,
            mm_forward_fa=NativeMM._forward_fa,
            attention_module=SimpleNamespace(vit_flash_attn_wrapper=original_attention),
            wrappers_module=SimpleNamespace(vit_flash_attn_wrapper=original_attention),
        )
        self.state.wrapped_forward = candidate._wrap_encoder(
            self.state.original_forward, lambda _: True
        )
        self.state.wrapped_attention = candidate._wrap_attention(original_attention)
        self.state.attention_module.vit_flash_attn_wrapper = (
            self.state.wrapped_attention
        )
        patcher = patch.object(NativeEncoder, "forward", self.state.wrapped_forward)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.encoder = NativeEncoder()

    def eligible(self):
        return candidate._encoder_eligible(self.encoder, self.state)

    def test_native_shape_and_dispatch_bindings_are_eligible(self):
        self.assertTrue(self.eligible())

    def test_class_method_changes_are_rejected(self):
        for cls, name in (
            (NativeLayer, "forward"),
            (NativeAttention, "forward"),
            (NativeMM, "forward"),
            (NativeMM, "forward_cuda"),
            (NativeMM, "_forward_fa"),
        ):
            with self.subTest(method=f"{cls.__name__}.{name}"):
                with patch.object(cls, name, lambda *args, **kwargs: None):
                    self.assertFalse(self.eligible())

    def test_instance_dispatch_override_and_wrong_binding_are_rejected(self):
        mm = self.encoder.layers[0].self_attn.attn
        for name in ("_forward_method", "forward_cuda", "_forward_fa"):
            with self.subTest(method=name):
                with patch.object(mm, name, MethodType(lambda *args: None, mm)):
                    self.assertFalse(self.eligible())
        other = NativeMM()
        with patch.object(mm, "_forward_method", other.forward_cuda):
            self.assertFalse(self.eligible())

    def test_instance_forward_hooks_and_compilation_are_rejected(self):
        layer = self.encoder.layers[0]
        with patch.object(layer, "forward", lambda *args, **kwargs: None):
            self.assertFalse(self.eligible())
        hook = layer.register_forward_pre_hook(lambda *args: None)
        self.addCleanup(hook.remove)
        self.assertFalse(self.eligible())
        hook.remove()
        with patch.object(layer, "_compiled_call_impl", lambda *args: None):
            self.assertFalse(self.eligible())

    def test_global_hooks_and_unsupported_profile_are_rejected(self):
        from torch.nn.modules import module as nn_module

        with patch.object(nn_module, "_global_forward_hooks", {1: object()}):
            self.assertFalse(self.eligible())
        with patch.object(self.encoder.layers[0].self_attn, "tp_size", 2):
            self.assertFalse(self.eligible())
        with patch.object(self.encoder.config, "hidden_size", 1024):
            self.assertFalse(self.eligible())
        with patch.object(self.encoder.layers[0].self_attn.attn, "_fa_version", 3):
            self.assertFalse(self.eligible())

    def test_metadata_and_idempotent_install_detect_replaced_hooks(self):
        with patch.object(candidate, "_INSTALLATION", self.state):
            self.assertTrue(candidate.metadata()["enabled"])
            self.assertTrue(candidate.install()["hook_identity_ok"])
            replacement = lambda **kwargs: None
            self.state.attention_module.vit_flash_attn_wrapper = replacement
            info = candidate.metadata()
            self.assertTrue(info["installed"])
            self.assertFalse(info["enabled"])
            self.assertFalse(info["hook_identity_ok"])
            with self.assertRaisesRegex(RuntimeError, "replaced"):
                candidate.install()
            with self.assertRaisesRegex(RuntimeError, "Refusing"):
                candidate.uninstall()
            self.assertIs(
                self.state.attention_module.vit_flash_attn_wrapper, replacement
            )

    def test_native_method_change_is_reported_and_install_fails_closed(self):
        with patch.object(candidate, "_INSTALLATION", self.state):
            with patch.object(NativeMM, "_forward_fa", lambda *args: None):
                info = candidate.metadata()
                self.assertTrue(info["hook_identity_ok"])
                self.assertFalse(info["native_method_identity_ok"])
                self.assertFalse(info["enabled"])
                with self.assertRaisesRegex(RuntimeError, "replaced"):
                    candidate.install()

    def test_uninstall_restores_only_owned_hooks(self):
        with patch.object(candidate, "_INSTALLATION", self.state):
            candidate.uninstall()
            self.assertIs(NativeEncoder.forward, self.state.original_forward)
            self.assertIs(
                self.state.attention_module.vit_flash_attn_wrapper,
                self.state.original_attention,
            )
            self.assertFalse(candidate.metadata()["installed"])
            self.assertFalse(candidate.metadata()["enabled"])

    def test_native_function_provenance_rejects_decorators_and_other_source_files(self):
        function = self.state.original_forward
        module = sys.modules[__name__]
        self.assertTrue(
            candidate._native_python_function(function, module, "NativeEncoder.forward")
        )

        @wraps(function)
        def decorated(*args, **kwargs):
            return function(*args, **kwargs)

        self.assertFalse(
            candidate._native_python_function(
                decorated, module, "NativeEncoder.forward"
            )
        )
        other_module = SimpleNamespace(
            __name__=module.__name__, __file__="elsewhere.py"
        )
        self.assertFalse(
            candidate._native_python_function(
                function, other_module, "NativeEncoder.forward"
            )
        )


if __name__ == "__main__":
    unittest.main()
