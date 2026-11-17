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

"""CPU checks against real vLLM 0.19 classes; CUDA_VISIBLE_DEVICES='' required."""
import importlib.metadata
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch as mock_patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hps_batch import cpu_grid as patch


def has_native_vllm019():
    try:
        return importlib.metadata.version("vllm") == "0.19.0"
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(has_native_vllm019(), "CPU-grid routing tests require vLLM 0.19.0")
class CPUGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global native, MultiModalFieldConfig
        from vllm.model_executor.models import paddleocr_vl as native
        from vllm.multimodal.inputs import MultiModalFieldConfig

        # A new resolved class inherits the real native factory and methods;
        # tests must never modify the process-wide vLLM model registration.
        cls.target = type(
            patch.ARCH,
            (native.PaddleOCRVLForConditionalGeneration,),
            {"__module__": patch.NATIVE_MODULE},
        )
        cls.original_factory = cls.target._processor_factory
        cls.native_factory = (
            native.PaddleOCRVLForConditionalGeneration._processor_factory
        )

    def setUp(self):
        self.target._processor_factory = self.original_factory

    def tearDown(self):
        self.target._processor_factory = self.original_factory

    def test_changes_only_grid_policy_without_mutating_inputs(self):
        original = native.PaddleOCRVLMultiModalProcessor._get_mm_fields_config(
            None, None, {}
        )
        result = patch.cpu_grid_fields(original)
        self.assertTrue(result["image_grid_thw"].field.keep_on_cpu)
        self.assertFalse(original["image_grid_thw"].field.keep_on_cpu)
        self.assertIs(result["pixel_values"], original["pixel_values"])
        self.assertFalse(result["pixel_values"].field.keep_on_cpu)
        self.assertEqual(result["image_grid_thw"].modality, "image")

    def test_real_final_factory_preserves_native_parent_and_model_methods(self):
        methods = {
            n: getattr(self.target, n)
            for n in [
                "forward",
                "compute_logits",
                "_process_image_input",
                "embed_multimodal",
            ]
        }
        out = patch.adapt_target_class(self.target)
        self.assertFalse(out["model_methods_changed"])
        self.assertIs(
            native.PaddleOCRVLForConditionalGeneration._processor_factory,
            self.native_factory,
        )
        self.assertIs(self.target._processor_factory.info, self.original_factory.info)
        self.assertIs(
            self.target._processor_factory.dummy_inputs,
            self.original_factory.dummy_inputs,
        )
        for n, m in methods.items():
            self.assertIs(getattr(self.target, n), m)
        processor = object.__new__(self.target._processor_factory.processor)
        fields = processor._get_mm_fields_config(None, {})
        self.assertTrue(fields["image_grid_thw"].field.keep_on_cpu)
        self.assertFalse(fields["pixel_values"].field.keep_on_cpu)
        first = self.target._processor_factory
        patch.adapt_target_class(self.target)
        self.assertIs(self.target._processor_factory, first)

    def test_real_collation_keeps_grid_cpu_and_pixels_on_requested_device(self):
        import torch

        original = native.PaddleOCRVLMultiModalProcessor._get_mm_fields_config(
            None, None, {}
        )
        fields = patch.cpu_grid_fields(original)
        grid = torch.tensor([[1, 4, 6], [1, 8, 10]], dtype=torch.int64)
        cfg = fields["image_grid_thw"]
        result = cfg.field.reduce_data(
            cfg.build_elems("image_grid_thw", grid), device="meta", pin_memory=True
        )
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.equal(result, grid))
        old = original["image_grid_thw"]
        self.assertEqual(
            old.field.reduce_data(
                old.build_elems("image_grid_thw", grid), device="meta"
            ).device.type,
            "meta",
        )
        pixels = fields["pixel_values"]
        self.assertEqual(
            pixels.field.reduce_data(
                pixels.build_elems("pixel_values", torch.ones(2, 3)), device="meta"
            ).device.type,
            "meta",
        )

    def test_unknown_processor_rejected_without_replacement(self):
        class Custom(native.PaddleOCRVLMultiModalProcessor):
            pass

        custom = replace(self.original_factory, processor=Custom)
        self.target._processor_factory = custom
        with self.assertRaisesRegex(RuntimeError, "customized"):
            patch.adapt_target_class(self.target)
        self.assertIs(self.target._processor_factory, custom)

    def test_unknown_info_factory_and_changed_image_method_rejected(self):
        custom = replace(self.original_factory, info=lambda x: x)
        self.target._processor_factory = custom
        with self.assertRaisesRegex(RuntimeError, "customized"):
            patch.adapt_target_class(self.target)
        self.assertIs(self.target._processor_factory, custom)
        self.target._processor_factory = self.original_factory
        before = self.target._process_image_input
        self.target._process_image_input = lambda self, x: x
        try:
            with self.assertRaisesRegex(RuntimeError, "differs"):
                patch.adapt_target_class(self.target)
            self.assertIs(self.target._processor_factory, self.original_factory)
        finally:
            self.target._process_image_input = before

    def test_unknown_model_and_fields_rejected(self):
        class Other(self.target):
            pass

        with self.assertRaisesRegex(RuntimeError, "Unknown"):
            patch.adapt_target_class(Other)
        fields = native.PaddleOCRVLMultiModalProcessor._get_mm_fields_config(
            None, None, {}
        )
        fields["custom"] = MultiModalFieldConfig.batched("image")
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            patch.cpu_grid_fields(fields)

    def test_lazy_hook_observes_final_registration_and_preserves_other_arch(self):
        target = self.target

        class Registry:
            def _try_load_model_cls(self, architecture):
                return target if architecture == patch.ARCH else str

        patch.install_registry_hook(Registry)
        self.assertIs(Registry()._try_load_model_cls("Other"), str)
        self.assertIs(self.target._processor_factory, self.original_factory)
        self.assertIs(Registry()._try_load_model_cls(patch.ARCH), self.target)
        self.assertIs(self.target._processor_factory.processor, patch._PROCESSOR_CLASS)
        first = Registry._try_load_model_cls
        patch.install_registry_hook(Registry)
        self.assertIs(first, Registry._try_load_model_cls)

    def test_disabled_plugin_leaves_registry_untouched(self):
        from vllm.model_executor.models.registry import _ModelRegistry

        before = _ModelRegistry._try_load_model_cls
        old = os.environ.pop(patch.ENV, None)
        try:
            patch.register()
        finally:
            if old is not None:
                os.environ[patch.ENV] = old
        self.assertIs(_ModelRegistry._try_load_model_cls, before)

    def test_combined_hooks_audit_actual_factory_without_mixing_record_types(self):
        from hps_batch import eagle3_adapter

        target = self.target

        class Registry:
            def _try_load_model_cls(self, architecture):
                return target

        with tempfile.TemporaryDirectory() as directory, mock_patch.dict(
            os.environ, {patch.AUDIT_ENV: directory}
        ), mock_patch.object(
            eagle3_adapter, "_ENABLE_MODEL_ADAPTER", False
        ), mock_patch.object(
            eagle3_adapter, "_ENABLE_BATCH_VISION", False
        ), mock_patch.object(
            eagle3_adapter, "_AUDIT_DIR", directory
        ):
            patch.install_registry_hook(Registry)
            eagle3_adapter.install_registry_hook(Registry)
            Registry()._try_load_model_cls(patch.ARCH)
            root = Path(directory)
            class_reports = list(root.glob("*.json"))
            self.assertEqual(len(class_reports), 1)
            identity = json.loads(class_reports[0].read_text())["identity"]
            self.assertTrue(identity["cpu_grid_enabled"])
            self.assertFalse(identity["adapter_installed"])
            processor_reports = list((root / "cpu_grid").glob("*.json"))
            self.assertEqual(len(processor_reports), 1)
            self.assertTrue(
                json.loads(processor_reports[0].read_text())[
                    "image_grid_thw_keep_on_cpu"
                ]
            )


if __name__ == "__main__":
    unittest.main()
