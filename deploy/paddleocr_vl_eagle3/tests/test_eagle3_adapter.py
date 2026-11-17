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

"""CPU contract checks; no model weights, GPU execution, or service is needed."""

import copy
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hps_batch import eagle3_adapter as adapter


class FakeDecoder:
    def __init__(self):
        self.embed_tokens = object()
        self.aux_layers = ()

    def _set_aux_hidden_state_layers(self, layers):
        self.aux_layers = layers


def target_class(*, direct=False):
    class Target:
        def __init__(self, *, vllm_config, prefix=""):
            self.config = vllm_config.model_config.hf_config
            self.prefix = prefix
            if direct:
                self.model = FakeDecoder()
                self.language_model = self.model
                self.lm_head = object()
            else:
                self.language_model = NS(model=FakeDecoder(), lm_head=object())

        def get_language_model(self):
            # Both observed registrations: native LM wrapper and legacy bare LM.
            return self.language_model

        def forward(self, token_rows):
            return token_rows

        def compute_logits(self, hidden):
            return hidden

        def encode_image(self, images):
            return images

        def get_multimodal_embeddings(self, images):
            return self.encode_image(images)

    return Target


def model_config(**kwargs):
    return NS(model_config=NS(hf_config=NS(image_token_id=100295, **kwargs)))


class AdapterTests(unittest.TestCase):
    def test_aux_contract_preserves_forward_and_vision_for_both_layouts(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                target = target_class(direct=direct)
                original_methods = {
                    name: getattr(target, name)
                    for name in (
                        "forward",
                        "compute_logits",
                        "encode_image",
                        "get_multimodal_embeddings",
                    )
                }
                adapter.adapt_target_class(target)
                configured = model_config(model_type="paddleocr_vl", dtype="unchanged")
                model = target(vllm_config=configured, prefix="target")
                self.assertTrue(model.supports_eagle3)
                self.assertEqual(model.prefix, "target")
                self.assertEqual(model.config.image_token_index, 100295)
                self.assertEqual(model.config.dtype, "unchanged")
                self.assertEqual(model.config.model_type, "paddleocr_vl")
                self.assertEqual(
                    model.get_eagle3_default_aux_hidden_state_layers(), (4, 10, 16)
                )
                model.set_aux_hidden_state_layers([4, 10, 16])
                language = model.get_language_model()
                self.assertIs(language, model if direct else model.language_model)
                self.assertEqual(language.model.aux_layers, (4, 10, 16))
                for name, original in original_methods.items():
                    self.assertIs(getattr(target, name), original)
                for batch_size in (1, 8, 32, 200):
                    rows = [object() for _ in range(batch_size)]
                    self.assertIs(model.forward(rows), rows)
                    self.assertIs(model.get_multimodal_embeddings(rows), rows)
                with self.assertRaisesRegex(ValueError, "auxiliary layers"):
                    model.set_aux_hidden_state_layers([3, 9, 15])

    def test_adaptation_is_idempotent_and_preserves_valid_language_override(self):
        target = target_class()
        language = NS(model=FakeDecoder(), lm_head=object())
        target.get_language_model = lambda self: language
        adapter.adapt_target_class(target)
        init = target.__init__
        adapter.adapt_target_class(target)
        self.assertIs(init, target.__init__)
        self.assertIs(target(vllm_config=model_config()).get_language_model(), language)

    def test_image_alias_fails_on_conflict_and_does_not_rewrite_model_type(self):
        target = adapter.adapt_target_class(target_class())
        with self.assertRaisesRegex(ValueError, "differs"):
            target(vllm_config=model_config(image_token_index=10))
        config = model_config()
        config.model_config.hf_config.image_token_id = 12
        with self.assertRaisesRegex(ValueError, "Image token ID"):
            target(vllm_config=config)

    def test_legacy_b1_adapter_cannot_be_combined(self):
        target = target_class()
        target._ocr_native_dflash_adapter = True
        with self.assertRaisesRegex(RuntimeError, "B1"):
            adapter.adapt_target_class(target)

    def test_lazy_registry_hook_respects_later_plugin_registration(self):
        class Registry:
            def __init__(self):
                self.models = {}

            def _try_load_model_cls(self, architecture):
                return self.models.get(architecture)

        initial = target_class()
        replacement = target_class(direct=True)
        unrelated = target_class()
        registry = Registry()
        registry.models[adapter.TARGET_ARCHITECTURE] = initial
        registry.models["AnotherArchitecture"] = unrelated
        adapter.install_registry_hook(Registry)
        hooked = Registry._try_load_model_cls
        adapter.install_registry_hook(Registry)
        self.assertIs(hooked, Registry._try_load_model_cls)
        # A plugin registered after ours still wins; no model entry is replaced.
        registry.models[adapter.TARGET_ARCHITECTURE] = replacement
        with patch.object(adapter, "_ENABLE_MODEL_ADAPTER", True):
            with patch.object(adapter, "_AUDIT_DIR", None):
                self.assertIs(
                    registry._try_load_model_cls(adapter.TARGET_ARCHITECTURE),
                    replacement,
                )
                self.assertIs(
                    registry._try_load_model_cls("AnotherArchitecture"), unrelated
                )
                self.assertIsNone(registry._try_load_model_cls("Missing"))
        self.assertTrue(replacement.supports_eagle3)
        self.assertFalse(hasattr(initial, "_paddlex_hps_eagle3_adapter"))
        self.assertFalse(hasattr(unrelated, "_paddlex_hps_eagle3_adapter"))

    def test_baseline_audit_does_not_modify_selected_class(self):
        target = target_class()

        class Registry:
            def _try_load_model_cls(self, architecture):
                return target

        original = dict(target.__dict__)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(adapter, "_ENABLE_MODEL_ADAPTER", False):
                with patch.object(adapter, "_AUDIT_DIR", directory):
                    adapter.install_registry_hook(Registry)
                    Registry()._try_load_model_cls(adapter.TARGET_ARCHITECTURE)
            reports = list(Path(directory).glob("*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertFalse(report["eagle3_enabled"])
            self.assertFalse(report["identity"]["adapter_installed"])
            self.assertEqual(report["identity"], adapter.collect_class_identity(target))
            self.assertEqual(set(Path(directory).iterdir()), set(reports))
        self.assertEqual(dict(target.__dict__), original)

    def test_audit_preserves_math_hashes_and_detects_wraps_wrapper(self):
        target = target_class()
        baseline = adapter.collect_class_identity(target)
        adapter.adapt_target_class(target)
        enabled = adapter.collect_class_identity(target)
        for method in (
            "forward",
            "compute_logits",
            "encode_image",
            "get_multimodal_embeddings",
        ):
            self.assertEqual(baseline["methods"][method], enabled["methods"][method])
        original = target.forward

        @functools.wraps(original)
        def changed(self, rows):
            return original(self, list(rows))

        target.forward = changed
        changed_identity = adapter.collect_class_identity(target)
        self.assertNotEqual(
            enabled["methods"]["forward"]["code_sha256"],
            changed_identity["methods"]["forward"]["code_sha256"],
        )

    def test_register_disabled_is_inert_and_explicit_modes_are_required(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            adapter, "install"
        ) as install:
            adapter.register()
            install.assert_not_called()
        with patch.dict(os.environ, {adapter.ENABLED_ENV: "yes"}, clear=True):
            with self.assertRaisesRegex(ValueError, "0 or 1"):
                adapter.register()
        environment = {adapter.ENABLED_ENV: "0", adapter.AUDIT_DIR_ENV: "/tmp/audit"}
        with patch.dict(os.environ, environment, clear=True), patch.object(
            adapter, "install"
        ) as install:
            adapter.register()
            install.assert_called_once_with(
                enabled=False, audit_dir="/tmp/audit", batch_vision=False
            )

    def test_batch_vision_can_be_requested_independently_of_eagle3(self):
        with patch.dict(
            os.environ, {adapter.BATCH_VISION_ENV: "1"}, clear=True
        ), patch.object(adapter, "install") as install:
            adapter.register()
            install.assert_called_once_with(
                enabled=False, audit_dir=None, batch_vision=True
            )
        with patch.dict(os.environ, {adapter.BATCH_VISION_ENV: "yes"}, clear=True):
            with self.assertRaisesRegex(ValueError, "0 or 1"):
                adapter.register()

    def test_cpu_grid_uses_existing_entrypoint_before_model_auditing(self):
        from hps_batch import cpu_grid

        calls = []
        with patch.dict(
            os.environ,
            {cpu_grid.ENV: "1", adapter.AUDIT_DIR_ENV: "/tmp/audit"},
            clear=True,
        ), patch.object(
            cpu_grid, "register", side_effect=lambda: calls.append("cpu_grid")
        ), patch.object(
            adapter, "install", side_effect=lambda **kw: calls.append("model")
        ):
            adapter.register()
        self.assertEqual(calls, ["cpu_grid", "model"])
        with patch.dict(
            os.environ, {cpu_grid.ENV: "1", adapter.BATCH_VISION_ENV: "1"}, clear=True
        ), patch.object(cpu_grid, "register") as install:
            with self.assertRaisesRegex(ValueError, "not been validated"):
                adapter.register()
            install.assert_not_called()


def has_native_vllm019():
    try:
        return importlib.metadata.version("vllm") == "0.19.0"
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(has_native_vllm019(), "Native config tests require vLLM 0.19.0")
class NativeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hps_batch import eagle3_config

        cls.config = eagle3_config

    def make_spec(self):
        hf = NS(
            architectures=[adapter.TARGET_ARCHITECTURE],
            model_type="paddleocr_vl",
            hidden_size=1024,
            num_hidden_layers=18,
            vocab_size=103424,
            image_token_id=100295,
            max_position_embeddings=131072,
        )
        draft = NS(
            architectures=["Eagle3LlamaForCausalLM"],
            hidden_size=1024,
            num_hidden_layers=1,
            vocab_size=103424,
            draft_vocab_size=40000,
            has_own_lm_head=True,
            has_own_embed_tokens=False,
            eagle_aux_hidden_state_layer_ids=[4, 10, 16],
            max_position_embeddings=4096,
        )
        draft_model = NS(
            hf_config=draft,
            hf_text_config=copy.deepcopy(draft),
            max_model_len=4096,
            verify_with_parallel_config=Mock(),
        )
        return NS(
            method="eagle3",
            target_model_config=NS(
                hf_config=hf, hf_text_config=hf, max_model_len=16384
            ),
            target_parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
            draft_model_config=draft_model,
            draft_parallel_config=object(),
            num_speculative_tokens=4,
            parallel_drafting=False,
            max_model_len=None,
            tensor_parallel_size=None,
            verify_equal_vocab_size_if_draft_model=Mock(),
        )

    def test_context_extension_does_not_change_target_config(self):
        for context in (16384, 32768):
            spec = self.make_spec()
            spec.target_model_config.max_model_len = context
            original = copy.deepcopy(spec.target_model_config)
            for k in (1, 2, 4, 8):
                spec.num_speculative_tokens = k
                self.config.validate_profile(spec)
            self.config.extend_draft_context(spec)
            self.assertEqual(spec.target_model_config, original)
            self.assertEqual(spec.draft_model_config.max_model_len, context)
            self.assertEqual(
                spec.draft_model_config.hf_config.max_position_embeddings, context
            )
            self.assertEqual(
                spec.draft_model_config.hf_text_config.max_position_embeddings, context
            )

    def test_incompatible_shapes_and_parallelism_are_rejected(self):
        for path, value in (
            ("target_model_config.hf_config.vocab_size", 32000),
            ("target_model_config.hf_config.num_hidden_layers", 24),
            ("draft_model_config.hf_config.draft_vocab_size", 32000),
            (
                "draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids",
                [3, 9, 15],
            ),
            ("target_parallel_config.tensor_parallel_size", 2),
            ("target_parallel_config.pipeline_parallel_size", 2),
            ("num_speculative_tokens", 0),
            ("parallel_drafting", True),
        ):
            with self.subTest(path=path):
                spec = self.make_spec()
                node = spec
                fields = path.split(".")
                for field in fields[:-1]:
                    node = getattr(node, field)
                setattr(node, fields[-1], value)
                with self.assertRaises(ValueError):
                    self.config.validate_profile(spec)

    def test_only_target_allowlist_is_extended_other_native_checks_still_run(self):
        spec = self.make_spec()
        target = spec.target_model_config
        self.assertIs(self.config.verify_args_with_paddle(spec), spec)
        self.assertIs(spec.target_model_config, target)
        self.assertEqual(target.hf_config.model_type, "paddleocr_vl")
        spec.draft_model_config.verify_with_parallel_config.assert_called_once_with(
            spec.draft_parallel_config
        )
        spec.tensor_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "not a valid argument"):
            self.config.verify_args_with_paddle(spec)

    def test_other_models_keep_upstream_validator(self):
        spec = self.make_spec()
        spec.target_model_config.hf_config.model_type = "another_unsupported_model"
        with self.assertRaisesRegex(ValueError, "only supported"):
            self.config.verify_args_with_paddle(spec)

    def test_factory_preserves_caller_options_and_other_methods(self):
        class EngineArgs:
            def create_speculative_config(
                self, target_model_config, target_parallel_config
            ):
                return "original"

        spec = self.make_spec()
        args = EngineArgs()
        args.speculative_config = {
            "method": "eagle3",
            "model": "draft",
            "num_speculative_tokens": 2,
        }
        original = dict(args.speculative_config)
        self.config.install_config_factory(EngineArgs)
        with patch.object(
            self.config, "PaddleOCRSpeculativeConfig", return_value="adapted"
        ) as constructor:
            self.assertEqual(
                args.create_speculative_config(
                    spec.target_model_config, spec.target_parallel_config
                ),
                "adapted",
            )
            constructor.assert_called_once_with(
                **original,
                target_model_config=spec.target_model_config,
                target_parallel_config=spec.target_parallel_config
            )
        self.assertEqual(args.speculative_config, original)
        args.speculative_config["method"] = "ngram"
        self.assertEqual(
            args.create_speculative_config(
                spec.target_model_config, spec.target_parallel_config
            ),
            "original",
        )


if __name__ == "__main__":
    unittest.main()
