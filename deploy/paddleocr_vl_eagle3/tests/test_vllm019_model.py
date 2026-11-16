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

"""CPU contracts for the version-specific model/config adapter."""

import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_model as compat


def profile():
    hf = SimpleNamespace(
        model_type="paddleocr_vl",
        architectures=[compat.TARGET_ARCHITECTURE],
        hidden_size=1024,
        num_hidden_layers=18,
        vocab_size=103424,
        image_token_id=100295,
    )
    draft = SimpleNamespace(
        architectures=[compat.DRAFT_ARCHITECTURE],
        hidden_size=1024,
        num_hidden_layers=1,
        vocab_size=103424,
        draft_vocab_size=40000,
        has_own_lm_head=True,
        has_own_embed_tokens=False,
        eagle_aux_hidden_state_layer_ids=[4, 10, 16],
    )
    return SimpleNamespace(
        method="eagle3",
        target_model_config=SimpleNamespace(hf_config=hf, max_model_len=12288),
        target_parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        num_speculative_tokens=4,
        parallel_drafting=False,
        draft_model_config=SimpleNamespace(hf_config=draft),
    )


class Vllm019ModelTests(unittest.TestCase):
    def test_profile_rejects_incompatible_target_draft_and_parallelism(self):
        compat._validate_profile(profile())
        changes = (
            ("target_model_config.hf_config", "architectures", ["QwenForCausalLM"]),
            (
                "draft_model_config.hf_config",
                "eagle_aux_hidden_state_layer_ids",
                [3, 9, 15],
            ),
            ("draft_model_config.hf_config", "draft_vocab_size", 32768),
            ("target_parallel_config", "tensor_parallel_size", 2),
            ("target_model_config", "max_model_len", 16384),
            ("", "num_speculative_tokens", 5),
            ("", "parallel_drafting", True),
        )
        for path, name, value in changes:
            with self.subTest(path=path, name=name):
                item = profile()
                obj = item
                for part in path.split(".") if path else []:
                    obj = getattr(obj, part)
                setattr(obj, name, value)
                with self.assertRaises(ValueError):
                    compat._validate_profile(item)

    def test_factory_is_scoped_and_idempotent(self):
        class Args:
            def create_speculative_config(self, target, parallel):
                return "upstream"

        compat._install_config_factory(Args)
        first = Args.create_speculative_config
        compat._install_config_factory(Args)
        self.assertIs(first, Args.create_speculative_config)
        args = Args()
        data = profile()
        with patch.object(
            compat, "PaddleOCRSpeculativeConfig", return_value="paddle"
        ) as make:
            args.speculative_config = {"method": "eagle3"}
            self.assertEqual(
                args.create_speculative_config(
                    data.target_model_config, data.target_parallel_config
                ),
                "paddle",
            )
            make.assert_called_once()
            args.speculative_config = {"method": "draft_model"}
            self.assertEqual(
                args.create_speculative_config(
                    data.target_model_config, data.target_parallel_config
                ),
                "upstream",
            )
            args.speculative_config = {"method": "eagle3"}
            data.target_model_config.hf_config.model_type = "llama"
            self.assertEqual(
                args.create_speculative_config(
                    data.target_model_config, data.target_parallel_config
                ),
                "upstream",
            )
            self.assertEqual(make.call_count, 1)

    def test_image_alias_preserves_id_and_rejects_conflicts(self):
        class Target:
            def __init__(self, *, vllm_config, prefix=""):
                self.config = vllm_config.model_config.hf_config
                self.prefix = prefix

        compat._install_image_token_alias(Target)
        first = Target.__init__
        compat._install_image_token_alias(Target)
        self.assertIs(first, Target.__init__)
        hf = SimpleNamespace(image_token_id=100295, model_type="paddleocr_vl")
        config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf))
        model = Target(vllm_config=config, prefix="target")
        self.assertEqual(model.config.image_token_index, 100295)
        self.assertEqual(model.config.model_type, "paddleocr_vl")
        self.assertEqual(model.prefix, "target")
        hf.image_token_index = 1
        with self.assertRaises(ValueError):
            Target(vllm_config=config)

    def test_allowlist_exception_still_runs_upstream_validation(self):
        item = profile()
        target = item.target_model_config
        with patch.object(
            compat, "_ORIGINAL_VERIFY_ARGS", return_value=None
        ) as upstream:
            result = compat.PaddleOCRSpeculativeConfig._verify_args(item)
        self.assertIs(result, item)
        self.assertIs(item.target_model_config, target)
        self.assertEqual(target.hf_config.model_type, "paddleocr_vl")
        validation_view = upstream.call_args.args[0]
        self.assertIsNot(validation_view, item)
        self.assertIsNone(validation_view.target_model_config)
        self.assertIs(validation_view.draft_model_config, item.draft_model_config)
        with patch.object(
            compat,
            "_ORIGINAL_VERIFY_ARGS",
            side_effect=ValueError("invalid parallel config"),
        ):
            with self.assertRaisesRegex(ValueError, "invalid parallel"):
                compat.PaddleOCRSpeculativeConfig._verify_args(item)

    def test_native_auxiliary_contract_and_missing_states(self):
        import torch
        from vllm.model_executor.models.interfaces import supports_eagle3
        from vllm.model_executor.models.paddleocr_vl import (
            PaddleOCRVLForConditionalGeneration as Target,
        )
        import native_dflash_target

        native_dflash_target.install()
        hidden = torch.arange(4, dtype=torch.float32)
        auxiliary = [hidden + index for index in (4, 10, 16)]

        class Language(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNamespace(
                    _set_aux_hidden_state_layers=lambda ids: None
                )
                self.result = (hidden, auxiliary)

            def forward(self, *args):
                return self.result

        target = Target.__new__(Target)
        torch.nn.Module.__init__(target)
        target.language_model = Language()
        self.assertTrue(supports_eagle3(target))
        self.assertEqual(
            target.get_eagle3_default_aux_hidden_state_layers(), (4, 10, 16)
        )
        target.set_aux_hidden_state_layers((4, 10, 16))
        output = target(None, torch.zeros(1, dtype=torch.long))
        self.assertIs(output[0], hidden)
        self.assertIs(output[1], auxiliary)
        with self.assertRaises(ValueError):
            target.set_aux_hidden_state_layers((3, 9, 15))
        target.language_model.result = (hidden, [])
        with self.assertRaisesRegex(RuntimeError, "all three"):
            target(None, torch.zeros(1, dtype=torch.long))

    def test_adapter_rejects_other_versions_before_installation(self):
        with patch.object(importlib.metadata, "version", return_value="0.19.1"):
            with self.assertRaisesRegex(RuntimeError, "0.19.0"):
                compat.install()


if __name__ == "__main__":
    unittest.main()
