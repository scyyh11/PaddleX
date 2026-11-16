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

"""CPU contract checks for the old proposer's distinct initialization lifecycle."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class GraphWrapper:
    def __init__(self, runnable):
        self.runnable = runnable

    def unwrap(self):
        return self.runnable


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Vllm019WorkerTests(unittest.TestCase):
    def setUp(self):
        self.plans = {42: {"private_vocab_size": 40000, "target_vocab_size": 103424}}
        self.ar1_installs = []
        modules = {
            "vllm019_model": SimpleNamespace(install=lambda: None),
            "vllm019_ar1_attention": SimpleNamespace(
                install=lambda: self.ar1_installs.append(True),
                COUNTS={"row_calls": 36, "fallback_calls": 18},
            ),
            "vllm.compilation.cuda_graph": SimpleNamespace(
                CUDAGraphWrapper=GraphWrapper
            ),
            "round3_eagle_fused_int8_top1": SimpleNamespace(
                install=lambda: None,
                metadata=lambda: {"plans": len(self.plans), "top1_dispatches": {}},
            ),
            "round3_eagle_fused_int8_head": SimpleNamespace(PLANS=self.plans),
        }
        with patch.dict(sys.modules, modules):
            self.base = load("hps_vllm019_worker")
            with patch.dict(sys.modules, {"hps_vllm019_worker": self.base}):
                self.int8 = load("hps_vllm019_int8_worker")
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="eagle3", num_speculative_tokens=4
            ),
            model_config=SimpleNamespace(max_model_len=12288),
            scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=8),
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
                cudagraph_capture_sizes=[5],
            ),
        )
        draft = type("Eagle3LlamaForCausalLM", (), {})()
        draft.config = SimpleNamespace(
            max_position_embeddings=12288, draft_vocab_size=40000, vocab_size=103424
        )
        draft.lm_head = SimpleNamespace(
            weight=SimpleNamespace(dtype="torch.bfloat16", data_ptr=lambda: 42)
        )
        draft.draft_id_to_target_id = object()
        target = type("PaddleOCRVLForConditionalGeneration", (), {})()
        target._ocr_native_aux_layers = (4, 10, 16)
        proposer = type("EagleProposer", (), {})()
        proposer.vllm_config = config
        proposer.model = draft
        proposer.draft_model_config = SimpleNamespace(model="/models/draft")
        proposer.use_local_argmax_reduction = True
        proposer.cudagraph_dispatcher = SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE")
        )
        self.runner = SimpleNamespace(drafter=proposer, model=target)

    def worker(self, cls):
        worker = cls()
        worker.model_runner = self.runner
        return worker

    def test_prepared_int8_before_first_sampling_is_valid(self):
        result = self.worker(self.int8.Vllm019Int8Worker).hps_eagle_metadata()
        self.assertTrue(result["draft_head_prepared"])
        self.assertFalse(result["draft_top1_in_forward_graph"])
        self.assertEqual(result["draft_head_startup"]["top1_dispatches"], {})
        self.assertFalse(result["cuda13_locked_kernels"])

    def test_bf16_and_int8_share_the_target_attention_compatibility_path(self):
        self.assertEqual(self.ar1_installs, [True])
        for cls in (self.base.Vllm019EagleWorker, self.int8.Vllm019Int8Worker):
            result = self.worker(cls).hps_eagle_metadata()
            self.assertEqual(
                result["runtime_profile"], "vllm-0.19.0-ar1-attention-target"
            )
            self.assertEqual(result["ar1_attention_startup"]["row_calls"], 36)
            self.assertIn("native FA2 AR1 row batch", result["target_attention"])

    def test_missing_or_wrong_head_plan_is_rejected(self):
        worker = self.worker(self.int8.Vllm019Int8Worker)
        self.plans[42]["target_vocab_size"] = 1
        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            worker.hps_eagle_metadata()
        self.plans.clear()
        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            worker.hps_eagle_metadata()

    def test_mixed_mode_requires_actual_draft_graph_dispatch(self):
        proposer = self.runner.drafter
        proposer.vllm_config.compilation_config.cudagraph_mode.name = (
            "FULL_AND_PIECEWISE"
        )
        worker = self.worker(self.base.Vllm019EagleWorker)
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            worker.hps_eagle_metadata()
        proposer.cudagraph_dispatcher.cudagraph_mode.name = "PIECEWISE"
        self.assertEqual(
            worker.hps_eagle_metadata()["draft_cudagraph_mode"], "PIECEWISE"
        )

    def test_int8_requires_actual_local_argmax_path(self):
        self.runner.drafter.use_local_argmax_reduction = False
        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            self.worker(self.int8.Vllm019Int8Worker).hps_eagle_metadata()

    def test_unexpanded_draft_cannot_advertise_long_context_ready(self):
        self.runner.drafter.model.config.max_position_embeddings = 4096
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            self.worker(self.base.Vllm019EagleWorker).hps_eagle_metadata()

    def test_graph_wrapper_preserves_actual_target_identity(self):
        self.runner.model = GraphWrapper(self.runner.model)
        result = self.worker(self.base.Vllm019EagleWorker).hps_eagle_metadata()
        self.assertEqual(result["target"], "PaddleOCRVLForConditionalGeneration")
        wrong = type("OtherModel", (), {"_ocr_native_aux_layers": (4, 10, 16)})()
        self.runner.model = GraphWrapper(wrong)
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            self.worker(self.base.Vllm019EagleWorker).hps_eagle_metadata()


if __name__ == "__main__":
    unittest.main()
