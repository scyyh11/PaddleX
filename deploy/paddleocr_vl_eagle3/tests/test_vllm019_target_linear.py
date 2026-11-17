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

"""CPU contracts for target-only Q1-equivalent BF16 projection dispatch."""

from math import prod
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_target_linear as adapter


class NativeLinear:
    pass


class NativeEmbedding:
    pass


class NativeHead:
    pass


class WeightMetadata:
    next_pointer = 0x1000000

    def __init__(self, shape):
        self.shape, self.ndim = tuple(shape), len(shape)
        self.dtype, self.device = torch.bfloat16, torch.device("cuda:0")
        self.is_cuda = True
        self.pointer = WeightMetadata.next_pointer
        WeightMetadata.next_pointer += 0x10000000

    def numel(self):
        return prod(self.shape)

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return True

    def reshape(self, *shape):
        value = WeightMetadata(shape)
        value.pointer = self.pointer
        return value

    def contiguous(self):
        return self


def model_fixture():
    def projection(kind):
        return NS(
            weight=WeightMetadata(adapter.SHAPES[kind]),
            quant_method=NativeLinear(),
            bias=None,
        )

    layers = [
        NS(
            self_attn=NS(qkv_proj=projection("qkv"), o_proj=projection("o")),
            mlp=NS(gate_up_proj=projection("gateup"), down_proj=projection("down")),
        )
        for _ in range(18)
    ]
    head = NativeHead()
    head.weight, head.quant_method, head.bias = (
        WeightMetadata(adapter.SHAPES["head"]),
        NativeEmbedding(),
        None,
    )
    hf = NS(
        model_type="paddleocr_vl",
        architectures=["PaddleOCRVLForConditionalGeneration"],
        hidden_size=1024,
        num_hidden_layers=18,
        vocab_size=103424,
        intermediate_size=3072,
        num_attention_heads=16,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    target = NS(
        language_model=NS(
            model=NS(layers=layers, embed_tokens=projection("head")),
            lm_head=head,
            config=hf,
        ),
        visual=NS(decoy=projection("qkv")),
        config=hf,
    )
    config = NS(
        model_config=NS(hf_config=hf, dtype=torch.bfloat16),
        quant_config=None,
        lora_config=None,
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )
    return target, config


class TargetLinearDispatchTests(unittest.TestCase):
    def test_unmarked_layers_keep_original_arguments_and_result(self):
        original = Mock(return_value=object())
        wrapper = adapter._make_apply_wrapper(original)
        method, layer, inputs, bias = object(), NS(), object(), object()
        with patch.object(adapter, "target_linear") as candidate:
            self.assertIs(wrapper(method, layer, inputs, bias), original.return_value)
        candidate.assert_not_called()
        original.assert_called_once_with(method, layer, inputs, bias)

    def test_marked_head_without_prefix_uses_target_dispatch(self):
        original = Mock()
        wrapper = adapter._make_apply_wrapper(original)
        layer = NS(
            weight=WeightMetadata(adapter.SHAPES["head"]),
            _vllm019_target_linear_kind="head",
        )
        inputs, method = object(), object()
        with patch.object(adapter, "target_linear", return_value=object()) as candidate:
            self.assertIs(wrapper(method, layer, inputs), candidate.return_value)
        candidate.assert_called_once_with(inputs, layer.weight, "head")
        original.assert_not_called()

    def test_marked_bias_never_falls_back_to_native_q5(self):
        original = Mock()
        wrapper = adapter._make_apply_wrapper(original)
        layer = NS(weight=object(), _vllm019_target_linear_kind="qkv")
        with patch.object(adapter, "target_linear") as candidate:
            with self.assertRaises((RuntimeError, ValueError)):
                wrapper(object(), layer, object(), object())
        candidate.assert_not_called()
        original.assert_not_called()

    def test_q1_and_large_query_keep_native_linear(self):
        weight = torch.empty((3, 4), dtype=torch.bfloat16)
        for shape in ((1, 4), (6, 4), (2, 3, 4)):
            inputs = torch.empty(shape, dtype=torch.bfloat16)
            with patch.object(
                adapter.torch.nn.functional, "linear", return_value=object()
            ) as native:
                self.assertIs(
                    adapter._execute(inputs, weight, "o"), native.return_value
                )
            native.assert_called_once_with(inputs, weight)

    def test_unprepared_small_queries_fail_without_native_batch_fallback(self):
        weight = torch.empty((3, 4), dtype=torch.bfloat16)
        with patch.dict(adapter.PLANS, {}, clear=True):
            for query in range(2, 6):
                with patch.object(adapter.torch.nn.functional, "linear") as native:
                    with self.assertRaises((RuntimeError, ValueError)):
                        adapter._execute(
                            torch.empty((query, 4), dtype=torch.bfloat16), weight, "o"
                        )
                native.assert_not_called()

    def test_prepared_weight_storage_accepts_customop_alias_and_preserves_leading_dims(
        self,
    ):
        owner = WeightMetadata(adapter.SHAPES["o"])
        alias = WeightMetadata(owner.shape)
        alias.pointer = owner.pointer
        self.assertIsNot(alias, owner)
        plan = Mock(
            side_effect=lambda inputs: WeightMetadata((inputs.shape[0], owner.shape[0]))
        )
        plan.weight, plan.plans = owner, {query: object() for query in range(2, 6)}
        with patch.dict(adapter.PLANS, {(owner.data_ptr(), "o"): plan}, clear=True):
            for query in range(2, 6):
                output = adapter._execute(
                    WeightMetadata((1, query, owner.shape[1])), alias, "o"
                )
                self.assertEqual(output.shape, (1, query, owner.shape[0]))
                self.assertEqual(plan.call_args.args[0].shape, (query, owner.shape[1]))
            self.assertIs(adapter.PLANS[(owner.data_ptr(), "o")].weight, owner)

    def test_fake_customop_preserves_shape_dtype_without_cuda(self):
        inputs = torch.empty((1, 5, 8), dtype=torch.bfloat16, device="meta")
        weight = torch.empty((13, 8), dtype=torch.bfloat16, device="meta")
        result = adapter._target_linear_fake(inputs, weight, "o")
        self.assertEqual(result.shape, (1, 5, 13))
        self.assertEqual(result.dtype, inputs.dtype)
        self.assertEqual(result.device, inputs.device)


class TargetLinearLifecycleTests(unittest.TestCase):
    def test_real_decoder_objects_and_prefixless_head_are_marked(self):
        target, config = model_fixture()
        adapter._mark_target(target, config, NativeLinear, NativeEmbedding, NativeHead)
        marked = target._vllm019_target_linear_layers
        self.assertEqual(len(marked), 73)
        self.assertEqual(len({id(layer) for _, layer in marked}), 73)
        self.assertIs(marked[-1][1], target.language_model.lm_head)
        self.assertEqual(marked[-1][0], "head")
        self.assertFalse(hasattr(target.language_model.lm_head, "prefix"))
        for kind, layer in marked:
            self.assertEqual(layer._vllm019_target_linear_kind, kind)
        for decoy in (target.visual.decoy, target.language_model.model.embed_tokens):
            self.assertFalse(hasattr(decoy, "_vllm019_target_linear_kind"))

    def test_draft_vocabulary_head_is_rejected(self):
        target, config = model_fixture()
        target.language_model.lm_head.weight = WeightMetadata((40000, 1024))
        with self.assertRaises((RuntimeError, ValueError)):
            adapter._mark_target(
                target, config, NativeLinear, NativeEmbedding, NativeHead
            )

    def test_load_weights_prepares_after_loading_and_keeps_native_result(self):
        events, result = [], {"loaded_parameter"}
        target, weights = NS(), object()
        original = Mock(side_effect=lambda *args: (events.append("load"), result)[1])
        with patch.object(
            adapter,
            "_prepare_target",
            side_effect=lambda *args: events.append("prepare"),
        ) as prepare:
            self.assertIs(adapter._make_load_weights(original)(target, weights), result)
        self.assertEqual(events, ["load", "prepare"])
        original.assert_called_once_with(target, weights)
        prepare.assert_called_once_with(target)

    def test_failed_load_does_not_create_plans(self):
        error = RuntimeError("load failed")
        original = Mock(side_effect=error)
        with patch.object(adapter, "_prepare_target") as prepare:
            with self.assertRaises(RuntimeError) as caught:
                adapter._make_load_weights(original)(NS(), object())
        self.assertIs(caught.exception, error)
        prepare.assert_not_called()

    def test_prepared_target_cannot_reload_or_mutate_weights(self):
        original = Mock()
        with patch.object(adapter, "_prepare_target") as prepare:
            with self.assertRaises(RuntimeError):
                adapter._make_load_weights(original)(
                    NS(_vllm019_target_linear_prepared=True), object()
                )
        original.assert_not_called()
        prepare.assert_not_called()

    def test_target_init_marks_after_native_construction(self):
        events = []
        original = Mock(side_effect=lambda *a, **k: events.append("init"))
        target, config = object(), object()
        wrapper = adapter._make_target_init(
            original, NativeLinear, NativeEmbedding, NativeHead
        )
        with patch.object(
            adapter, "_mark_target", side_effect=lambda *a: events.append("mark")
        ) as mark:
            wrapper(target, vllm_config=config, prefix="outer")
        self.assertEqual(events, ["init", "mark"])
        original.assert_called_once_with(target, vllm_config=config, prefix="outer")
        mark.assert_called_once_with(
            target, config, NativeLinear, NativeEmbedding, NativeHead
        )

    def test_unsupported_target_contract_is_rejected_before_marking_any_layer(self):
        for defect in (
            "quantized",
            "parallel",
            "dtype",
            "custom_method",
            "shared_module",
        ):
            target, config = model_fixture()
            first = target.language_model.model.layers[0].self_attn.qkv_proj
            if defect == "quantized":
                config.quant_config = object()
            elif defect == "parallel":
                config.parallel_config.tensor_parallel_size = 2
            elif defect == "dtype":
                config.model_config.dtype = torch.float16
            elif defect == "custom_method":
                target.language_model.model.layers[-1].mlp.down_proj.quant_method = (
                    object()
                )
            else:
                target.language_model.model.layers[-1].self_attn.qkv_proj = first
            with self.subTest(defect=defect), self.assertRaises(RuntimeError):
                adapter._mark_target(
                    target, config, NativeLinear, NativeEmbedding, NativeHead
                )
            self.assertFalse(hasattr(first, "_vllm019_target_linear_kind"))


class TargetLinearPreparationTests(unittest.TestCase):
    def setUp(self):
        self.target, config = model_fixture()
        adapter._mark_target(
            self.target, config, NativeLinear, NativeEmbedding, NativeHead
        )
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(adapter.PLANS, {}, clear=True))
        self.runtime = self.stack.enter_context(patch.object(adapter, "_check_runtime"))
        self.created = []

    def make_plan(self, weight, recipe, *, workspace_bytes):
        plan = NS(
            weight=weight,
            recipe=recipe,
            library_version=120804,
            plans={},
            workspace=NS(numel=lambda: workspace_bytes),
            close=Mock(),
        )
        plan.prepare = Mock(
            side_effect=lambda query: plan.plans.setdefault(query, object())
        )
        self.created.append(plan)
        return plan

    def test_all_73_owned_plans_are_prepared_before_graph_capture(self):
        with patch.object(
            adapter, "LockedLt", side_effect=self.make_plan
        ) as constructor:
            adapter._prepare_target(self.target)
        self.assertTrue(self.target._vllm019_target_linear_prepared)
        self.assertEqual(len(adapter.PLANS), 73)
        self.assertEqual(sum(len(plan.plans) for plan in adapter.PLANS.values()), 292)
        for (kind, layer), call in zip(
            self.target._vllm019_target_linear_layers, constructor.call_args_list
        ):
            self.assertIs(call.args[0], layer.weight)
            self.assertEqual(call.args[1], adapter.RECIPES[kind])
            self.assertEqual(
                call.kwargs["workspace_bytes"], 4096 if kind == "head" else 0
            )
            plan = adapter.PLANS[(layer.weight.data_ptr(), kind)]
            self.assertIs(plan.weight, layer.weight)
            self.assertEqual(
                [entry.args[0] for entry in plan.prepare.call_args_list], [2, 3, 4, 5]
            )
            plan.close.assert_not_called()
        self.runtime.assert_called_once_with(self.created[0].weight.device)

    def test_failed_preparation_closes_pending_plans_and_does_not_publish_cache(self):
        error = RuntimeError("unsupported recipe")

        def failing_plan(*args, **kwargs):
            plan = self.make_plan(*args, **kwargs)
            if len(self.created) == 2:
                plan.prepare.side_effect = error
            return plan

        with patch.object(adapter, "LockedLt", side_effect=failing_plan):
            with self.assertRaises(RuntimeError) as caught:
                adapter._prepare_target(self.target)
        self.assertIs(caught.exception, error)
        self.assertEqual(adapter.PLANS, {})
        self.assertFalse(self.target._vllm019_target_linear_prepared)
        for plan in self.created:
            plan.close.assert_called_once()

    def test_changed_loaded_weight_and_library_version_fail_closed(self):
        first = self.target._vllm019_target_linear_layers[0][1]
        first.weight.dtype = torch.float16
        with patch.object(adapter, "LockedLt") as constructor:
            with self.assertRaises(RuntimeError):
                adapter._prepare_target(self.target)
        constructor.assert_not_called()
        first.weight.dtype = torch.bfloat16

        def wrong_version(*args, **kwargs):
            plan = self.make_plan(*args, **kwargs)
            plan.library_version = 120803
            return plan

        with patch.object(adapter, "LockedLt", side_effect=wrong_version):
            with self.assertRaises(RuntimeError):
                adapter._prepare_target(self.target)
        self.assertEqual(adapter.PLANS, {})
        self.created[0].close.assert_called_once()


class TargetLinearRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.versions = {"vllm": "0.19.0", "nvidia-cublas-cu12": "12.8.4.1"}
        self.envs = NS(VLLM_BATCH_INVARIANT=False)
        self.flags = NS(
            allow_bf16_reduced_precision_reduction=True,
            allow_bf16_reduced_precision_reduction_split_k=True,
        )
        self.properties = NS(
            major=8,
            minor=0,
            multi_processor_count=108,
            name="NVIDIA A100 80GB PCIe",
            total_memory=80 * 1024**3,
        )
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules, {"vllm": NS(envs=self.envs)}))
        self.stack.enter_context(
            patch.object(
                adapter.importlib.metadata,
                "version",
                side_effect=self.versions.__getitem__,
            )
        )
        self.stack.enter_context(
            patch.object(adapter.torch, "__version__", "2.10.0+cu128")
        )
        self.stack.enter_context(patch.object(adapter.torch.version, "cuda", "12.8"))
        self.stack.enter_context(
            patch.object(adapter.torch.backends.cuda, "matmul", self.flags)
        )
        self.backend = self.stack.enter_context(
            patch.object(
                adapter.torch.backends.cuda,
                "preferred_blas_library",
                return_value="_BlasBackend.Cublas",
            )
        )
        self.stack.enter_context(
            patch.object(
                adapter.torch.cuda,
                "get_device_properties",
                return_value=self.properties,
            )
        )

    def test_exact_runtime_and_device_are_admitted_without_mutating_flags(self):
        adapter._check_runtime()
        adapter._check_runtime(torch.device("cuda:0"))
        self.assertEqual(self.backend.call_args_list[0].args, ())
        self.assertIs(self.flags.allow_bf16_reduced_precision_reduction, True)
        self.assertIs(self.flags.allow_bf16_reduced_precision_reduction_split_k, True)

    def test_other_versions_and_blas_precision_modes_are_rejected(self):
        for package, wrong in (("vllm", "0.20.0"), ("nvidia-cublas-cu12", "12.8.3.0")):
            prior = self.versions[package]
            self.versions[package] = wrong
            with self.assertRaises(RuntimeError):
                adapter._check_runtime()
            self.versions[package] = prior
        for flags in (
            "allow_bf16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction_split_k",
        ):
            setattr(self.flags, flags, False)
            with self.assertRaises(RuntimeError):
                adapter._check_runtime()
            setattr(self.flags, flags, True)
        self.backend.return_value = "_BlasBackend.Cublaslt"
        with self.assertRaises(RuntimeError):
            adapter._check_runtime()
        self.backend.return_value = "_BlasBackend.Cublas"
        self.envs.VLLM_BATCH_INVARIANT = True
        with self.assertRaises(RuntimeError):
            adapter._check_runtime()

    def test_unmeasured_hardware_is_rejected(self):
        for field, wrong in (
            ("major", 9),
            ("minor", 6),
            ("multi_processor_count", 107),
            ("name", "NVIDIA A30"),
            ("total_memory", 40 * 1024**3),
        ):
            prior = getattr(self.properties, field)
            setattr(self.properties, field, wrong)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                adapter._check_runtime(torch.device("cuda:0"))
            setattr(self.properties, field, prior)


if __name__ == "__main__":
    unittest.main()
