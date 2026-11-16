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

"""CPU contracts for the CUDA 12 cuBLASLt ABI and explicit BF16 recipes."""

import ctypes as C
from contextlib import ExitStack, nullcontext
from math import prod
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_locked_lt as adapter


class LockedLtAbiTests(unittest.TestCase):
    def test_algorithm_matches_cuda12_uint64_array_abi(self):
        self.assertEqual(C.sizeof(adapter.Algo), 64)
        self.assertEqual(C.alignment(adapter.Algo), 8)
        self.assertEqual(adapter.Algo.data.offset, 0)

    def test_heuristic_matches_cuda12_64bit_abi(self):
        self.assertEqual(C.sizeof(C.c_void_p), 8)
        self.assertEqual(C.sizeof(adapter.Heuristic), 96)
        self.assertEqual(C.alignment(adapter.Heuristic), 8)
        for name, offset in {
            "algo": 0,
            "workspaceSize": 64,
            "state": 72,
            "wavesCount": 76,
            "reserved": 80,
        }.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(adapter.Heuristic, name).offset, offset)


class LockedLtRecipeTests(unittest.TestCase):
    def test_algorithm_is_explicit_and_options_have_safe_defaults(self):
        recipe = {"algo_id": 7}
        normalized = adapter.normalize_recipe(recipe)
        self.assertEqual(recipe, {"algo_id": 7})
        self.assertEqual(normalized["algo_id"], 7)
        self.assertIs(normalized["row_batch"], False)
        self.assertIsNone(normalized["sm_count_target"])
        self.assertIsNot(normalized, recipe)
        with self.assertRaises((TypeError, ValueError)):
            adapter.normalize_recipe({})

    def test_valid_recipe_values_are_preserved(self):
        recipe = {
            "algo_id": 7,
            "tile": 15,
            "stages": 17,
            "split_k": 2,
            "reduction": 1,
            "swizzle": 1,
            "custom_option": 3,
            "inner_shape": 1,
            "cluster_shape": 0,
            "row_batch": True,
            "sm_count_target": 108,
        }
        actual = adapter.normalize_recipe(recipe)
        for key, value in recipe.items():
            with self.subTest(key=key):
                self.assertEqual(actual[key], value)

    def test_unknown_recipe_attributes_and_implicit_coercions_are_rejected(self):
        cases = [
            {"algo_id": 7, "typo_tile": 15},
            {"algo_id": True},
            {"algo_id": 7.0},
            {"algo_id": "7"},
            {"algo_id": -1},
            {"algo_id": 7, "row_batch": 1},
            {"algo_id": 7, "sm_count_target": True},
        ]
        cases.extend(
            {"algo_id": 7, field: value}
            for field in (
                "tile",
                "stages",
                "split_k",
                "reduction",
                "swizzle",
                "custom_option",
                "inner_shape",
                "cluster_shape",
            )
            for value in (True, -1, 1.0, "1")
        )
        for recipe in cases:
            with self.subTest(recipe=recipe):
                with self.assertRaises((TypeError, ValueError)):
                    adapter.normalize_recipe(recipe)

    def test_attribute_widths_are_checked_before_calling_ctypes(self):
        maximums = {
            "algo_id": 2**31 - 1,
            "split_k": 2**31 - 1,
            "sm_count_target": 2**31 - 1,
            "tile": 2**32 - 1,
            "stages": 2**32 - 1,
            "reduction": 2**32 - 1,
            "swizzle": 2**32 - 1,
            "custom_option": 2**32 - 1,
            "inner_shape": 2**16 - 1,
            "cluster_shape": 2**16 - 1,
        }
        for key, maximum in maximums.items():
            with self.subTest(key=key):
                with self.assertRaises((TypeError, ValueError)):
                    adapter.normalize_recipe({"algo_id": 7, key: maximum + 1})


class LockedLtLayoutTests(unittest.TestCase):
    def test_normal_batch_uses_transposed_weight_and_column_major_views(self):
        self.assertEqual(
            adapter._layout_specs(2048, 1024, 5, False),
            [
                (2048, 1024, 2048, 1, 0),
                (2048, 5, 2048, 1, 2048),
                (1024, 5, 1024, 1, 1024),
            ],
        )

    def test_row_batch_preserves_q1_geometry_and_element_strides(self):
        self.assertEqual(
            adapter._layout_specs(3072, 1024, 5, True),
            [
                (3072, 1024, 3072, 5, 0),
                (3072, 1, 3072, 5, 3072),
                (1024, 1, 1024, 5, 1024),
            ],
        )


class FakeTensor:
    """Tensor metadata/storage identity only; never allocates CUDA memory."""

    next_pointer = 0x1000000

    def __init__(self, shape, *, dtype=torch.bfloat16, device=None, pointer=None):
        self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        self.ndim, self.dtype = len(self.shape), dtype
        self.device = torch.device("cuda:0") if device is None else torch.device(device)
        self.is_cuda, self.contiguous = self.device.type == "cuda", True
        self.pointer = pointer if pointer is not None else FakeTensor.next_pointer
        FakeTensor.next_pointer += 0x1000000
        self.record_stream = Mock()

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return 1 if self.dtype == torch.uint8 else 2

    def is_contiguous(self):
        return self.contiguous


class FakeLibrary:
    def __init__(self):
        self.next_handle = 0x1000
        self.required_workspace, self.check_status, self.result_state = 64, 0, 0
        self.alignment_written, self.alignment = 4, 16
        self.layout_specs = {}
        self.cublasLtCreate = Mock(side_effect=self.create)
        self.cublasLtDestroy = Mock(return_value=0)
        self.cublasLtMatmulDescCreate = Mock(side_effect=self.create)
        self.cublasLtMatmulDescDestroy = Mock(return_value=0)
        self.cublasLtMatmulDescSetAttribute = Mock(return_value=0)
        self.cublasLtMatrixLayoutCreate = Mock(side_effect=self.layout)
        self.cublasLtMatrixLayoutDestroy = Mock(return_value=0)
        self.cublasLtMatrixLayoutSetAttribute = Mock(return_value=0)
        self.cublasLtMatmulAlgoInit = Mock(side_effect=self.algo_init)
        self.cublasLtMatmulAlgoConfigSetAttribute = Mock(return_value=0)
        self.cublasLtMatmulAlgoCapGetAttribute = Mock(side_effect=self.capability)
        self.cublasLtMatmulAlgoCheck = Mock(side_effect=self.check)
        self.cublasLtMatmul = Mock(return_value=0)
        self.cublasLtGetVersion = Mock(return_value=120804)

    def create(self, pointer, *args):
        C.cast(pointer, C.POINTER(C.c_void_p))[0] = self.next_handle
        self.next_handle += 0x100
        return 0

    def layout(self, pointer, dtype, rows, columns, leading_dimension):
        self.create(pointer)
        handle = C.cast(pointer, C.POINTER(C.c_void_p))[0]
        self.layout_specs[handle] = (dtype, rows, columns, leading_dimension)
        return 0

    def algo_init(self, *args):
        C.cast(args[-1], C.POINTER(adapter.Algo)).contents.data[0] = 0xABC
        return 0

    def capability(self, algo, attribute, value, size, written):
        C.cast(value, C.POINTER(C.c_uint32))[0] = self.alignment
        C.cast(written, C.POINTER(C.c_size_t))[0] = self.alignment_written
        return 0

    def check(self, *args):
        result = C.cast(args[-1], C.POINTER(adapter.Heuristic)).contents
        result.workspaceSize, result.state, result.wavesCount = (
            self.required_workspace,
            self.result_state,
            1.5,
        )
        # The real API does not update result.algo. A distinguishable value
        # detects accidentally replacing the explicitly configured algorithm.
        result.algo.data[0] = 0xBAD
        return self.check_status


class LockedLtBindingTests(unittest.TestCase):
    def test_library_functions_have_explicit_cuda12_argument_and_return_types(self):
        library = FakeLibrary()
        with patch.object(adapter.C, "CDLL", return_value=library):
            self.assertIs(adapter._load_library("unused/libcublasLt.so.12"), library)
        self.assertEqual(library.cublasLtGetVersion.argtypes, [])
        self.assertIs(library.cublasLtGetVersion.restype, C.c_size_t)
        self.assertEqual(len(library.cublasLtMatmul.argtypes), 16)
        self.assertIs(library.cublasLtMatmul.argtypes[-2], C.c_size_t)
        self.assertIs(library.cublasLtMatmul.argtypes[-1], C.c_void_p)
        self.assertIs(library.cublasLtMatmul.argtypes[12], C.POINTER(adapter.Algo))
        self.assertIs(
            library.cublasLtMatmulAlgoCheck.argtypes[-1], C.POINTER(adapter.Heuristic)
        )
        self.assertEqual(
            library.cublasLtMatrixLayoutCreate.argtypes[2:],
            [C.c_uint64, C.c_uint64, C.c_int64],
        )

    def test_loaded_cuda13_library_is_rejected_even_with_a_cuda12_filename(self):
        library = FakeLibrary()
        library.cublasLtGetVersion.return_value = 130101
        with patch.object(adapter.C, "CDLL", return_value=library):
            with self.assertRaises(RuntimeError):
                adapter._load_library("unused/libcublasLt.so.12")


class LockedLtPlanTests(unittest.TestCase):
    def setUp(self):
        self.library = FakeLibrary()
        self.stream = NS(cuda_stream=0xCAFE)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(
            patch.object(
                adapter,
                "cublaslt_path",
                return_value=Path("/virtual/libcublasLt.so.12"),
            )
        )
        self.patches.enter_context(
            patch.object(adapter, "_load_library", return_value=self.library)
        )
        self.patches.enter_context(patch.object(adapter.torch, "Tensor", FakeTensor))
        self.patches.enter_context(
            patch.object(
                adapter.torch,
                "empty",
                side_effect=lambda shape, **kwargs: FakeTensor(shape, **kwargs),
            )
        )
        self.patches.enter_context(
            patch.object(
                adapter.torch.cuda, "device", side_effect=lambda device: nullcontext()
            )
        )
        self.capturing = self.patches.enter_context(
            patch.object(
                adapter.torch.cuda, "is_current_stream_capturing", return_value=False
            )
        )
        self.current_stream = self.patches.enter_context(
            patch.object(adapter.torch.cuda, "current_stream", return_value=self.stream)
        )
        self.weight = FakeTensor((16, 32))

    def make(self, **recipe):
        instance = adapter.LockedLt(
            self.weight, {"algo_id": 7, **recipe}, workspace_bytes=128
        )
        self.addCleanup(instance.close)
        return instance

    def test_attribute_scalar_widths_and_host_float_scaling(self):
        instance = self.make(
            split_k=2, tile=15, inner_shape=1, cluster_shape=0, sm_count_target=108
        )
        attrs = {
            call.args[1]: call.args
            for call in self.library.cublasLtMatmulAlgoConfigSetAttribute.call_args_list
        }
        self.assertIs(type(attrs[2][2]._obj), C.c_int32)
        self.assertIs(type(attrs[1][2]._obj), C.c_uint32)
        self.assertEqual(attrs[7][-1], 2)
        self.assertEqual(attrs[8][-1], 2)
        desc = {
            call.args[1]: call.args[2]._obj.value
            for call in self.library.cublasLtMatmulDescSetAttribute.call_args_list
        }
        self.assertEqual(desc, {2: 0, 3: 1, 15: 108})
        self.assertEqual(instance.alpha.value, 1.0)
        self.assertEqual(instance.beta.value, 0.0)
        self.assertEqual(
            [
                call.args[1]
                for call in self.library.cublasLtMatmulAlgoCapGetAttribute.call_args_list
            ],
            [16, 17, 18, 19],
        )

    def test_prepare_caches_validated_layouts_and_ignores_result_algorithm(self):
        instance = self.make()
        first = instance.prepare(5)
        self.assertIs(instance.prepare(5), first)
        self.assertEqual(
            (first.workspace_bytes, first.waves_count, first.rows), (64, 1.5, 5)
        )
        self.assertEqual(instance.algo.data[0], 0xABC)
        self.assertEqual(self.library.cublasLtMatrixLayoutCreate.call_count, 3)
        self.library.cublasLtMatmulAlgoCheck.assert_called_once()
        self.library.cublasLtMatrixLayoutSetAttribute.assert_not_called()
        self.assertEqual(
            list(self.library.layout_specs.values()),
            [(14, 32, 16, 32), (14, 32, 5, 32), (14, 16, 5, 16)],
        )

    def test_row_batch_attributes_are_counts_and_element_offsets(self):
        instance = self.make(row_batch=True)
        instance.prepare(5)
        attrs = [
            (call.args[1], call.args[2]._obj.value, call.args[3])
            for call in self.library.cublasLtMatrixLayoutSetAttribute.call_args_list
        ]
        self.assertEqual(
            attrs, [(5, 5, 4), (6, 0, 8), (5, 5, 4), (6, 32, 8), (5, 5, 4), (6, 16, 8)]
        )

    def test_algocheck_api_and_result_failures_destroy_layouts_without_caching(self):
        for api_status, result_state in ((15, 0), (0, 15)):
            with self.subTest(api_status=api_status, result_state=result_state):
                instance = self.make()
                self.library.check_status, self.library.result_state = (
                    api_status,
                    result_state,
                )
                before = self.library.cublasLtMatrixLayoutDestroy.call_count
                with self.assertRaises(adapter.LtError) as error:
                    instance.prepare(5)
                self.assertEqual(error.exception.status, 15)
                self.assertEqual(instance.plans, {})
                self.assertEqual(
                    self.library.cublasLtMatrixLayoutDestroy.call_count - before, 3
                )
                instance.close()

    def test_insufficient_workspace_rejects_before_launch_and_does_not_cache(self):
        instance = self.make()
        self.library.required_workspace = 129
        with self.assertRaises(ValueError):
            instance.prepare(5)
        self.assertEqual(instance.plans, {})
        self.assertEqual(self.library.cublasLtMatrixLayoutDestroy.call_count, 3)
        self.library.cublasLtMatmul.assert_not_called()

    def test_capture_uses_cached_shape_but_never_creates_new_descriptors(self):
        instance = self.make()
        cached = instance.prepare(5)
        self.capturing.return_value = True
        self.assertIs(instance.prepare(5), cached)
        with self.assertRaises(RuntimeError):
            instance.prepare(4)
        self.assertEqual(self.library.cublasLtMatrixLayoutCreate.call_count, 3)

    def test_explicit_output_and_current_stream_reach_matmul_unchanged(self):
        instance = self.make()
        inputs, output = FakeTensor((5, 32)), FakeTensor((5, 16))
        self.assertIs(instance(inputs, output), output)
        args = self.library.cublasLtMatmul.call_args.args
        self.assertEqual(
            [args[i] for i in (3, 5, 8, 10)],
            [
                self.weight.data_ptr(),
                inputs.data_ptr(),
                output.data_ptr(),
                output.data_ptr(),
            ],
        )
        self.assertEqual(C.cast(args[2], C.POINTER(C.c_float)).contents.value, 1.0)
        self.assertEqual(C.cast(args[7], C.POINTER(C.c_float)).contents.value, 0.0)
        self.assertEqual(
            C.cast(args[12], C.POINTER(adapter.Algo)).contents.data[0], 0xABC
        )
        self.assertEqual(args[-2:], (64, 0xCAFE))
        self.current_stream.assert_called_once_with(self.weight.device)
        inputs.record_stream.assert_called_once_with(self.stream)
        output.record_stream.assert_called_once_with(self.stream)

    def test_invalid_output_storage_is_rejected_before_raw_launch(self):
        instance = self.make()
        inputs = FakeTensor((5, 32))
        invalid = [
            FakeTensor((4, 16)),
            FakeTensor((5, 16), dtype=torch.float32),
            FakeTensor((5, 16), device="cpu"),
            FakeTensor((5, 16), device="cuda:1"),
            FakeTensor((5, 16), pointer=inputs.data_ptr()),
            FakeTensor((5, 16), pointer=self.weight.data_ptr()),
        ]
        strided = FakeTensor((5, 16))
        strided.contiguous = False
        invalid.append(strided)
        for output in invalid:
            with self.subTest(shape=output.shape, device=output.device):
                with self.assertRaises(ValueError):
                    instance(inputs, output)
        self.library.cublasLtMatmul.assert_not_called()

    def test_pointer_alignment_and_changed_weight_storage_are_rejected(self):
        instance = self.make()
        inputs, output = FakeTensor((5, 32)), FakeTensor((5, 16))
        inputs.pointer += 2
        with self.assertRaises(ValueError):
            instance(inputs, output)
        inputs.pointer -= 2
        self.weight.pointer += 16
        with self.assertRaises(ValueError):
            instance(inputs, output)
        self.library.cublasLtMatmul.assert_not_called()

    def test_leading_dimension_alignment_is_required_in_both_batch_modes(self):
        for row_batch in (False, True):
            for shape in ((16, 33), (17, 32)):
                with self.subTest(row_batch=row_batch, shape=shape):
                    self.weight = FakeTensor(shape)
                    instance = self.make(row_batch=row_batch)
                    inputs, output = FakeTensor((5, shape[1])), FakeTensor(
                        (5, shape[0])
                    )
                    with self.assertRaises(ValueError):
                        instance(inputs, output)
        self.library.cublasLtMatmul.assert_not_called()

    def test_bad_capability_width_cleans_up_constructor_resources(self):
        self.library.alignment_written = 8
        with self.assertRaises(RuntimeError):
            adapter.LockedLt(self.weight, {"algo_id": 7})
        self.library.cublasLtMatmulDescDestroy.assert_called_once()
        self.library.cublasLtDestroy.assert_called_once()

    def test_close_destroys_every_resource_once_and_blocks_further_use(self):
        instance = self.make()
        instance.prepare(5)
        instance.close()
        instance.close()
        self.assertEqual(self.library.cublasLtMatrixLayoutDestroy.call_count, 3)
        self.library.cublasLtMatmulDescDestroy.assert_called_once()
        self.library.cublasLtDestroy.assert_called_once()
        self.assertTrue(instance.metadata()["closed"])
        with self.assertRaises(RuntimeError):
            instance.prepare(5)
        with self.assertRaises(RuntimeError):
            instance(FakeTensor((5, 32)))


if __name__ == "__main__":
    unittest.main()
