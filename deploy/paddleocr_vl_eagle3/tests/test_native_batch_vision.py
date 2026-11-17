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

"""Native CPU projection/dispatch contracts for the opt-in vision batch bridge."""

import functools
import importlib.metadata
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hps_batch import native_batch_vision as bridge


def has_native_vllm019():
    try:
        return importlib.metadata.version("vllm") == "0.19.0"
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(has_native_vllm019(), "Native vision tests require vLLM 0.19.0")
class NativeBatchVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from vllm.model_executor.models.paddleocr_vl import (
            PaddleOCRVLForConditionalGeneration,
            Projector,
        )

        cls.torch = torch
        cls.target_cls = PaddleOCRVLForConditionalGeneration
        cls.projector_cls = Projector

    def native_dispatch_fixture(self):
        # Copy native method objects without mutating the live vLLM registry.
        target = type(bridge.NATIVE_CLASS, (), {"__module__": bridge.NATIVE_MODULE})
        for name in (*bridge.NATIVE_METHOD_HASHES, "forward", "compute_logits"):
            setattr(target, name, getattr(self.target_cls, name))
        return target

    def test_only_native_unmodified_dispatch_is_adapted_and_is_idempotent(self):
        target = self.native_dispatch_fixture()
        unchanged = {
            name: getattr(target, name)
            for name in (
                "forward",
                "compute_logits",
                "encode_image",
                "embed_multimodal",
            )
        }
        result = bridge.adapt_target_class(target)
        self.assertTrue(result["applied"])
        self.assertIs(target._process_image_input, bridge.process_image_input_batch)
        for name, method in unchanged.items():
            self.assertIs(getattr(target, name), method)
        self.assertTrue(bridge.adapt_target_class(target)["applied"])

    def test_registered_custom_class_and_wrapped_native_vision_take_precedence(self):
        target = self.native_dispatch_fixture()
        target.__module__ = "paddlex.custom_plugin"
        original = target._process_image_input
        self.assertFalse(bridge.adapt_target_class(target)["applied"])
        self.assertIs(target._process_image_input, original)
        for name in bridge.NATIVE_METHOD_HASHES:
            with self.subTest(method=name):
                target = self.native_dispatch_fixture()
                original = getattr(target, name)

                @functools.wraps(original)
                def custom(*args, **kwargs):
                    return original(*args, **kwargs)

                setattr(target, name, custom)
                self.assertFalse(bridge.adapt_target_class(target)["applied"])
                self.assertIs(getattr(target, name), custom)

    def make_host(self):
        torch = self.torch

        class SegmentedVision(torch.nn.Module):
            """CPU stand-in verifies ragged segmentation and image isolation."""

            dtype = torch.float32

            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(
                self,
                *,
                pixel_values,
                image_grid_thw,
                position_ids,
                interpolate_pos_encoding,
                cu_seqlens
            ):
                boundaries = cu_seqlens.tolist()
                self.calls.append((image_grid_thw, boundaries, position_ids.clone()))
                assert interpolate_pos_encoding
                assert boundaries[0] == 0 and boundaries[-1] == len(pixel_values)
                assert len(boundaries) == len(image_grid_thw) + 1
                patch_means = pixel_values.flatten(1).mean(-1)
                segments = []
                for grid, start, end in zip(image_grid_thw, boundaries, boundaries[1:]):
                    t, h, w = grid
                    assert end - start == t * h * w
                    torch.testing.assert_close(
                        position_ids[start:end], torch.arange(end - start) % (h * w)
                    )
                    values = patch_means[start:end]
                    # A segment-local reduction catches accidental image mixing.
                    segments.append(
                        (values + values.mean())[:, None] + torch.arange(8)[None, :]
                    )
                return torch.cat(segments, dim=0)[None]

        torch.manual_seed(4)
        host = NS(
            visual=SegmentedVision(),
            mlp_AR=self.projector_cls(NS(hidden_size=6), NS(hidden_size=8)),
        )
        host.encode_image = MethodType(self.target_cls.encode_image, host)
        return host

    def test_b8_ragged_encoder_once_preserves_native_projector_results(self):
        torch = self.torch
        host = self.make_host()
        grids = torch.tensor(
            [
                [1, 4, 4],
                [1, 2, 8],
                [1, 8, 4],
                [1, 2, 2],
                [1, 4, 6],
                [1, 6, 2],
                [1, 2, 10],
                [1, 6, 8],
            ]
        )
        pixels = [
            torch.randn(int(grid.prod()), 3, 2, 2) + index
            for index, grid in enumerate(grids)
        ]
        inputs = NS(pixel_values=pixels, image_grid_thw=grids)
        original_grids = grids.clone()
        with torch.no_grad():
            reference = self.target_cls._process_image_input(host, inputs)
            self.assertEqual(len(host.visual.calls), 8)
            host.visual.calls.clear()
            actual = bridge.process_image_input_batch(host, inputs)
        self.assertEqual(len(host.visual.calls), 1)
        self.assertEqual(len(actual), 8)
        for original, batched, grid in zip(reference, actual, grids):
            self.assertEqual(tuple(batched.shape), (int(grid.prod()) // 4, 6))
            torch.testing.assert_close(original, batched, rtol=0, atol=0)
        torch.testing.assert_close(grids, original_grids)
        self.assertFalse(torch.cuda.is_initialized())

    def test_equal_patch_counts_and_single_image_preserve_output_boundaries(self):
        torch = self.torch
        for grid_list in ([[1, 4, 4]], [[1, 4, 4], [1, 2, 8]]):
            host = self.make_host()
            grids = torch.tensor(grid_list)
            pixels = torch.randn(len(grids), 16, 3, 2, 2)
            inputs = NS(pixel_values=pixels, image_grid_thw=grids)
            with torch.no_grad():
                expected = self.target_cls._process_image_input(host, inputs)
                actual = bridge.process_image_input_batch(host, inputs)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_empty_batch_and_malformed_patch_counts(self):
        torch = self.torch
        host = self.make_host()
        self.assertEqual(
            bridge.process_image_input_batch(
                host,
                NS(
                    pixel_values=[], image_grid_thw=torch.empty(0, 3, dtype=torch.int64)
                ),
            ),
            (),
        )
        self.assertEqual(host.visual.calls, [])
        with self.assertRaisesRegex(ValueError, "patch count"):
            bridge.process_image_input_batch(
                host,
                NS(
                    pixel_values=[torch.zeros(3, 3, 2, 2)],
                    image_grid_thw=torch.tensor([[1, 2, 2]]),
                ),
            )
        with self.assertRaisesRegex(ValueError, "same image count"):
            bridge.process_image_input_batch(
                host, NS(pixel_values=[], image_grid_thw=torch.tensor([[1, 2, 2]]))
            )


if __name__ == "__main__":
    unittest.main()
