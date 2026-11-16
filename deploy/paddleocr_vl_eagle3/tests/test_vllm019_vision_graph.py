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

"""CPU contracts for the native vision layer-stack CUDA graph adapter.

No CUDA tensor, CUDA API, model weight, or graph capture is used here. These
checks do not establish GPU numerical equivalence or performance.
"""

import inspect
from functools import wraps
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_vision_graph as candidate


_PREFIX_TAG = "native-prefix-global"


class RecordingLayer:
    def __init__(self, name, result, records):
        self.name = name
        self.result = result
        self.records = records

    def __call__(self, hidden_states, **kwargs):
        self.records.append((self.name, hidden_states, kwargs))
        return self.result


class EncoderFixture:
    """The loop matches vLLM's native SiglipEncoder.forward exactly."""

    def __init__(self, layers=()):
        self.layers = list(layers)
        self.prefix_calls = 0
        self.events = []
        self.rotary = object()
        self.maximum = object()

    def forward(
        self,
        inputs_embeds,
        cu_seqlens=None,
        image_grid_thw=None,
        height_position_ids=None,
        width_position_ids=None,
    ):
        self.prefix_calls += 1
        self.events.append((_PREFIX_TAG, image_grid_thw))
        device = inputs_embeds.device
        self.events.append((device, height_position_ids, width_position_ids))
        rotary_pos_emb = self.rotary
        max_seqlen = self.maximum
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
            )
        return hidden_states


class UnsupportedEncoderFixture:
    def forward(self, inputs_embeds, cu_seqlens=None):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=None,
                max_seqlen=None,
            )
        return hidden_states


class SignatureFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((2, 2)))
        self.normalized_shape = (2,)
        self.layout = [2, 3]

    def forward(self, hidden):
        return hidden

    def split_qkv(self, hidden):
        return hidden

    @staticmethod
    def view_qkv_to_4d(*args):
        return args


class NativeStackContracts(unittest.TestCase):
    def test_preserves_layer_order_inputs_and_metadata_objects(self):
        records = []
        initial, one, two, three = (object() for _ in range(4))
        cu, rotary, maximum = (object() for _ in range(3))
        encoder = SimpleNamespace(
            layers=[
                RecordingLayer("one", one, records),
                RecordingLayer("two", two, records),
                RecordingLayer("three", three, records),
            ]
        )
        result = candidate._native_stack(encoder, initial, cu, rotary, maximum)
        self.assertIs(result, three)
        self.assertEqual([row[0] for row in records], ["one", "two", "three"])
        for record, expected_input in zip(records, (initial, one, two)):
            self.assertIs(record[1], expected_input)
            self.assertEqual(
                set(record[2]), {"cu_seqlens", "rotary_pos_emb", "max_seqlen"}
            )
            self.assertIs(record[2]["cu_seqlens"], cu)
            self.assertIs(record[2]["rotary_pos_emb"], rotary)
            self.assertIs(record[2]["max_seqlen"], maximum)

    def test_empty_stack_returns_original_input(self):
        initial = object()
        result = candidate._native_stack(
            SimpleNamespace(layers=[]), initial, object(), object(), object()
        )
        self.assertIs(result, initial)

    def test_native_exception_stops_later_layers(self):
        records = []

        def fail(*args, **kwargs):
            raise ValueError("native layer failure")

        encoder = SimpleNamespace(
            layers=[fail, RecordingLayer("unreached", object(), records)]
        )
        with self.assertRaisesRegex(ValueError, "native layer failure"):
            candidate._native_stack(encoder, object(), object(), object(), object())
        self.assertEqual(records, [])


class NativePrefixContracts(unittest.TestCase):
    def test_replacement_keeps_prefix_signature_and_real_metadata(self):
        seen = []
        result = object()

        def dispatch(encoder, hidden, cu, rotary, maximum):
            seen.append((encoder, hidden, cu, rotary, maximum))
            return result

        native = EncoderFixture.forward
        rewritten = candidate._replace_native_loop(native, dispatch)
        self.assertEqual(inspect.signature(rewritten), inspect.signature(native))
        encoder = EncoderFixture()
        hidden = torch.zeros((1, 4, 3), dtype=torch.bfloat16)
        cu, grid, height, width = (object() for _ in range(4))
        actual = rewritten(
            encoder,
            hidden,
            cu,
            grid,
            height_position_ids=height,
            width_position_ids=width,
        )
        self.assertIs(actual, result)
        self.assertEqual(encoder.prefix_calls, 1)
        self.assertEqual(encoder.events[0], (_PREFIX_TAG, grid))
        self.assertIs(encoder.events[1][1], height)
        self.assertIs(encoder.events[1][2], width)
        self.assertEqual(len(seen), 1)
        for actual_arg, expected_arg in zip(
            seen[0], (encoder, hidden, cu, encoder.rotary, encoder.maximum)
        ):
            self.assertIs(actual_arg, expected_arg)

    def test_each_call_dispatches_fresh_metadata_and_inputs(self):
        seen = []

        def dispatch(encoder, hidden, cu, rotary, maximum):
            seen.append((hidden, cu, rotary, maximum))
            return hidden

        rewritten = candidate._replace_native_loop(EncoderFixture.forward, dispatch)
        encoder = EncoderFixture()
        expected = []
        for _ in range(2):
            hidden = torch.empty((1, 4, 3), dtype=torch.bfloat16)
            cu, encoder.rotary, encoder.maximum = (object() for _ in range(3))
            expected.append((hidden, cu, encoder.rotary, encoder.maximum))
            self.assertIs(rewritten(encoder, hidden, cu), hidden)
        self.assertEqual(encoder.prefix_calls, 2)
        for actual, wanted in zip(seen, expected):
            for actual_arg, expected_arg in zip(actual, wanted):
                self.assertIs(actual_arg, expected_arg)

    def test_native_fallback_keeps_prefix_and_layer_invocations(self):
        records = []
        hidden = torch.empty((1, 4, 3), dtype=torch.bfloat16)
        end = object()
        encoder = EncoderFixture([RecordingLayer("only", end, records)])
        native = EncoderFixture.forward
        rewritten = candidate._replace_native_loop(native, candidate._native_stack)
        self.assertIs(rewritten(encoder, hidden), end)
        self.assertEqual(encoder.prefix_calls, 1)
        self.assertEqual(len(records), 1)
        self.assertIs(records[0][1], hidden)
        self.assertIs(EncoderFixture.forward, native)
        self.assertIs(native(encoder, hidden), end)
        self.assertEqual(encoder.prefix_calls, 2)
        self.assertEqual(len(records), 2)

    def test_modified_native_loop_is_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            candidate._replace_native_loop(
                UnsupportedEncoderFixture.forward, candidate._native_stack
            )


class CacheKeyContracts(unittest.TestCase):
    def test_tensor_key_captures_layout_dtype_shape_and_device(self):
        base = torch.empty((2, 3), dtype=torch.bfloat16)
        self.assertEqual(
            candidate._tensor_key(base),
            ("cpu", "torch.bfloat16", (2, 3), (3, 1)),
        )
        for changed in (
            base.t(),
            torch.empty((2, 3), dtype=torch.float32),
            torch.empty_strided((2, 3), (1, 2), dtype=torch.bfloat16),
        ):
            self.assertNotEqual(
                candidate._tensor_key(base), candidate._tensor_key(changed)
            )
        self.assertEqual(
            candidate._tensor_key(base), candidate._tensor_key(base.clone())
        )

    def test_cache_key_tracks_encoder_identity_actual_max_and_all_layouts(self):
        encoder = object()
        hidden = torch.empty((1, 8, 1152), dtype=torch.bfloat16)
        rotary = torch.empty((8, 36), dtype=torch.float32)
        cu = torch.tensor([0, 8], dtype=torch.int32)
        baseline = candidate._cache_key(encoder, hidden, rotary, cu, 8)
        self.assertEqual(
            baseline,
            candidate._cache_key(
                encoder, hidden.clone(), rotary.clone(), cu.clone(), 8
            ),
        )
        changed_inputs = (
            (object(), hidden, rotary, cu, 8),
            (encoder, hidden, rotary, cu, 7),
            (encoder, hidden[:, :, ::2], rotary, cu, 8),
            (encoder, hidden, rotary[:, ::2], cu, 8),
            (encoder, hidden, rotary, cu.to(torch.int64), 8),
        )
        for args in changed_inputs:
            self.assertNotEqual(baseline, candidate._cache_key(*args))


class CacheAdmissionContracts(unittest.TestCase):
    def setUp(self):
        self.encoder = SimpleNamespace(layers=[])
        self.hidden = torch.empty((1, 8, 1152), dtype=torch.bfloat16)
        self.rotary = torch.empty((8, 36), dtype=torch.float32)
        self.cu = torch.tensor([0, 8], dtype=torch.int32)
        self.maximum = torch.tensor(8, dtype=torch.int32)
        self.captures = []
        self.audit_ok = True
        self.audit_error = None
        self.stream = object()
        owner = self

        class FakePlan:
            accounted_bytes = 64

            def __init__(self, encoder, hidden, cu, rotary, maximum):
                owner.captures.append(self)
                self.replays = 0
                self.audit_calls = 0

            def audit(self, *args, **kwargs):
                self.audit_calls += 1
                if owner.audit_error is not None:
                    raise owner.audit_error
                return owner.audit_ok

            def replay(self, hidden, cu, rotary, *, borrow_output=False):
                self.replays += 1
                return hidden if borrow_output else hidden.clone()

            def metadata(self):
                return {"accounted_bytes": self.accounted_bytes}

        class FakeFastGuard:
            module_count = 1
            tensor_count = 0

            def __init__(self, encoder):
                self.encoder = encoder
                self.baseline_signature = candidate._module_signature(encoder)

            def matches(self, encoder):
                return (
                    encoder is self.encoder
                    and candidate._module_signature(encoder) == self.baseline_signature
                )

        for patcher in (
            patch.object(candidate, "_inputs_eligible", return_value=True),
            patch.object(candidate, "_module_signature", return_value=("native",)),
            patch.object(candidate, "LayerStackPlan", FakePlan),
            patch.object(candidate, "FastModuleGuard", FakeFastGuard),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "current_stream", return_value=self.stream),
            patch.object(
                torch.cuda, "mem_get_info", return_value=(2 * 1024**3, 4 * 1024**3)
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_cache(self, cache, maximum=None):
        return cache.run(
            self.encoder,
            self.hidden,
            self.cu,
            self.rotary,
            self.maximum if maximum is None else maximum,
        )

    def set_sequence_length(self, length):
        self.hidden = torch.empty((1, length, 1152), dtype=torch.bfloat16)
        self.rotary = torch.empty((length, 36), dtype=torch.float32)
        self.cu = torch.tensor([0, length], dtype=torch.int32)
        self.maximum = torch.tensor(length, dtype=torch.int32)

    def test_constructor_rejects_invalid_cache_limits(self):
        for kwargs in (
            {"max_graphs": 0},
            {"max_graphs": 33},
            {"max_cached_bytes": 0},
            {"max_cached_bytes": 1024**3 + 1},
            {"minimum_reuse": 0},
            {"audit_replays": -1},
            {"audit_replays": 9},
            {"max_sequence_length": 0},
            {"max_sequence_length": 2049},
            {"minimum_free_bytes": -1},
            {"borrow_output": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    candidate.LayerStackCache(**kwargs)

    def test_new_cache_admission_limits_accept_their_boundaries(self):
        cache = candidate.LayerStackCache(
            max_graphs=32, max_sequence_length=1, minimum_free_bytes=0
        )
        self.assertEqual(cache.max_graphs, 32)
        self.assertEqual(cache.max_sequence_length, 1)
        self.assertEqual(cache.minimum_free_bytes, 0)

    def test_production_defaults_keep_bounded_owned_output_scope(self):
        cache = candidate.LayerStackCache()
        self.assertEqual(cache.max_graphs, 32)
        self.assertEqual(cache.max_cached_bytes, 1024**3)
        self.assertEqual(cache.max_sequence_length, 2048)
        self.assertEqual(cache.minimum_reuse, 2)
        self.assertEqual(cache.minimum_free_bytes, 1536 * 1024**2)
        self.assertFalse(cache.borrow_output)
        self.assertEqual(cache.audit_replays, 0)

    def test_runtime_cache_policy_matches_effective_adapter_defaults(self):
        from runtime_config import VISION_GRAPHS_POLICY

        cache = candidate.LayerStackCache()
        self.assertEqual(
            VISION_GRAPHS_POLICY,
            {
                "max_graphs": cache.max_graphs,
                "max_cached_bytes": cache.max_cached_bytes,
                "max_sequence_length": cache.max_sequence_length,
                "minimum_reuse": cache.minimum_reuse,
                "minimum_free_bytes": cache.minimum_free_bytes,
                "actual_raw_bit_admission": True,
                "clone_output": not cache.borrow_output,
                "audit_changed_inputs": cache.audit_replays,
            },
        )

    def test_minimum_reuse_then_capture_and_independent_outputs(self):
        cache = candidate.LayerStackCache(minimum_reuse=2)
        self.assertIs(self.run_cache(cache), self.hidden)
        self.assertEqual(self.captures, [])
        self.assertIsNot(self.run_cache(cache), self.hidden)
        self.assertIsNot(self.run_cache(cache), self.hidden)
        self.assertEqual(len(self.captures), 1)
        self.assertEqual(self.captures[0].replays, 2)
        self.assertEqual(self.captures[0].audit_calls, 1)
        self.assertEqual(cache.counts["fallback_unprepared"], 1)
        self.assertEqual(cache.counts["cold_captures"], 1)
        self.assertEqual(cache.counts["cache_hits"], 1)
        self.assertEqual(cache.cache_bytes, 64)

    def test_full_cache_falls_back_without_eviction(self):
        cache = candidate.LayerStackCache(max_graphs=1, minimum_reuse=1)
        self.run_cache(cache)
        original_plan = next(iter(cache.plans.values()))
        self.set_sequence_length(7)
        self.assertIs(self.run_cache(cache), self.hidden)
        self.assertEqual(len(self.captures), 1)
        self.assertIs(next(iter(cache.plans.values())), original_plan)
        self.assertEqual(cache.counts["fallback_capacity"], 1)

    def test_failed_audit_is_not_cached_or_retried(self):
        self.audit_ok = False
        cache = candidate.LayerStackCache(minimum_reuse=1)
        self.assertIs(self.run_cache(cache), self.hidden)
        self.assertIs(self.run_cache(cache), self.hidden)
        self.assertEqual(len(self.captures), 1)
        self.assertEqual(len(cache.plans), 0)
        self.assertEqual(len(cache.failed), 1)
        self.assertEqual(cache.counts["fallback_failed_validation"], 2)

    def test_memory_budget_rejection_stops_future_captures(self):
        cache = candidate.LayerStackCache(minimum_reuse=1, max_cached_bytes=63)
        self.assertIs(self.run_cache(cache), self.hidden)
        self.assertTrue(cache.budget_exhausted)
        self.assertEqual(len(cache.plans), 0)
        self.set_sequence_length(7)
        self.assertIs(self.run_cache(cache), self.hidden)
        self.assertEqual(len(self.captures), 1)
        self.assertEqual(cache.counts["fallback_memory_budget"], 1)
        self.assertEqual(self.captures[0].audit_calls, 0)

    def test_foreign_stream_cannot_replay_a_prepared_plan(self):
        cache = candidate.LayerStackCache(minimum_reuse=1)
        self.run_cache(cache)
        with patch.object(torch.cuda, "current_stream", return_value=object()):
            self.assertIs(self.run_cache(cache), self.hidden)
        self.assertEqual(self.captures[0].replays, 1)
        self.assertEqual(cache.counts["fallback_stream"], 1)

    def test_outer_capture_cannot_prepare_or_replay(self):
        cache = candidate.LayerStackCache(minimum_reuse=1)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertIs(self.run_cache(cache), self.hidden)
        self.assertEqual(self.captures, [])
        self.assertEqual(cache.counts["fallback_outer_capture"], 1)
        self.assertEqual(cache.counts["gpu_scalar_reads"], 0)

    def assert_once_scalar_fallback(self, cache, reason, value=8):
        class Source:
            reads = 0

            def item(self):
                self.reads += 1
                return value

        source = Source()
        records = []
        self.encoder.layers = [RecordingLayer("native", self.hidden, records)]
        self.assertIs(self.run_cache(cache, source), self.hidden)
        self.assertEqual(source.reads, 1)
        self.assertEqual(cache.counts["gpu_scalar_reads"], 1)
        self.assertEqual(cache.counts["fallback_" + reason], 1)
        self.assertEqual(len(records), 1)
        actual_max = records[0][2]["max_seqlen"]
        self.assertIsInstance(actual_max, torch.Tensor)
        self.assertEqual(actual_max.device.type, "cpu")
        self.assertEqual(actual_max.dtype, torch.int32)
        self.assertEqual(actual_max.ndim, 0)
        self.assertEqual(actual_max.item(), value)

    def test_sequence_limit_fallback_retains_once_scalar_path(self):
        cache = candidate.LayerStackCache(minimum_reuse=1, max_sequence_length=7)
        with patch.object(
            torch.cuda, "current_stream", side_effect=AssertionError("stream queried")
        ):
            self.assert_once_scalar_fallback(cache, "sequence_length")
        self.assertEqual(self.captures, [])

    def test_partial_sequence_length_keeps_once_scalar_native_fallback(self):
        cache = candidate.LayerStackCache(minimum_reuse=1)
        self.assert_once_scalar_fallback(cache, "incomplete_sequence", value=7)
        self.assertEqual(self.captures, [])

    def test_capture_error_propagates_and_releases_the_admission_lock(self):
        cache = candidate.LayerStackCache(minimum_reuse=1)
        records = []
        self.encoder.layers = [RecordingLayer("native", self.hidden, records)]
        with patch.object(
            candidate, "LayerStackPlan", side_effect=RuntimeError("capture failure")
        ):
            with self.assertRaises(RuntimeError):
                self.run_cache(cache)
        self.assertEqual(records, [])
        self.assertEqual(cache.plans, {})
        self.assertTrue(cache.capture_failed)
        self.assertEqual(cache.counts["capture_errors"], 1)
        self.assertTrue(cache.lock.acquire(blocking=False))
        cache.lock.release()
        with patch.object(
            candidate,
            "_inputs_eligible",
            side_effect=AssertionError("admission retried"),
        ):
            with self.assertRaisesRegex(RuntimeError, "restart"):
                self.run_cache(cache)

    def test_validation_execution_error_requires_worker_restart(self):
        self.audit_error = RuntimeError("validation launch failure")
        cache = candidate.LayerStackCache(minimum_reuse=1)
        records = []
        self.encoder.layers = [RecordingLayer("native", self.hidden, records)]
        with self.assertRaisesRegex(RuntimeError, "restart"):
            self.run_cache(cache)
        self.assertTrue(cache.capture_failed)
        self.assertEqual(cache.counts["graph_execution_errors"], 1)
        self.assertEqual(records, [])
        self.assertEqual(cache.plans, {})
        with self.assertRaisesRegex(RuntimeError, "restart"):
            self.run_cache(cache)

    def test_prepared_replay_error_disables_adapter_and_all_future_calls(self):
        cache = candidate.LayerStackCache(minimum_reuse=1)
        self.run_cache(cache)
        plan = self.captures[0]
        with (
            patch.object(
                plan, "replay", side_effect=RuntimeError("replay launch failure")
            ) as replay,
            patch.object(
                candidate,
                "_native_stack",
                side_effect=AssertionError("native fallback after GPU failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "restart"):
                self.run_cache(cache)
            self.assertTrue(cache.capture_failed)
            self.assertEqual(cache.counts["graph_execution_errors"], 1)
            self.assertTrue(cache.lock.acquire(blocking=False))
            cache.lock.release()
            with patch.object(
                candidate,
                "_inputs_eligible",
                side_effect=AssertionError("admission after GPU failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "restart"):
                    self.run_cache(cache)
                with self.assertRaisesRegex(RuntimeError, "restart"):
                    cache.run(self.encoder, None, None, None, None)
            replay.assert_called_once()
        state = SimpleNamespace(
            guards=SimpleNamespace(
                _owned_hooks_intact=lambda _: True,
                _native_methods_intact=lambda _: True,
                COUNTS={},
            ),
            cache=cache,
        )
        with patch.object(candidate, "_STATE", state):
            self.assertFalse(candidate.metadata()["enabled"])

    def test_low_free_memory_fallback_retains_once_scalar_path(self):
        cache = candidate.LayerStackCache(minimum_reuse=1)
        with patch.object(torch.cuda, "mem_get_info", return_value=(0, 4 * 1024**3)):
            self.assert_once_scalar_fallback(cache, "free_memory")
        self.assertEqual(self.captures, [])

    def test_busy_fallback_retains_once_scalar_path(self):
        cache = candidate.LayerStackCache(minimum_reuse=1)
        cache.lock = SimpleNamespace(acquire=lambda **kwargs: False)
        self.assert_once_scalar_fallback(cache, "busy")
        self.assertEqual(self.captures, [])

    def test_changed_module_fallback_retains_once_scalar_path(self):
        cache = candidate.LayerStackCache(minimum_reuse=1)
        with patch.object(candidate, "_module_signature", return_value=None):
            self.assert_once_scalar_fallback(cache, "module_changed")
        self.assertEqual(self.captures, [])

    def test_unsupported_inputs_fall_back_before_any_cuda_api(self):
        cache = candidate.LayerStackCache(minimum_reuse=1)
        with (
            patch.object(candidate, "_inputs_eligible", return_value=False),
            patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                side_effect=AssertionError("CUDA called"),
            ),
            patch.object(
                torch.cuda, "current_stream", side_effect=AssertionError("CUDA called")
            ),
        ):
            self.assertIs(self.run_cache(cache), self.hidden)
        self.assertEqual(self.captures, [])
        self.assertEqual(cache.counts["fallback_inputs"], 1)


class InstallationContracts(unittest.TestCase):
    def test_metadata_disables_adapter_after_capture_failure(self):
        state = SimpleNamespace(
            guards=SimpleNamespace(
                _owned_hooks_intact=lambda _: True,
                _native_methods_intact=lambda _: True,
                COUNTS={},
            ),
            cache=SimpleNamespace(
                capture_failed=True,
                metadata=lambda: {"capture_failed": True},
            ),
        )
        with patch.object(candidate, "_STATE", state):
            info = candidate.metadata()
        self.assertTrue(info["installed"])
        self.assertFalse(info["enabled"])
        self.assertTrue(info["capture_failed"])

    def test_invalid_audit_option_never_changes_existing_installation(self):
        sentinel = object()
        with (
            patch.object(candidate, "_STATE", sentinel),
            patch.object(
                candidate.importlib.metadata,
                "version",
                side_effect=AssertionError("version queried"),
            ),
        ):
            for value in (-1, 9, True, "1", 1.5):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "audit_changed_inputs"):
                        candidate.install(audit_changed_inputs=value)
                    self.assertIs(candidate._STATE, sentinel)

    def test_unsupported_version_is_rejected_before_vllm_import_or_hooks(self):
        with (
            patch.object(candidate, "_STATE", None),
            patch.object(
                candidate.importlib.metadata, "version", return_value="0.32.0"
            ),
            patch.object(
                candidate.scalar,
                "_native_python_function",
                side_effect=AssertionError("guard module loaded"),
            ),
        ):
            before = {name for name in sys.modules if name.startswith("vllm")}
            with self.assertRaisesRegex(RuntimeError, "0.19.0"):
                candidate.install()
            self.assertIsNone(candidate._STATE)
            after = {name for name in sys.modules if name.startswith("vllm")}
            self.assertEqual(before, after)

    def test_repeated_install_detects_changed_hook_and_native_method(self):
        for owned, native in ((False, True), (True, False)):
            state = SimpleNamespace(
                guards=SimpleNamespace(
                    _owned_hooks_intact=lambda _: owned,
                    _native_methods_intact=lambda _: native,
                )
            )
            with self.subTest(owned=owned, native=native):
                with patch.object(candidate, "_STATE", state):
                    with self.assertRaisesRegex(RuntimeError, "replaced"):
                        candidate.install()
                    self.assertIs(candidate._STATE, state)

    def test_intact_repeated_install_does_not_reinstall_or_import(self):
        state = SimpleNamespace(
            guards=SimpleNamespace(
                _owned_hooks_intact=lambda _: True,
                _native_methods_intact=lambda _: True,
            )
        )
        result = {"installed": True, "sentinel": object()}
        with (
            patch.object(candidate, "_STATE", state),
            patch.object(candidate, "metadata", return_value=result),
            patch.object(
                candidate.importlib.metadata,
                "version",
                side_effect=AssertionError("version queried"),
            ),
        ):
            self.assertIs(candidate.install(), result)
            self.assertIs(candidate._STATE, state)

    def test_uninstall_refuses_to_overwrite_later_hooks(self):
        state = SimpleNamespace(
            guards=SimpleNamespace(_owned_hooks_intact=lambda _: False),
            cache=SimpleNamespace(capture_failed=False),
        )
        with patch.object(candidate, "_STATE", state):
            with self.assertRaisesRegex(RuntimeError, "Refusing"):
                candidate.uninstall()
            self.assertIs(candidate._STATE, state)

    def test_failed_context_cannot_be_bypassed_by_uninstall(self):
        current_forward, current_attention = object(), object()
        restore = Mock()
        state = SimpleNamespace(
            guards=SimpleNamespace(_owned_hooks_intact=lambda _: True),
            cache=SimpleNamespace(capture_failed=True),
            encoder_type=SimpleNamespace(forward=current_forward),
            attention_module=SimpleNamespace(vit_flash_attn_wrapper=current_attention),
            previous_forward=object(),
            original_attention=object(),
            scalar_module_to_restore=SimpleNamespace(install=restore),
        )
        with patch.object(candidate, "_STATE", state):
            with self.assertRaisesRegex(RuntimeError, "restart"):
                candidate.uninstall()
            self.assertIs(candidate._STATE, state)
            self.assertIs(state.encoder_type.forward, current_forward)
            self.assertIs(
                state.attention_module.vit_flash_attn_wrapper, current_attention
            )
            restore.assert_not_called()

    def test_uninstall_restores_native_hooks_then_production_scalar(self):
        old_forward, old_attention = object(), object()
        restored = Mock()
        state = SimpleNamespace(
            guards=SimpleNamespace(_owned_hooks_intact=lambda _: True),
            cache=SimpleNamespace(owner_stream=None, capture_failed=False),
            encoder_type=SimpleNamespace(forward=object()),
            attention_module=SimpleNamespace(vit_flash_attn_wrapper=object()),
            previous_forward=old_forward,
            original_attention=old_attention,
            scalar_module_to_restore=SimpleNamespace(install=restored),
        )
        with patch.object(candidate, "_STATE", state):
            candidate.uninstall()
            self.assertIsNone(candidate._STATE)
            self.assertIs(state.encoder_type.forward, old_forward)
            self.assertIs(state.attention_module.vit_flash_attn_wrapper, old_attention)
            restored.assert_called_once_with()

    def test_shared_scalar_guard_checks_plain_native_source_provenance(self):
        guards = candidate.scalar
        function = EncoderFixture.forward
        module = sys.modules[__name__]
        qualname = "EncoderFixture.forward"
        self.assertTrue(guards._native_python_function(function, module, qualname))

        @wraps(function)
        def decorated(*args, **kwargs):
            return function(*args, **kwargs)

        self.assertFalse(guards._native_python_function(decorated, module, qualname))
        different_source = SimpleNamespace(
            __name__=module.__name__, __file__="different_native_source.py"
        )
        self.assertFalse(
            guards._native_python_function(function, different_source, qualname)
        )


class NativeSignatureContracts(unittest.TestCase):
    def setUp(self):
        self.module = SignatureFixture().eval()
        self.signature = candidate._module_signature(self.module)
        self.assertIsNotNone(self.signature)

    def test_class_and_instance_helper_replacements_change_signature(self):
        for owner, name in (
            (SignatureFixture, "split_qkv"),
            (SignatureFixture, "view_qkv_to_4d"),
            (self.module, "split_qkv"),
            (self.module, "view_qkv_to_4d"),
        ):
            with self.subTest(owner=type(owner).__name__, name=name):
                with patch.object(owner, name, lambda *args: args):
                    self.assertNotEqual(
                        self.signature, candidate._module_signature(self.module)
                    )
        self.assertEqual(self.signature, candidate._module_signature(self.module))

    def test_normalized_shape_and_mutable_public_list_are_snapshotted(self):
        with patch.object(self.module, "normalized_shape", (1, 2)):
            self.assertNotEqual(
                self.signature, candidate._module_signature(self.module)
            )
        self.module.layout[0] = 5
        self.assertNotEqual(self.signature, candidate._module_signature(self.module))

    def test_parameter_storage_replacement_changes_signature(self):
        old_parameter = self.module.weight
        self.module.weight.data = self.module.weight.data.clone()
        self.assertIs(self.module.weight, old_parameter)
        self.assertNotEqual(self.signature, candidate._module_signature(self.module))

    def test_in_place_weight_value_update_keeps_captured_pointer_signature(self):
        with torch.no_grad():
            self.module.weight.add_(1)
        self.assertEqual(self.signature, candidate._module_signature(self.module))

    def test_training_hooks_and_instance_forward_disable_graph_admission(self):
        with patch.object(self.module, "training", True):
            self.assertIsNone(candidate._module_signature(self.module))
        hook = self.module.register_forward_hook(lambda *args: None)
        try:
            self.assertIsNone(candidate._module_signature(self.module))
        finally:
            hook.remove()
        with patch.object(self.module, "forward", lambda hidden: hidden):
            self.assertIsNone(candidate._module_signature(self.module))
        with patch.object(self.module, "_compiled_call_impl", lambda hidden: hidden):
            self.assertIsNone(candidate._module_signature(self.module))


class FastGuardContracts(unittest.TestCase):
    def setUp(self):
        self.module = SignatureFixture().eval()

    def make_guard(self, module=None):
        return candidate.FastModuleGuard(self.module if module is None else module)

    def test_baseline_metadata_and_encoder_identity(self):
        guard = self.make_guard()
        self.assertEqual(
            guard.baseline_signature, candidate._module_signature(self.module)
        )
        self.assertEqual(guard.module_count, 1)
        self.assertEqual(guard.tensor_count, 1)
        self.assertTrue(guard.matches(self.module))
        self.assertFalse(guard.matches(SignatureFixture().eval()))

    def test_constructor_rejects_unsupported_training_or_hook_state(self):
        with patch.object(self.module, "training", True):
            with self.assertRaises(RuntimeError):
                self.make_guard()
        hook = self.module.register_forward_hook(lambda *args: None)
        try:
            with self.assertRaises(RuntimeError):
                self.make_guard()
        finally:
            hook.remove()

    def test_hot_check_never_rebuilds_recursive_signature_or_string_tensor_key(self):
        guard = self.make_guard()
        forbidden = AssertionError("cold recursive helper used in hot guard")
        with (
            patch.object(candidate, "_module_signature", side_effect=forbidden),
            patch.object(candidate, "_tensor_key", side_effect=forbidden),
            patch.object(torch.nn.Module, "modules", side_effect=forbidden),
            patch.object(torch.nn.Module, "named_parameters", side_effect=forbidden),
            patch.object(torch.nn.Module, "named_buffers", side_effect=forbidden),
        ):
            self.assertTrue(guard.matches(self.module))

    def test_child_replacement_and_in_place_modulelist_order_change_are_rejected(self):
        tree = torch.nn.ModuleList([SignatureFixture(), SignatureFixture()]).eval()
        guard = self.make_guard(tree)
        original = tree[0]
        tree[0] = SignatureFixture().eval()
        self.assertFalse(guard.matches(tree))
        tree[0] = original
        self.assertTrue(guard.matches(tree))
        first = tree._modules.pop("0")
        tree._modules["0"] = first
        self.assertFalse(guard.matches(tree))

    def test_parameter_and_buffer_slot_replacements_are_rejected(self):
        self.module.register_buffer("state", torch.zeros(2))
        guard = self.make_guard()
        self.assertEqual(guard.tensor_count, 2)
        old_weight = self.module.weight
        self.module.weight = torch.nn.Parameter(old_weight.detach().clone())
        self.assertFalse(guard.matches(self.module))
        self.module.weight = old_weight
        self.assertTrue(guard.matches(self.module))
        self.module.state = self.module.state.clone()
        self.assertFalse(guard.matches(self.module))

    def test_added_parameter_or_buffer_slots_are_rejected(self):
        guard = self.make_guard()
        self.module.register_buffer("extra_state", torch.zeros(1))
        self.assertFalse(guard.matches(self.module))
        del self.module.extra_state
        self.assertTrue(guard.matches(self.module))
        self.module.register_parameter(
            "extra_weight", torch.nn.Parameter(torch.zeros(1))
        )
        self.assertFalse(guard.matches(self.module))

    def test_same_parameter_object_new_storage_is_rejected(self):
        guard = self.make_guard()
        original = self.module.weight
        self.module.weight.data = self.module.weight.data.clone()
        self.assertIs(self.module.weight, original)
        self.assertFalse(guard.matches(self.module))

    def test_same_storage_shape_or_stride_changes_are_rejected(self):
        guard = self.make_guard()
        original_data = self.module.weight.data
        self.module.weight.data = original_data.view(1, 4)
        self.assertFalse(guard.matches(self.module))
        self.module.weight.data = original_data
        self.assertTrue(guard.matches(self.module))
        self.module.weight.data = original_data.t()
        self.assertFalse(guard.matches(self.module))

    def test_in_place_weight_values_remain_supported(self):
        guard = self.make_guard()
        with torch.no_grad():
            self.module.weight.add_(1)
        self.assertTrue(guard.matches(self.module))

    def test_primitive_and_public_sequence_mutations_are_rejected(self):
        guard = self.make_guard()
        with patch.object(self.module, "normalized_shape", (1, 2)):
            self.assertFalse(guard.matches(self.module))
        self.module.layout[0] = 5
        self.assertFalse(guard.matches(self.module))
        self.module.layout[0] = 2
        self.assertTrue(guard.matches(self.module))
        self.module.new_public_flag = True
        self.assertFalse(guard.matches(self.module))

    def test_opaque_public_attribute_replacement_and_reclassification_are_rejected(
        self,
    ):
        self.module.extra_config = object()
        guard = self.make_guard()
        self.module.extra_config = object()
        self.assertFalse(guard.matches(self.module))
        self.module.extra_config = 1
        self.assertFalse(guard.matches(self.module))

    def test_bound_helper_and_dispatch_cannot_switch_owner(self):
        other = SignatureFixture().eval()
        self.module.split_qkv = MethodType(SignatureFixture.split_qkv, self.module)
        self.module._forward_method = self.module.forward
        guard = self.make_guard()
        self.assertTrue(guard.matches(self.module))
        original_helper = self.module.split_qkv
        self.module.split_qkv = other.split_qkv
        self.assertFalse(guard.matches(self.module))
        self.module.split_qkv = original_helper
        self.assertTrue(guard.matches(self.module))
        self.module._forward_method = other.forward
        self.assertFalse(guard.matches(self.module))

    def test_class_forward_helper_and_compiled_binding_changes_are_rejected(self):
        guard = self.make_guard()
        for name in ("forward", "split_qkv", "view_qkv_to_4d", "_compiled_call_impl"):
            with self.subTest(name=name):
                with patch.object(SignatureFixture, name, lambda *args: args):
                    self.assertFalse(guard.matches(self.module))
        self.assertTrue(guard.matches(self.module))

    def test_training_and_late_hooks_are_rejected(self):
        guard = self.make_guard()
        self.module.train()
        self.assertFalse(guard.matches(self.module))
        self.module.eval()
        self.assertTrue(guard.matches(self.module))
        hook = self.module.register_forward_pre_hook(lambda *args: None)
        try:
            self.assertFalse(guard.matches(self.module))
        finally:
            hook.remove()

    def test_quant_method_identity_and_apply_function_are_guarded(self):
        class UnquantizedLinearMethod:
            def apply(self, *args):
                return args

        self.module.quant_method = UnquantizedLinearMethod()
        guard = self.make_guard()
        with patch.object(UnquantizedLinearMethod, "apply", lambda *args: None):
            self.assertFalse(guard.matches(self.module))
        self.assertTrue(guard.matches(self.module))
        self.module.quant_method = UnquantizedLinearMethod()
        self.assertFalse(guard.matches(self.module))


if __name__ == "__main__":
    unittest.main()
