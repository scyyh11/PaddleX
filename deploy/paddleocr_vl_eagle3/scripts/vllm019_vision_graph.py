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

"""Opt-in exact-shape CUDA graphs for the native Siglip layer stack.

Only the final 27-layer loop is replaced. The pinned native metadata prefix
still computes this invocation's actual rotary values, cu_seqlens and CUDA max.
Every replay copies hidden, rotary and cu into fixed buffers. Actual max is read
once and included in the cache key because the native custom op captures it as
a Python integer. There is no padding or image/result cache.

Each graph has an independent pool, a strong encoder owner, and fixed inputs.
Calls must use the stream on which the cache was first prepared. Unsupported,
nested, training, outer-capture, unprepared and capacity-limited calls use native
layers. Cache entries are never evicted. Outputs are independent clones. The
fixed production scope is B1/TP1, BF16 H16/D72, 27 native layers, FA2, N<=2048,
and a complete sequence whose actual maximum equals N. Cache limits are 32
graphs and 1 GiB, with 1536 MiB free before attempting a capture.

Cold construction runs warmup, capture and a raw-bit comparison on actual input;
audit_changed_inputs optionally adds changed hidden/rotary checks. These synchronize
and are reported separately. torch.cuda.graph also synchronizes/empties unused
allocator cache. No GPU work or monkey patch occurs on import.
"""

import ast
from collections import Counter
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib.metadata
import inspect
import textwrap
import threading
import time
from types import SimpleNamespace

import torch

import vllm019_vision_scalar as scalar

NATIVE_FORWARD_SHA256 = (
    "69117f1188c810d9b74246952a72f97f4e802bfbca41ec09c6d077d79ddf6ca5"
)
MAX_GRAPHS = 32
MAX_CACHED_BYTES = 1024**3
MAX_SEQUENCE_LENGTH = 2048
MINIMUM_REUSE = 2
MINIMUM_FREE_BYTES = 1536 * 1024**2
_STATE = None
_ACTIVE = ContextVar("vision_encoder_graph_active", default=False)


def _tensor_key(tensor):
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _cache_key(encoder, hidden, rotary, cu, max_value):
    return (
        id(encoder),
        _tensor_key(hidden),
        _tensor_key(rotary),
        _tensor_key(cu),
        int(max_value),
    )


def _native_stack(encoder, hidden, cu, rotary, maximum):
    for encoder_layer in encoder.layers:
        hidden = encoder_layer(
            hidden, cu_seqlens=cu, rotary_pos_emb=rotary, max_seqlen=maximum
        )
    return hidden


def _replace_native_loop(native_forward, dispatch):
    """Keep the entire native prefix AST; replace only its exact final loop."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(native_forward)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError("Expected one plain Python native forward")
    function = tree.body[0]
    expected = ast.parse(
        "for encoder_layer in self.layers:\n"
        "    hidden_states = encoder_layer(\n"
        "        hidden_states, cu_seqlens=cu_seqlens,\n"
        "        rotary_pos_emb=rotary_pos_emb, max_seqlen=max_seqlen)\n"
    ).body[0]
    if (
        function.decorator_list
        or len(function.body) < 2
        or ast.dump(function.body[-2]) != ast.dump(expected)
        or ast.dump(function.body[-1])
        != ast.dump(ast.parse("return hidden_states").body[0])
    ):
        raise RuntimeError("Native encoder layer-loop structure changed")
    replacement = ast.parse(
        "hidden_states = _vision_graph_dispatch("
        "self, hidden_states, cu_seqlens, rotary_pos_emb, max_seqlen)"
    ).body[0]
    function.body[-2] = ast.copy_location(replacement, function.body[-2])
    ast.fix_missing_locations(tree)
    # A new globals dictionary avoids changing the installed vLLM module.
    namespace = dict(native_forward.__globals__)
    namespace["_vision_graph_dispatch"] = dispatch
    exec(compile(tree, "<vision_encoder_graph_native_prefix>", "exec"), namespace)
    return namespace[function.name]


def _inputs_eligible(hidden, rotary, cu, maximum):
    return (
        all(isinstance(t, torch.Tensor) for t in (hidden, rotary, cu, maximum))
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and hidden.ndim == 3
        and hidden.shape[0] == 1
        and hidden.shape[1] > 0
        and hidden.shape[2] == 1152
        and hidden.is_contiguous()
        and rotary.device == hidden.device
        and rotary.dtype == torch.float32
        and tuple(rotary.shape) == (hidden.shape[1], 36)
        and rotary.is_contiguous()
        and cu.device == hidden.device
        and cu.dtype == torch.int32
        and tuple(cu.shape) == (2,)
        and cu.is_contiguous()
        and maximum.device == hidden.device
        and maximum.dtype == torch.int32
        and maximum.ndim == 0
    )


def _module_signature(encoder):
    """CPU-only immutable-dispatch/storage guard; no tensor value transfers."""
    signature = []
    for module in encoder.modules():
        if (
            module.training
            or module._forward_hooks
            or module._forward_pre_hooks
            or getattr(module, "_compiled_call_impl", None) is not None
            or "forward" in vars(module)
        ):
            return None
        method = module.forward
        dispatch = getattr(module, "_forward_method", None)
        signature.append(
            (
                id(module),
                type(module),
                getattr(method, "__func__", None),
                getattr(method, "__self__", None) is module,
                getattr(dispatch, "__func__", None),
                dispatch is None or getattr(dispatch, "__self__", None) is module,
                tuple(
                    sorted(
                        (name, value)
                        for name, value in vars(module).items()
                        if not name.startswith("_")
                        and type(value) in (str, int, float, bool, type(None))
                    )
                ),
            )
        )
        for name in (
            "split_qkv",
            "view_qkv_to_4d",
            "_forward_fa",
            "forward_cuda",
            "forward_native",
            "forward_hip",
            "forward_xpu",
        ):
            if name not in vars(module) and name not in type(module).__dict__:
                continue
            helper = getattr(module, name, None)
            if helper is not None:
                signature.append(
                    (
                        name,
                        getattr(helper, "__func__", helper),
                        getattr(helper, "__self__", None) is module,
                    )
                )
        signature.append(
            tuple(
                sorted(
                    (name, tuple(value))
                    for name, value in vars(module).items()
                    if not name.startswith("_")
                    and type(value) in (tuple, list)
                    and all(
                        type(item) in (str, int, float, bool, type(None))
                        for item in value
                    )
                )
            )
        )
        quant = getattr(module, "quant_method", None)
        if quant is not None:
            if type(quant).__name__ != "UnquantizedLinearMethod":
                return None
            signature.append(
                (id(quant), type(quant), getattr(quant.apply, "__func__", None))
            )
    for name, tensor in list(encoder.named_parameters()) + list(
        encoder.named_buffers()
    ):
        signature.append((name, id(tensor), tensor.data_ptr(), _tensor_key(tensor)))
    return tuple(signature)


_PRIMITIVE_TYPES = (str, int, float, bool, type(None))
_HELPER_NAMES = (
    "split_qkv",
    "view_qkv_to_4d",
    "_forward_fa",
    "forward_cuda",
    "forward_native",
    "forward_hip",
    "forward_xpu",
)
_MISSING = object()


class FastModuleGuard:
    """Preflatten immutable bindings; hot checks never recurse or build a signature.

    Captured bindings and storage metadata are retained. This rejects changed module dict
    keysets, opaque public attribute identities, and registration-map identities,
    even if a change would be harmless. In-place weight value updates remain
    allowed: graphs read the same storage, and no tensor version is consulted.
    """

    def __init__(self, encoder):
        baseline = _module_signature(encoder)
        if baseline is None:
            raise RuntimeError("Encoder is not an immutable native inference module")
        self.encoder = encoder
        self.baseline_signature = baseline
        self.module_records = []
        self.class_records = []
        self.quant_records = []
        self.quant_classes = []
        self.mapping_records = []
        self.tensor_records = []
        classes, quant_classes, tensors = set(), set(), set()
        # Recursion, sorting and string conversion occur only in this cold setup.
        for module in encoder.modules():
            cls = type(module)
            data = vars(module)
            if cls not in classes:
                classes.add(cls)
                self.class_records.append(
                    (
                        cls,
                        frozenset(cls.__dict__),
                        cls.forward,
                        getattr(cls, "_compiled_call_impl", None),
                        getattr(cls, "_forward_method", None),
                        tuple(
                            (name, cls.__dict__.get(name, _MISSING))
                            for name in _HELPER_NAMES
                        ),
                    )
                )
            scalars, sequences, opaque, opaque_sequences = [], [], [], []
            for name, value in data.items():
                if name.startswith("_"):
                    continue
                if type(value) in _PRIMITIVE_TYPES:
                    scalars.append((name, value))
                elif type(value) in (tuple, list):
                    if all(type(item) in _PRIMITIVE_TYPES for item in value):
                        sequences.append((name, tuple(value)))
                    else:
                        opaque_sequences.append((name, value))
                else:
                    opaque.append((name, value))
            dispatch = data.get("_forward_method", None)
            dispatch_record = (
                getattr(dispatch, "__func__", None),
                dispatch is None or getattr(dispatch, "__self__", None) is module,
            )
            instance_helpers = tuple(
                (
                    name,
                    data[name],
                    getattr(data[name], "__func__", data[name]),
                    getattr(data[name], "__self__", None) is module,
                )
                for name in _HELPER_NAMES
                if name in data
            )
            self.module_records.append(
                (
                    module,
                    cls,
                    data,
                    frozenset(data),
                    tuple(scalars),
                    tuple(sequences),
                    tuple(opaque),
                    tuple(opaque_sequences),
                    dispatch_record,
                    instance_helpers,
                )
            )
            for name in ("_modules", "_parameters", "_buffers"):
                mapping = data[name]
                self.mapping_records.append(
                    (data, name, mapping, tuple(mapping.items()))
                )
                if name == "_modules":
                    continue
                for tensor in mapping.values():
                    if tensor is None or id(tensor) in tensors:
                        continue
                    tensors.add(id(tensor))
                    self.tensor_records.append(
                        (
                            tensor,
                            tensor.data_ptr(),
                            tensor.shape,
                            tensor.stride(),
                            tensor.dtype,
                            tensor.device,
                        )
                    )
            quant = data.get("quant_method", None)
            if quant is not None:
                quant_cls = type(quant)
                self.quant_records.append(
                    (
                        data,
                        quant,
                        quant_cls,
                        vars(quant),
                        vars(quant).get("apply", _MISSING),
                    )
                )
                if quant_cls not in quant_classes:
                    quant_classes.add(quant_cls)
                    self.quant_classes.append((quant_cls, quant_cls.apply))
        self.module_count = len(self.module_records)
        self.tensor_count = len(self.tensor_records)
        if not self.matches(encoder):
            raise RuntimeError("Cannot flatten this encoder's native method bindings")

    def matches(self, encoder):
        if encoder is not self.encoder:
            return False
        try:
            for (
                cls,
                keys,
                forward,
                compiled,
                class_dispatch,
                helpers,
            ) in self.class_records:
                if (
                    cls.__dict__.keys() != keys
                    or cls.forward is not forward
                    or getattr(cls, "_compiled_call_impl", None) is not compiled
                    or getattr(cls, "_forward_method", None) is not class_dispatch
                ):
                    return False
                for name, expected in helpers:
                    if cls.__dict__.get(name, _MISSING) is not expected:
                        return False
            for (
                module,
                cls,
                data,
                keys,
                scalars,
                sequences,
                opaque,
                opaque_sequences,
                dispatch_record,
                instance_helpers,
            ) in self.module_records:
                if (
                    type(module) is not cls
                    or vars(module) is not data
                    or data.keys() != keys
                    or data["training"]
                    or data["_forward_hooks"]
                    or data["_forward_pre_hooks"]
                    or data.get("_compiled_call_impl", None) is not None
                    or "forward" in data
                ):
                    return False
                dispatch = data.get("_forward_method", None)
                if (
                    getattr(dispatch, "__func__", None) is not dispatch_record[0]
                    or (
                        dispatch is None
                        or getattr(dispatch, "__self__", None) is module
                    )
                    != dispatch_record[1]
                ):
                    return False
                for name, expected in scalars:
                    value = data[name]
                    if type(value) not in _PRIMITIVE_TYPES or value != expected:
                        return False
                for name, expected in sequences:
                    value = data[name]
                    if (
                        type(value) not in (tuple, list)
                        or not all(type(item) in _PRIMITIVE_TYPES for item in value)
                        or tuple(value) != expected
                    ):
                        return False
                for name, expected in opaque:
                    if data[name] is not expected:
                        return False
                for name, expected in opaque_sequences:
                    if data[name] is not expected or all(
                        type(item) in _PRIMITIVE_TYPES for item in expected
                    ):
                        return False
                for name, raw, function, bound in instance_helpers:
                    helper = data[name]
                    if (
                        helper is not raw
                        or getattr(helper, "__func__", helper) is not function
                        or (getattr(helper, "__self__", None) is module) != bound
                    ):
                        return False
            for data, name, mapping, expected_items in self.mapping_records:
                if data.get(name) is not mapping or len(mapping) != len(expected_items):
                    return False
                # Registration order matters for ModuleList and recursive named
                # parameter traversal; equal unordered maps are insufficient.
                for (actual_name, actual), (expected_name, expected) in zip(
                    mapping.items(), expected_items
                ):
                    if actual_name != expected_name or actual is not expected:
                        return False
            for data, quant, cls, quant_data, apply_override in self.quant_records:
                if (
                    data.get("quant_method") is not quant
                    or type(quant) is not cls
                    or vars(quant) is not quant_data
                    or quant_data.get("apply", _MISSING) is not apply_override
                ):
                    return False
            for cls, apply in self.quant_classes:
                if cls.apply is not apply:
                    return False
            for tensor, pointer, shape, stride, dtype, device in self.tensor_records:
                if (
                    tensor.data_ptr() != pointer
                    or tensor.shape != shape
                    or tensor.stride() != stride
                    or tensor.dtype != dtype
                    or tensor.device != device
                ):
                    return False
        except (AttributeError, KeyError, TypeError, RuntimeError):
            return False
        return True


def _native_encoder_eligible(encoder, state):
    return (
        not torch.is_grad_enabled()
        and not torch.is_autocast_enabled("cuda")
        and state.guards._encoder_eligible(encoder, state)
        and (
            getattr(state.cache, "fast_guard", None) is not None
            or all(not module.training for module in encoder.modules())
        )
    )


class LayerStackPlan:
    """One exact metadata shape/max; no weights or arithmetic are rewritten."""

    def __init__(self, encoder, hidden, cu, rotary, maximum, *, warmups=2):
        started = time.perf_counter()
        self.encoder = encoder
        self.layers = tuple(encoder.layers)
        self.device = hidden.device
        self.owner_stream = torch.cuda.current_stream(self.device)
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.signature = _module_signature(encoder)
        if self.signature is None:
            raise RuntimeError("Encoder is not an immutable native inference module")
        allocated_before = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        self.hidden = torch.empty_like(hidden)
        self.cu = torch.empty_like(cu)
        self.rotary = torch.empty_like(rotary)
        self.maximum = torch.tensor(maximum, dtype=torch.int32, device="cpu")
        self.key = _cache_key(encoder, hidden, rotary, cu, maximum)
        self.copy_inputs(hidden, cu, rotary)
        self.capture_stream.wait_stream(self.owner_stream)
        with torch.cuda.stream(self.capture_stream):
            for _ in range(warmups):
                _native_stack(encoder, self.hidden, self.cu, self.rotary, self.maximum)
        self.owner_stream.wait_stream(self.capture_stream)
        # torch.cuda.graph itself empties unused allocator cache. Establish the
        # graph-pool baseline after warmup cleanup so unrelated freed blocks do
        # not subtract from this graph's retained private-pool footprint.
        self.capture_stream.synchronize()
        torch.cuda.empty_cache()
        reserved_before_capture = torch.cuda.memory_reserved(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.output = _native_stack(
                encoder, self.hidden, self.cu, self.rotary, self.maximum
            )
        self.owner_stream.wait_stream(self.capture_stream)
        self.static_input_bytes = sum(
            t.numel() * t.element_size() for t in (self.hidden, self.cu, self.rotary)
        )
        self.allocated_delta = max(
            0, torch.cuda.memory_allocated(self.device) - allocated_before
        )
        self.reserved_delta = max(
            0, torch.cuda.memory_reserved(self.device) - reserved_before
        )
        self.graph_pool_reserved_delta = max(
            0, torch.cuda.memory_reserved(self.device) - reserved_before_capture
        )
        self.accounted_bytes = max(
            self.allocated_delta,
            self.graph_pool_reserved_delta + self.static_input_bytes,
        )
        self.cold_capture_seconds = time.perf_counter() - started
        self.audit_records = []
        self.replay_calls = 0
        self.audit_replay_calls = 0

    def copy_inputs(self, hidden, cu, rotary):
        self.hidden.copy_(hidden)
        self.cu.copy_(cu)
        self.rotary.copy_(rotary)

    def replay(self, hidden, cu, rotary, *, borrow_output=False):
        self.copy_inputs(hidden, cu, rotary)
        # Capture records work; the first normal call must also replay it.
        self.graph.replay()
        self.replay_calls += 1
        return self.output if borrow_output else self.output.clone()

    def audit(self, hidden, cu, rotary, *, changed_replays=0):
        """Explicit GPU gate on this encoder's actual loaded weights.

        Actual metadata/image features are checked first. Optional changed finite
        hidden/rotary inputs test stale-buffer errors; original inputs are then
        restored. This gate synchronizes and is excluded from steady replay cost.
        """
        started = time.perf_counter()
        generator = torch.Generator(device=self.device).manual_seed(190190)
        records = []
        for step in range(changed_replays + 1):
            if step == 0:
                test_hidden, test_rotary = hidden, rotary
            else:
                test_hidden = torch.randn(
                    hidden.shape,
                    dtype=hidden.dtype,
                    device=self.device,
                    generator=generator,
                )
                test_rotary = torch.randn(
                    rotary.shape,
                    dtype=rotary.dtype,
                    device=self.device,
                    generator=generator,
                )
            expected = _native_stack(
                self.encoder, test_hidden, cu, test_rotary, self.maximum
            )
            actual = self.replay(test_hidden, cu, test_rotary, borrow_output=True)
            self.audit_replay_calls += 1
            exact = torch.equal(
                actual.contiguous().view(torch.int16),
                expected.contiguous().view(torch.int16),
            )
            record = {
                "phase": "actual" if step == 0 else "changed_hidden_rotary",
                "step": step,
                "raw_bits_exact": exact,
            }
            if not exact:
                record["different_elements"] = int(
                    (actual.view(torch.int16) != expected.view(torch.int16)).sum()
                )
                record["max_abs"] = float(
                    (actual.float() - expected.float()).abs().nan_to_num().max()
                )
            records.append(record)
        self.audit_records.extend(records)
        self.audit_seconds = time.perf_counter() - started
        # Never return a synthetic audit result as an image embedding.
        self.copy_inputs(hidden, cu, rotary)
        return all(row["raw_bits_exact"] for row in records)

    def metadata(self):
        return {
            "hidden": list(self.hidden.shape),
            "rotary": list(self.rotary.shape),
            "cu": list(self.cu.shape),
            "max_seqlen": self.maximum.item(),
            "static_input_bytes": self.static_input_bytes,
            "allocated_delta": self.allocated_delta,
            "reserved_delta": self.reserved_delta,
            "graph_pool_reserved_delta": self.graph_pool_reserved_delta,
            "accounted_bytes": self.accounted_bytes,
            "cold_capture_seconds": self.cold_capture_seconds,
            "capture_ms": self.cold_capture_seconds * 1000,
            "replays": self.replay_calls - self.audit_replay_calls,
            "audit_replays": self.audit_replay_calls,
            "audit_seconds": getattr(self, "audit_seconds", 0),
            "audits": self.audit_records,
            "pool": "independent",
            "owner_stream": self.owner_stream.cuda_stream,
        }


class LayerStackCache:
    def __init__(
        self,
        max_graphs=MAX_GRAPHS,
        max_cached_bytes=MAX_CACHED_BYTES,
        borrow_output=False,
        minimum_reuse=2,
        audit_replays=0,
        allow_capture=True,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        minimum_free_bytes=MINIMUM_FREE_BYTES,
    ):
        if type(max_graphs) is not int or not 1 <= max_graphs <= 32:
            raise ValueError("max_graphs must be in 1..32")
        if (
            type(max_cached_bytes) is not int
            or not 0 < max_cached_bytes <= MAX_CACHED_BYTES
        ):
            raise ValueError("max_cached_bytes must be positive and at most 1 GiB")
        if type(minimum_reuse) is not int or minimum_reuse < 1:
            raise ValueError("minimum_reuse must be a positive integer")
        if type(audit_replays) is not int or not 0 <= audit_replays <= 8:
            raise ValueError("audit_replays must be in 0..8")
        if (
            type(max_sequence_length) is not int
            or not 0 < max_sequence_length <= MAX_SEQUENCE_LENGTH
        ):
            raise ValueError("max_sequence_length must be in 1..2048")
        if borrow_output:
            raise ValueError(
                "Production encoder graphs require independent output clones"
            )
        if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes must be nonnegative")
        self.max_graphs = max_graphs
        self.max_cached_bytes = max_cached_bytes
        self.borrow_output = bool(borrow_output)
        self.minimum_reuse = minimum_reuse
        self.audit_replays = audit_replays
        self.allow_capture = bool(allow_capture)
        self.max_sequence_length = max_sequence_length
        self.minimum_free_bytes = minimum_free_bytes
        self.plans = {}
        self.seen = Counter()
        self.failed = set()
        self.counts = Counter()
        self.lock = threading.Lock()
        self.owner_stream = None
        self.owner_device = None
        self.encoder = None
        self.signature = None
        self.fast_guard = None
        self.cache_bytes = 0
        self.budget_exhausted = False
        self.capture_failed = False
        self.rejected_plan_records = []

    def fallback(self, reason, encoder, hidden, cu, rotary, maximum):
        self.counts["fallback_" + reason] += 1
        return _native_stack(encoder, hidden, cu, rotary, maximum)

    def run(self, encoder, hidden, cu, rotary, maximum):
        if self.capture_failed:
            raise RuntimeError(
                "Vision graph capture previously failed; restart the worker before retrying"
            )
        self.counts["stack_calls"] += 1
        if not _inputs_eligible(hidden, rotary, cu, maximum):
            return self.fallback("inputs", encoder, hidden, cu, rotary, maximum)
        if torch.cuda.is_current_stream_capturing():
            return self.fallback("outer_capture", encoder, hidden, cu, rotary, maximum)
        # Every eligible native fallback retains scalar-only's one GPU read.
        value = maximum.item()
        self.counts["gpu_scalar_reads"] += 1
        cpu_max = torch.tensor(value, dtype=torch.int32, device="cpu")
        if not 0 < value <= hidden.shape[1]:
            return self.fallback("max_range", encoder, hidden, cu, rotary, cpu_max)
        if value != hidden.shape[1]:
            return self.fallback(
                "incomplete_sequence", encoder, hidden, cu, rotary, cpu_max
            )
        if hidden.shape[1] > self.max_sequence_length:
            return self.fallback(
                "sequence_length", encoder, hidden, cu, rotary, cpu_max
            )
        if not self.lock.acquire(blocking=False):
            return self.fallback("busy", encoder, hidden, cu, rotary, cpu_max)
        try:
            stream = torch.cuda.current_stream(hidden.device)
            if self.owner_stream is not None and (
                hidden.device != self.owner_device or stream != self.owner_stream
            ):
                return self.fallback("stream", encoder, hidden, cu, rotary, cpu_max)
            if self.fast_guard is None:
                signature = _module_signature(encoder)
                if signature is None:
                    return self.fallback(
                        "module_changed", encoder, hidden, cu, rotary, cpu_max
                    )
                self.fast_guard = FastModuleGuard(encoder)
                self.encoder = encoder
                self.signature = self.fast_guard.baseline_signature
                self.counts["guard_initializations"] += 1
            if not self.fast_guard.matches(encoder):
                return self.fallback(
                    "module_changed", encoder, hidden, cu, rotary, cpu_max
                )
            key = _cache_key(encoder, hidden, rotary, cu, value)
            plan = self.plans.get(key)
            if plan is None:
                self.counts["cache_misses"] += 1
                if key in self.failed:
                    return self.fallback(
                        "failed_validation", encoder, hidden, cu, rotary, cpu_max
                    )
                if len(self.plans) >= self.max_graphs or self.budget_exhausted:
                    return self.fallback(
                        "capacity", encoder, hidden, cu, rotary, cpu_max
                    )
                if key not in self.seen and len(self.seen) >= 64:
                    return self.fallback(
                        "tracked_shape_limit", encoder, hidden, cu, rotary, cpu_max
                    )
                self.seen[key] += 1
                if not self.allow_capture or self.seen[key] < self.minimum_reuse:
                    return self.fallback(
                        "unprepared", encoder, hidden, cu, rotary, cpu_max
                    )
                free_bytes, _ = torch.cuda.mem_get_info(hidden.device)
                input_bytes = sum(
                    t.numel() * t.element_size() for t in (hidden, cu, rotary)
                )
                if free_bytes < self.minimum_free_bytes + input_bytes:
                    return self.fallback(
                        "free_memory", encoder, hidden, cu, rotary, cpu_max
                    )
                self.counts["cold_captures"] += 1
                try:
                    plan = LayerStackPlan(encoder, hidden, cu, rotary, value)
                except Exception as exc:
                    # A failed CUDA capture may leave the stream/context unsafe.
                    # Never mask it by launching native work on that context.
                    self.capture_failed = True
                    self.counts["capture_errors"] += 1
                    raise RuntimeError(
                        "Native vision graph capture failed; restart the worker or disable the opt-in"
                    ) from exc
                if self.cache_bytes + plan.accounted_bytes > self.max_cached_bytes:
                    # No eviction/capture thrashing. Actual measurement requires
                    # one transient capture; report this rejected graph's cost.
                    self.budget_exhausted = True
                    self.rejected_plan_records.append(plan.metadata())
                    return self.fallback(
                        "memory_budget", encoder, hidden, cu, rotary, cpu_max
                    )
                try:
                    exact = plan.audit(
                        hidden, cu, rotary, changed_replays=self.audit_replays
                    )
                except Exception as exc:
                    self.capture_failed = True
                    self.counts["graph_execution_errors"] += 1
                    raise RuntimeError(
                        "Native vision graph validation could not complete; restart the worker or disable the opt-in"
                    ) from exc
                if not exact:
                    self.failed.add(key)
                    self.rejected_plan_records.append(plan.metadata())
                    return self.fallback(
                        "failed_validation", encoder, hidden, cu, rotary, cpu_max
                    )
                self.plans[key] = plan
                self.cache_bytes += plan.accounted_bytes
                self.owner_stream, self.owner_device = stream, hidden.device
            else:
                self.counts["cache_hits"] += 1
            self.counts["replays"] += 1
            try:
                return plan.replay(hidden, cu, rotary, borrow_output=self.borrow_output)
            except Exception as exc:
                self.capture_failed = True
                self.counts["graph_execution_errors"] += 1
                raise RuntimeError(
                    "Native vision graph replay failed; restart the worker or disable the opt-in"
                ) from exc
        finally:
            self.lock.release()

    def metadata(self):
        return {
            "graphs": len(self.plans),
            "max_graphs": self.max_graphs,
            "cache_bytes": self.cache_bytes,
            "max_cached_bytes": self.max_cached_bytes,
            "budget_exhausted": self.budget_exhausted,
            "capture_failed": self.capture_failed,
            "minimum_reuse": self.minimum_reuse,
            "allow_capture": self.allow_capture,
            "max_sequence_length": self.max_sequence_length,
            "minimum_free_bytes": self.minimum_free_bytes,
            "borrow_output": self.borrow_output,
            "clone_output": True,
            "actual_raw_bit_admission": True,
            "audit_changed_inputs": self.audit_replays,
            "eviction": False,
            "pool": "independent; shared pools deliberately disabled",
            "module_guard": "preflattened class/method/instance/ordered-map/storage checks; no hot recursive signature",
            "memory_limit_note": "Measured post-capture; one rejected capture may transiently exceed the cache budget",
            "counts": dict(self.counts),
            "shapes_seen": [
                {
                    "hidden": list(key[1][2]),
                    "max_seqlen": key[-1],
                    "miss_observations": count,
                }
                for key, count in self.seen.items()
            ],
            "plans": [plan.metadata() for plan in self.plans.values()],
            "rejected_plans": self.rejected_plan_records,
        }


def install(*, audit_changed_inputs=0):
    """Explicit worker opt-in; extra audits are diagnostics, never timed replay."""
    global _STATE
    if type(audit_changed_inputs) is not int or not 0 <= audit_changed_inputs <= 8:
        raise ValueError("audit_changed_inputs must be an integer in 0..8")
    if _STATE is not None:
        if not (
            _STATE.guards._owned_hooks_intact(_STATE)
            and _STATE.guards._native_methods_intact(_STATE)
        ):
            raise RuntimeError("Encoder graph hook or native methods were replaced")
        return metadata()
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("Vision encoder graphs require vLLM 0.19.0")
    from vllm.model_executor.models import paddleocr_vl as model
    from vllm.model_executor.layers.attention import mm_encoder_attention as mm
    from vllm.model_executor import custom_op
    from vllm.v1.attention.ops import vit_attn_wrappers as wrappers
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    guards = scalar
    previous_forward = model.SiglipEncoder.forward
    native_forward = previous_forward
    scalar_module = scalar
    prior = getattr(scalar_module, "_INSTALLATION", None)
    if prior is not None:
        if not (
            scalar_module._owned_hooks_intact(prior)
            and scalar_module._native_methods_intact(prior)
        ):
            raise RuntimeError("Existing scalar optimization is not intact")
        native_forward = prior.original_forward
    elif mm.vit_flash_attn_wrapper is not wrappers.vit_flash_attn_wrapper:
        raise RuntimeError("Unrecognized native attention wrapper")
    native_source = textwrap.dedent(inspect.getsource(native_forward)).rstrip() + "\n"
    if hashlib.sha256(native_source.encode()).hexdigest() != NATIVE_FORWARD_SHA256:
        raise RuntimeError(
            "Native encoder source differs from the frozen reviewed prefix"
        )
    state = SimpleNamespace(
        guards=guards,
        cache=LayerStackCache(
            max_graphs=MAX_GRAPHS,
            max_cached_bytes=MAX_CACHED_BYTES,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            minimum_reuse=MINIMUM_REUSE,
            minimum_free_bytes=MINIMUM_FREE_BYTES,
            audit_replays=audit_changed_inputs,
            borrow_output=False,
        ),
        encoder_type=model.SiglipEncoder,
        layer_type=model.SiglipEncoderLayer,
        attention_type=model.SiglipAttention,
        mm_type=mm.MMEncoderAttention,
        attention_module=mm,
        wrappers_module=wrappers,
        backend=AttentionBackendEnum.FLASH_ATTN,
        original_forward=native_forward,
        previous_forward=native_forward,
        original_attention=wrappers.vit_flash_attn_wrapper,
        wrapped_attention=guards._wrap_attention(wrappers.vit_flash_attn_wrapper),
        layer_forward=model.SiglipEncoderLayer.forward,
        self_attention_forward=model.SiglipAttention.forward,
        mm_forward=mm.MMEncoderAttention.forward,
        mm_forward_cuda=mm.MMEncoderAttention.forward_cuda,
        mm_forward_fa=mm.MMEncoderAttention._forward_fa,
        scalar_module_to_restore=scalar_module if prior is not None else None,
    )
    expected = (
        (native_forward, model, "SiglipEncoder.forward"),
        (state.layer_forward, model, "SiglipEncoderLayer.forward"),
        (state.self_attention_forward, model, "SiglipAttention.forward"),
        (state.mm_forward, custom_op, "CustomOp.forward"),
        (state.mm_forward_cuda, mm, "MMEncoderAttention.forward_cuda"),
        (state.mm_forward_fa, mm, "MMEncoderAttention._forward_fa"),
        (state.original_attention, wrappers, "vit_flash_attn_wrapper"),
    )
    if not all(guards._native_python_function(*entry) for entry in expected):
        raise RuntimeError("Native vision methods have been patched")
    prefix = _replace_native_loop(native_forward, state.cache.run)
    fallback = guards._wrap_encoder(
        native_forward, lambda encoder: guards._encoder_eligible(encoder, state)
    )

    @wraps(native_forward)
    def forward(encoder, *args, **kwargs):
        if state.cache.capture_failed:
            raise RuntimeError(
                "Vision graph capture previously failed; restart the worker before retrying"
            )
        if _ACTIVE.get() or not _native_encoder_eligible(encoder, state):
            state.cache.counts["encoder_fallbacks"] += 1
            return fallback(encoder, *args, **kwargs)
        token = _ACTIVE.set(True)
        scalar_token = guards._ACTIVE.set(guards._ScalarScope())
        try:
            return prefix(encoder, *args, **kwargs)
        finally:
            guards._ACTIVE.reset(scalar_token)
            _ACTIVE.reset(token)

    state.wrapped_forward = forward
    if prior is not None:
        scalar_module.uninstall()
    mm.vit_flash_attn_wrapper = state.wrapped_attention
    model.SiglipEncoder.forward = forward
    _STATE = state
    return metadata()


def metadata():
    if _STATE is None:
        return {"installed": False}
    return {
        "installed": True,
        "enabled": _STATE.guards._owned_hooks_intact(_STATE)
        and _STATE.guards._native_methods_intact(_STATE)
        and not _STATE.cache.capture_failed,
        "scope": "native Siglip 27-layer stack; B1 TP1 BF16 H16 D72 FA2; exact N<=2048, actual maximum=N",
        "native_prefix": "unchanged AST before final layer loop; actual rotary/cu/max every call",
        "output": "independent clone included in replay cost",
        "scalar_takeover": "scalar hook explicitly uninstalled; vision graph adapter owns the same scalar-only fallback",
        "scalar_fallback_counts": dict(_STATE.guards.COUNTS),
        "native_forward_sha256": NATIVE_FORWARD_SHA256,
        "admission": "mandatory raw-bit comparison against native layers on actual input",
        **_STATE.cache.metadata(),
    }


def uninstall():
    global _STATE
    if _STATE is None:
        return
    if _STATE.cache.capture_failed:
        raise RuntimeError(
            "Vision graph execution failed; restart the worker instead of restoring native hooks"
        )
    if not _STATE.guards._owned_hooks_intact(_STATE):
        raise RuntimeError("Refusing to replace a later encoder hook")
    # Keep cache/graphs alive until queued work finishes before releasing them.
    if _STATE.cache.owner_stream is not None:
        _STATE.cache.owner_stream.synchronize()
    _STATE.encoder_type.forward = _STATE.previous_forward
    _STATE.attention_module.vit_flash_attn_wrapper = _STATE.original_attention
    restore_scalar = _STATE.scalar_module_to_restore
    _STATE = None
    if restore_scalar is not None:
        restore_scalar.install()
