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

"""Explicit CUDA 12 cuBLASLt BF16 plans for isolated numeric experiments.

No recipe is supplied or claimed to match native Q1. Pass algorithm attributes
obtained from this exact cuBLAS version, then compare against native rowwise
F.linear on current weights/inputs. AlgoCheck confirms descriptor support,
not numerical equivalence. This module installs no vLLM hooks and changes no
PyTorch backend/precision options.

prepare(n) creates layouts and checks the algorithm before CUDA graph capture.
Keep this object (including its weight/workspace), input, and output storage
alive for every graph that references them. One plan's workspace may only be
used by ordered, nonconcurrent calls. close() is explicit: the caller must first
finish outstanding work and retire all graphs using the object.
"""

import ctypes as C
from dataclasses import dataclass
import importlib.metadata
from pathlib import Path
import sysconfig

import torch


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class Heuristic(C.Structure):
    _fields_ = [
        ("algo", Algo),
        ("workspaceSize", C.c_size_t),
        ("state", C.c_int),
        ("wavesCount", C.c_float),
        ("reserved", C.c_int * 4),
    ]


# CUDA 12.8 cublasLtMatmulAlgoConfigAttributes_t, including exact value widths.
_ATTRIBUTES = {
    "tile": (1, C.c_uint32),
    "split_k": (2, C.c_int32),
    "reduction": (3, C.c_uint32),
    "swizzle": (4, C.c_uint32),
    "custom_option": (5, C.c_uint32),
    "stages": (6, C.c_uint32),
    "inner_shape": (7, C.c_uint16),
    "cluster_shape": (8, C.c_uint16),
}


def _integer(value, name, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return value


def normalize_recipe(recipe):
    if not isinstance(recipe, dict) or "algo_id" not in recipe:
        raise ValueError("An explicit recipe dictionary with algo_id is required")
    unknown = set(recipe) - {*_ATTRIBUTES, "algo_id", "row_batch", "sm_count_target"}
    if unknown:
        raise ValueError(f"Unsupported recipe fields: {sorted(unknown)}")
    result = {"algo_id": _integer(recipe["algo_id"], "algo_id", 2**31 - 1)}
    for name, (_, ctype) in _ATTRIBUTES.items():
        value = recipe.get(name)
        if value is not None:
            bits = C.sizeof(ctype) * 8 - int(ctype is C.c_int32)
            result[name] = _integer(value, name, 2**bits - 1)
    row_batch = recipe.get("row_batch", False)
    if type(row_batch) is not bool:
        raise ValueError("row_batch must be bool")
    result["row_batch"] = row_batch
    sm_count = recipe.get("sm_count_target")
    result["sm_count_target"] = (
        None if sm_count is None else _integer(sm_count, "sm_count_target", 2**31 - 1)
    )
    return result


def cublaslt_path(library_path=None):
    if library_path is not None:
        candidate = Path(library_path)
        if (
            not candidate.name.startswith("libcublasLt.so.12")
            or not candidate.is_file()
        ):
            raise ValueError("library_path must point to an existing libcublasLt.so.12")
        return candidate.resolve()
    distribution = importlib.metadata.distribution("nvidia-cublas-cu12")
    if not distribution.version.startswith("12."):
        raise RuntimeError("This adapter requires the current CUDA 12 cuBLAS package")
    relative = Path("nvidia/cublas/lib/libcublasLt.so.12")
    for candidate in (
        Path(distribution.locate_file(relative)),
        Path(sysconfig.get_path("purelib")) / relative,
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("libcublasLt.so.12 was not found in this Python environment")


def _load_library(path):
    lib = C.CDLL(str(path))
    P, I, S = C.c_void_p, C.c_int, C.c_size_t
    signatures = {
        "cublasLtCreate": [C.POINTER(P)],
        "cublasLtDestroy": [P],
        "cublasLtMatmulDescCreate": [C.POINTER(P), I, I],
        "cublasLtMatmulDescDestroy": [P],
        "cublasLtMatmulDescSetAttribute": [P, I, P, S],
        "cublasLtMatrixLayoutCreate": [
            C.POINTER(P),
            I,
            C.c_uint64,
            C.c_uint64,
            C.c_int64,
        ],
        "cublasLtMatrixLayoutDestroy": [P],
        "cublasLtMatrixLayoutSetAttribute": [P, I, P, S],
        "cublasLtMatmulAlgoInit": [P, I, I, I, I, I, I, I, C.POINTER(Algo)],
        "cublasLtMatmulAlgoConfigSetAttribute": [C.POINTER(Algo), I, P, S],
        "cublasLtMatmulAlgoCapGetAttribute": [C.POINTER(Algo), I, P, S, C.POINTER(S)],
        "cublasLtMatmulAlgoCheck": [
            P,
            P,
            P,
            P,
            P,
            P,
            C.POINTER(Algo),
            C.POINTER(Heuristic),
        ],
        "cublasLtMatmul": [
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            P,
            C.POINTER(Algo),
            P,
            S,
            P,
        ],
    }
    for name, args in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, I
    lib.cublasLtGetVersion.argtypes, lib.cublasLtGetVersion.restype = [], S
    version = int(lib.cublasLtGetVersion())
    if version // 10000 != 12:
        raise RuntimeError(f"Expected CUDA 12 cuBLASLt; loaded version {version}")
    return lib


class LtError(RuntimeError):
    def __init__(self, operation, status):
        self.operation, self.status = operation, int(status)
        super().__init__(f"{operation} failed: cuBLASLt status={self.status}")


def _check(status, operation):
    if status:
        raise LtError(operation, status)


def _layout_specs(k, o, n, row_batch):
    if type(n) is not int or not 1 <= n <= 2**31 - 1:
        raise ValueError("n must be a positive int32 row count")
    cols, count = (1, n) if row_batch else (n, 1)
    return [(k, o, k, count, 0), (k, cols, k, count, k), (o, cols, o, count, o)]


@dataclass(frozen=True)
class _Plan:
    a: C.c_void_p
    b: C.c_void_p
    c: C.c_void_p
    workspace_bytes: int
    waves_count: float
    rows: int


def _bf16_matrix(value, name, device=None, shape=None):
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_cuda
        or value.dtype != torch.bfloat16
        or value.ndim != 2
        or not value.is_contiguous()
        or any(size <= 0 for size in value.shape)
        or (device is not None and value.device != device)
        or (shape is not None and tuple(value.shape) != tuple(shape))
    ):
        raise ValueError(
            f"{name} must be a contiguous CUDA BF16 matrix of the required shape/device"
        )


def _overlap(left, right):
    a, b = left.data_ptr(), right.data_ptr()
    return (
        a < b + right.numel() * right.element_size()
        and b < a + left.numel() * left.element_size()
    )


class LockedLt:
    def __init__(
        self, weight, recipe, *, workspace_bytes=8 * 1024**2, library_path=None
    ):
        self.parameters = normalize_recipe(recipe)
        _integer(workspace_bytes, "workspace_bytes", 2**63 - 1)
        _bf16_matrix(weight, "weight")
        self.weight, self.device = weight, weight.device
        self.o, self.k = weight.shape
        self.weight_ptr = weight.data_ptr()
        self.row_batch = self.parameters["row_batch"]
        self.library_path = cublaslt_path(library_path)
        self.lib = _load_library(self.library_path)
        self.library_version = int(self.lib.cublasLtGetVersion())
        self.handle, self.desc = C.c_void_p(), C.c_void_p()
        self.algo, self.plans = Algo(), {}
        self.workspace = None
        self.closed = False
        self.alpha, self.beta = C.c_float(1.0), C.c_float(0.0)
        try:
            with torch.cuda.device(self.device):
                _check(self.lib.cublasLtCreate(C.byref(self.handle)), "Create")
                _check(
                    self.lib.cublasLtMatmulDescCreate(C.byref(self.desc), 68, 0),
                    "MatmulDescCreate",
                )
                self._desc_attribute(2, C.c_int32(0))  # host alpha/beta
                self._desc_attribute(3, C.c_int32(1))  # transpose A
                if self.parameters["sm_count_target"] is not None:
                    self._desc_attribute(
                        15, C.c_int32(self.parameters["sm_count_target"])
                    )
                _check(
                    self.lib.cublasLtMatmulAlgoInit(
                        self.handle,
                        68,
                        0,
                        14,
                        14,
                        14,
                        14,
                        self.parameters["algo_id"],
                        C.byref(self.algo),
                    ),
                    "MatmulAlgoInit",
                )
                for name, (attribute, ctype) in _ATTRIBUTES.items():
                    if name in self.parameters:
                        value = ctype(self.parameters[name])
                        _check(
                            self.lib.cublasLtMatmulAlgoConfigSetAttribute(
                                C.byref(self.algo),
                                attribute,
                                C.byref(value),
                                C.sizeof(value),
                            ),
                            f"AlgoConfigSetAttribute({name})",
                        )
                self.min_alignments = []
                for attribute in (16, 17, 18, 19):
                    value, written = C.c_uint32(), C.c_size_t()
                    _check(
                        self.lib.cublasLtMatmulAlgoCapGetAttribute(
                            C.byref(self.algo),
                            attribute,
                            C.byref(value),
                            C.sizeof(value),
                            C.byref(written),
                        ),
                        f"AlgoCapGetAttribute({attribute})",
                    )
                    if written.value != C.sizeof(value) or value.value == 0:
                        raise RuntimeError(
                            "cuBLASLt returned an invalid alignment capability"
                        )
                    self.min_alignments.append(value.value)
                self.workspace = torch.empty(
                    workspace_bytes, dtype=torch.uint8, device=self.device
                )
        except Exception:
            # Preserve the creation/configuration error if cleanup also fails.
            try:
                self.close()
            except Exception:
                pass
            raise

    def _desc_attribute(self, attribute, value):
        _check(
            self.lib.cublasLtMatmulDescSetAttribute(
                self.desc,
                attribute,
                C.byref(value),
                C.sizeof(value),
            ),
            f"MatmulDescSetAttribute({attribute})",
        )

    def prepare(self, n):
        if self.closed:
            raise RuntimeError("LockedLt is closed")
        specs = _layout_specs(self.k, self.o, n, self.row_batch)
        if n in self.plans:
            return self.plans[n]
        layouts = []
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Call prepare(n) before CUDA graph capture")
            try:
                for rows, cols, ld, count, stride in specs:
                    layout = C.c_void_p()
                    _check(
                        self.lib.cublasLtMatrixLayoutCreate(
                            C.byref(layout), 14, rows, cols, ld
                        ),
                        "MatrixLayoutCreate",
                    )
                    layouts.append(layout)
                    if self.row_batch and n > 1:
                        for attribute, value in (
                            (5, C.c_int32(count)),
                            (6, C.c_int64(stride)),
                        ):
                            _check(
                                self.lib.cublasLtMatrixLayoutSetAttribute(
                                    layout,
                                    attribute,
                                    C.byref(value),
                                    C.sizeof(value),
                                ),
                                f"MatrixLayoutSetAttribute({attribute})",
                            )
                a, b, c = layouts
                result = Heuristic()
                _check(
                    self.lib.cublasLtMatmulAlgoCheck(
                        self.handle,
                        self.desc,
                        a,
                        b,
                        c,
                        c,
                        C.byref(self.algo),
                        C.byref(result),
                    ),
                    "MatmulAlgoCheck",
                )
                _check(result.state, "MatmulAlgoCheck.result.state")
                if result.workspaceSize > self.workspace.numel():
                    raise ValueError(
                        f"Algorithm requires {result.workspaceSize} workspace bytes; allocated {self.workspace.numel()}"
                    )
                plan = _Plan(
                    a, b, c, int(result.workspaceSize), float(result.wavesCount), n
                )
            except Exception:
                for layout in reversed(layouts):
                    self.lib.cublasLtMatrixLayoutDestroy(layout)
                raise
        self.plans[n] = plan
        return plan

    def __call__(self, x, out=None):
        if self.closed:
            raise RuntimeError("LockedLt is closed")
        _bf16_matrix(self.weight, "weight", self.device, (self.o, self.k))
        if self.weight.data_ptr() != self.weight_ptr:
            raise ValueError("Weight storage changed after the plan was constructed")
        _bf16_matrix(x, "x", self.device)
        n = x.shape[0]
        if x.shape[1] != self.k:
            raise ValueError("Input K dimension does not match weight")
        if out is not None:
            _bf16_matrix(out, "out", self.device, (n, self.o))
            if _overlap(out, x) or _overlap(out, self.weight):
                raise ValueError("Output must not overlap input or weight storage")
        with torch.cuda.device(self.device):
            plan = self.prepare(n)
            if out is None:
                out = torch.empty((n, self.o), dtype=torch.bfloat16, device=self.device)
            for name, tensor, alignment, ld_bytes in zip(
                ("A", "B", "C", "D"),
                (self.weight, x, out, out),
                self.min_alignments,
                (self.k * 2, self.k * 2, self.o * 2, self.o * 2),
            ):
                # Row-batch strides are either zero (A) or exactly these LDs.
                if tensor.data_ptr() % alignment or ld_bytes % alignment:
                    raise ValueError(
                        f"{name} storage/leading dimension violates {alignment}-byte algorithm alignment"
                    )
            stream = torch.cuda.current_stream(self.device)
            # Raw ctypes launches are invisible to PyTorch's stream tracking.
            # Graph replay still requires explicit owners for all these buffers.
            for tensor in (self.weight, x, out, self.workspace):
                tensor.record_stream(stream)
            _check(
                self.lib.cublasLtMatmul(
                    self.handle,
                    self.desc,
                    C.byref(self.alpha),
                    self.weight.data_ptr(),
                    plan.a,
                    x.data_ptr(),
                    plan.b,
                    C.byref(self.beta),
                    out.data_ptr(),
                    plan.c,
                    out.data_ptr(),
                    plan.c,
                    C.byref(self.algo),
                    self.workspace.data_ptr() if plan.workspace_bytes else None,
                    plan.workspace_bytes,
                    stream.cuda_stream,
                ),
                "Matmul",
            )
        return out

    def metadata(self):
        return {
            "library_path": str(self.library_path),
            "library_version": self.library_version,
            "recipe": dict(self.parameters),
            "weight_shape": [self.o, self.k],
            "workspace_capacity": (
                0 if self.workspace is None else self.workspace.numel()
            ),
            "minimum_alignment_bytes": list(self.min_alignments),
            "plans": {
                str(n): {
                    "workspace_bytes": plan.workspace_bytes,
                    "waves_count": plan.waves_count,
                }
                for n, plan in self.plans.items()
            },
            "closed": self.closed,
            "numeric_equivalence_validated": False,
        }

    def close(self):
        """Release native resources after callers retire all in-flight work/graphs."""
        if self.closed:
            return
        self.closed = True
        failures = []
        with torch.cuda.device(self.device):
            for plan in self.plans.values():
                for layout in (plan.c, plan.b, plan.a):
                    status = self.lib.cublasLtMatrixLayoutDestroy(layout)
                    if status:
                        failures.append(("MatrixLayoutDestroy", status))
            self.plans.clear()
            for name, descriptor in (
                ("MatmulDescDestroy", self.desc),
                ("Destroy", self.handle),
            ):
                if descriptor.value:
                    status = getattr(self.lib, "cublasLt" + name)(descriptor)
                    if status:
                        failures.append((name, status))
                    descriptor.value = None
            self.workspace = None
        if failures:
            raise LtError(*failures[0])
