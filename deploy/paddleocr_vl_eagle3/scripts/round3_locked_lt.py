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

"""Explicit cuBLASLt plans reproducing logged AR algorithms (experimental)."""
import ctypes as C
from pathlib import Path
import importlib.metadata
import sysconfig
import torch


def cublaslt_path():
    """Resolve the pinned CUDA 13 library from this Python environment."""
    distribution = importlib.metadata.distribution("nvidia-cublas")
    if distribution.version != "13.1.1.3":
        raise RuntimeError(
            "The validated cuBLASLt recipes require nvidia-cublas==13.1.1.3"
        )
    relative = Path("nvidia/cu13/lib/libcublasLt.so.13")
    candidates = [
        Path(distribution.locate_file(relative)),
        Path(sysconfig.get_path("purelib")) / relative,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Cannot find libcublasLt.so.13 in the nvidia-cublas installation"
    )


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


class LockedLt:
    def __init__(
        self,
        weight,
        algo_id=6,
        tile=18,
        stages=12,
        split_k=2,
        reduction=1,
        swizzle=1,
        custom_option=None,
        workspace_bytes=8 * 1024**2,
        row_batch=False,
    ):
        assert (
            weight.is_cuda and weight.dtype == torch.bfloat16 and weight.is_contiguous()
        )
        self.weight = weight
        self.o, self.k = weight.shape
        self.lib = C.CDLL(str(cublaslt_path()))
        P = C.c_void_p
        I = C.c_int
        S = C.c_size_t
        signatures = {
            "cublasLtCreate": [C.POINTER(P)],
            "cublasLtMatmulDescCreate": [C.POINTER(P), I, I],
            "cublasLtMatmulDescSetAttribute": [P, I, P, S],
            "cublasLtMatrixLayoutCreate": [
                C.POINTER(P),
                I,
                C.c_uint64,
                C.c_uint64,
                C.c_int64,
            ],
            "cublasLtMatrixLayoutSetAttribute": [P, I, P, S],
            "cublasLtMatmulAlgoInit": [P, I, I, I, I, I, I, I, C.POINTER(Algo)],
            "cublasLtMatmulAlgoConfigSetAttribute": [C.POINTER(Algo), I, P, S],
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
            f = getattr(self.lib, name)
            f.argtypes = args
            f.restype = I
        self.handle = P()
        self.check(self.lib.cublasLtCreate(C.byref(self.handle)))
        self.desc = P()
        self.check(self.lib.cublasLtMatmulDescCreate(C.byref(self.desc), 68, 0))
        trans = C.c_int(1)
        self.check(
            self.lib.cublasLtMatmulDescSetAttribute(
                self.desc, 3, C.byref(trans), C.sizeof(trans)
            )
        )
        sm = C.c_int(
            torch.cuda.get_device_properties(weight.device).multi_processor_count
        )
        self.check(
            self.lib.cublasLtMatmulDescSetAttribute(
                self.desc, 15, C.byref(sm), C.sizeof(sm)
            )
        )
        self.algo = Algo()
        self.check(
            self.lib.cublasLtMatmulAlgoInit(
                self.handle, 68, 0, 14, 14, 14, 14, algo_id, C.byref(self.algo)
            )
        )
        self.row_batch = row_batch
        self.parameters = dict(
            algo_id=algo_id,
            tile=tile,
            stages=stages,
            split_k=split_k,
            reduction=reduction,
            swizzle=swizzle,
            custom_option=custom_option,
            row_batch=row_batch,
        )
        for attr, value in [
            (1, tile),
            (6, stages),
            (2, split_k),
            (3, reduction),
            (4, swizzle),
            (5, custom_option),
        ]:
            if value is None:
                continue
            v = C.c_uint32(value)
            self.check(
                self.lib.cublasLtMatmulAlgoConfigSetAttribute(
                    C.byref(self.algo), attr, C.byref(v), C.sizeof(v)
                )
            )
        self.workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=weight.device
        )
        self.plans = {}
        self.alpha = C.c_float(1)
        self.beta = C.c_float(0)

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"cuBLASLt status={status}")

    def prepare(self, n):
        if n not in self.plans:
            layouts = []
            cols_n = 1 if self.row_batch else n
            for rows, cols, ld in [
                (self.k, self.o, self.k),
                (self.k, cols_n, self.k),
                (self.o, cols_n, self.o),
            ]:
                p = C.c_void_p()
                self.check(
                    self.lib.cublasLtMatrixLayoutCreate(C.byref(p), 14, rows, cols, ld)
                )
                layouts.append(p)
            if self.row_batch and n > 1:
                count = C.c_int32(n)
                for layout, stride in zip(layouts, [0, self.k, self.o]):
                    offset = C.c_int64(stride)
                    self.check(
                        self.lib.cublasLtMatrixLayoutSetAttribute(
                            layout, 5, C.byref(count), C.sizeof(count)
                        )
                    )
                    self.check(
                        self.lib.cublasLtMatrixLayoutSetAttribute(
                            layout, 6, C.byref(offset), C.sizeof(offset)
                        )
                    )
            a, b, c = layouts
            result = Heuristic()
            self.check(
                self.lib.cublasLtMatmulAlgoCheck(
                    self.handle,
                    self.desc,
                    a,
                    b,
                    c,
                    c,
                    C.byref(self.algo),
                    C.byref(result),
                )
            )
            self.check(result.state)
            if result.workspaceSize > self.workspace.numel():
                raise ValueError("Insufficient preallocated workspace")
            self.plans[n] = (a, b, c, result.workspaceSize)
        return self.plans[n]

    def __call__(self, x, out=None):
        assert (
            x.is_contiguous() and x.dtype == self.weight.dtype and x.shape[1] == self.k
        )
        n = x.shape[0]
        a, b, c, size = self.prepare(n)
        if out is None:
            out = torch.empty((n, self.o), dtype=x.dtype, device=x.device)
        self.check(
            self.lib.cublasLtMatmul(
                self.handle,
                self.desc,
                C.byref(self.alpha),
                self.weight.data_ptr(),
                a,
                x.data_ptr(),
                b,
                C.byref(self.beta),
                out.data_ptr(),
                c,
                out.data_ptr(),
                c,
                C.byref(self.algo),
                self.workspace.data_ptr(),
                size,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
        )
        return out
