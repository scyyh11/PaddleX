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

"""Isolated finite-input FA2 row-packing prototype; not a production fallback.

PackedFA2 preallocates CUDA workspaces. Calls accept original target Q/K/V and
GPU-only causal metadata; no GPU scalar is read by the host. The default output
is persistent and overwritten by the next call: clone it if retaining a result.
Every tensor/plan must outlive captured graphs. One plan supports ordered calls
across streams with explicit caller-provided waits; concurrent reuse is invalid.
The caller must retain this plan and all graph-bound tensors until every captured
graph using them is retired. record_stream protects in-flight allocator lifetime,
not future replays after the caller destroys a graph input.

Scope: BF16 finite Q/K/V, A100 SM80/108SM, Hq16/Hkv2/D128, page16,
Q2..Q7, context bound4096 or12288. Caller-provided GPU endpoints must stay
within the bound, and page IDs must address the supplied cache. Nonfinite future V is a known counterexample:
this research kernel deliberately makes no arbitrary-BF16-bit-pattern claim.
"""
from pathlib import Path
import ctypes as C
import hashlib
import importlib.metadata
import json

import torch

_PROFILE = "paddleocr-vl-fa2-sm80-v3-finite"
_SOURCE_HASHES = {
    "packed_fa2_v3.cu": "dff47d6c8c7ca7a66005cdbca8f6d4040351a2314b06adf58691d308d4c290bc",
    "packed_mask.h": "1f05457bfa4ff85105fe4ae9efae1ad31231c3551b7013171b05cfac0c2051eb",
}
_HEADER_HASH = "f400f3ef44174f6397ed20a6e53d346eac4367dc591d04aa7e989abea4de423c"
_SOURCE_PINS = {
    "flash-attention": (
        "29210221863736a08f71a866459e368ad1ac4a95",
        "fe98fd6aa07e30203ee0bb9619cb53f30703f71c22bbfa91909aa3b43b7018f1",
    ),
    "cutlass": (
        "62750a2b75c802660e4894434dc55e839f322277",
        "78816d6c6d97793b5b59ef2a702174cb85b78dfcefc8fe2489964de2e42f17d2",
    ),
}


def _verify_artifact(library_path):
    """CPU-only construction-time check; never run inside a captured forward."""
    path = Path(library_path).resolve()
    manifest = json.loads(path.with_name("build_manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("profile") != _PROFILE
        or manifest.get("gpu_arch") != "sm80"
        or manifest.get("nvcc_version") != "12.8.93"
        or manifest.get("torch_headers_version") not in ("2.10.0", "2.10.0+cu128")
        or manifest.get("cuda_source_sha256") != _SOURCE_HASHES
        or manifest.get("patched_header_sha256") != _HEADER_HASH
        or manifest.get("library_name") != path.name
    ):
        raise ValueError("The built library does not match the validated FA2 profile")
    for name, (commit, archive_hash) in _SOURCE_PINS.items():
        pin = manifest.get("source_pins", {}).get(name, {})
        if pin.get("commit") != commit or pin.get("sha256") != archive_hash:
            raise ValueError(f"Unexpected upstream source pin: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("library_sha256"):
        raise ValueError("Built FA2 library checksum mismatch")
    return manifest


def _record_stream(tensors, stream):
    # ctypes bypasses the dispatcher, so explicitly tell the caching allocator
    # which ordered stream uses every input/output/workspace. Do not lock the
    # plan to one stream: warmup/capture/replay may use different ordered streams.
    for tensor in tensors:
        tensor.record_stream(stream)


class PackedFA2:
    def __init__(
        self,
        queries,
        max_seqlen_k,
        block_table_columns,
        device="cuda",
        *,
        library_path=None,
    ):
        if type(queries) is not int or not 2 <= queries <= 7:
            raise ValueError("queries must be an integer in 2..7")
        if max_seqlen_k not in (4096, 12288):
            raise ValueError("Only native AR1 bounds4096/12288 are supported")
        if (
            type(block_table_columns) is not int
            or block_table_columns < max_seqlen_k // 16
        ):
            raise ValueError("The page table must cover the full context bound")
        self.library_path = Path(
            library_path
            or Path(__file__).parent / "build" / "libpaddleocr_vl_fa2_sm80.so"
        ).resolve()
        self.build_manifest = _verify_artifact(self.library_path)
        if (
            torch.__version__ != "2.10.0+cu128"
            or torch.version.cuda != "12.8"
            or importlib.metadata.version("vllm") != "0.19.0"
        ):
            raise RuntimeError(
                "This profile requires PyTorch2.10.0+cu128 and vLLM0.19.0"
            )
        self.queries, self.max_seqlen_k, self.columns = (
            queries,
            max_seqlen_k,
            block_table_columns,
        )
        self.splits = max_seqlen_k // 128
        device = torch.device(device)
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Construct PackedFA2 before CUDA graph capture")
            props = torch.cuda.get_device_properties(device)
            if (
                (props.major, props.minor, props.multi_processor_count) != (8, 0, 108)
                or "A100" not in props.name
                or props.total_memory < 79 * 1024**3
            ):
                raise ValueError("The experiment is restricted to A100 SM80/108SM")
            self.output = torch.empty(
                (queries, 16, 128), device=device, dtype=torch.bfloat16
            )
            self.device = self.output.device
            self.packed_q = torch.empty(
                (queries * 8, 2, 128), device=device, dtype=torch.bfloat16
            )
            self.packed_out = torch.empty_like(self.packed_q)
            self.endpoints = torch.empty(queries, device=device, dtype=torch.int32)
            self.maximum = torch.empty(1, device=device, dtype=torch.int32)
            self.lse = torch.empty((2, queries * 8), device=device, dtype=torch.float32)
            self.partial_lse = torch.empty(
                (self.splits, 1, 2, queries * 8), device=device, dtype=torch.float32
            )
            self.partial_out = torch.empty(
                (self.splits, 1, 2, queries * 8, 128),
                device=device,
                dtype=torch.float32,
            )
            self.lib = C.CDLL(str(self.library_path))
            P, I = C.c_void_p, C.c_int
            self.lib.packed_fa2_prepare.argtypes = []
            self.lib.packed_fa2_prepare.restype = I
            self.lib.packed_fa2_smem.argtypes = []
            self.lib.packed_fa2_smem.restype = I
            self.lib.packed_fa2_launch.argtypes = (
                [P] * 14 + [C.POINTER(C.c_int64)] + [I] * 4 + [C.c_float, P]
            )
            self.lib.packed_fa2_launch.restype = I
            result = self.lib.packed_fa2_prepare()
            if result:
                raise RuntimeError(f"packed_fa2_prepare CUDA status={result}")
            self.shared_memory_bytes = self.lib.packed_fa2_smem()
        self.workspace_bytes = sum(
            t.numel() * t.element_size()
            for t in (
                self.packed_q,
                self.packed_out,
                self.endpoints,
                self.maximum,
                self.lse,
                self.partial_lse,
                self.partial_out,
            )
        )

    def _tensor(self, tensor, name, dtype, shape=None):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != self.device
            or tensor.dtype != dtype
        ):
            raise ValueError(f"{name} must have dtype{dtype} on {self.device}")
        if shape is not None and tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} shape must be {shape}, got{tuple(tensor.shape)}")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have contiguous last dimension")

    def __call__(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        out=None,
        softmax_scale=None,
    ):
        out = self.output if out is None else out
        for tensor, name in ((q, "q"), (out, "out")):
            self._tensor(tensor, name, torch.bfloat16, (self.queries, 16, 128))
        for tensor, name in ((k, "k"), (v, "v")):
            self._tensor(tensor, name, torch.bfloat16)
            if (
                tensor.ndim != 4
                or tuple(tensor.shape[1:]) != (16, 2, 128)
                or tensor.shape[0] < 1
            ):
                raise ValueError(f"{name} must have shape[blocks,16,2,128]")
        if k.shape != v.shape:
            raise ValueError("k/v shapes differ")
        self._tensor(cu_seqlens_q, "cu_seqlens_q", torch.int32, (2,))
        self._tensor(seqused_k, "seqused_k", torch.int32, (1,))
        self._tensor(block_table, "block_table", torch.int32, (1, self.columns))
        for tensor, name in ((q, "q"), (k, "k"), (v, "v"), (out, "out")):
            if tensor.data_ptr() % 16 or any(
                stride <= 0 or stride % 8 for stride in tensor.stride()[:-1]
            ):
                raise ValueError(f"{name} requires16-byte-aligned rows/heads")
        if any(
            out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
            for tensor in (q, k, v)
        ):
            raise ValueError("out must not alias Q/K/V storage")
        strides = (C.c_int64 * 10)(
            q.stride(0),
            q.stride(1),
            *k.stride()[:3],
            *v.stride()[:3],
            out.stride(0),
            out.stride(1),
        )
        tensors = (
            q,
            k,
            v,
            cu_seqlens_q,
            seqused_k,
            block_table,
            self.packed_q,
            self.packed_out,
            out,
            self.endpoints,
            self.maximum,
            self.lse,
            self.partial_lse,
            self.partial_out,
        )
        scale = 128**-0.5 if softmax_scale is None else float(softmax_scale)
        if C.c_float(scale).value != C.c_float(128**-0.5).value:
            raise ValueError(
                "This target-specific prototype requires the native head128 softmax scale"
            )
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            _record_stream(tensors, stream)
            status = self.lib.packed_fa2_launch(
                *(tensor.data_ptr() for tensor in tensors),
                strides,
                self.queries,
                self.max_seqlen_k,
                16,
                self.columns,
                scale,
                stream.cuda_stream,
            )
        if status:
            raise RuntimeError(f"packed_fa2_launch CUDA status={status}")
        return out

    def metadata(self):
        return {
            "scope": "experimental finite Q/K/V only",
            "queries": self.queries,
            "max_seqlen_k": self.max_seqlen_k,
            "splits": self.splits,
            "shared_memory_bytes": self.shared_memory_bytes,
            "workspace_bytes": self.workspace_bytes,
            "library": str(self.library_path),
            "library_sha256": hashlib.sha256(
                self.library_path.read_bytes()
            ).hexdigest(),
            "build_manifest": self.build_manifest,
        }
