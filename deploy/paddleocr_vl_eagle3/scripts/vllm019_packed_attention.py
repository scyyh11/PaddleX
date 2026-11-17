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

"""Explicit finite-QKV packed FA2 for the pinned vLLM 0.19 B1/K4 service.

Import performs no installation and imports neither torch nor vLLM. Artifact
admission can run in the parent before CUDA is imported. Plans are strongly
owned until process exit and are prepared before target CUDA graph capture.
Unsupported layouts or absent plans retain the existing exact AR1 adapter.

The packed numerical scope requires finite Q/K/V, valid page IDs, and causal
endpoints within the configured bound. No per-call GPU scan enforces these
model/runner invariants. Nonfinite future V is a known counterexample; this is
not an arbitrary-BF16-bit-pattern equivalence claim. Plan buffers must only be
used by ordered calls, with explicit waits for any stream handoff.
"""
from collections import Counter
from functools import wraps
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

LIBRARY_ENV = "PADDLEOCR_VLLM019_PACKED_ATTENTION_LIBRARY"
SOURCE_NAMES = (
    "vllm019_packed_attention.py",
    "../kernels/fa2_sm80/build.py",
    "../kernels/fa2_sm80/packed_fa2.py",
    "../kernels/fa2_sm80/src/packed_fa2_v3.cu",
    "../kernels/fa2_sm80/src/packed_mask.h",
)
COUNTS = Counter()
PLANS = {}
_INSTALLATION = None
_ARTIFACT = None
_PLAN_TYPE = None


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_artifact(library_path, *, bundle_root=None):
    """Verify pinned local sources, upstream identities, manifest and binary on CPU."""
    root = Path(bundle_root or Path(__file__).resolve().parents[1] / "kernels/fa2_sm80")
    build = _load_source("_paddleocr_fa2_build_contract", root / "build.py")
    for name, expected in build.EXPECTED_SOURCE_HASHES.items():
        if build.sha256(root / "src" / name) != expected:
            raise ValueError(f"Validated packed FA2 source was changed: {name}")
    path = Path(library_path).expanduser().resolve(strict=True)
    manifest_path = path.with_name("build_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": build.SCHEMA_VERSION,
        "profile": "paddleocr-vl-fa2-sm80-v3-finite",
        "gpu_arch": "sm80",
        "nvcc_version": build.EXPECTED_NVCC_VERSION,
        "cuda_source_sha256": build.EXPECTED_SOURCE_HASHES,
        "patched_header_sha256": build.PATCHED_KERNEL_SHA256,
        "library_name": path.name,
    }
    if (
        any(manifest.get(key) != value for key, value in expected.items())
        or path.name != build.LIBRARY_NAME
        or manifest.get("torch_headers_version") not in ("2.10.0", "2.10.0+cu128")
    ):
        raise ValueError(
            "Packed attention artifact does not match the pinned FA2 profile"
        )
    for name, pin in build.SOURCES.items():
        actual = manifest.get("source_pins", {}).get(name, {})
        if any(actual.get(key) != pin[key] for key in ("commit", "sha256")):
            raise ValueError(f"Unexpected packed FA2 upstream source: {name}")
    library_hash = build.sha256(path)
    if library_hash != manifest.get("library_sha256"):
        raise ValueError("Packed attention library checksum mismatch")
    return {
        "library": str(path),
        "library_sha256": library_hash,
        "manifest_sha256": build.sha256(manifest_path),
        "profile": manifest["profile"],
    }


def validate_configuration(args, config, *, version):
    """Called after recovery config resolution, including in the parent process."""
    path = getattr(args, "vllm019_packed_attention_library", None)
    if path is None:
        return
    spec = config.get("speculative_config", {})
    if (
        version != "0.19.0"
        or getattr(args, "mode", None) not in ("eagle3-recovery", "accelerated")
        or getattr(args, "recovery_decode_graphs", None) != "int8"
        or config.get("worker_extension_cls")
        != "hps_vllm019_int8_worker.Vllm019Int8Worker"
        or config.get("max_num_seqs") != 1
        or config.get("tensor_parallel_size") != 1
        or config.get("max_model_len") not in (4096, 12288)
        or spec.get("method") != "eagle3"
        or spec.get("num_speculative_tokens") != 4
    ):
        raise ValueError(
            "--vllm019-packed-attention-library requires vLLM 0.19 recovery/int8, "
            "B1/K4/TP1 and context 4096 or 12288"
        )


def _fallback_reason(kw, output, kernel):
    import ctypes
    import torch

    if kernel is not None:
        return "explicit_kernel"
    q, k, v = (kw.get(name) for name in ("q", "k", "v"))
    cu, lengths, table = (
        kw.get(name) for name in ("cu_seqlens_q", "seqused_k", "block_table")
    )
    if not all(
        isinstance(t, torch.Tensor) for t in (q, k, v, cu, lengths, table, output)
    ):
        return "tensor_type"
    if kw.get("max_seqlen_k") not in (4096, 12288):
        return "context_bound"
    if (
        not q.is_cuda
        or q.ndim != 3
        or not 2 <= q.shape[0] <= 5
        or tuple(q.shape[1:]) != (16, 128)
        or output.shape != q.shape
        or k.ndim != 4
        or v.shape != k.shape
        or tuple(k.shape[1:]) != (16, 2, 128)
        or k.shape[0] < 1
        or any(
            t.device != q.device or t.dtype != torch.bfloat16 for t in (q, k, v, output)
        )
    ):
        return "qkv_layout"
    if (
        tuple(cu.shape) != (2,)
        or tuple(lengths.shape) != (1,)
        or table.ndim != 2
        or table.shape[0] != 1
        or table.shape[1] < kw["max_seqlen_k"] // 16
        or any(
            t.device != q.device or t.dtype != torch.int32 or t.stride(-1) != 1
            for t in (cu, lengths, table)
        )
    ):
        return "metadata_layout"
    if any(
        t.stride(-1) != 1
        or t.data_ptr() % 16
        or any(s <= 0 or s % 8 for s in t.stride()[:-1])
        for t in (q, k, v, output)
    ):
        return "alignment"
    if any(
        output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
        for t in (q, k, v)
    ):
        return "output_alias"
    scale = kw.get("softmax_scale")
    if scale is not None and (
        type(scale) not in (float, int)
        or ctypes.c_float(float(scale)).value != ctypes.c_float(128**-0.5).value
    ):
        return "softmax_scale"
    return None


def _plan_key(queries, bound, columns, device):
    return (queries, bound, columns, device)


def _make_attention(original):
    @wraps(original)
    def attention(kw, output, kernel=None):
        reason = _fallback_reason(kw, output, kernel)
        plan = None
        if reason is None:
            q, table = kw["q"], kw["block_table"]
            key = _plan_key(q.shape[0], kw["max_seqlen_k"], table.shape[1], q.device)
            plan = PLANS.get(key)
            if plan is None:
                reason = "unprepared_plan"
        if reason is not None:
            COUNTS["ar1_fallback_calls"] += 1
            COUNTS[f"fallback_{reason}"] += 1
            return original(kw, output, kernel)
        result = plan(
            kw["q"],
            kw["k"],
            kw["v"],
            kw["cu_seqlens_q"],
            kw["seqused_k"],
            kw["block_table"],
            out=output,
            softmax_scale=kw.get("softmax_scale"),
        )
        COUNTS["packed_calls"] += 1
        return result

    return attention


def _prepare_runner(runner):
    config = runner.vllm_config
    spec = config.speculative_config
    if (
        spec is None
        or spec.method != "eagle3"
        or spec.num_speculative_tokens != 4
        or config.model_config.max_model_len not in (4096, 12288)
        or config.scheduler_config.max_num_seqs != 1
    ):
        raise RuntimeError(
            "Packed attention worker requires B1/K4 and context 4096 or 12288"
        )
    table = runner.input_batch.block_table[0].block_table.gpu
    queries, bound = spec.num_speculative_tokens + 1, config.model_config.max_model_len
    key = _plan_key(queries, bound, table.shape[1], table.device)
    if key not in PLANS:
        PLANS[key] = _PLAN_TYPE(
            queries,
            bound,
            table.shape[1],
            device=table.device,
            library_path=_ARTIFACT["library"],
        )
        COUNTS["prepared_plans"] += 1


def _make_capture(original):
    @wraps(original)
    def capture(runner, *args, **kwargs):
        _prepare_runner(runner)
        return original(runner, *args, **kwargs)

    return capture


def install(library_path=None):
    global _INSTALLATION, _ARTIFACT, _PLAN_TYPE
    path = library_path or os.environ.get(LIBRARY_ENV)
    if path is None:
        return metadata()
    artifact = verify_artifact(path)
    if _INSTALLATION is not None:
        if artifact != _ARTIFACT:
            raise RuntimeError(
                "Cannot change packed attention library while graphs are alive"
            )
        return metadata()
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("Packed attention requires vLLM 0.19.0")
    import vllm019_ar1_attention as attention
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    root = Path(__file__).resolve().parents[1] / "kernels/fa2_sm80"
    adapter = _load_source("_paddleocr_packed_fa2", root / "packed_fa2.py")
    original_attention, original_capture = (
        attention._batched_attention,
        GPUModelRunner.capture_model,
    )
    _PLAN_TYPE = adapter.PackedFA2
    _ARTIFACT = artifact
    attention._batched_attention = _make_attention(original_attention)
    GPUModelRunner.capture_model = _make_capture(original_capture)
    _INSTALLATION = (
        adapter,
        attention,
        GPUModelRunner,
        original_attention,
        original_capture,
    )
    return metadata()


def metadata():
    return {
        "enabled": _INSTALLATION is not None,
        "artifact": _ARTIFACT,
        "scope": "finite BF16 Q/K/V; B1/K4/TP1; A100 80GB SM80/108SM; context 4096 or 12288",
        "metadata_contract": "valid page IDs and causal endpoints within the configured bound",
        "fallback": "original exact FA2 AR1 for unsupported layout/bound or unprepared plan",
        "counter_scope": "Python startup/capture/eager calls; graph replays do not increment counters",
        "counts": dict(COUNTS),
        "plans": [plan.metadata() for plan in PLANS.values()],
    }
