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

"""Portable configuration for the frozen VL-1.6 EAGLE3 service.

This replaces the development CLI's dependency on experiment manifests and
precompiled benchmark binaries. The service continues to use vLLM's native
image processor and the same decode-only CUDA graph worker.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    os.environ.get("OCR_EAGLE_TARGET", str(ROOT / "models/PaddleOCR-VL-1.6"))
).expanduser()
DEFAULT_DRAFT = Path(
    os.environ.get(
        "OCR_EAGLE_DRAFT", str(ROOT / "models/PaddleOCR-VL-1.6-EAGLE3-K4-dense")
    )
).expanduser()
DEFAULT_EXTENSION = "hps_decode_graph_int8_worker.DecodeGraphInt8Worker"
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}
PINNED_VERSIONS = {
    "vllm": "0.32.0",
    "torch": "2.13.0+cu130",
    "transformers": "5.19.0",
    "triton": "3.7.1",
    "nvidia-cublas": "13.1.1.3",
}
PINNED_VERSIONS_019 = {
    "vllm": "0.19.0",
    "torch": "2.10.0",
    "transformers": "4.57.6",
    "triton": "3.6.0",
    "nvidia-cublas-cu12": "12.8.4.1",
    "pydantic": "2.14.3",
}
RUNTIME_SOURCES = (
    "runtime_config.py",
    "serve_ocr.py",
    "serve_hps_vlm.py",
    "native_dflash_target.py",
    "round3_attention_rows.py",
    "round3_fast_engine_locked_target.py",
    "round3_locked_lt.py",
    "round3_eagle_int8_head.py",
    "round3_eagle_fused_int8_head.py",
    "round3_eagle_fused_int8_top1.py",
    "hps_native_eagle_worker.py",
    "hps_decode_graph_eagle_worker.py",
    "hps_decode_graph_int8_worker.py",
)


def runtime_version():
    version = importlib.metadata.version("vllm")
    if version not in ("0.19.0", "0.32.0"):
        raise RuntimeError(f"Unsupported vLLM runtime: {version}")
    return version


def is_vllm019():
    return runtime_version() == "0.19.0"


def engine_config(args):
    if args.mode not in ("baseline", "accelerated", "eagle3-recovery"):
        raise ValueError("Unsupported service mode")
    if args.mode != "baseline" and args.k != 4:
        raise ValueError("The validated speculative worker requires K=4")
    directories = [args.model] + ([] if args.mode == "baseline" else [args.draft_model])
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {directory}")
    config = dict(
        model=str(args.model.resolve()),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        seed=args.seed,
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.45),
        kv_cache_memory_bytes=getattr(args, "kv_cache_memory_mib", 4096) * 1024 * 1024,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        enforce_eager=False,
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=False,
        max_cudagraph_capture_size=80,
    )
    if not is_vllm019():
        config["per_request_spec_decode_metrics"] = (
            "none" if args.mode == "baseline" else "summary"
        )
    if args.mode != "baseline":
        config["worker_extension_cls"] = args.worker_extension
        config["speculative_config"] = dict(
            method="eagle3",
            model=str(args.draft_model.resolve()),
            num_speculative_tokens=args.k,
            use_local_argmax_reduction=True,
        )
        if not is_vllm019():
            config["speculative_config"].update(
                draft_sample_method="greedy", enable_adaptive_verification=False
            )
    return config


def configure_environment(args, config):
    # This runs before CUDA/vLLM imports, including imports in spawned workers.
    legacy = is_vllm019()
    pins = PINNED_VERSIONS_019 if legacy else PINNED_VERSIONS
    versions = {name: importlib.metadata.version(name) for name in pins}
    mismatches = {
        name: (pins[name], actual)
        for name, actual in versions.items()
        if actual != pins[name]
    }
    if mismatches:
        raise RuntimeError(
            f"Unvalidated runtime versions (expected, actual): {mismatches}"
        )
    source_names = RUNTIME_SOURCES + (
        (
            "vllm019_model.py",
            "hps_vllm019_worker.py",
            "hps_vllm019_int8_worker.py",
            "vllm019_ar1_attention.py",
        )
        if legacy
        else ()
    )
    sources = {
        name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
        for name in source_names
    }
    basis = dict(
        engine_config=config,
        runtime_package_versions=versions,
        current_dependency_sha256=sources,
    )
    fingerprint = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
    cache = (
        getattr(args, "cache_root", None) or ROOT / "runtime_cache" / fingerprint[:20]
    ).resolve()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.update(
        OMP_WAIT_POLICY="PASSIVE",
        OMP_NUM_THREADS="4",
        VLLM_BATCH_INVARIANT="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_PLUGINS="",
        VLLM_USE_V2_MODEL_RUNNER="0" if legacy else "1",
    )
    scripts = str(ROOT / "scripts")
    search_paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [scripts] + [p for p in search_paths if p != scripts]
    )
    os.environ["VLLM_CACHE_ROOT"] = str(cache / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torchinductor")
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "triton_cache"))
    return {
        "cache_root": str(cache),
        "cache_identity": {"fingerprint": fingerprint, "basis": basis},
    }


def request_for(prompt, picture):
    return dict(
        prompt=prompt,
        multi_modal_data={"image": picture},
        multi_modal_uuids={"image": [str(uuid.uuid4())]},
    )
