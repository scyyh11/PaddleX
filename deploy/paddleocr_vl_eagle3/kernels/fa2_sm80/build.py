#!/usr/bin/env python3
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

"""Build the pinned, experimental CUDA 12.8 FA2 packed-query kernel.

Only the Python standard library is used for download, extraction and build
orchestration. PyTorch must already be installed to provide the exact headers.
This script never imports CUDA, runs a GPU kernel or modifies vLLM sources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
FA2_COMMIT = "29210221863736a08f71a866459e368ad1ac4a95"
CUTLASS_COMMIT = "62750a2b75c802660e4894434dc55e839f322277"
SOURCES = {
    "flash-attention": {
        "commit": FA2_COMMIT,
        "url": f"https://codeload.github.com/vllm-project/flash-attention/tar.gz/{FA2_COMMIT}",
        "sha256": "fe98fd6aa07e30203ee0bb9619cb53f30703f71c22bbfa91909aa3b43b7018f1",
        "prefixes": ("csrc/flash_attn/src/",),
        "files": ("LICENSE", "AUTHORS"),
    },
    "cutlass": {
        "commit": CUTLASS_COMMIT,
        "url": f"https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_COMMIT}",
        "sha256": "78816d6c6d97793b5b59ef2a702174cb85b78dfcefc8fe2489964de2e42f17d2",
        "prefixes": ("include/",),
        "files": ("LICENSE.txt",),
    },
}
EXPECTED_SOURCE_HASHES = {
    "packed_fa2_v3.cu": "dff47d6c8c7ca7a66005cdbca8f6d4040351a2314b06adf58691d308d4c290bc",
    "packed_mask.h": "1f05457bfa4ff85105fe4ae9efae1ad31231c3551b7013171b05cfac0c2051eb",
}
UPSTREAM_KERNEL_SHA256 = (
    "242cbc331bd09d6ccfd023aae2d31461484b2a4801dcf66733b4151f9018622a"
)
PATCHED_KERNEL_SHA256 = (
    "f400f3ef44174f6397ed20a6e53d346eac4367dc591d04aa7e989abea4de423c"
)
EXPECTED_NVCC_VERSION = "12.8.93"
LIBRARY_NAME = "libpaddleocr_vl_fa2_sm80.so"
SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_archive(name: str, spec: dict, cache: Path, offline: bool) -> Path:
    path = cache / f"{name}-{spec['commit']}.tar.gz"
    if path.is_file():
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Cached archive checksum mismatch: {path}")
        return path
    if offline:
        raise FileNotFoundError(f"Offline build requires the verified archive: {path}")
    cache.mkdir(parents=True, exist_ok=True)
    # A unique partial file prevents interrupted/concurrent downloads from
    # becoming trusted cache entries. The final rename follows checksum verify.
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            request = urllib.request.Request(
                spec["url"], headers={"User-Agent": "paddleocr-vl-fa2-build/1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            handle.close()
            if sha256(temporary) != spec["sha256"]:
                raise ValueError(f"Downloaded archive checksum mismatch: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def extract_headers(archive: Path, name: str, spec: dict, destination: Path) -> Path:
    """Extract only pinned headers/licenses; never follow archive links."""
    expected_root = f"{name}-{spec['commit']}"
    target_root = destination / expected_root
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not path.parts or path.parts[0] != expected_root:
                raise ValueError(f"Unexpected archive root: {member.name}")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            selected = relative in spec["files"] or any(
                relative.startswith(prefix) for prefix in spec["prefixes"]
            )
            if not selected or member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(
                    f"Only regular source files may be extracted: {member.name}"
                )
            output = target_root / Path(*path.parts[1:])
            if not output.resolve().is_relative_to(destination.resolve()):
                raise ValueError(
                    f"Extraction destination escapes through an existing symlink: {member.name}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"Missing source payload: {member.name}")
            with source, output.open("wb") as stream:
                shutil.copyfileobj(source, stream)
    return target_root


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError("Pinned FA2 source differs from the expected patch context")
    return text.replace(before, after)


def patch_kernel(upstream: Path, output: Path) -> None:
    if sha256(upstream) != UPSTREAM_KERNEL_SHA256:
        raise ValueError("Unexpected pinned flash_fwd_kernel.h checksum")
    text = upstream.read_text()
    start = text.index("inline __device__ void compute_attn_1rowblock_splitkv(")
    end = text.index("inline __device__ void compute_attn_splitkv(", start)
    section = replace_once(
        text[start:end],
        "FLASH_NAMESPACE::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);",
        "FLASH_NAMESPACE::PackedRowMask mask(params.packed_endpoints, params.packed_queries);",
    )
    text = text[:start] + section + text[end:]
    start = text.index("inline __device__ void combine_attn_seqk_parallel(")
    prefix, section = text[:start], text[start:]
    section = replace_once(
        section,
        "    // Load Oaccum in then scale and accumulate to O\n    for (int split = 0; split < params.num_splits; ++split) {",
        """    // PACKED V3: every nonempty split has exactly one 128-key tile. LSE's
    // original full tree above is intentionally untouched. Skip only Oaccum
    // traffic beyond the GPU maximum endpoint, preserving active FMA order.
    const int active_splits = std::min(params.num_splits,
        (std::max(params.seqused_k[0], 0) + 127) / 128);
    for (int split = 0; split < active_splits; ++split) {""",
    )
    section = replace_once(
        section,
        "    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(tOrO);\n    // Write to gO",
        """    // The omitted native tail is repeated +0 * +0 + accumulator. Retain
    // one exact native FTZ FMA to preserve negative-zero/denormal behavior.
    // Inline PTX prevents the optimizer from deleting this operation.
    if (active_splits < params.num_splits) {
        #pragma unroll
        for (int i = 0; i < size(tOrO); ++i) {
            float accumulator = tOrO(i);
            asm volatile("fma.rn.ftz.f32 %0, 0f00000000, 0f00000000, %1;"
                : "=f"(accumulator) : "f"(accumulator));
            tOrO(i) = accumulator;
        }
    }
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(tOrO);
    // Write to gO""",
    )
    output.write_text(prefix + section)
    if sha256(output) != PATCHED_KERNEL_SHA256:
        raise ValueError("Generated header does not match the GPU-validated v3 source")


def find_torch_include(explicit: Path | None) -> tuple[Path, str]:
    version = importlib.metadata.version("torch")
    if version not in ("2.10.0", "2.10.0+cu128"):
        raise ValueError(
            f"Expected the validated PyTorch 2.10.0 headers, found {version}"
        )
    if explicit is None:
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.origin is None:
            raise ValueError("Install PyTorch 2.10.0+cu128 before building")
        explicit = Path(spec.origin).parent / "include"
    include = explicit.resolve()
    if not (include / "ATen/cuda/CUDAGeneratorImpl.h").is_file():
        raise FileNotFoundError(f"PyTorch CUDA headers were not found: {include}")
    return include, version


def toolchain(nvcc_path: Path | None) -> tuple[Path, Path, str]:
    if nvcc_path is None:
        found = shutil.which("nvcc")
        if found is None:
            raise FileNotFoundError(
                "CUDA 12.8 nvcc is required; pass --nvcc /path/to/nvcc"
            )
        nvcc_path = Path(found)
    nvcc = nvcc_path.resolve()
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if f"V{EXPECTED_NVCC_VERSION}" not in version:
        raise ValueError(
            f"This numerical profile requires nvcc {EXPECTED_NVCC_VERSION}"
        )
    root = nvcc.parent.parent
    directories = (root / "lib64", root / "lib", root / "targets/x86_64-linux/lib")
    library_dir = next(
        (path for path in directories if (path / "libcudart.so").is_file()), None
    )
    if library_dir is None:
        raise FileNotFoundError(
            "The matching CUDA 12.8 development runtime libcudart.so is required"
        )
    return nvcc, library_dir, version


def main(args) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / LIBRARY_NAME).exists():
        raise FileExistsError(
            "Choose a fresh output directory; do not rebuild a library used by active CUDA graphs"
        )
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        if sha256(ROOT / "src" / name) != expected:
            raise ValueError(f"The validated local CUDA source was changed: {name}")
    cache = args.cache_dir.resolve()
    roots = {}
    for name, spec in SOURCES.items():
        archive = checked_archive(name, spec, cache, args.offline)
        roots[name] = extract_headers(archive, name, spec, output / "sources")
    fa2 = roots["flash-attention"] / "csrc/flash_attn/src"
    generated = output / "packed_flash_fwd_kernel_v3.h"
    patch_kernel(fa2 / "flash_fwd_kernel.h", generated)
    for name in EXPECTED_SOURCE_HASHES:
        shutil.copy2(ROOT / "src" / name, output / name)
    shutil.copytree(ROOT / "LICENSES", output / "LICENSES", dirs_exist_ok=True)
    shutil.copy2(ROOT / "NOTICE.md", output / "NOTICE.md")
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    print(
        json.dumps(
            {"prepared": str(output), "patched_header_sha256": sha256(generated)}
        ),
        flush=True,
    )
    if args.prepare_only:
        return
    nvcc, cuda_lib, nvcc_version = toolchain(args.nvcc)
    torch_include, torch_version = find_torch_include(args.torch_include)
    command = [
        str(nvcc),
        "-std=c++17",
        "-O3",
        "-arch=sm_80",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-Xcompiler=-fPIC",
        "--shared",
        "--cudart=shared",
        f"-I{fa2}",
        f"-I{roots['cutlass'] / 'include'}",
        f"-I{torch_include}",
        f"-I{torch_include / 'torch/csrc/api/include'}",
        f"-I{output}",
        str(output / "packed_fa2_v3.cu"),
        "-o",
        str(output / LIBRARY_NAME),
        f"-L{cuda_lib}",
        "-Xlinker=-rpath",
        f"-Xlinker={cuda_lib}",
    ]
    # Compilation cannot execute a GPU kernel. Hiding devices also prevents
    # incidental discovery by compiler support programs.
    if any(
        os.environ.get(name) for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
    ):
        raise ValueError(
            "Unset NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS; this profile fixes the numerical compiler flags"
        )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with (output / "build.log").open("w") as log:
        subprocess.run(
            command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": "paddleocr-vl-fa2-sm80-v3-finite",
        "source_pins": {
            name: {key: spec[key] for key in ("commit", "url", "sha256")}
            for name, spec in SOURCES.items()
        },
        "cuda_source_sha256": EXPECTED_SOURCE_HASHES,
        "patched_header_sha256": PATCHED_KERNEL_SHA256,
        "nvcc_version": EXPECTED_NVCC_VERSION,
        "nvcc_version_output": nvcc_version,
        "torch_headers_version": torch_version,
        "gpu_arch": "sm80",
        "command": command,
        "library_name": LIBRARY_NAME,
        "library_sha256": sha256(output / LIBRARY_NAME),
        "gpu_validation_performed_by_build": False,
    }
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "library": str(output / LIBRARY_NAME),
                "sha256": manifest["library_sha256"],
                "gpu_validation": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--torch-include", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify/extract sources and generate the header; do not compile",
    )
    main(parser.parse_args())
