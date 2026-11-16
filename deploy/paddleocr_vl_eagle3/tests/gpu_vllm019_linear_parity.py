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

"""Explicit CUDA 12 native BF16 rowwise/batched linear comparison.

Example (choose the GPU externally):
  python gpu_vllm019_linear_parity.py --model /path/to/PaddleOCR-VL-1.6 \
    --layers 0 17 --queries 2 3 4 5 --output /tmp/linear-parity.json --log-cublas

This diagnostic leaves all PyTorch precision/backend settings unchanged. The
ordered O/down candidate reproduces the old stride-32 FMA/tree arithmetic; it
does not assume a CUDA 13 algorithm ID or that CUDA 12 native Q1 selects it.
Finite input equality is not an end-to-end target token guarantee. Only scalar
summaries and kernel traces are emitted, never activation values or token IDs.
"""

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path


def _load_runtime():
    # The logging environment is configured by main before any CUDA import.
    global torch, triton, tl
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def ordered_kernel(
        X,
        W,
        Y,
        N: tl.constexpr,
        K: tl.constexpr,
        Q: tl.constexpr,
        BQ: tl.constexpr,
        BN: tl.constexpr,
        UNROLL: tl.constexpr,
    ):
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        lanes = tl.arange(0, 32)
        queries = tl.arange(0, BQ)
        accumulators = tl.full((BQ, BN, 32), 0, tl.float32)
        for block in tl.range(0, K // 32, loop_unroll_factor=UNROLL):
            columns = block * 32 + lanes
            weights = tl.load(
                W + rows[:, None] * K + columns[None, :],
                rows[:, None] < N,
                0,
            ).to(tl.float32)
            inputs = tl.load(
                X + queries[:, None] * K + columns[None, :],
                queries[:, None] < Q,
                0,
            ).to(tl.float32)
            accumulators = tl.fma(weights[None, :, :], inputs[:, None, :], accumulators)
        result = tl.sum(accumulators, axis=2)
        tl.store(
            Y + queries[:, None] * N + rows[None, :],
            result,
            (queries[:, None] < Q) & (rows[None, :] < N),
        )

    return ordered_kernel


class Checkpoint:
    def __init__(self, model):
        from safetensors import safe_open

        self.safe_open, self.model = safe_open, model
        index = model / "model.safetensors.index.json"
        if index.is_file():
            self.weights = json.loads(index.read_text())["weight_map"]
        else:
            self.weights = {}
            files = sorted(model.glob("*.safetensors"))
            if not files:
                raise ValueError(f"No safetensors checkpoint found in {model}")
            for path in files:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key in self.weights:
                            raise ValueError(f"Ambiguous checkpoint key: {key}")
                        self.weights[key] = path.name

    def tensor(self, suffix):
        matches = [key for key in self.weights if key.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one checkpoint key ending {suffix!r}; found {len(matches)}"
            )
        key = matches[0]
        with self.safe_open(
            str(self.model / self.weights[key]), framework="pt", device="cpu"
        ) as handle:
            tensor = handle.get_tensor(key)
        if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
            raise ValueError(f"{key} must be a two-dimensional BF16 weight")
        return tensor, key

    def weight(self, kind, layer):
        prefix = f"model.layers.{layer}."
        suffixes = {
            "qkv": [
                prefix + f"self_attn.{part}_proj.weight" for part in ("q", "k", "v")
            ],
            "o": [prefix + "self_attn.o_proj.weight"],
            "gateup": [prefix + f"mlp.{part}_proj.weight" for part in ("gate", "up")],
            "down": [prefix + "mlp.down_proj.weight"],
            "head": ["lm_head.weight"],
        }[kind]
        values = [self.tensor(suffix) for suffix in suffixes]
        weight = (
            torch.cat([value for value, _ in values], dim=0)
            if len(values) > 1
            else values[0][0]
        )
        return weight.contiguous(), [key for _, key in values]


def _comparison(actual, expected, *, head=False):
    difference = (actual.float() - expected.float()).abs()
    maximum = float(difference.max().item())
    result = {
        "exact": bool(torch.equal(actual, expected)),
        "different_elements": int((actual != expected).sum().item()),
        "elements": actual.numel(),
        "max_absolute": maximum if math.isfinite(maximum) else None,
        "nonfinite_differences": int((~torch.isfinite(difference)).sum().item()),
    }
    if head:
        result["top1_agree"] = int(
            (actual.argmax(-1) == expected.argmax(-1)).sum().item()
        )
        result["top1_total"] = actual.shape[0]
    return result


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs = {}
    if args.log_cublas:
        logs = {
            "cublas": str(args.output.with_suffix(".cublas.log")),
            "cublaslt": str(args.output.with_suffix(".cublaslt.log")),
        }
        os.environ["CUBLAS_LOGINFO_DBG"] = "1"
        os.environ["CUBLAS_LOGDEST_DBG"] = logs["cublas"]
        os.environ["CUBLASLT_LOG_LEVEL"] = "5"
        os.environ["CUBLASLT_LOG_FILE"] = logs["cublaslt"]
    ordered_kernel = _load_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("This explicit diagnostic requires a CUDA device")
    torch.set_num_threads(4)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    checkpoint = Checkpoint(args.model)
    records = []
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        if args.trace
        else nullcontext()
    )
    with torch.inference_mode(), profiler as active_profiler:
        for kind in args.kinds:
            for layer in [None] if kind == "head" else args.layers:
                cpu_weight, keys = checkpoint.weight(kind, layer)
                weight = cpu_weight.to(device="cuda")
                for queries in args.queries:
                    for scale in args.scales:
                        inputs = (
                            torch.randn(
                                (queries, weight.shape[1]),
                                device="cuda",
                                dtype=torch.float32,
                                generator=generator,
                            )
                            * scale
                        ).to(torch.bfloat16)
                        label = f"{kind}/layer{layer}/q{queries}/scale{scale}"
                        with torch.profiler.record_function(label + "/rowwise"):
                            reference = torch.cat(
                                [
                                    torch.nn.functional.linear(
                                        inputs[i : i + 1], weight
                                    )
                                    for i in range(queries)
                                ],
                                dim=0,
                            )
                        with torch.profiler.record_function(label + "/batched"):
                            batched = torch.nn.functional.linear(inputs, weight)
                        comparisons = {
                            "batch_vs_rows": _comparison(
                                batched, reference, head=kind == "head"
                            )
                        }
                        if kind in ("o", "down"):
                            if weight.shape[1] % 32:
                                raise ValueError(
                                    "The ordered candidate requires K divisible by 32"
                                )
                            ordered = torch.empty_like(reference)
                            with torch.profiler.record_function(label + "/ordered"):
                                ordered_kernel[(triton.cdiv(weight.shape[0], 4),)](
                                    inputs,
                                    weight,
                                    ordered,
                                    weight.shape[0],
                                    weight.shape[1],
                                    queries,
                                    triton.next_power_of_2(queries),
                                    4,
                                    8,
                                    num_warps=4,
                                )
                            comparisons["ordered_vs_rows"] = _comparison(
                                ordered, reference
                            )
                        records.append(
                            {
                                "kind": kind,
                                "layer": layer,
                                "weight_keys": keys,
                                "weight_shape": list(weight.shape),
                                "dtype": str(weight.dtype),
                                "query": queries,
                                "scale": scale,
                                "comparisons": comparisons,
                            }
                        )
                del weight, cpu_weight
        torch.cuda.synchronize()
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        active_profiler.export_chrome_trace(str(args.trace))
    summary = {}
    for record in records:
        for method, result in record["comparisons"].items():
            total = summary.setdefault(
                method,
                {
                    "cases": 0,
                    "exact_cases": 0,
                    "different_elements": 0,
                    "elements": 0,
                    "max_absolute": 0.0,
                },
            )
            total["cases"] += 1
            total["exact_cases"] += int(result["exact"])
            total["different_elements"] += result["different_elements"]
            total["elements"] += result["elements"]
            if result["max_absolute"] is not None:
                total["max_absolute"] = max(
                    total["max_absolute"], result["max_absolute"]
                )
            if "top1_agree" in result:
                total["head_top1_agree"] = (
                    total.get("head_top1_agree", 0) + result["top1_agree"]
                )
                total["head_top1_total"] = (
                    total.get("head_top1_total", 0) + result["top1_total"]
                )
    environment = {
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "cublas_package": _version("nvidia-cublas-cu12"),
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "preferred_blas": str(torch.backends.cuda.preferred_blas_library()),
        "allow_bf16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", None
        ),
        "allow_bf16_reduced_precision_reduction_split_k": getattr(
            torch.backends.cuda.matmul,
            "allow_bf16_reduced_precision_reduction_split_k",
            None,
        ),
        "allow_bf16_accumulation": getattr(
            torch.backends.cuda.matmul, "allow_bf16_accumulation", None
        ),
    }
    result = {
        "scope": "Native Q1 versus batched BF16 projections on current weights; seeded synthetic inputs only. "
        "Ordered O/down is a numeric candidate, not evidence of the default Q1 algorithm choice.",
        "model": str(args.model),
        "seed": args.seed,
        "environment": environment,
        "summary": summary,
        "records": records,
        "logs": logs,
        "trace": str(args.trace) if args.trace else None,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": summary,
                "environment": environment,
                "logs": logs,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=("qkv", "o", "gateup", "down", "head"),
        default=["qkv", "o", "gateup", "down", "head"],
    )
    parser.add_argument("--queries", nargs="+", type=int, default=[5])
    parser.add_argument("--seed", type=int, default=19008)
    parser.add_argument("--scales", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--log-cublas", action="store_true")
    arguments = parser.parse_args()
    arguments.model = arguments.model.resolve()
    if not arguments.model.is_dir():
        parser.error("--model must be an existing Hugging Face checkpoint directory")
    if any(layer < 0 for layer in arguments.layers):
        parser.error("--layers must be nonnegative")
    if any(query not in (2, 3, 4, 5) for query in arguments.queries):
        parser.error("--queries must contain only 2, 3, 4 or 5")
    if any(not math.isfinite(scale) or scale <= 0 for scale in arguments.scales):
        parser.error("--scales must be finite and positive")
    main(arguments)
