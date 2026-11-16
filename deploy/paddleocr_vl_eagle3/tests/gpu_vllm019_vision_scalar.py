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

"""Explicit GPU mixed-device scalar gate; no model weights or installed hooks.

Native vision has head dimension 72; FA2 dispatch rounds this to a head-96
kernel. Both actual dimensions 72 and 96 are covered. Optional graph checks
hold max_seqlen fixed while changing Q/K/V; captured Python scalars are static.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--include-nan", action="store_true")
    args = parser.parse_args()
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("This gate requires vLLM 0.19.0")
    from vllm.v1.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper

    torch.set_num_threads(2)
    torch.manual_seed(190190)
    records = []
    cases = []
    for dim in (72, 96):
        for seq in (1, 7, 63, 128, 129, 256, 1024):
            cases.append(
                dict(
                    dim=dim,
                    batch=1,
                    seq=seq,
                    lengths=[seq],
                    layout="contiguous",
                    nonfinite=None,
                )
            )
        for batch, seq, lengths in (
            (2, 63, [58, 68]),
            (2, 128, [128, 128]),
            (3, 64, [32, 64, 96]),
        ):
            cases.append(
                dict(
                    dim=dim,
                    batch=batch,
                    seq=seq,
                    lengths=lengths,
                    layout="qkv_strided",
                    nonfinite=None,
                )
            )
    if args.include_nan:
        for where in ("q", "v"):
            cases.append(
                dict(
                    dim=72,
                    batch=1,
                    seq=128,
                    lengths=[128],
                    layout="contiguous",
                    nonfinite=where,
                )
            )

    def compare(actual, expected, label):
        actual_bits = actual.contiguous().view(torch.int16)
        expected_bits = expected.contiguous().view(torch.int16)
        exact = torch.equal(actual_bits, expected_bits)
        record = {
            **label,
            "exact": exact,
            "different_bits_elements": int((actual_bits != expected_bits).sum()),
            "nan_actual": int(torch.isnan(actual).sum()),
            "nan_expected": int(torch.isnan(expected).sum()),
        }
        records.append(record)
        if not exact:
            print(json.dumps(record), flush=True)

    error = None
    try:
        with torch.inference_mode():
            for case in cases:
                dim, batch, seq = case["dim"], case["batch"], case["seq"]
                if case["layout"] == "qkv_strided":
                    storage = torch.randn(
                        (batch, seq, 3, 16, dim), device="cuda", dtype=torch.bfloat16
                    )
                    q, k, v = storage.unbind(dim=2)
                else:
                    q, k, v = [
                        torch.randn(
                            (batch, seq, 16, dim), device="cuda", dtype=torch.bfloat16
                        )
                        for _ in range(3)
                    ]
                if case["nonfinite"] is not None:
                    (q if case["nonfinite"] == "q" else v)[0, 0, 0, 0] = float("nan")
                cu_cpu = torch.tensor([0] + case["lengths"], dtype=torch.int32).cumsum(
                    0, dtype=torch.int32
                )
                assert int(cu_cpu[-1]) == batch * seq
                cu = cu_cpu.to(device="cuda")
                gpu_max = (cu[1:] - cu[:-1]).max()
                cpu_max = torch.tensor(gpu_max.item(), dtype=torch.int32, device="cpu")

                def run(maximum):
                    return vit_flash_attn_wrapper(
                        q=q,
                        k=k,
                        v=v,
                        batch_size=batch,
                        is_rocm_aiter=False,
                        fa_version=2,
                        scale=dim**-0.5,
                        cu_seqlens=cu,
                        max_seqlen=maximum,
                    )

                expected = run(gpu_max)
                actual = run(cpu_max)
                compare(actual, expected, {**case, "phase": "eager"})
                if (
                    args.graph
                    and case["nonfinite"] is None
                    and (
                        (dim, batch, seq) == (72, 1, 128)
                        or (dim, batch, seq) == (96, 2, 63)
                    )
                ):
                    for _ in range(3):
                        run(cpu_max)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured_out = run(cpu_max)
                    for step in range(4):
                        q.normal_()
                        k.normal_()
                        v.normal_()
                        graph.replay()
                        compare(
                            captured_out,
                            run(gpu_max),
                            {**case, "phase": "graph", "step": step},
                        )
                    del graph
                print(
                    json.dumps(
                        {
                            "dim": dim,
                            "batch": batch,
                            "seq": seq,
                            "checked": len(records),
                            "failures": sum(not row["exact"] for row in records),
                        }
                    ),
                    flush=True,
                )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    all_exact = bool(records) and all(row["exact"] for row in records) and error is None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "scope": "Original FA2 vision custom op; CUDA QKV/cu plus CPU 0-d int32 max scalar",
        "hooks_installed": False,
        "weights_loaded": False,
        "cases": len(records),
        "exact_cases": sum(row["exact"] for row in records),
        "all_exact": all_exact,
        "error": error,
        "records": records,
        "graph_scope": "Fixed dimensions, cu_seqlens and max_seqlen; changed Q/K/V. Does not test whole Siglip graph capture.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(records),
                "all_exact": all_exact,
                "error": error,
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if all_exact else 1)


if __name__ == "__main__":
    main()
