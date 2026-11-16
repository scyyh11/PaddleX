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

"""Explicit packed FA2 component GPU gate, excluded from CPU test discovery.

Requires the pinned vLLM 0.19 / PyTorch 2.10 CUDA 12.8 environment, an A100
80GB/108-SM device, and a library built by kernels/fa2_sm80/build.py. No model
weights, image data, runtime hooks, or private experiment paths are used.

The default suites reproduce 1956 core, 224 zero/subnormal, and 1476 strided
checks. Q2..7 is a kernel component scope; this does not expand the production
B1/K4 admission guard. Inputs are finite BF16, page IDs are valid, and causal
endpoints stay within their bound. This is not an end-to-end token or arbitrary
nonfinite-input equivalence test. Graph tests change both tensors and metadata.
"""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_bundle():
    path = ROOT / "kernels" / "fa2_sm80" / "packed_fa2.py"
    spec = importlib.util.spec_from_file_location("_packed_fa2_gpu_gate_bundle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PackedFA2


def _lengths(queries, bound, quick):
    if quick:
        return [queries, 127, 128, 129, 3916, bound]
    return sorted(
        {
            0,
            1,
            queries,
            queries + 1,
            63,
            64,
            65,
            127,
            128,
            129,
            130,
            131,
            132,
            255,
            256,
            257,
            3916,
            4095,
            4096,
            bound - 1,
            bound,
        }
    )


class Checker:
    def __init__(self, torch, reference, plan_type, library, device):
        self.torch = torch
        self.reference = reference
        self.plan_type = plan_type
        self.library = library
        self.device = device
        self.records = []
        self.failures = []

    def compare(self, actual, expected, **case):
        torch = self.torch
        actual_bits = actual.view(torch.int16)
        expected_bits = expected.view(torch.int16)
        exact = torch.equal(actual_bits, expected_bits)
        record = {
            "case": case,
            "bitwise_exact": exact,
            "different_bits_elements": int((actual_bits != expected_bits).sum()),
            "different_values_elements": int((actual != expected).sum()),
            "nan_actual": int(torch.isnan(actual).sum()),
            "nan_reference": int(torch.isnan(expected).sum()),
        }
        if not exact:
            difference = (actual.float() - expected.float()).abs()
            record["max_abs_finite"] = float(difference.nan_to_num().max())
            self.failures.append(record)
            if len(self.failures) <= 8:
                print(json.dumps(record), flush=True)
        self.records.append(record)

    def kwargs(self, q, k, v, cu, lengths, table, bound):
        return dict(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            seqused_k=lengths,
            block_table=table,
            max_seqlen_q=q.shape[0],
            max_seqlen_k=bound,
            softmax_scale=128**-0.5,
            causal=True,
            fa_version=2,
            num_splits=0,
        )

    def native_ar1(self, kwargs, output):
        """Independent Q2..7 metadata; never invoke production fusion/dispatch.

        The reference still executes native FA2's independent AR1 rows with
        its original A100 split count. Only ordinary PyTorch integer operators
        construct per-row endpoints and page tables, so the production B1/K4
        metadata admission range cannot restrict this kernel component gate.
        """
        torch = self.torch
        q = kwargs["q"]
        count = q.shape[0]
        row_cu = torch.arange(count + 1, device=q.device, dtype=torch.int32)
        rows = row_cu[:-1]
        cu = kwargs["cu_seqlens_q"]
        lengths = kwargs["seqused_k"]
        valid = (rows >= cu[:1]) & (rows < cu[1:2]) & (lengths > 0)
        endpoints = (lengths - cu[1:2] + rows + 1).clamp_min(0)
        endpoints = torch.where(valid, endpoints, 0)
        tables = torch.where(valid[:, None], kwargs["block_table"].expand(count, -1), 0)
        dummy_cu = torch.zeros(count + 1, device=q.device, dtype=torch.int32)
        splits = self.reference.ar1_num_splits(kwargs["max_seqlen_k"], 108)
        return torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            kwargs["k"],
            kwargs["v"],
            output,
            row_cu,
            dummy_cu,
            endpoints,
            None,
            tables,
            None,
            1,
            kwargs["max_seqlen_k"],
            0.0,
            kwargs["softmax_scale"],
            False,
            True,
            -1,
            -1,
            0.0,
            False,
            splits,
            None,
        )[0]

    def plan(self, queries, bound):
        return self.plan_type(
            queries,
            bound,
            bound // 16,
            device=self.device,
            library_path=self.library,
        )

    def random_suite(self, *, strided=False, quick=False):
        torch = self.torch
        torch.manual_seed(190190)
        suite = "strides" if strided else "core"
        start = 1 if strided else 0
        for bound in (4096, 12288):
            for queries in (3, 4, 6) if strided else (2, 5, 7):
                columns = bound // 16
                plan = self.plan(queries, bound)
                storage = torch.empty(
                    queries, 2560, device=self.device, dtype=torch.bfloat16
                )
                q = storage[:, :2048].view(queries, 16, 128)
                kv_storage = torch.empty(
                    2,
                    columns + 17,
                    16,
                    2,
                    256 if strided else 128,
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                kv = kv_storage[..., :128]
                k, v = kv.unbind(0)
                if strided:
                    output_storage = torch.empty(
                        queries, 16, 256, device=self.device, dtype=torch.bfloat16
                    )
                    out = output_storage[..., :128]
                else:
                    out = torch.empty_like(q)
                expected = torch.empty_like(q)
                cu = torch.tensor(
                    [start, queries], device=self.device, dtype=torch.int32
                )
                lengths = torch.tensor([64], device=self.device, dtype=torch.int32)
                table = (
                    torch.randperm(columns + 17, device=self.device, dtype=torch.int32)[
                        :columns
                    ]
                    .reshape(1, columns)
                    .contiguous()
                )
                kwargs = self.kwargs(q, k, v, cu, lengths, table, bound)
                scales = (1.0,) if quick else (0.1, 1.0, 5.0, 10.0)
                valids = (
                    (queries,)
                    if quick
                    else (
                        (1, queries - 1, queries)
                        if strided
                        else (0, 1, queries - 1, queries)
                    )
                )
                for scale in scales:
                    q.normal_().mul_(scale)
                    kv.normal_().mul_(scale)
                    for valid in valids:
                        cu.copy_(
                            torch.tensor(
                                [start, valid], device=self.device, dtype=torch.int32
                            )
                        )
                        for length in _lengths(queries, bound, quick):
                            lengths.fill_(length)
                            self.native_ar1(kwargs, expected)
                            plan(q, k, v, cu, lengths, table, out=out)
                            self.compare(
                                out,
                                expected,
                                suite=suite,
                                bound=bound,
                                q=queries,
                                start=start,
                                valid=valid,
                                length=length,
                                input_scale=scale,
                                phase="eager",
                            )

                # Explicit stream handoff, then retain every graph-bound tensor
                # and the plan through the last changed-input replay comparison.
                stream = torch.cuda.Stream(device=self.device)
                current = torch.cuda.current_stream(self.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        plan(q, k, v, cu, lengths, table, out=out)
                current.wait_stream(stream)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    plan(q, k, v, cu, lengths, table, out=out)
                for step in range(6):
                    valid = (1, queries - 1, queries)[step % 3]
                    length = (128, 129, bound)[step % 3]
                    q.normal_()
                    kv.normal_()
                    cu.copy_(
                        torch.tensor(
                            [start, valid], device=self.device, dtype=torch.int32
                        )
                    )
                    lengths.fill_(length)
                    table.copy_(
                        torch.randperm(
                            columns + 17, device=self.device, dtype=torch.int32
                        )[:columns].reshape(1, columns)
                    )
                    graph.replay()
                    self.native_ar1(kwargs, expected)
                    self.compare(
                        out,
                        expected,
                        suite=suite,
                        bound=bound,
                        q=queries,
                        start=start,
                        valid=valid,
                        length=length,
                        phase="graph",
                        step=step,
                    )
                del graph
                self.progress(suite, bound, queries)

    def zero_suite(self):
        torch = self.torch
        for bound in (4096, 12288):
            for queries in (5, 7):
                columns = bound // 16
                plan = self.plan(queries, bound)
                q = torch.zeros(
                    queries, 16, 128, device=self.device, dtype=torch.bfloat16
                )
                k = torch.zeros(
                    columns, 16, 2, 128, device=self.device, dtype=torch.bfloat16
                )
                v = torch.empty_like(k)
                cu = torch.tensor([0, queries], device=self.device, dtype=torch.int32)
                lengths = torch.zeros(1, device=self.device, dtype=torch.int32)
                table = torch.arange(columns, device=self.device, dtype=torch.int32)[
                    None, :
                ]
                out, expected = torch.empty_like(q), torch.empty_like(q)
                kwargs = self.kwargs(q, k, v, cu, lengths, table, bound)
                # Positive/negative zero, subnormals, and smallest BF16 normals.
                for pattern in (0, 0x8000, 1, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080):
                    v.view(torch.int16).fill_(
                        pattern if pattern < 32768 else pattern - 65536
                    )
                    for length in (0, 1, 127, 128, 129, 512, bound):
                        lengths.fill_(length)
                        self.native_ar1(kwargs, expected)
                        plan(q, k, v, cu, lengths, table, out=out)
                        self.compare(
                            out,
                            expected,
                            suite="zeros",
                            bound=bound,
                            q=queries,
                            value_bits=pattern,
                            length=length,
                            phase="eager",
                        )
                self.progress("zeros", bound, queries)

    def progress(self, suite, bound, queries):
        print(
            json.dumps(
                {
                    "suite": suite,
                    "finished_bound": bound,
                    "q": queries,
                    "cases": len(self.records),
                    "failures": len(self.failures),
                }
            ),
            flush=True,
        )


def main(args):
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    import vllm019_ar1_attention as reference
    from vllm019_packed_attention import verify_artifact

    # Load the native FA2 operator registration without installing any adapter.
    importlib.import_module("vllm.v1.attention.backends.flash_attn")

    reference._verify_fa2_schema()
    artifact = verify_artifact(args.library)
    plan_type = _load_bundle()
    torch.set_num_threads(2)
    torch.manual_seed(190190)
    device = torch.device(args.device)
    checker = Checker(torch, reference, plan_type, artifact["library"], device)
    error = None
    try:
        with torch.cuda.device(device), torch.inference_mode():
            if "core" in args.suites:
                checker.random_suite(quick=args.quick)
            if "zeros" in args.suites:
                checker.zero_suite()
            if "strides" in args.suites:
                checker.random_suite(strided=True, quick=args.quick)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    counts = Counter(row["case"]["suite"] for row in checker.records)
    all_exact = bool(checker.records) and not checker.failures and error is None
    result = {
        "component_only": True,
        "scope": "Finite BF16 Q/K/V, Q2..7, S32/96, valid paged storage, causal endpoints within bound",
        "reference": "Unmodified native FA2 AR1 row-batch with original per-query endpoints and split count",
        "not_claimed": "End-to-end token equality, nonfinite Q/K/V, or expanded production admission",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "artifact": artifact,
        "quick": args.quick,
        "cases": len(checker.records),
        "suite_cases": dict(counts),
        "bitwise_exact_cases": sum(row["bitwise_exact"] for row in checker.records),
        "all_exact": all_exact,
        "error": error,
        "failures": checker.failures,
        "records": checker.records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("records", "failures")
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if all_exact else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=("core", "zeros", "strides"),
        default=["core", "zeros", "strides"],
    )
    raise SystemExit(main(parser.parse_args()))
