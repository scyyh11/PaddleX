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

"""Explicit GPU check of user-supplied CUDA 12 Lt recipes, without vLLM hooks.

The --recipes JSON maps qkv/o/gateup/down/head to explicit recipe dictionaries.
No default algorithm IDs are supplied. The comparison target is current native
rowwise BF16 F.linear, including for the full vocabulary head. Passing finite
synthetic cases is a candidate gate, not proof of end-to-end token equivalence.
The script also checks graph replay after replacing input values in place.
"""

import argparse
import json
from pathlib import Path
import sys


def main(args):
    import torch
    import gpu_vllm019_linear_parity as reference

    # Reuse checkpoint loading and aggregate comparisons, with no model hooks.
    reference._load_runtime()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from vllm019_locked_lt import LockedLt, normalize_recipe

    recipes = json.loads(args.recipes.read_text())
    if not isinstance(recipes, dict) or not recipes:
        raise ValueError("--recipes must contain a nonempty object of per-kind recipes")
    if set(recipes) - {"qkv", "o", "gateup", "down", "head"}:
        raise ValueError("Unknown projection kind in --recipes")
    recipes = {kind: normalize_recipe(recipe) for kind, recipe in recipes.items()}
    if not torch.cuda.is_available():
        raise RuntimeError("This explicit diagnostic requires a CUDA device")
    checkpoint = reference.Checkpoint(args.model)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records = []
    with torch.inference_mode():
        for kind, recipe in recipes.items():
            for layer in [None] if kind == "head" else args.layers:
                weight, keys = checkpoint.weight(kind, layer)
                weight = weight.to(device="cuda")
                candidate = None
                try:
                    candidate = LockedLt(
                        weight, recipe, workspace_bytes=args.workspace_bytes
                    )
                    for queries in args.queries:
                        record = {
                            "kind": kind,
                            "layer": layer,
                            "query": queries,
                            "weight_keys": keys,
                            "weight_shape": list(weight.shape),
                            "comparisons": [],
                        }
                        graph = None
                        try:
                            plan = candidate.prepare(queries)
                            record["workspace_bytes"] = plan.workspace_bytes
                            record["waves_count"] = plan.waves_count
                            inputs = torch.empty(
                                (queries, weight.shape[1]),
                                device="cuda",
                                dtype=torch.bfloat16,
                            )
                            output = torch.empty(
                                (queries, weight.shape[0]),
                                device="cuda",
                                dtype=torch.bfloat16,
                            )

                            def set_inputs(scale):
                                inputs.copy_(
                                    (
                                        torch.randn(
                                            inputs.shape,
                                            device="cuda",
                                            dtype=torch.float32,
                                            generator=generator,
                                        )
                                        * scale
                                    ).to(torch.bfloat16)
                                )

                            def compare(mode, case, scale):
                                expected = torch.cat(
                                    [
                                        torch.nn.functional.linear(
                                            inputs[i : i + 1], weight
                                        )
                                        for i in range(queries)
                                    ],
                                    dim=0,
                                )
                                value = reference._comparison(
                                    output, expected, head=kind == "head"
                                )
                                record["comparisons"].append(
                                    {
                                        "mode": mode,
                                        "case": case,
                                        "scale": scale,
                                        **value,
                                    }
                                )

                            for case, scale in enumerate((0.1, 1.0, 10.0)):
                                set_inputs(scale)
                                candidate(inputs, output)
                                compare("eager", case, scale)
                            if args.graph_replays:
                                # Warm cuBLAS entry points on a side stream before capture.
                                current = torch.cuda.current_stream()
                                warm = torch.cuda.Stream()
                                warm.wait_stream(current)
                                with torch.cuda.stream(warm):
                                    for _ in range(3):
                                        candidate(inputs, output)
                                current.wait_stream(warm)
                                torch.cuda.synchronize()
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    candidate(inputs, output)
                                for case in range(args.graph_replays):
                                    scale = (0.1, 1.0, 10.0)[case % 3]
                                    set_inputs(scale)
                                    graph.replay()
                                    compare("graph", case, scale)
                            record["supported"] = True
                            record["all_exact"] = all(
                                row["exact"] for row in record["comparisons"]
                            )
                        except (RuntimeError, ValueError) as error:
                            record["supported"] = False
                            record["error"] = str(error)
                        finally:
                            torch.cuda.synchronize()
                            del graph
                        records.append(record)
                    candidate_info = candidate.metadata()
                    for record in records:
                        if record["kind"] == kind and record["layer"] == layer:
                            record["candidate"] = candidate_info
                except (RuntimeError, ValueError) as error:
                    records.append(
                        {
                            "kind": kind,
                            "layer": layer,
                            "supported": False,
                            "recipe": recipe,
                            "error": str(error),
                        }
                    )
                finally:
                    torch.cuda.synchronize()
                    if candidate is not None:
                        candidate.close()
    comparisons = [
        comparison for record in records for comparison in record.get("comparisons", [])
    ]
    result = {
        "scope": "Explicit CUDA12 recipe candidates versus native Q1; synthetic BF16 inputs, no model hooks",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "native_reference_flags": {
            "preferred_blas": str(torch.backends.cuda.preferred_blas_library()),
            "allow_bf16_reduced_precision_reduction": getattr(
                torch.backends.cuda.matmul,
                "allow_bf16_reduced_precision_reduction",
                None,
            ),
            "allow_bf16_reduced_precision_reduction_split_k": getattr(
                torch.backends.cuda.matmul,
                "allow_bf16_reduced_precision_reduction_split_k",
                None,
            ),
        },
        "seed": args.seed,
        "graph_replays_per_plan": args.graph_replays,
        "recipes": recipes,
        "records": records,
        "summary": {
            "plans": len(records),
            "supported_plans": sum(row["supported"] for row in records),
            "comparison_cases": len(comparisons),
            "exact_cases": sum(row["exact"] for row in comparisons),
            "unsupported_or_mismatch": any(
                not row["supported"] or not row.get("all_exact", False)
                for row in records
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"output": str(args.output), "summary": result["summary"]}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0])
    parser.add_argument("--queries", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--workspace-bytes", type=int, default=8 * 1024**2)
    parser.add_argument("--graph-replays", type=int, default=6)
    parser.add_argument("--seed", type=int, default=19019)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if any(layer < 0 for layer in arguments.layers) or any(
        query not in range(1, 6) for query in arguments.queries
    ):
        parser.error("--layers must be nonnegative; --queries must be 1..5")
    if arguments.workspace_bytes < 0 or arguments.graph_replays < 0:
        parser.error("workspace size and graph replay count must be nonnegative")
    main(arguments)
