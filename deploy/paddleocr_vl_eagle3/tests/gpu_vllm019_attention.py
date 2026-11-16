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

"""Explicit GPU check; excluded from CPU unittest discovery."""
import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm019_ar1_attention as adapter
from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func as native


@torch.inference_mode()
def main(output=None):
    torch.manual_seed(17)
    adapter._verify_fa2_schema()
    cases = []
    for bound in (4096, 12288):
        for count in (2, 5):
            for length in (64, 3916, bound):
                q = torch.randn(count, 16, 128, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(
                    bound // 16, 16, 2, 128, device="cuda", dtype=torch.bfloat16
                )
                v = torch.randn_like(k)
                kw = dict(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=torch.tensor(
                        [0, count], device="cuda", dtype=torch.int32
                    ),
                    seqused_k=torch.tensor([length], device="cuda", dtype=torch.int32),
                    block_table=torch.arange(
                        bound // 16, device="cuda", dtype=torch.int32
                    )[None, :],
                    max_seqlen_q=count,
                    max_seqlen_k=bound,
                    softmax_scale=128**-0.5,
                    causal=True,
                    fa_version=2,
                    num_splits=0,
                )
                ref = torch.empty_like(q)
                for row in range(count):
                    native(
                        **dict(
                            kw,
                            q=q[row : row + 1],
                            out=ref[row : row + 1],
                            cu_seqlens_q=torch.tensor(
                                [0, 1], device="cuda", dtype=torch.int32
                            ),
                            max_seqlen_q=1,
                            seqused_k=torch.tensor(
                                [length - count + row + 1],
                                device="cuda",
                                dtype=torch.int32,
                            ),
                        )
                    )
                got = adapter._batched_attention(kw, torch.empty_like(q))
                torch.cuda.synchronize()
                exact = torch.equal(got, ref)
                cases.append(
                    dict(
                        bound=bound,
                        queries=count,
                        kv_length=length,
                        splits=adapter.ar1_num_splits(bound, 108),
                        exact=exact,
                        max_abs=float((got - ref).abs().max()),
                    )
                )
    result = {"cases": cases, "all_exact": all(c["exact"] for c in cases)}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["all_exact"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Opt-in A100 GPU parity check for the vLLM 0.19 FA2 adapter"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.output))
