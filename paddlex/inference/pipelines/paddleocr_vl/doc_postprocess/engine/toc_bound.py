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

"""Safe multiset upper bound for SequenceMatcher; GPU rejects impossible pairs.

Only pairs whose exact similarity cannot reach .92 are pruned. Surviving pairs
still use the original SequenceMatcher and original monotone title alignment.
"""
import collections
import importlib
import time

import numpy as np

from .runtime_context import backend

telemetry = []
KERNEL = r"""
extern "C" __global__ void upper_bound(const unsigned short* hist,const int* indices,const int* counts,const int* offsets,const int* qlength,const int* clength,int nq,int nc,unsigned char* out){
 int c=blockIdx.x*blockDim.x+threadIdx.x;int q=blockIdx.y;
 if(c>=nc||q>=nq)return;
 double ratio=(double)clength[c]/qlength[q];
 if(!(ratio>.85&&ratio<1.15)){out[q*nc+c]=0;return;}
 int matches=0;for(int j=offsets[q];j<offsets[q+1];++j)if(indices[j]>=0)matches+=min((int)hist[indices[j]*nc+c],counts[j]);
 out[q*nc+c]=(2.0*matches/(qlength[q]+clength[c])>=.92);
}
"""
_kernel = None


def admissible(queries, candidates):
    queries = list(dict.fromkeys(queries))
    candidates = list(candidates)
    if not queries or not candidates:
        return {q: [] for q in queries}
    cc = [collections.Counter(c) for c in candidates]
    qq = [collections.Counter(q) for q in queries]
    if backend.get() != "cuda":
        return {
            q: [
                c
                for c, h in zip(candidates, cc)
                if 0.85 < len(c) / len(q) < 1.15
                and 2
                * sum(min(n, h.get(ch, 0)) for ch, n in bag.items())
                / (len(q) + len(c))
                >= 0.92
            ]
            for q, bag in zip(queries, qq)
        }
    cp = importlib.import_module("cupy")
    global _kernel
    started = time.perf_counter()
    vocab = {ch: i for i, ch in enumerate(sorted({ch for bag in cc for ch in bag}))}
    nc = len(cc)
    hist = np.zeros((len(vocab), nc), dtype=np.uint16)
    for c, bag in enumerate(cc):
        for ch, n in bag.items():
            hist[vocab[ch], c] = n
    indices = []
    counts = []
    offsets = [0]
    for bag in qq:
        for ch, n in bag.items():
            indices.append(vocab.get(ch, -1))
            counts.append(n)
        offsets.append(len(indices))
    if _kernel is None:
        _kernel = cp.RawKernel(
            KERNEL, "upper_bound", options=("--std=c++11", "--fmad=false")
        )
    args = [cp.asarray(hist)] + [
        cp.asarray(a, dtype=cp.int32)
        for a in [
            indices,
            counts,
            offsets,
            list(map(len, queries)),
            list(map(len, candidates)),
        ]
    ]
    out = cp.empty((len(qq), nc), dtype=cp.uint8)
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    _kernel(((nc + 127) // 128, len(qq)), (128,), (*args, len(qq), nc, out))
    end.record()
    mask = out.get()
    end.synchronize()
    telemetry.append(
        {
            "queries": len(qq),
            "candidates": nc,
            "kernel_ms": cp.cuda.get_elapsed_time(start, end),
            "call_ms": (time.perf_counter() - started) * 1000,
            "surviving_pairs": int(mask.sum()),
        }
    )
    return {
        q: [candidates[i] for i in np.flatnonzero(mask[j])]
        for j, q in enumerate(queries)
    }
