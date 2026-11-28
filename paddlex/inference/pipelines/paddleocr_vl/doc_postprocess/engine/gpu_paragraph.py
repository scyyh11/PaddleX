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

"""CUDA dynamic programming for OCR glyph/line-width alignment."""
import importlib
import time

import numpy as np


def _cp():
    return importlib.import_module("cupy")


KERNEL = r"""
extern "C" __global__ void align(const double* pref,const double* targets,int m,int n,double* costs,int* back) {
 int tid=threadIdx.x;
 double* prev=costs;double* curr=costs+m;
 for(int j=tid;j<m;j+=blockDim.x)prev[j]=(j==0?0.:1.0/0.0);
 __syncthreads();
 for(int row=1;row<=n;++row){
  double target=targets[row-1];
  for(int end=tid;end<m;end+=blockDim.x){
   double best=1.0/0.0;int arg=-1;
   if(end>0){
    double low=(pref[end]-target)-4.0;double high=pref[end]-fmax(1.0,target-4.0);
    int a=0,b=m;while(a<b){int mid=(a+b)/2;if(pref[mid]<low)a=mid+1;else b=mid;}int lo=a;
    a=0;b=m;while(a<b){int mid=(a+b)/2;if(pref[mid]<=high)a=mid+1;else b=mid;}int hi=min(end,a);
    for(int k=lo;k<hi;++k){double delta=(pref[end]-pref[k])-target;double v=prev[k]+delta*delta;if(v<best){best=v;arg=k;}}
   }
   curr[end]=best;back[row*m+end]=arg;
  }
  __syncthreads();double* tmp=prev;prev=curr;curr=tmp;
 }
 if(tid==0)costs[2*m]=prev[m-1];
}
"""
_kernel = None
telemetry = []


def solve(pref, targets):
    cp = _cp()
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(KERNEL, "align", options=("--std=c++11", "--fmad=false"))
    started = time.perf_counter()
    m = len(pref)
    n = len(targets)
    p = cp.asarray(pref)
    t = cp.asarray(targets)
    c = cp.empty(2 * m + 1, dtype=cp.float64)
    b = cp.full((n + 1, m), -1, dtype=cp.int32)
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    _kernel((1,), (256,), (p, t, m, n, c, b))
    stop.record()
    last = float(c[-1].get())
    back = b.get()
    stop.synchronize()
    cost = np.full((n + 1, m), np.inf)
    cost[n, -1] = last
    telemetry.append(
        {
            "states": n * m,
            "kernel_ms": cp.cuda.get_elapsed_time(start, stop),
            "call_ms": (time.perf_counter() - started) * 1000,
        }
    )
    return cost, back
