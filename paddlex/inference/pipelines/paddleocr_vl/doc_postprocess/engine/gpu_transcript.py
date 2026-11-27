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

"""Exact CUDA DP for fixed-pitch transcript line alignment."""
import importlib
import time

import numpy as np


def _cp():
    return importlib.import_module("cupy")


KERNEL = r"""
extern "C" __global__ void align(const int* pos,const int* trim,const double* targets,int m,int n,double* costs,int* back){
 int tid=threadIdx.x;double* prev=costs;double* curr=costs+m;
 for(int j=tid;j<m;j+=blockDim.x)prev[j]=j==0?0.:1.0/0.;
 __syncthreads();
 for(int row=1;row<=n;++row){
  double target=targets[row-1];
  for(int end=tid;end<m;end+=blockDim.x){
   double best=1.0/0.;int arg=-1;
   for(int start=end-1;start>=0;--start){
    int length=pos[end]-trim[end]-pos[start];
    if(length>target+18.)break;
    if(length<fmax(2.,target-18.)||!isfinite(prev[start]))continue;
    double delta=length-target;double value=prev[start]+delta*delta;
    if(value<best){best=value;arg=start;}
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


def solve(positions, trim, targets):
    cp = _cp()
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(KERNEL, "align", options=("--std=c++11", "--fmad=false"))
    started = time.perf_counter()
    m = len(positions)
    n = len(targets)
    p = cp.asarray(positions, dtype=cp.int32)
    t = cp.asarray(trim, dtype=cp.int32)
    v = cp.asarray(targets, dtype=cp.float64)
    cost = cp.empty(2 * m + 1, dtype=cp.float64)
    back = cp.full((n + 1, m), -1, dtype=cp.int32)
    a = cp.cuda.Event()
    b = cp.cuda.Event()
    a.record()
    _kernel((1,), (256,), (p, t, v, m, n, cost, back))
    b.record()
    last = float(cost[-1].get())
    bb = back.get()
    b.synchronize()
    result = np.full((n + 1, m), np.inf)
    result[n, -1] = last
    telemetry.append(
        {
            "states": m * n,
            "kernel_ms": cp.cuda.get_elapsed_time(a, b),
            "call_ms": (time.perf_counter() - started) * 1000,
        }
    )
    return result, bb
