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

"""Inference-only CUDA traversal of frozen sklearn trees (no model retraining).

Double precision thresholds/leaf values, deterministic tree-order accumulation.
ExtraTrees inputs retain sklearn's float32 conversion; HGB uses float64.
"""
import importlib
import time

import numpy as np


def _cp():
    return importlib.import_module("cupy")


KERNEL = r"""
extern "C" __global__ void traverse(const double* x, int rows, int cols, int trees,
 const int* roots,const int* left,const int* right,const int* feature,
 const double* threshold,const unsigned char* missing_left,const double* leaf,
 int outputs,double* predictions) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;
 int row=blockIdx.y;
 if(t>=trees || row>=rows) return;
 int node=roots[t];
 while(feature[node]>=0) {
  double value=x[row*cols+feature[node]];
  node=(isnan(value)?missing_left[node]:(value<=threshold[node]))?left[node]:right[node];
 }
 for(int c=0;c<outputs;++c)predictions[((long long)row*trees+t)*outputs+c]=leaf[(long long)node*outputs+c];
}
extern "C" __global__ void reduce_ordered(const double* p,int rows,int trees,int outputs,int groups,const double* base,double* y) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows)return;
 for(int c=0;c<outputs;++c){
  double value=base[c];
  for(int t=0;t<trees;++t)value+=p[((long long)row*trees+t)*outputs+c];
  y[row*outputs+c]=value;
 }
}
"""
_kernel = None
telemetry = []


def kernels():
    global _kernel
    if _kernel is None:
        cp = _cp()
        module = cp.RawModule(code=KERNEL, options=("--std=c++11", "--fmad=false"))
        _kernel = (
            module.get_function("traverse"),
            module.get_function("reduce_ordered"),
        )
    return _kernel


class GpuTrees:
    def __init__(self, model):
        cp = _cp()
        self.classes_ = model.classes_
        self.kind = "extra" if hasattr(model, "estimators_") else "hgb"
        self.original = model
        roots = []
        left = []
        right = []
        feature = []
        threshold = []
        missing = []
        values = []
        self.outputs = (
            len(self.classes_) if self.kind == "extra" else model.n_trees_per_iteration_
        )

        def add(l, r, f, th, miss, val):
            offset = len(left)
            roots.append(offset)
            left.extend((np.asarray(l) + offset).tolist())
            right.extend((np.asarray(r) + offset).tolist())
            feature.extend(np.asarray(f).tolist())
            threshold.extend(np.asarray(th).tolist())
            missing.extend(np.asarray(miss).tolist())
            values.extend(np.asarray(val).tolist())

        if self.kind == "extra":
            self.base = np.zeros(self.outputs)
            for est in model.estimators_:
                tr = est.tree_
                v = tr.value[:, 0, :].copy()
                den = v.sum(axis=1, keepdims=True)
                v /= np.where(den == 0, 1, den)
                f = tr.feature.copy()
                f[tr.children_left < 0] = -1
                add(
                    tr.children_left,
                    tr.children_right,
                    f,
                    tr.threshold,
                    getattr(tr, "missing_go_to_left", np.zeros(tr.node_count)),
                    v,
                )
        else:
            self.base = model._baseline_prediction.reshape(-1)
            for iteration in model._predictors:
                for c, tree in enumerate(iteration):
                    n = tree.nodes
                    if np.any(n["is_categorical"]):
                        raise ValueError(
                            "Categorical trees require a different CUDA decoder"
                        )
                    f = n["feature_idx"].astype(np.int32)
                    f[n["is_leaf"] != 0] = -1
                    v = np.zeros((len(n), self.outputs))
                    v[:, c] = n["value"]
                    add(
                        n["left"],
                        n["right"],
                        f,
                        n["num_threshold"],
                        n["missing_go_to_left"],
                        v,
                    )
        cp = _cp()
        self.trees = len(roots)
        self.arrays = [
            cp.asarray(a, dtype=d)
            for a, d in [
                (roots, np.int32),
                (left, np.int32),
                (right, np.int32),
                (feature, np.int32),
                (threshold, np.float64),
                (missing, np.uint8),
                (values, np.float64),
            ]
        ]
        self.base_gpu = cp.asarray(self.base)
        kernels()
        cp.cuda.get_current_stream().synchronize()

    def predict_proba(self, x):
        cp = _cp()
        from scipy.special import expit, softmax

        started = time.perf_counter()
        x = np.asarray(
            x, dtype=np.float32 if self.kind == "extra" else np.float64
        ).astype(np.float64, copy=False)
        n, d = x.shape
        if not n:
            return np.empty((0, len(self.classes_)))
        gx = cp.asarray(x)
        p = cp.empty((n, self.trees, self.outputs), dtype=cp.float64)
        y = cp.empty((n, self.outputs), dtype=cp.float64)
        a, b = kernels()
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        a(
            ((self.trees + 127) // 128, n),
            (128,),
            (gx, n, d, self.trees, *self.arrays, self.outputs, p),
        )
        b(
            ((n + 127) // 128,),
            (128,),
            (p, n, self.trees, self.outputs, self.outputs, self.base_gpu, y),
        )
        stop.record()
        result = cp.asnumpy(y)
        stop.synchronize()
        telemetry.append(
            {
                "kind": self.kind,
                "rows": n,
                "trees": self.trees,
                "kernel_ms": cp.cuda.get_elapsed_time(start, stop),
                "call_ms": (time.perf_counter() - started) * 1000,
            }
        )
        if self.kind == "extra":
            return result / self.trees
        if self.outputs == 1:
            p = expit(result[:, 0])
            return np.column_stack([1 - p, p])
        return softmax(result, axis=1)


def convert_package(package):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

    result = dict(package)
    for k, v in result.items():
        if isinstance(v, (ExtraTreesClassifier, HistGradientBoostingClassifier)):
            result[k] = GpuTrees(v)
        elif type(v).__name__ == "LGBMClassifier":
            from . import gpu_lgbm

            result[k] = gpu_lgbm.convert(v)
    return result
