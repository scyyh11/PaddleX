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

"""Exact CUDA inference for frozen LightGBM classifiers (no retraining, same outputs).

Reproduces LightGBM 4.6 prediction bit for bit:
  * dense float32/float64 input is converted to double; values with |v| <= kZeroThreshold
    (1e-35f) become 0.0, exactly like LightGBM's sparse row conversion;
  * Tree::NumericalDecision (missing type None/Zero/NaN, default_left) is applied per node;
  * leaf values are accumulated in double, sequentially in LightGBM's tree order;
  * the sigmoid / softmax output transform runs on the CPU with the C library exp, the same
    function LightGBM calls.
Categorical or linear trees, unusual dtypes, shape mismatches and empty inputs fall back to
the original LightGBM object.  FAST_LGBM_VERIFY=1 runs both paths and requires identical bits.

One CUDA block evaluates one input row: the row is staged in shared memory, each thread walks
a subset of the trees (node records are packed into one 24-byte struct per node), and one
thread per class adds the tree outputs in tree order.
"""
import importlib
import math
import time

import numpy as np


def _cp():
    return importlib.import_module("cupy")


KZERO = float(np.float32(1e-35))
VERIFY = False
telemetry = {
    "calls": 0,
    "rows": 0,
    "verified_calls": 0,
    "verified_rows": 0,
    "fallback": 0,
}
TIMING = False
timing = {}


def _tick(key, t0):
    t = time.perf_counter()
    v = timing.setdefault(key, [0, 0.0])
    v[0] += 1
    v[1] += t - t0
    return t


NODE = np.dtype(
    [
        ("thr", "<f8"),
        ("feat", "<i4"),
        ("left", "<i4"),
        ("right", "<i4"),
        ("dt", "u1"),
        ("pad", "V3"),
    ]
)
assert NODE.itemsize == 24
KERNEL = r"""
struct __align__(8) Node{double thr;int feat;int left;int right;unsigned char dt;unsigned char pad[3];};
struct DenseF{const float* x;int cols;
  __device__ __forceinline__ double get(int row,int f)const{return (double)x[(long long)row*cols+f];}};
struct DenseD{const double* x;int cols;
  __device__ __forceinline__ double get(int row,int f)const{return x[(long long)row*cols+f];}};
/* Row r = [X[ia[r]] (0 if ia<0), X[ib[r]], mode 0: X[ib]-X[ia] in float32 | mode 1: X[ic[r]] (0 if ic<0), tail[r]] */
struct Gather{const float* X;int D;const int* ia;const int* ib;const int* ic;int mode;const float* tail;int tl;
  __device__ __forceinline__ double get(int row,int f)const{
    if(f<D){int a=ia[row];return a>=0?(double)X[(long long)a*D+f]:0.0;}
    if(f<2*D){return (double)X[(long long)ib[row]*D+(f-D)];}
    if(f<3*D){int j=f-2*D;
      if(mode==0){int a=ia[row];float av=a>=0?X[(long long)a*D+j]:0.0f;float bv=X[(long long)ib[row]*D+j];return (double)__fsub_rn(bv,av);}
      int c=ic[row];return c>=0?(double)X[(long long)c*D+j]:0.0;}
    return (double)tail[(long long)row*tl+(f-3*D)];}};
template<typename A> __device__ __forceinline__ void lgbm_body(const A& acc,int rows,int nfeat,int trees,int k,
    const int* nleaf,const int* base,const Node* nodes,const int* leafoff,const double* leafval,double kzero,double* y){
  extern __shared__ double sm[];
  int row=blockIdx.x;
  if(row>=rows)return;
  double* xs=sm;double* vals=sm+nfeat;
  /* LightGBM drops |v|<=kZeroThreshold from each input row, so it predicts with 0.0 */
  for(int f=threadIdx.x;f<nfeat;f+=blockDim.x){double v=acc.get(row,f);if(fabs(v)<=kzero)v=0.0;xs[f]=v;}
  __syncthreads();
  for(int t=threadIdx.x;t<trees;t+=blockDim.x){
    double v;
    if(nleaf[t]<=1){v=leafval[leafoff[t]];}
    else{
      const Node* tn=nodes+base[t];int node=0;
      while(node>=0){
        const Node nd=tn[node];double f=xs[nd.feat];int mt=(nd.dt>>2)&3;
        if(isnan(f)&&mt!=2)f=0.0;
        int goleft;
        if((mt==1&&f>=-kzero&&f<=kzero)||(mt==2&&isnan(f)))goleft=(nd.dt&2)!=0;
        else goleft=f<=nd.thr;
        node=goleft?nd.left:nd.right;
      }
      v=leafval[leafoff[t]+(~node)];
    }
    vals[t]=v;
  }
  __syncthreads();
  if(threadIdx.x<k){
    double s=0.0;
    for(int t=threadIdx.x;t<trees;t+=k)s+=vals[t];
    y[(long long)row*k+threadIdx.x]=s;
  }
}
#define MODEL_ARGS int nfeat,int trees,int k,const int* nleaf,const int* base,const Node* nodes,const int* leafoff,const double* leafval,double kzero
#define MODEL_PASS rows,nfeat,trees,k,nleaf,base,nodes,leafoff,leafval,kzero
extern "C" __global__ void lgbm_f32(const float* x,int rows,MODEL_ARGS,double* y){
  DenseF a{x,nfeat};lgbm_body(a,MODEL_PASS,y);}
extern "C" __global__ void lgbm_f64(const double* x,int rows,MODEL_ARGS,double* y){
  DenseD a{x,nfeat};lgbm_body(a,MODEL_PASS,y);}
/* rows assembled from X; index and tail arrays live in one int32 buffer at the given offsets */
extern "C" __global__ void lgbm_gather(const float* X,int D,const int* buf,int ia,int ib,int ic,int mode,int tail,int tl,
    int rows,MODEL_ARGS,double* yb,int yoff){
  Gather a{X,D,buf+ia,buf+ib,buf+ic,mode,(const float*)(buf+tail),tl};lgbm_body(a,MODEL_PASS,yb+yoff);}
"""
_kernels = None


def kernels():
    global _kernels
    if _kernels is None:
        cp = _cp()
        module = cp.RawModule(code=KERNEL, options=("--std=c++11", "--fmad=false"))
        _kernels = {
            np.dtype(np.float32): module.get_function("lgbm_f32"),
            np.dtype(np.float64): module.get_function("lgbm_f64"),
            "gather": module.get_function("lgbm_gather"),
        }
    return _kernels


def _parse(model_string):
    head, _, body = model_string.partition("\nTree=")
    header = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
    trees = []
    for block in ("Tree=" + body).split("\nTree=")[0:]:
        block = block if block.startswith("Tree=") else "Tree=" + block
        text = block.split("end of trees")[0]
        kv = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        if "num_leaves" not in kv:
            continue
        trees.append(kv)
    return header, trees


class ExactLgbm:
    def __init__(self, booster):
        # Booster.predict() uses best_iteration when one was recorded, otherwise all trees.
        best = booster.best_iteration
        text = booster.model_to_string(num_iteration=best if best and best > 0 else -1)
        header, trees = _parse(text)
        if any(
            line.strip() == "average_output"
            for line in text.partition("\nTree=")[0].splitlines()
        ):
            raise ValueError("random forest averaging")
        self.k = int(header["num_tree_per_iteration"])
        self.num_class = int(header["num_class"])
        self.n_features = int(header["max_feature_idx"]) + 1
        objective = header["objective"].split()
        self.kind = objective[0]
        if self.kind == "binary":
            params = dict(p.split(":", 1) for p in objective[1:] if ":" in p)
            self.sigmoid = float(params.get("sigmoid", "1"))
        elif self.kind != "multiclass":
            raise ValueError("unsupported objective %s" % header["objective"])
        base = []
        nleaf = []
        leafoff = []
        feat = []
        thr = []
        dt = []
        left = []
        right = []
        leafval = []
        for kv in trees:
            nl = int(kv["num_leaves"])
            if int(kv.get("num_cat", "0")) or kv.get("is_linear", "0") != "0":
                raise ValueError("categorical or linear tree")
            base.append(len(feat))
            nleaf.append(nl)
            leafoff.append(len(leafval))
            leafval.extend(float(v) for v in kv["leaf_value"].split())
            if nl > 1:
                f = [int(v) for v in kv["split_feature"].split()]
                feat.extend(f)
                thr.extend(float(v) for v in kv["threshold"].split())
                dt.extend(int(v) for v in kv["decision_type"].split())
                left.extend(int(v) for v in kv["left_child"].split())
                right.extend(int(v) for v in kv["right_child"].split())
                if any(d & 1 for d in dt[-len(f) :]):
                    raise ValueError("categorical split")
        self.trees = len(base)
        if not (
            self.trees % self.k == 0 and (self.trees + self.n_features) * 8 <= 48 * 1024
        ):
            raise ValueError("tree layout")
        if feat and max(feat) >= self.n_features:
            raise ValueError("feature index")
        cp = _cp()
        nodes = np.zeros(max(len(feat), 1), dtype=NODE)
        if feat:
            nodes["thr"] = thr
            nodes["feat"] = feat
            nodes["left"] = left
            nodes["right"] = right
            nodes["dt"] = dt
        self.arrays = [
            cp.asarray(np.asarray(nleaf, dtype=np.int32)),
            cp.asarray(np.asarray(base, dtype=np.int32)),
            cp.asarray(nodes.view(np.uint8)),
            cp.asarray(np.asarray(leafoff, dtype=np.int32)),
            cp.asarray(np.asarray(leafval, dtype=np.float64)),
        ]
        self.threads = 256 if self.trees >= 256 else 128
        self.shared = (self.trees + self.n_features) * 8
        self.model_args = (
            np.int32(self.n_features),
            np.int32(self.trees),
            np.int32(self.k),
            *self.arrays,
            np.float64(KZERO),
        )
        kernels()
        # compile and run once outside document timing
        self.raw(np.zeros((1, self.n_features), dtype=np.float32))
        self.raw(np.zeros((1, self.n_features)))
        cp.cuda.get_current_stream().synchronize()

    def raw(self, x):
        cp = _cp()
        n, d = x.shape
        gx = cp.asarray(np.ascontiguousarray(x))
        gy = cp.empty((n, self.k), dtype=cp.float64)
        kernels()[gx.dtype](
            (n,),
            (self.threads,),
            (gx, np.int32(n), *self.model_args, gy),
            shared_mem=self.shared,
        )
        return cp.asnumpy(gy)

    def launch_gather(self, X, D, buf, ia, ib, ic, mode, tail, tl, rows, out, yoff):
        kernels()["gather"](
            (rows,),
            (self.threads,),
            (
                X,
                np.int32(D),
                buf,
                np.int32(ia),
                np.int32(ib),
                np.int32(ic),
                np.int32(mode),
                np.int32(tail),
                np.int32(tl),
                np.int32(rows),
                *self.model_args,
                out,
                np.int32(yoff),
            ),
            shared_mem=self.shared,
        )

    def predict(self, x):
        """Same values as Booster.predict(x) (default output transform)."""
        return self.transform(self.raw(x))

    def transform(self, raw):
        if self.kind == "binary":
            s = self.sigmoid
            exp = math.exp
            values = []
            for r in raw[:, 0].tolist():
                try:
                    values.append(1.0 / (1.0 + exp(-s * r)))
                except OverflowError:
                    values.append(0.0)
            return np.asarray(values, dtype=np.float64)
        exp = math.exp
        values = []
        for row in raw.tolist():
            wmax = row[0]
            for v in row[1:]:
                wmax = wmax if v < wmax else v
            e = [exp(v - wmax) for v in row]
            wsum = 0.0
            for v in e:
                wsum += v
            values.append([v / wsum for v in e])
        return np.asarray(values, dtype=np.float64).reshape(raw.shape)


def _usable(x, n_features):
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.dtype in (np.float32, np.float64)
        and x.shape[0] > 0
        and x.shape[1] == n_features
    )


def _check(fast, original, what):
    if not (
        fast.shape == original.shape
        and fast.dtype == original.dtype
        and np.array_equal(fast, original)
    ):
        raise AssertionError(
            "fast LightGBM mismatch in %s: max abs diff %r"
            % (what, np.max(np.abs(fast - original)))
        )


class FastBooster:
    def __init__(self, booster, exact):
        self.original = booster
        self.exact = exact

    def predict(self, x, **kwargs):
        extra = {k: v for k, v in kwargs.items() if k != "num_threads"}
        if extra or not _usable(x, self.exact.n_features):
            telemetry["fallback"] += 1
            return self.original.predict(x, **kwargs)
        out = self.exact.predict(x)
        telemetry["calls"] += 1
        telemetry["rows"] += len(x)
        if VERIFY:
            _check(out, self.original.predict(x, **kwargs), "Booster.predict")
            telemetry["verified_calls"] += 1
            telemetry["verified_rows"] += len(x)
        return out

    def __getattr__(self, name):
        return getattr(self.original, name)


class FastLGBMClassifier:
    """Drop-in for LGBMClassifier.predict_proba / booster_.predict; other attributes delegate."""

    def __init__(self, model):
        self.original = model
        self.exact = ExactLgbm(model.booster_)
        self.classes_ = model.classes_
        self.booster_ = FastBooster(model.booster_, self.exact)

    def predict_proba(self, x, **kwargs):
        if kwargs or not _usable(x, self.exact.n_features):
            telemetry["fallback"] += 1
            return self.original.predict_proba(x, **kwargs)
        result = self.exact.predict(x)
        out = (
            result
            if self.exact.kind == "multiclass"
            else np.vstack((1.0 - result, result)).transpose()
        )
        telemetry["calls"] += 1
        telemetry["rows"] += len(x)
        if VERIFY:
            _check(out, self.original.predict_proba(x), "predict_proba")
            telemetry["verified_calls"] += 1
            telemetry["verified_rows"] += len(x)
        return out

    def __getattr__(self, name):
        return getattr(self.original, name)


def convert(model):
    try:
        return FastLGBMClassifier(model)
    except ValueError:
        return model


class _Staging:
    """Pinned host buffers and device buffers reused across calls (grown on demand)."""

    def __init__(self):
        self.cap_in = 0
        self.cap_out = 0

    def inputs(self, n):
        if n > self.cap_in:
            cp = _cp()
            self.cap_in = max(n, 2 * self.cap_in, 1 << 16)
            self.pin_in = cp.cuda.alloc_pinned_memory(self.cap_in * 4)
            self.host_in = np.frombuffer(self.pin_in, dtype=np.int32, count=self.cap_in)
            self.host_in_f = self.host_in.view(np.float32)
            self.dev_in = cp.empty(self.cap_in, dtype=cp.int32)
        return self.host_in, self.host_in_f, self.dev_in

    def outputs(self, n):
        if n > self.cap_out:
            cp = _cp()
            self.cap_out = max(n, 2 * self.cap_out, 1 << 14)
            self.pin_out = cp.cuda.alloc_pinned_memory(self.cap_out * 8)
            self.host_out = np.frombuffer(
                self.pin_out, dtype=np.float64, count=self.cap_out
            )
            self.dev_out = cp.empty(self.cap_out, dtype=cp.float64)
        return self.host_out, self.dev_out


_staging = None


class RowGather:
    """A float32 feature matrix X (n, D) kept on the GPU, and batched model calls on rows
    assembled from it without building them on the CPU.  Row r of a job is
        [X[ia[r]] (zeros if ia[r] < 0), X[ib[r]],
         mode 0: X[ib[r]] - X[ia[r]] (float32, zeros for ia < 0) | mode 1: X[ic[r]] (zeros if ic[r] < 0),
         tail[r]]
    which is how the heading pair / parent / span / relation inputs are laid out.  Results are
    identical to calling predict_proba / booster_.predict on the materialized rows; all jobs of
    one run() share one pinned upload, one download and one synchronization."""

    def __init__(self, x):
        global _staging
        if not (isinstance(x, np.ndarray) and x.ndim == 2 and x.dtype == np.float32):
            raise ValueError("RowGather needs a 2-D float32 matrix")
        cp = _cp()
        self.x = np.ascontiguousarray(x)
        self.D = x.shape[1]
        self.dev = cp.asarray(self.x) if len(self.x) else None
        if _staging is None:
            _staging = _Staging()

    def dense(self, ia, ib, ic, mode, tail):
        x = self.x
        za = ia < 0
        xa = x[np.where(za, 0, ia)]
        xa[za] = 0
        xb = x[ib]
        if mode == 0:
            third = xb - xa
        else:
            zc = ic < 0
            third = x[np.where(zc, 0, ic)]
            third[zc] = 0
        return np.concatenate([xa, xb, third, tail], axis=1)

    def run(self, jobs):
        """jobs: (model, kind, ia, ib, ic, mode, tail) with kind 'proba' or 'booster'."""
        if TIMING:
            t0 = time.perf_counter()
        results = [None] * len(jobs)
        plan = []
        ints = 0
        outs = 0
        for n, (model, kind, ia, ib, ic, mode, tail) in enumerate(jobs):
            rows = len(ib)
            exact = getattr(model, "exact", None)
            if (
                exact is None
                or rows == 0
                or self.dev is None
                or getattr(tail, "ndim", 0) != 2
                or tail.shape[0] != rows
                or 3 * self.D + tail.shape[1] != exact.n_features
            ):
                x = self.dense(
                    np.asarray(ia),
                    np.asarray(ib),
                    None if ic is None else np.asarray(ic),
                    mode,
                    np.asarray(tail, dtype=np.float32),
                )
                results[n] = (
                    model.predict_proba(x)
                    if kind == "proba"
                    else model.booster_.predict(x, num_threads=1)
                )
                continue
            tl = tail.shape[1]
            plan.append((n, model, kind, rows, tl, ints, mode, outs, ia, ib, ic, tail))
            ints += 3 * rows + rows * tl
            outs += rows * exact.k
        if plan:
            cp = _cp()
            host, host_f, dev = _staging.inputs(ints)
            hout, dout = _staging.outputs(outs)
            for n, model, kind, rows, tl, io, mode, oo, ia, ib, ic, tail in plan:
                host[io : io + rows] = ia
                host[io + rows : io + 2 * rows] = ib
                host[io + 2 * rows : io + 3 * rows] = -1 if ic is None else ic
                host_f[io + 3 * rows : io + 3 * rows + rows * tl] = np.asarray(
                    tail, dtype=np.float32
                ).reshape(-1)
            stream = cp.cuda.get_current_stream()
            cp.cuda.runtime.memcpyAsync(
                dev.data.ptr,
                host.ctypes.data,
                ints * 4,
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )
            if TIMING:
                t0 = _tick("prep_h2d", t0)
            for n, model, kind, rows, tl, io, mode, oo, *_ in plan:
                model.exact.launch_gather(
                    self.dev,
                    self.D,
                    dev,
                    io,
                    io + rows,
                    io + 2 * rows,
                    mode,
                    io + 3 * rows,
                    tl,
                    rows,
                    dout,
                    oo,
                )
            cp.cuda.runtime.memcpyAsync(
                hout.ctypes.data,
                dout.data.ptr,
                outs * 8,
                cp.cuda.runtime.memcpyDeviceToHost,
                stream.ptr,
            )
            if TIMING:
                t0 = _tick("launch", t0)
            stream.synchronize()
            if TIMING:
                t0 = _tick("wait", t0)
            for n, model, kind, rows, tl, io, mode, oo, ia, ib, ic, tail in plan:
                exact = model.exact
                value = exact.transform(
                    hout[oo : oo + rows * exact.k].reshape(rows, exact.k)
                )
                if kind == "proba" and exact.kind == "binary":
                    value = np.vstack((1.0 - value, value)).transpose()
                telemetry["calls"] += 1
                telemetry["rows"] += rows
                if VERIFY:
                    x = self.dense(
                        np.asarray(ia),
                        np.asarray(ib),
                        None if ic is None else np.asarray(ic),
                        mode,
                        np.asarray(tail, dtype=np.float32),
                    )
                    ref = (
                        model.original.predict_proba(x)
                        if kind == "proba"
                        else model.original.booster_.predict(x, num_threads=1)
                    )
                    _check(value, ref, "gathered " + kind)
                    telemetry["verified_calls"] += 1
                    telemetry["verified_rows"] += rows
                results[n] = value
            if TIMING:
                t0 = _tick("transform", t0)
        return results
