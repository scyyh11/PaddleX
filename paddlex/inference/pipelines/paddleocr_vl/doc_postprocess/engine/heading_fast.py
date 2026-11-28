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

"""Vectorized, bit-identical builders for the heading outline model inputs.

Every matrix equals the row-by-row construction in heading_ml_features.pair_features,
heading_parent_decode.parent_feature, span_features_fast.prepare and heading_graph
(graph_edges, relation_features): the same double expressions are evaluated elementwise and
rounded to float32 once, exactly as np.asarray(list, dtype=float32) does.
HEADING_FAST_VERIFY=1 rebuilds every matrix the original way and requires identical bits.
"""
import math

import numpy as np

from .heading_ml_features import fsize, norm, number
from .heading_structure_features import sequence

VERIFY = False


def _ids(values):
    table = {}
    return np.asarray([table.setdefault(v, len(table)) for v in values], dtype=np.int64)


def _rawkey(v):
    return (
        (type(v).__name__, tuple(v))
        if isinstance(v, (list, tuple))
        else (type(v).__name__, v)
    )


_NUM_CACHE = {}


def _numfeat(na, nb):
    key = (na, nb)
    v = _NUM_CACHE.get(key)
    if v is None:
        common = 0
        for x, y in zip(na, nb):
            if x != y:
                break
            common += 1
        v = (
            common,
            len(na),
            len(nb),
            int(
                bool(
                    na
                    and nb
                    and len(na) == len(nb)
                    and na[:-1] == nb[:-1]
                    and nb[-1] == na[-1] + 1
                )
            ),
            int(bool(na and nb and len(nb) > len(na) and nb[: len(na)] == na)),
            int(bool(nb and nb[-1] == 1)),
            int(bool(na and nb and nb[0] > na[0])),
        )
        if len(_NUM_CACHE) > 200000:
            _NUM_CACHE.clear()
        _NUM_CACHE[key] = v
    return v


_LOG1P32 = {}


def _gap_log(gap):
    v = _LOG1P32.get(gap)
    if v is None:
        v = np.array([False, min(gap, 500), np.log1p(gap)], dtype=np.float32)[2]
        _LOG1P32[gap] = v
    return v


def pairwise_sums(a, off, cnt):
    """np.add.reduce(a[o:o + c]) for many (o, c) pairs (c >= 1) with positive values a, in numpy's
    own pairwise order: fewer than 8 values are added left to right; up to 128 values use 8 strided
    partial sums r[j] = a[j] + a[8+j] + ... combined as ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) followed
    by the remaining values left to right; longer runs are split at n2 = n//2 - (n//2)%8 and the two
    halves added.  Sequential sums are taken with cumsum (strictly left to right); unused slots are
    padded with +0.0, which leaves the positive partial sums unchanged."""
    off = np.asarray(off, dtype=np.int64)
    cnt = np.asarray(cnt, dtype=np.int64)
    out = np.empty(len(off), dtype=np.float64)
    ext = np.concatenate([a, np.zeros(136)]) if len(off) else a
    zero = len(a)  # index of a padding +0.0
    small = cnt < 8
    if small.any():
        idx = np.flatnonzero(small)
        o = off[idx]
        c = cnt[idx]
        j = np.arange(7)[None, :]
        g = ext[np.where(j < c[:, None], o[:, None] + j, zero)]
        out[idx] = np.cumsum(g, axis=1)[:, -1]
    mid = (cnt >= 8) & (cnt <= 128)
    if mid.any():
        idx = np.flatnonzero(mid)
        o = off[idx]
        c = cnt[idx]
        nb = c // 8
        blk = np.arange(16)[None, :, None]
        lane = np.arange(8)[None, None, :]
        pos = o[:, None, None] + 8 * blk + lane
        g = ext[np.where(blk < nb[:, None, None], pos, zero)]
        r = np.cumsum(g, axis=1)[:, -1, :]
        res = ((r[:, 0] + r[:, 1]) + (r[:, 2] + r[:, 3])) + (
            (r[:, 4] + r[:, 5]) + (r[:, 6] + r[:, 7])
        )
        rem = c - 8 * nb
        j = np.arange(7)[None, :]
        t = ext[np.where(j < rem[:, None], (o + 8 * nb)[:, None] + j, zero)]
        out[idx] = np.cumsum(np.concatenate([res[:, None], t], axis=1), axis=1)[:, -1]
    big = cnt > 128
    if big.any():
        idx = np.flatnonzero(big)
        o = off[idx]
        c = cnt[idx]
        n2 = c // 2
        n2 = n2 - n2 % 8
        out[idx] = pairwise_sums(a, o, n2) + pairwise_sums(a, o + n2, c - n2)
    return out


def _require(fast, ref, what):
    if not (
        fast.dtype == ref.dtype
        and fast.shape == ref.shape
        and np.array_equal(fast, ref)
    ):
        raise AssertionError("heading_fast mismatch: " + what)


class Records:
    """Per-candidate scalars used by the pairwise features."""

    def __init__(self, meta, x):
        self.meta = meta
        self.x = x
        self.n = len(meta)
        self.fs = np.asarray([fsize(r) for r in meta], dtype=np.float64)
        self.dens = np.asarray(
            [r["visual"].get("density", 0.25) for r in meta], dtype=np.float64
        )
        self.bx = np.asarray([r["box"][0] / r["width"] for r in meta], dtype=np.float64)
        self.ctr = np.asarray(
            [(r["box"][0] + r["box"][2]) / r["width"] / 2 for r in meta],
            dtype=np.float64,
        )
        self.page = np.asarray([r["page"] for r in meta], dtype=np.int64)
        self.level = np.asarray([r["level"] for r in meta], dtype=np.int64)
        self.fam = _ids([_rawkey(r["family"]) for r in meta])
        self.lab = _ids([_rawkey(r["label"]) for r in meta])
        self.normk = _ids([norm(r["text"]) for r in meta])
        self.num = [number(r["text"]) for r in meta]
        self._ords = None
        # numbering tuples as padded int64 arrays for the vectorized _numfeat (None if any part is huge)
        width = max((len(q) for q in self.num), default=0)
        self.numlen = np.asarray([len(q) for q in self.num], dtype=np.int64)
        if all(abs(v) < 2**62 for q in self.num for v in q):
            arr = np.zeros((self.n, max(width, 1)), dtype=np.int64)
            for i, q in enumerate(self.num):
                if q:
                    arr[i, : len(q)] = q
            self.numarr = arr
            self.numfirst = arr[:, 0].copy()
            self.numlast = (
                arr[np.arange(self.n), np.maximum(self.numlen - 1, 0)]
                if self.n
                else np.zeros(0, dtype=np.int64)
            )
        else:
            self.numarr = None

    @property
    def ords(self):
        if self._ords is None:
            from .heading_graph import ordinal

            self._ords = [ordinal(r["text"]) for r in self.meta]
        return self._ords

    def f17(self, a, b):
        """pair_features scalar block for records a (first) and b (second), float32."""
        f = np.empty((len(a), 17), dtype=np.float64)
        f[:, 0] = self.fs[b] / self.fs[a]
        f[:, 1] = self.dens[b] / np.maximum(self.dens[a], 0.01)
        f[:, 2] = self.bx[b] - self.bx[a]
        f[:, 3] = self.ctr[b] - self.ctr[a]
        f[:, 4] = self.page[b] - self.page[a]
        f[:, 5] = self.page[b] == self.page[a]
        f[:, 6] = self.level[b] - self.level[a]
        f[:, 7] = self.fam[a] == self.fam[b]
        f[:, 8] = self.lab[a] == self.lab[b]
        f[:, 9] = self.normk[a] == self.normk[b]
        if len(a) and self.numarr is not None:
            f[:, 10:17] = self.numfeat(a, b)
            if VERIFY:
                num = self.num
                _require(
                    f[:, 10:17],
                    np.asarray(
                        [
                            _numfeat(num[i], num[j])
                            for i, j in zip(a.tolist(), b.tolist())
                        ],
                        dtype=np.float64,
                    ),
                    "number features",
                )
        else:
            num = self.num
            f[:, 10:17] = (
                [_numfeat(num[i], num[j]) for i, j in zip(a.tolist(), b.tolist())]
                if len(a)
                else np.empty((0, 7))
            )
        return f.astype(np.float32)

    def numfeat(self, a, b):
        """_numfeat(num[a], num[b]) for index arrays a, b: common prefix length, both lengths and the
        four sibling/descendant/first-item/order flags, with the same integer comparisons.
        """
        la = self.numlen[a]
        lb = self.numlen[b]
        A = self.numarr[a]
        Bm = self.numarr[b]
        k = np.arange(A.shape[1])[None, :]
        eq = (A == Bm) & (k < np.minimum(la, lb)[:, None])
        common = np.cumprod(eq, axis=1).sum(axis=1)
        fa = self.numfirst[a]
        fb = self.numfirst[b]
        za = self.numlast[a]
        zb = self.numlast[b]
        c1 = (la > 0) & (lb > 0) & (la == lb) & (common >= la - 1) & (zb == za + 1)
        c2 = (la > 0) & (lb > la) & (common >= la)
        c3 = (lb > 0) & (zb == 1)
        c4 = (la > 0) & (lb > 0) & (fb > fa)
        return np.stack([common, la, lb, c1, c2, c3, c4], axis=1).astype(np.float64)

    def pairs(self, a, b, zero_first=None):
        """Rows of pair_features(meta[a], meta[b], x[a], x[b]); zero_first rows use fa=0."""
        xa = self.x[a]
        if zero_first is not None and zero_first.any():
            xa = xa.copy()
            xa[zero_first] = 0
        xb = self.x[b]
        return np.concatenate([xa, xb, xb - xa, self.f17(a, b)], axis=1)

    def consecutive_pairs(self):
        from .heading_ml_features import pair_features

        n = self.n
        if n < 2:
            return None
        a = np.arange(0, n - 1)
        b = np.arange(1, n)
        out = self.pairs(a, b)
        if VERIFY:
            meta, x = self.meta, self.x
            _require(
                out,
                np.asarray(
                    [
                        pair_features(meta[i - 1], meta[i], x[i - 1], x[i])
                        for i in range(1, n)
                    ]
                ),
                "consecutive pairs",
            )
        return out

    def consecutive_job(self):
        """(ia, ib, tail) for gpu_lgbm.RowGather mode 0: the rows of consecutive_pairs()."""
        a = np.arange(0, self.n - 1, dtype=np.int64)
        b = np.arange(1, self.n, dtype=np.int64)
        return a, b, self.f17(a, b)

    def parent_job(self, i, parents):
        """(ia, ib, tail) for gpu_lgbm.RowGather mode 0: the rows of parent_matrix(i, parents)."""
        none = np.asarray([j is None for j in parents])
        a = np.asarray([i if j is None else j for j in parents], dtype=np.int64)
        b = np.full(len(parents), i, dtype=np.int64)
        gaps = [i if j is None else i - j for j in parents]
        extra = np.empty((len(parents), 3), dtype=np.float32)
        extra[:, 0] = none
        extra[:, 1] = [min(g, 500) for g in gaps]
        extra[:, 2] = [_gap_log(g) for g in gaps]
        return np.where(none, -1, a), b, np.concatenate([self.f17(a, b), extra], axis=1)

    def parent_job_pairs(self, js, cs):
        """parent_job rows for many (parent j or -1 for none, child c) pairs."""
        js = np.asarray(js, dtype=np.int64)
        cs = np.asarray(cs, dtype=np.int64)
        none = js < 0
        a = np.where(none, cs, js)
        gaps = np.where(none, cs, cs - js)
        extra = np.empty((len(js), 3), dtype=np.float32)
        if getattr(self, "_gap_table", None) is None:
            self._gap_table = np.asarray(
                [_gap_log(g) for g in range(self.n + 1)], dtype=np.float32
            )
        extra[:, 0] = none
        extra[:, 1] = np.minimum(gaps, 500)
        extra[:, 2] = self._gap_table[gaps]
        return (
            np.where(none, -1, a),
            cs,
            np.concatenate([self.f17(a, cs), extra], axis=1),
        )

    def relation_job(self, a, b):
        """(ia, ib, tail) for gpu_lgbm.RowGather mode 0: the rows of relation_matrix(self, a, b)."""
        ords = self.ords
        e = np.empty((len(a), 11), dtype=np.float64)
        e[:, :9] = (
            [_relation_extra(ords[i], ords[j]) for i, j in zip(a.tolist(), b.tolist())]
            if len(a)
            else np.empty((0, 9))
        )
        e[:, 9] = self.page[a] == 1
        e[:, 10] = self.page[b] == 1
        return a, b, np.concatenate([self.f17(a, b), e.astype(np.float32)], axis=1)

    def parent_matrix(self, i, parents):
        """Rows of heading_parent_decode.parent_feature for child i and each parent (None first)."""
        none = np.asarray([j is None for j in parents])
        a = np.asarray([i if j is None else j for j in parents], dtype=np.int64)
        b = np.full(len(parents), i, dtype=np.int64)
        gaps = [i if j is None else i - j for j in parents]
        extra = np.empty((len(parents), 3), dtype=np.float32)
        extra[:, 0] = none
        extra[:, 1] = [min(g, 500) for g in gaps]
        extra[:, 2] = [_gap_log(g) for g in gaps]
        out = np.concatenate([self.pairs(a, b, none), extra], axis=1)
        if VERIFY:
            from .heading_parent_decode import parent_feature

            meta, x = self.meta, self.x
            ref = np.asarray(
                [
                    parent_feature(
                        meta[j] if j is not None else None,
                        meta[i],
                        x[j] if j is not None else None,
                        x[i],
                        i - j if j is not None else i,
                    )
                    for j in parents
                ]
            )
            _require(out, ref, "parent matrix")
        return out


class Spans:
    """Vectorized span_features_fast.prepare(...)(start, end) for one end and many starts."""

    def __init__(self, rec):
        meta = rec.meta
        n = rec.n
        self.rec = rec
        self.n = n
        self.fonts = np.array([fsize(r) for r in meta])
        nums = [number(r["text"]) for r in meta]
        fams = [tuple(r["family"]) for r in meta]
        self.famid = _ids(fams)
        self.numid = _ids(nums)
        ufam = {}
        unum = {}
        for f, k in zip(fams, self.famid.tolist()):
            ufam.setdefault(k, f)
        for v, k in zip(nums, self.numid.tolist()):
            unum.setdefault(k, v)
        self.counts = np.zeros((len(ufam), n + 1), dtype=np.int64)
        for k in range(len(ufam)):
            self.counts[k] = np.r_[0, np.cumsum(self.famid == k)]
        # prefix[k][e] = #{q in nums[:e] : num_k and q[:len(num_k)] == num_k}
        holders = {}
        for pos, q in enumerate(nums):
            for l in range(1, len(q) + 1):
                holders.setdefault(q[:l], []).append(pos)
        self.prefix = np.zeros((len(unum), n + 1), dtype=np.int64)
        for k in range(len(unum)):
            num = unum[k]
            if num and num in holders:
                ind = np.zeros(n, dtype=np.int64)
                ind[holders[num]] = 1
                self.prefix[k, 1:] = np.cumsum(ind)
        if VERIFY:
            for k in range(len(unum)):
                num = unum[k]
                _require(
                    self.prefix[k],
                    np.r_[
                        0, np.cumsum([bool(num and q[: len(num)] == num) for q in nums])
                    ].astype(np.int64),
                    "span prefix counts",
                )
        self.rawfam = _ids([_rawkey(r["family"]) for r in meta])
        self.page = rec.page
        self._log1p = {}

    def job_pairs(self, ts, es):
        """(ia, ib, ic, tail) for many (start t, end e) pairs, t <= e: the rows of matrix([t], e)."""
        n = self.n
        fonts = self.fonts
        page = self.page
        t = np.asarray(ts, dtype=np.int64)
        e_ = np.asarray(es, dtype=np.int64)
        k = len(t)
        ln = e_ - t
        e = np.empty((k, 15), dtype=np.float64)
        e[:, 0] = e_ == n - 1
        e[:, 1] = t == 0
        e[:, 2] = ln
        e[:, 3] = self.log1p_table()[ln]
        e[:, 4] = ln / max(n, 1)
        e[:, 5] = page[e_] - page[t]
        inner = t < e_
        e[~inner, 6:9] = 1.0
        if inner.any():
            ii = np.flatnonzero(inner)
            ti = t[ii]
            ei = e_[ii]
            mx = np.empty(len(ii))
            mn = np.empty(len(ii))
            for end in np.unique(ei).tolist():
                sel = ei == end
                seg = fonts[: end + 1]
                sufmax = np.maximum.accumulate(seg[::-1])[::-1]
                sufmin = np.minimum.accumulate(seg[::-1])[::-1]
                mx[sel] = sufmax[ti[sel] + 1]
                mn[sel] = sufmin[ti[sel] + 1]
            ft = fonts[ti]
            e[ii, 6] = mx / ft
            e[ii, 7] = pairwise_sums(fonts, ti + 1, ei - ti) / (ei - ti) / ft
            e[ii, 8] = mn / ft
        width = np.maximum(1, ln)
        e[:, 9] = (
            self.counts[self.famid[t], e_ + 1] - self.counts[self.famid[t], t + 1]
        ) / width
        e[:, 10] = (
            self.prefix[self.numid[t], e_ + 1] - self.prefix[self.numid[t], t + 1]
        ) / width
        first = np.where(inner, t + 1, t)
        e[:, 11] = fonts[first] / fonts[t]
        has = e_ + 1 < n
        nx = np.minimum(e_ + 1, n - 1)
        e[:, 12] = np.where(has, fonts[nx] / fonts[t], 1.0)
        e[:, 13] = np.where(has, self.rawfam[nx] == self.rawfam[t], 0)
        e[:, 14] = np.where(has, page[nx] == page[e_], 0)
        return t, e_, np.where(has, e_ + 1, -1), e.astype(np.float32)

    def log1p_table(self):
        """math.log1p(k) for k = 0..n (span lengths), as float64."""
        if getattr(self, "_log1p_table", None) is None:
            self._log1p_table = np.asarray(
                [math.log1p(k) for k in range(self.n + 1)], dtype=np.float64
            )
        return self._log1p_table

    def job(self, starts, end):
        """(ia, ib, ic, tail) for gpu_lgbm.RowGather mode 1: the rows of matrix(starts, end)."""
        s = np.asarray(starts, dtype=np.int64)
        k = len(s)
        e = self.extra(s, end)
        nxt = end + 1 if end + 1 < self.n else -1
        return (
            s,
            np.full(k, end, dtype=np.int64),
            np.full(k, nxt, dtype=np.int64),
            e.astype(np.float32),
        )

    def extra(self, s, end):
        """The 15 span scalars (float64) for starts s and one end."""
        n = self.n
        fonts = self.fonts
        k = len(s)
        ln = end - s
        e = np.empty((k, 15), dtype=np.float64)
        e[:, 0] = end == n - 1
        e[:, 1] = s == 0
        e[:, 2] = ln
        lg = self._log1p
        e[:, 3] = [
            lg[v] if v in lg else lg.setdefault(v, math.log1p(v)) for v in ln.tolist()
        ]
        e[:, 4] = ln / max(n, 1)
        e[:, 5] = self.page[end] - self.page[s]
        inner = s < end
        if inner.any():
            seg = fonts[: end + 1]
            sufmax = np.maximum.accumulate(seg[::-1])[::-1]
            sufmin = np.minimum.accumulate(seg[::-1])[::-1]
            si = s[inner]
            e[inner, 6] = sufmax[si + 1] / fonts[si]
            e[inner, 7] = [
                float(np.mean(fonts[t + 1 : end + 1])) / fonts[t] for t in si.tolist()
            ]
            e[inner, 8] = sufmin[si + 1] / fonts[si]
        e[~inner, 6:9] = 1.0
        width = np.maximum(1, ln)
        e[:, 9] = (
            self.counts[self.famid[s], end + 1] - self.counts[self.famid[s], s + 1]
        ) / width
        e[:, 10] = (
            self.prefix[self.numid[s], end + 1] - self.prefix[self.numid[s], s + 1]
        ) / width
        first = np.where(inner, s + 1, s)
        e[:, 11] = fonts[first] / fonts[s]
        if end + 1 < n:
            e[:, 12] = fonts[end + 1] / fonts[s]
            e[:, 13] = self.rawfam[end + 1] == self.rawfam[s]
            e[:, 14] = self.page[end + 1] == self.page[end]
        else:
            e[:, 12] = 1.0
            e[:, 13] = 0
            e[:, 14] = 0
        return e

    def matrix(self, starts, end):
        rec = self.rec
        n = self.n
        x = rec.x
        fonts = self.fonts
        s = np.asarray(starts, dtype=np.int64)
        k = len(s)
        ln = end - s
        e = np.empty((k, 15), dtype=np.float64)
        e[:, 0] = end == n - 1
        e[:, 1] = s == 0
        e[:, 2] = ln
        lg = self._log1p
        e[:, 3] = [
            lg[v] if v in lg else lg.setdefault(v, math.log1p(v)) for v in ln.tolist()
        ]
        e[:, 4] = ln / max(n, 1)
        e[:, 5] = self.page[end] - self.page[s]
        inner = s < end
        if inner.any():
            seg = fonts[: end + 1]
            sufmax = np.maximum.accumulate(seg[::-1])[::-1]
            sufmin = np.minimum.accumulate(seg[::-1])[::-1]
            si = s[inner]
            e[inner, 6] = sufmax[si + 1] / fonts[si]
            e[inner, 7] = [
                float(np.mean(fonts[t + 1 : end + 1])) / fonts[t] for t in si.tolist()
            ]
            e[inner, 8] = sufmin[si + 1] / fonts[si]
        e[~inner, 6:9] = 1.0
        width = np.maximum(1, ln)
        e[:, 9] = (
            self.counts[self.famid[s], end + 1] - self.counts[self.famid[s], s + 1]
        ) / width
        e[:, 10] = (
            self.prefix[self.numid[s], end + 1] - self.prefix[self.numid[s], s + 1]
        ) / width
        first = np.where(inner, s + 1, s)
        e[:, 11] = fonts[first] / fonts[s]
        if end + 1 < n:
            e[:, 12] = fonts[end + 1] / fonts[s]
            e[:, 13] = self.rawfam[end + 1] == self.rawfam[s]
            e[:, 14] = self.page[end + 1] == self.page[end]
            tail = np.broadcast_to(x[end + 1], (k, x.shape[1]))
        else:
            e[:, 12] = 1.0
            e[:, 13] = 0
            e[:, 14] = 0
            tail = np.zeros((k, x.shape[1]), dtype=x.dtype)
        out = np.concatenate(
            [
                x[s],
                np.broadcast_to(x[end], (k, x.shape[1])),
                tail,
                e.astype(np.float32),
            ],
            axis=1,
        )
        return out


def graph_edges(rec):
    """Same edge set as heading_graph.graph_edges, with the O(n^2) Python scans vectorized."""
    meta = rec.meta
    n = rec.n
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)
    z = np.asarray(
        [
            [
                math.log(fsize(r)),
                r["box"][0] / r["width"] * 2,
                r["visual"].get("density", 0.25),
                float(r["text"].isupper()) * 0.2,
            ]
            for r in meta
        ]
    )
    ords = rec.ords
    seq = _ids([sequence(r) for r in meta])
    keys = _ids([norm(r["text"]) for r in meta])
    ofam = _ids([o[0] for o in ords])
    olen = np.asarray([len(o[1]) for o in ords])
    plain = np.asarray([o[0] == "plain" for o in ords])
    idx = np.arange(n)
    edges = set()
    head = set(range(min(n, 3)))
    for i in range(n):
        dist = np.sum((z - z[i]) ** 2, axis=1)
        dist += np.where(seq == seq[i], 0.0, 0.15)
        neighbors = np.argsort(dist, kind="stable")[: min(n, 9)]
        candidates = set(neighbors.tolist())
        candidates.update(range(max(0, i - 5), min(n, i + 6)))
        mask = keys == keys[i]
        if not plain[i]:
            mask |= (ofam == ofam[i]) & (olen == olen[i])
        mask[i] = False
        same = idx[mask]
        if len(same):
            candidates.update(
                same[np.argsort(np.abs(i - same), kind="stable")[:8]].tolist()
            )
        candidates.update(head)
        for j in candidates:
            if i != j:
                edges.add((i, j) if i < j else (j, i))
    out = np.asarray(sorted(edges), dtype=np.int32)
    if VERIFY:
        from .heading_graph import graph_edges as reference

        _require(out, reference(meta, rec.x), "graph edges")
    return out


_ORD_CACHE = {}


def _relation_extra(oa, ob):
    key = (oa, ob)
    v = _ORD_CACHE.get(key)
    if v is None:
        (fa, na), (fb, nb) = oa, ob
        common = 0
        for u, w in zip(na, nb):
            if u != w:
                break
            common += 1
        v = (
            int(fa == fb),
            int(fa == "plain"),
            int(fb == "plain"),
            len(na),
            len(nb),
            common,
            int(bool(na and nb and len(na) < len(nb) and nb[: len(na)] == na)),
            int(bool(na and nb and len(nb) < len(na) and na[: len(nb)] == nb)),
            int(bool(na and nb and len(na) == len(nb) and na[:-1] == nb[:-1])),
        )
        if len(_ORD_CACHE) > 200000:
            _ORD_CACHE.clear()
        _ORD_CACHE[key] = v
    return v


def relation_matrix(rec, a, b):
    """Rows of heading_graph.relation_features(meta[a], meta[b], x[a], x[b])."""
    meta = rec.meta
    ords = rec.ords
    e = np.empty((len(a), 11), dtype=np.float64)
    e[:, :9] = (
        [_relation_extra(ords[i], ords[j]) for i, j in zip(a.tolist(), b.tolist())]
        if len(a)
        else np.empty((0, 9))
    )
    e[:, 9] = rec.page[a] == 1
    e[:, 10] = rec.page[b] == 1
    out = np.concatenate([rec.pairs(a, b), e.astype(np.float32)], axis=1)
    if VERIFY:
        from .heading_graph import relation_features

        x = rec.x
        _require(
            out,
            np.asarray(
                [
                    relation_features(meta[i], meta[j], x[i], x[j])
                    for i, j in zip(a.tolist(), b.tolist())
                ]
            ),
            "relation matrix",
        )
    return out
