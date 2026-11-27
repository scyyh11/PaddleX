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

"""Nonlocal ordinal factors used inside a parent/subtree beam search.

factors() keeps the original (j, {class: log p}) lists; arrays() returns the same values as
per-heading arrays for the vectorized beam search.  Edges, features, model calls and logs are
identical to the original implementation (heading_joint_relations_reference.py).
"""
import contextvars
import math

import numpy as np

MODEL = contextvars.ContextVar("heading_joint_relation_model", default=None)
LOG_FLOOR = math.log(1e-5)


def factors(meta, x, rec=None):
    from . import heading_fast

    model = MODEL.get()
    n = len(meta)
    refs = [[] for _ in meta]
    if model is None or n < 2:
        return refs
    if rec is None:
        rec = heading_fast.Records(meta, x)
    edges = heading_fast.graph_edges(rec)
    classes = [int(c) for c in model.classes_]
    for start in range(0, len(edges), 4096):
        batch = edges[start : start + 4096]
        xx = heading_fast.relation_matrix(
            rec, batch[:, 0].astype(np.int64), batch[:, 1].astype(np.int64)
        )
        probs = model.predict_proba(xx)
        for (i, j), pr in zip(batch.tolist(), probs.tolist()):
            lp = {c: math.log(max(float(v), 1e-5)) for c, v in zip(classes, pr)}
            refs[j].append((i, lp))
    return refs


def factor_arrays(meta, x, rec, gather):
    """arrays(factors(meta,x)) computed directly: model rows assembled on the GPU, logs with
    math.log exactly as in factors(), grouped per heading in the same (edge) order."""
    from . import heading_fast

    model = MODEL.get()
    n = len(meta)
    if model is None or n < 2:
        return [None] * n
    edges = heading_fast.graph_edges(rec)
    if not len(edges):
        return [None] * n
    probs = []
    for start in range(0, len(edges), 4096):
        batch = edges[start : start + 4096]
        a = batch[:, 0].astype(np.int64)
        b = batch[:, 1].astype(np.int64)
        ia, ib, tail = rec.relation_job(a, b)
        if heading_fast.VERIFY:
            heading_fast._require(
                gather.dense(ia, ib, None, 0, tail),
                heading_fast.relation_matrix(rec, a, b),
                "relation rows",
            )
        probs.append(gather.run([(model, "proba", ia, ib, None, 0, tail)])[0])
    p = np.concatenate(probs)
    col = {int(c): k for k, c in enumerate(model.classes_)}
    log = math.log

    def logs(c):
        k = col.get(c)
        if k is None:
            return np.full(len(p), LOG_FLOOR)
        return np.asarray([log(v) for v in np.maximum(p[:, k], 1e-5).tolist()])

    order = np.argsort(edges[:, 1], kind="stable")
    js = edges[order, 1]
    iz = edges[order, 0].astype(np.int64)
    vm, v0, vp = logs(-1)[order], logs(0)[order], logs(1)[order]
    bounds = np.searchsorted(js, np.arange(n + 1)).tolist()
    out = [None] * n
    for j in range(n):
        lo, hi = bounds[j], bounds[j + 1]
        if hi > lo:
            out[j] = (iz[lo:hi], vm[lo:hi], v0[lo:hi], vp[lo:hi])
    return out


def arrays(refs):
    """Per heading: (indices, log p for sign -1, 0, +1) in reference order."""
    out = []
    for r in refs:
        if not r:
            out.append(None)
            continue
        out.append(
            (
                np.asarray([j for j, _ in r], dtype=np.int64),
                np.asarray([lp.get(-1, LOG_FLOOR) for _, lp in r]),
                np.asarray([lp.get(0, LOG_FLOOR) for _, lp in r]),
                np.asarray([lp.get(1, LOG_FLOOR) for _, lp in r]),
            )
        )
    return out


def score(refs, path, level):
    if not refs:
        return 0.0
    # Each heading contributes one normalized factor. Long documents therefore
    # do not overwhelm parent/closure evidence merely by having more edges.
    return sum(
        lp.get(int(np.sign(level - path[j])), math.log(1e-5)) for j, lp in refs
    ) / len(refs)
