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

"""Document-wide ordinal factors and dual decomposition; no annotation inputs."""
import math
import re
import unicodedata

import numpy as np

from .heading_ml_features import fsize, norm, pair_features
from .heading_structure_features import sequence


def ordinal(s):
    s = unicodedata.normalize("NFKC", s).strip()
    # Return a grammar family and components, never a prescribed depth.
    cn = "零〇一二三四五六七八九十百千两兩"
    m = re.match(r"^(第[" + cn + r"\d]+([章节節部篇卷]))", s)
    if m:
        return "cn_" + m[2], ()
    m = re.match(
        r"^(chapter|section|part|appendix|annex)\s+([IVXLCDM]+|\d+(?:\.\d+)*)\b",
        s,
        re.I,
    )
    if m:
        return m[1].lower(), (
            tuple(int(x) for x in m[2].split(".")) if m[2][0].isdigit() else ()
        )
    m = re.match(r"^(\d+(?:\.\d+)*)(?:[.)、:\s]|$)", s)
    if m:
        return "decimal", tuple(map(int, m[1].split(".")))
    for pattern, name in [
        (r"^\([0-9]+\)", "number_paren"),
        (r"^[" + cn + r"]+[、.\s]", "cn_list"),
        (r"^\([" + cn + r"]+\)", "cn_paren"),
        (r"^[IVXLCDM]+[.)\s]", "roman_upper"),
        (r"^[ivxlcdm]+[.)\s]", "roman_lower"),
        (r"^\([A-Z]\)", "letter_upper_paren"),
        (r"^\([a-z]\)", "letter_lower_paren"),
        (r"^[A-Z][.)]", "letter_upper"),
        (r"^[a-z][.)]", "letter_lower"),
    ]:
        if re.match(pattern, s):
            return name, ()
    return "plain", ()


def graph_edges(meta, x):
    n = len(meta)
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)
    # All references are chosen from predicted layout, in both temporal directions.
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
    ords = [ordinal(r["text"]) for r in meta]
    seq = [sequence(r) for r in meta]
    keys = [norm(r["text"]) for r in meta]
    edges = set()
    for i in range(n):
        dist = np.sum((z - z[i]) ** 2, axis=1)
        dist += np.asarray([0 if seq[j] == seq[i] else 0.15 for j in range(n)])
        neighbors = np.argsort(dist, kind="stable")[: min(n, 9)]
        candidates = set(map(int, neighbors))
        candidates.update(range(max(0, i - 5), min(n, i + 6)))
        same = [
            j
            for j in range(n)
            if j != i
            and (
                keys[j] == keys[i]
                or (
                    ords[i][0] != "plain"
                    and ords[j][0] == ords[i][0]
                    and len(ords[j][1]) == len(ords[i][1])
                )
            )
        ]
        candidates.update(sorted(same, key=lambda j: abs(i - j))[:8])
        candidates.update(range(min(n, 3)))
        for j in candidates:
            if i != j:
                edges.add((min(i, j), max(i, j)))
    return np.asarray(sorted(edges), dtype=np.int32)


def relation_features(a, b, fa, fb):
    oa, na = ordinal(a["text"])
    ob, nb = ordinal(b["text"])
    common = 0
    for u, v in zip(na, nb):
        if u != v:
            break
        common += 1
    extra = [
        int(oa == ob),
        int(oa == "plain"),
        int(ob == "plain"),
        len(na),
        len(nb),
        common,
        int(bool(na and nb and len(na) < len(nb) and nb[: len(na)] == na)),
        int(bool(na and nb and len(nb) < len(na) and na[: len(nb)] == nb)),
        int(bool(na and nb and len(na) == len(nb) and na[:-1] == nb[:-1])),
        int(a["page"] == 1),
        int(b["page"] == 1),
    ]
    return np.concatenate(
        [pair_features(a, b, fa, fb), np.asarray(extra, dtype=np.float32)]
    )


def chain_decode(unary, transition):
    """Exact Viterbi for valid preorder depths (zero based), including missing levels."""
    n, D = unary.shape
    if not n:
        return np.empty(0, dtype=int), 0.0
    dp = unary[0].copy()
    dp[1:] = -1e30
    back = np.zeros((n, D), int)
    legal = np.arange(D)[None, :] <= np.arange(D)[:, None] + 1
    for i in range(1, n):
        v = dp[:, None] + transition[i - 1]
        v = np.where(legal, v, -1e30)
        back[i] = np.argmax(v, axis=0)
        dp = np.max(v, axis=0) + unary[i]
    y = np.zeros(n, int)
    y[-1] = np.argmax(dp)
    score = float(dp[y[-1]])
    for i in range(n - 1, 0, -1):
        y[i - 1] = back[i, y[i]]
    return y, score


def objective(y, unary, transition, edges, factors):
    v = float(unary[np.arange(len(y)), y].sum())
    if len(y) > 1:
        v += float(transition[np.arange(len(y) - 1), y[:-1], y[1:]].sum())
    if len(edges):
        v += float(factors[np.arange(len(edges)), y[edges[:, 0]], y[edges[:, 1]]].sum())
    return v


def solve(unary, transition, edges, factors, iterations=100, initial=None):
    """Maximize a loopy graph by exact chain/edge subproblems.

    Returns a feasible lower bound and a Lagrangian upper bound, not a claim of
    global optimality. Every accepted candidate is scored under the same objective.
    """
    n, D = unary.shape
    m = len(edges)
    best, _ = chain_decode(unary, transition)
    bestscore = objective(best, unary, transition, edges, factors)
    if initial is not None:
        q = np.asarray(initial, dtype=int)
        if len(q) == n and (not n or (q[0] == 0 and np.all(q[1:] <= q[:-1] + 1))):
            s = objective(q, unary, transition, edges, factors)
            if s > bestscore:
                best, bestscore = q.copy(), s
    if not m:
        return best, {
            "objective": bestscore,
            "upper_bound": bestscore,
            "duality_gap": 0.0,
            "iterations": 0,
        }
    a, b = edges.T
    idx = np.arange(m)
    lam_a = np.zeros((m, D))
    lam_b = np.zeros((m, D))
    upper = float("inf")
    used = 0
    for it in range(iterations):
        u = unary.copy()
        np.add.at(u, a, lam_a)
        np.add.at(u, b, lam_b)
        y, cs = chain_decode(u, transition)
        ef = factors - lam_a[:, :, None] - lam_b[:, None, :]
        flat = np.argmax(ef.reshape(m, -1), axis=1)
        ea, eb = flat // D, flat % D
        bound = cs + float(ef[idx, ea, eb].sum())
        upper = min(upper, bound)
        s = objective(y, unary, transition, edges, factors)
        if s > bestscore:
            best, bestscore = y.copy(), s
        used = it + 1
        if upper - bestscore < 1e-7:
            break
        mismatches = int((y[a] != ea).sum() + (y[b] != eb).sum())
        if not mismatches:
            break
        # Polyak step to the best feasible solution, with conservative damping.
        step = min(1.0, 0.8 * max(bound - bestscore, 1e-6) / (2 * mismatches))
        lam_a[idx, y[a]] -= step
        lam_a[idx, ea] += step
        lam_b[idx, y[b]] -= step
        lam_b[idx, eb] += step
    # Monotone coordinate improvements preserve a realizable preorder tree.
    adjacency = [[] for _ in range(n)]
    for e, (i, j) in enumerate(edges):
        adjacency[i].append((e, j, 0))
        adjacency[j].append((e, i, 1))
    for sweep in range(8):
        changed = False
        for i in range(1, n):
            v = unary[i].copy() + transition[i - 1, best[i - 1], :]
            if i + 1 < n:
                v += transition[i, :, best[i + 1]]
            for e, j, side in adjacency[i]:
                v += factors[e, :, best[j]] if side == 0 else factors[e, best[j], :]
            v[np.arange(D) > best[i - 1] + 1] = -1e30
            if i + 1 < n:
                v[np.arange(D) < best[i + 1] - 1] = -1e30
            q = int(np.argmax(v))
            if v[q] > v[best[i]] + 1e-9:
                best[i] = q
                changed = True
        if not changed:
            break
    bestscore = objective(best, unary, transition, edges, factors)
    return best, {
        "objective": bestscore,
        "upper_bound": upper,
        "duality_gap": max(0.0, upper - bestscore),
        "iterations": used,
    }


def decode(meta, x, probs, classes, pair_model, graph_model, weight=1.0, initial=None):
    n = len(meta)
    D = 6
    if not n:
        return [], {"edges": 0}
    unary = np.full((n, D), 1e-5)
    for j, c in enumerate(classes):
        if 1 <= c <= D:
            unary[:, c - 1] = probs[:, j]
    unary = 0.3 * np.log(np.maximum(unary, 1e-5))
    transition = np.full((max(n - 1, 0), D, D), 0.35 * math.log(1e-5))
    deltas = np.arange(D)[None, :] - np.arange(D)[:, None]
    if n > 1:
        xx = np.asarray(
            [pair_features(meta[i - 1], meta[i], x[i - 1], x[i]) for i in range(1, n)]
        )
        pp = (
            pair_model.booster_.predict(xx, num_threads=1)
            if hasattr(pair_model, "booster_")
            else pair_model.predict_proba(xx)
        )
        for k, c in enumerate(pair_model.classes_):
            transition[:, deltas == int(c)] = 0.35 * np.log(
                np.maximum(pp[:, k, None], 1e-5)
            )
    edges = graph_edges(meta, x)
    factors = np.empty((len(edges), D, D))
    sign = np.sign(deltas)
    for start in range(0, len(edges), 4096):
        batch = edges[start : start + 4096]
        xx = np.asarray(
            [relation_features(meta[i], meta[j], x[i], x[j]) for i, j in batch]
        )
        pp = graph_model.booster_.predict(xx, num_threads=1)
        vv = np.full((len(batch), D, D), math.log(1e-5))
        for k, c in enumerate(graph_model.classes_):
            vv[:, sign == int(c)] = np.log(np.maximum(pp[:, k, None], 1e-5))
        factors[start : start + len(batch)] = vv
    degree = np.bincount(edges.ravel(), minlength=n) if len(edges) else np.zeros(n)
    if len(edges):
        factors -= factors.max(axis=(1, 2), keepdims=True)
        factors *= (
            weight
            / np.sqrt(np.maximum(1, degree[edges[:, 0]] * degree[edges[:, 1]]))[
                :, None, None
            ]
        )
    y, info = solve(
        unary,
        transition,
        edges,
        factors,
        initial=np.asarray(initial) - 1 if initial is not None else None,
    )
    info.update(
        edges=len(edges),
        nodes=n,
        weight=weight,
        algorithm="dual decomposition of preorder-chain and nonlocal ordinal factors",
    )
    return (y + 1).tolist(), info
