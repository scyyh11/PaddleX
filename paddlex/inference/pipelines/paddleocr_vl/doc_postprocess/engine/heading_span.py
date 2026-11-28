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

"""Learned subtree boundaries add following-context evidence to outline search.

Same beam search as the v44 implementation (heading_span_reference.py): identical model
inputs, identical floating-point operations in the same order, identical tie-breaking.
Per step, the candidate scores of all beam states are evaluated as arrays; logarithms use
math.log exactly as before.  HEADING_SPAN_VERIFY=1 runs the reference and requires the
same levels.
"""
import math
import time

import numpy as np

from . import runtime_context as ctx

TIMING = False
timing = {}


def _tick(key, t0):
    t = time.perf_counter()
    v = timing.setdefault(key, [0, 0.0])
    v[0] += 1
    v[1] += t - t0
    return t


from .heading_ml_features import fsize, number


def features(meta, x, start, end):
    n = len(meta)
    a = meta[start]
    last = meta[end]
    nxt = meta[end + 1] if end + 1 < n else None
    first = meta[start + 1] if start < end else a
    fs = np.asarray([fsize(r) for r in meta[start + 1 : end + 1]])
    nums = number(a["text"])
    same = sum(r["family"] == a["family"] for r in meta[start + 1 : end + 1])
    pref = sum(
        bool(nums and number(r["text"])[: len(nums)] == nums)
        for r in meta[start + 1 : end + 1]
    )
    extra = [
        end == n - 1,
        start == 0,
        end - start,
        math.log1p(end - start),
        (end - start) / max(n, 1),
        last["page"] - a["page"],
        float(np.max(fs)) / fsize(a) if len(fs) else 1.0,
        float(np.mean(fs)) / fsize(a) if len(fs) else 1.0,
        float(np.min(fs)) / fsize(a) if len(fs) else 1.0,
        same / max(1, end - start),
        pref / max(1, end - start),
        fsize(first) / fsize(a),
        fsize(nxt) / fsize(a) if nxt else 1.0,
        int(nxt["family"] == a["family"]) if nxt else 0,
        int(nxt["page"] == last["page"]) if nxt else 0,
    ]
    return np.concatenate(
        [
            x[start],
            x[end],
            x[end + 1] if end + 1 < n else np.zeros_like(x[start]),
            np.asarray(extra, dtype=np.float32),
        ]
    )


def decode(
    meta,
    x,
    probs,
    classes,
    pair_model,
    parent_model,
    span_model,
    weight=0.5,
    beam_width=48,
):
    gather = None
    if len(meta):
        try:
            from . import runtime_context

            if runtime_context.backend.get() == "cuda":
                from . import gpu_lgbm

                gather = gpu_lgbm.RowGather(x)
        except (ImportError, ValueError):
            gather = None
    if gather is not None and len(meta) + 1 <= _CODE_LIMIT:
        result = _decode_gpu(
            meta,
            x,
            probs,
            classes,
            pair_model,
            parent_model,
            span_model,
            weight,
            beam_width,
            gather,
        )
    else:
        result = _decode(
            meta,
            x,
            probs,
            classes,
            pair_model,
            parent_model,
            span_model,
            weight,
            beam_width,
        )
    return result


def _decode(
    meta, x, probs, classes, pair_model, parent_model, span_model, weight, beam_width
):
    from . import heading_fast, heading_joint_relations

    n = len(meta)
    if not n:
        return []
    rec = heading_fast.Records(meta, x)
    spans = heading_fast.Spans(rec)
    verify = heading_fast.VERIFY
    if verify:
        from .span_features_fast import prepare

        reference_span = prepare(meta, x)
    gather = None
    try:
        from . import runtime_context

        if runtime_context.backend.get() == "cuda":
            from . import gpu_lgbm

            gather = gpu_lgbm.RowGather(x)
    except (ImportError, ValueError):
        gather = None
    unary = np.full((n, 6), 1e-5)
    for j, c in enumerate(classes):
        if 1 <= c <= 6:
            unary[:, c - 1] = probs[:, j]
    if n > 1:
        if gather is not None:
            ia, ib, tail = rec.consecutive_job()
            if verify:
                heading_fast._require(
                    gather.dense(ia, ib, None, 0, tail),
                    rec.consecutive_pairs(),
                    "pair rows",
                )
            pairp = gather.run([(pair_model, "proba", ia, ib, None, 0, tail)])[0]
        else:
            pairp = pair_model.predict_proba(rec.consecutive_pairs())
    else:
        pairp = []
    delta_idx = {int(c): i for i, c in enumerate(pair_model.classes_)}
    if gather is not None:
        rel = heading_joint_relations.factor_arrays(meta, x, rec, gather)
    else:
        rel = heading_joint_relations.arrays(
            heading_joint_relations.factors(meta, x, rec)
        )
    if verify:
        from . import heading_joint_relations_reference

        expected = heading_joint_relations.arrays(
            heading_joint_relations_reference.factors(meta, x)
        )
        same = len(expected) == len(rel) and all(
            (a is None and b is None)
            or (
                a is not None
                and b is not None
                and all(
                    np.array_equal(u, v) and u.dtype == v.dtype for u, v in zip(a, b)
                )
            )
            for a, b in zip(rel, expected)
        )
        if not same:
            raise AssertionError("heading_fast mismatch: relation factors")
    cfg = ctx.get_config()
    rw = cfg.heading_joint_relation_weight
    grammar = None
    if cfg.heading_numbering_grammar:
        from .heading_numbering_grammar import induce

        grammar = induce(meta)
    lookahead = cfg.heading_span_lookahead

    def logits(p):
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return weight * np.log(p / (1 - p))

    def span_job(starts, end):
        ia, ib, ic, tail = spans.job(starts, end)
        if verify:
            heading_fast._require(
                gather.dense(ia, ib, ic, 1, tail),
                spans.matrix(starts, end),
                "span rows",
            )
            heading_fast._require(
                spans.matrix(starts, end),
                np.asarray([reference_span(k, end) for k in starts]),
                "span matrix",
            )
        return (span_model, "booster", ia, ib, ic, 1, tail)

    def close_score(starts, end):
        if not starts:
            return {}
        if gather is not None:
            return dict(zip(starts, logits(gather.run([span_job(starts, end)])[0])))
        xx = spans.matrix(starts, end)
        if verify:
            heading_fast._require(
                xx, np.asarray([reference_span(k, end) for k in starts]), "span matrix"
            )
        return dict(zip(starts, logits(span_model.booster_.predict(xx, num_threads=1))))

    def step_models(i, starts, parents):
        """parent probabilities for child i and span closing scores for heading i-1 (one GPU round trip)."""
        if gather is None:
            closing = close_score(starts, i - 1) if i else {}
            return (
                parent_model.predict_proba(rec.parent_matrix(i, parents))[:, 1],
                closing,
            )
        pa, pb, ptail = rec.parent_job(i, parents)
        if verify:
            heading_fast._require(
                gather.dense(pa, pb, None, 0, ptail),
                rec.parent_matrix(i, parents),
                "parent rows",
            )
        jobs = [(parent_model, "proba", pa, pb, None, 0, ptail)]
        if i and starts:
            jobs.append(span_job(starts, i - 1))
        out = gather.run(jobs)
        return out[0][:, 1], (dict(zip(starts, logits(out[1]))) if len(out) > 1 else {})

    levels = np.arange(1, 7)
    # .3*log(unary) and .35*log(transition) exactly as in the per-candidate expressions
    lu = [[0.3 * math.log(max(float(v), 1e-5)) for v in row] for row in unary.tolist()]
    pairl = pairp.tolist() if n > 1 else []
    scores = [0.0]
    stacks = [()]
    paths = np.zeros((1, n), dtype=np.int8)
    for i in range(n):
        if TIMING:
            t0 = time.perf_counter()
        starts = sorted(set(j for stack in stacks for j in stack))
        parents = [None] + starts
        pprob, closing = step_models(i, starts, parents)
        if TIMING:
            t0 = _tick("models", t0)
        logp = {
            j: math.log(max(float(v), 1e-6)) for j, v in zip(parents, pprob.tolist())
        }
        S = len(stacks)
        none = logp[None]
        lprows = []
        cvrows = []
        parrows = []
        ncut = []
        for stack in stacks:
            c = min(len(stack), 5)
            ncut.append(c)
            lprows.append([none] + [logp[j] for j in stack[:c]] + [0.0] * (5 - c))
            parrows.append([-1] + list(stack[:c]) + [-1] * (5 - c))
            cvrows.append([closing[j] for j in stack] + [0.0] * (6 - len(stack)))
        lp = np.asarray(lprows)
        cvt = np.asarray(cvrows).T.copy()
        # sum(closing[j] for j in stack[cut:]), left to right; zero padding adds exact +0.0
        sc = np.empty((6, S))
        for k in range(6):
            acc = cvt[k]
            for t in range(k + 1, 6):
                acc = acc + cvt[t]
            sc[k] = acc
        if TIMING:
            t0 = _tick("build", t0)
        value = np.asarray(scores)[:, None] + lp
        value = value + np.asarray(lu[i])[None, :]
        value = value + sc.T
        if grammar is not None and grammar[i]:
            par = np.asarray(parrows)
            pen = np.zeros((S, 6))
            for kind, j, w in grammar[i]:
                if kind == "equal":
                    pen = pen - (3.0 * w) * np.minimum(
                        2,
                        np.abs(levels[None, :] - paths[:, j].astype(np.int64)[:, None]),
                    )
                elif kind == "parent":
                    pen = pen - (3.0 * w) * (par != j)
            value = value + pen
        if TIMING:
            t0 = _tick("value_grammar", t0)
        if i:
            prev = paths[:, i - 1].tolist()
            row = pairl[i - 1]
            table = {}
            trows = []
            for pl in prev:
                t = table.get(pl)
                if t is None:
                    t = []
                    for k in range(6):
                        d = delta_idx.get(k + 1 - pl)
                        p = row[d] if d is not None else 1e-5
                        t.append(0.35 * math.log(max(float(p), 1e-5)))
                    table[pl] = t
                trows.append(t)
            value = value + np.asarray(trows)
        if TIMING:
            t0 = _tick("transition", t0)
        if rw:
            r = rel[i]
            if r is None:
                value = value + rw * 0.0
            else:
                jidx, vm, v0, vp = r
                pv = paths[:, jidx].astype(np.int64)[:, None, :]
                lv = levels[None, :, None]
                term = np.where(pv > lv, vm, np.where(pv == lv, v0, vp))
                # sequential left-to-right sum over the factors, like sum() in the reference
                total = (
                    np.cumsum(term, axis=2)[:, :, -1]
                    if term.shape[2] > 1
                    else term[:, :, 0]
                )
                value = value + rw * (total / len(jidx))
        if TIMING:
            t0 = _tick("relation", t0)
            r = rel[i]
            timing.setdefault("relation_terms", [0, 0.0])
            timing["relation_terms"][0] += 1
            timing["relation_terms"][1] += 0 if r is None else len(r[0]) * len(stacks)
        rows = value.tolist()
        best = {}
        for s, stack in enumerate(stacks):
            row = rows[s]
            for k in range(ncut[s] + 1):
                ns = stack[:k] + (i,)
                v = row[k]
                old = best.get(ns)
                if old is None or v > old[0]:
                    best[ns] = (v, ns, s, k + 1)
        chosen = sorted(best.values(), key=lambda t: t[0], reverse=True)[:beam_width]
        scores = [t[0] for t in chosen]
        stacks = [t[1] for t in chosen]
        paths = paths[[t[2] for t in chosen]]
        paths[:, i] = [t[3] for t in chosen]
        if TIMING:
            t0 = _tick("beam", t0)
    closing = close_score(sorted(set(j for stack in stacks for j in stack)), n - 1)
    keys = [scores[s] + sum(closing[j] for j in stacks[s]) for s in range(len(stacks))]
    top = max(range(len(stacks)), key=lambda s: keys[s])
    return [int(v) for v in paths[top].tolist()]


# Stack prefixes are coded as base-(n+1) integers of at most 5 digits; this must fit in int64.
_CODE_LIMIT = 6207
LOOKAHEAD = 32


def _decode_gpu(
    meta,
    x,
    probs,
    classes,
    pair_model,
    parent_model,
    span_model,
    weight,
    beam_width,
    gather,
):
    """Same search as _decode (and the v44 reference) with two changes in how values are obtained:
    * Parent probabilities p(j -> c) and subtree-closing probabilities p(t, e) depend only on the
      pair, not on the beam, and the stacks after step c only contain indices that were in the
      stacks at step i or are in i..c-1.  So they are predicted for blocks of LOOKAHEAD steps at a
      time (a superset of the needed pairs) in one GPU round trip, and each step looks up exactly the
      pairs it needs.  Logs/logits are then computed per step on the same arrays as before.
    * The beam is kept as arrays; candidate stacks are deduplicated with the same rule as the dict
      (first candidate with the strictly largest value wins, entries ranked by value with ties in
      first-occurrence order).
    """
    from . import heading_fast, heading_joint_relations

    n = len(meta)
    rec = heading_fast.Records(meta, x)
    spans = heading_fast.Spans(rec)
    verify = heading_fast.VERIFY
    unary = np.full((n, 6), 1e-5)
    for j, c in enumerate(classes):
        if 1 <= c <= 6:
            unary[:, c - 1] = probs[:, j]
    pairp = None
    if n > 1:
        ia, ib, tail = rec.consecutive_job()
        if verify:
            heading_fast._require(
                gather.dense(ia, ib, None, 0, tail),
                rec.consecutive_pairs(),
                "pair rows",
            )
        pairp = gather.run([(pair_model, "proba", ia, ib, None, 0, tail)])[0]
    delta_idx = {int(c): i for i, c in enumerate(pair_model.classes_)}
    rel = heading_joint_relations.factor_arrays(meta, x, rec, gather)
    if verify:
        from . import heading_joint_relations_reference

        expected = heading_joint_relations.arrays(
            heading_joint_relations_reference.factors(meta, x)
        )
        same = len(expected) == len(rel) and all(
            (a is None and b is None)
            or (
                a is not None
                and b is not None
                and all(
                    np.array_equal(u, v) and u.dtype == v.dtype for u, v in zip(a, b)
                )
            )
            for a, b in zip(rel, expected)
        )
        if not same:
            raise AssertionError("heading_fast mismatch: relation factors")
    cfg = ctx.get_config()
    rw = cfg.heading_joint_relation_weight
    grammar = None
    if cfg.heading_numbering_grammar:
        from .heading_numbering_grammar import induce

        grammar = induce(meta)
    lookahead = cfg.heading_span_lookahead

    def logits(p):
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return weight * np.log(p / (1 - p))

    levels = np.arange(1, 7)
    # .3*log(unary) and .35*log(transition) exactly as in the per-candidate expressions
    LU = np.asarray(
        [[0.3 * math.log(max(float(v), 1e-5)) for v in row] for row in unary.tolist()]
    )
    TR = None
    if n > 1:
        # TR[i-1, pl-1, k] = .35*log(max(p,1e-5)) of level k+1 after level pl (p = 1e-5 for an unseen delta)
        lpair = np.asarray(
            [
                [0.35 * math.log(max(float(p), 1e-5)) for p in row]
                for row in pairp.tolist()
            ]
        )
        D = np.asarray(
            [[delta_idx.get(k + 1 - pl, -1) for k in range(6)] for pl in range(1, 7)]
        )
        TR = np.where(
            D[None, :, :] >= 0,
            lpair[:, np.maximum(D, 0)],
            0.35 * math.log(max(float(1e-5), 1e-5)),
        )
    # parent_cache[c] = (sorted parent ids with -1 for "no parent", probabilities); span_cache[e] likewise
    parent_cache = {}
    span_cache = {}
    block = [0]

    def fill(i, starts):
        hi = min(n, i + lookahead)
        cand = np.asarray(sorted(set(starts) | set(range(i, hi - 1))), dtype=np.int64)
        js = []
        cs = []
        ts = []
        es = []
        pcuts = []
        scuts = []
        for c in range(i, hi):
            m = cand[cand < c]
            js.append(np.r_[-1, m])
            cs.append(np.full(len(m) + 1, c))
            pcuts.append((c, len(m) + 1))
            if c >= 1:
                m = cand[cand <= c - 1]
                ts.append(m)
                es.append(np.full(len(m), c - 1))
                scuts.append((c - 1, len(m)))
        js = np.concatenate(js)
        cs = np.concatenate(cs)
        pa, pb, ptail = rec.parent_job_pairs(js, cs)
        jobs = [(parent_model, "proba", pa, pb, None, 0, ptail)]
        if ts:
            ts = np.concatenate(ts)
            es = np.concatenate(es)
            if len(ts):
                sa, sb, sc_, stail = spans.job_pairs(ts, es)
                jobs.append((span_model, "booster", sa, sb, sc_, 1, stail))
        out = gather.run(jobs)
        pp = out[0][:, 1]
        at = 0
        for c, k in pcuts:
            parent_cache[c] = (js[at : at + k], pp[at : at + k])
            at += k
        if len(out) > 1:
            sp = out[1]
            at = 0
            for e, k in scuts:
                span_cache[e] = (ts[at : at + k], sp[at : at + k])
                at += k
        block[0] = hi

    def lookup(cache, key, need):
        hit = cache.get(key)
        if hit is None:
            return None
        ids, vals = hit
        idx = np.searchsorted(ids, need)
        if len(need) and (idx.max() >= len(ids) or not np.array_equal(ids[idx], need)):
            return None
        return vals[idx]

    def step_values(i, starts):
        need_p = np.asarray([-1] + starts, dtype=np.int64)
        need_s = np.asarray(starts, dtype=np.int64)
        for attempt in range(2):
            if attempt or i >= block[0] or i not in parent_cache:
                fill(i, starts)
            pp = lookup(parent_cache, i, need_p)
            sp = lookup(span_cache, i - 1, need_s) if i and starts else np.zeros(0)
            if pp is not None and sp is not None:
                return pp.tolist(), sp
        raise AssertionError(
            "heading_span lookahead: missing pair values at step %d" % i
        )

    reference_span = None

    def check_step(i, starts, pp, closing):
        nonlocal reference_span
        parents = [None] + starts
        ref = parent_model.predict_proba(rec.parent_matrix(i, parents))[:, 1].tolist()
        if ref != pp:
            raise AssertionError(
                "heading_span lookahead: parent probabilities differ at step %d" % i
            )
        if i and starts:
            xx = spans.matrix(starts, i - 1)
            if reference_span is None:
                from .span_features_fast import prepare

                reference_span = prepare(meta, x)
            heading_fast._require(
                xx,
                np.asarray([reference_span(k, i - 1) for k in starts]),
                "span matrix",
            )
            want = logits(span_model.booster_.predict(xx, num_threads=1))
            if not np.array_equal(want, closing):
                raise AssertionError(
                    "heading_span lookahead: closing scores differ at step %d" % i
                )

    B = n + 1
    powers = np.asarray([B**t for t in range(6)], dtype=np.int64)
    cols = np.arange(6)
    St = np.full((1, 6), -1, dtype=np.int64)
    Ln = np.zeros(1, dtype=np.int64)
    scores = np.zeros(1)
    paths = np.zeros((1, n), dtype=np.int8)
    LP = np.zeros(n + 1)
    CL = np.zeros(n + 1)
    for i in range(n):
        if TIMING:
            t0 = time.perf_counter()
        starts = np.unique(St[St >= 0]).tolist()
        pp, sp = step_values(i, starts)
        logp = [math.log(max(float(v), 1e-6)) for v in pp]
        none = logp[0]
        closing = logits(sp) if i and starts else np.zeros(0)
        if verify:
            check_step(i, starts, pp, closing)
        if TIMING:
            t0 = _tick("models", t0)
        S = len(St)
        cut = np.minimum(Ln, 5)
        LP[starts] = logp[1:]
        CL[starts] = closing
        lp = np.zeros((S, 6))
        lp[:, 0] = none
        inner = cols[None, :5] < cut[:, None]
        lp[:, 1:] = np.where(inner, LP[St[:, :5]], 0.0)
        cvt = np.where(St >= 0, CL[St], 0.0).T.copy()
        # sum(closing[j] for j in stack[cut:]), left to right; zero padding adds exact +0.0
        sc = np.empty((6, S))
        for k in range(6):
            acc = cvt[k]
            for t in range(k + 1, 6):
                acc = acc + cvt[t]
            sc[k] = acc
        if TIMING:
            t0 = _tick("build", t0)
        value = scores[:, None] + lp
        value = value + LU[i][None, :]
        value = value + sc.T
        if grammar is not None and grammar[i]:
            par = np.full((S, 6), -1, dtype=np.int64)
            par[:, 1:] = np.where(inner, St[:, :5], -1)
            pen = np.zeros((S, 6))
            for kind, j, w in grammar[i]:
                if kind == "equal":
                    pen = pen - (3.0 * w) * np.minimum(
                        2,
                        np.abs(levels[None, :] - paths[:, j].astype(np.int64)[:, None]),
                    )
                elif kind == "parent":
                    pen = pen - (3.0 * w) * (par != j)
            value = value + pen
        if TIMING:
            t0 = _tick("value_grammar", t0)
        if i:
            value = value + TR[i - 1][paths[:, i - 1].astype(np.int64) - 1]
        if TIMING:
            t0 = _tick("transition", t0)
        if rw:
            r = rel[i]
            if r is None:
                value = value + rw * 0.0
            else:
                jidx, vm, v0, vp = r
                pv = paths[:, jidx].astype(np.int64)[:, None, :]
                lv = levels[None, :, None]
                term = np.where(pv > lv, vm, np.where(pv == lv, v0, vp))
                # sequential left-to-right sum over the factors, like sum() in the reference
                total = (
                    np.cumsum(term, axis=2)[:, :, -1]
                    if term.shape[2] > 1
                    else term[:, :, 0]
                )
                value = value + rw * (total / len(jidx))
        if TIMING:
            t0 = _tick("relation", t0)
        # candidate (state s, cut k) for k <= min(len(stack), 5): new stack = stack[:k] + (i,)
        valid = cols[None, :] <= cut[:, None]
        code = np.zeros((S, 6), dtype=np.int64)
        if S:
            code[:, 1:] = np.cumsum(
                np.where(St[:, :5] >= 0, St[:, :5] + 1, 0) * powers[None, :5], axis=1
            )
        flat = np.flatnonzero(valid.ravel())
        keys = code.ravel()[flat]
        vals = value.ravel()[flat]
        pos = np.arange(len(flat))
        order = np.lexsort((pos, -vals, keys))
        sk = keys[order]
        head = np.empty(len(sk), dtype=bool)
        head[0] = True
        np.not_equal(sk[1:], sk[:-1], out=head[1:])
        winners = order[head]
        # first occurrence of every key (same groups, in key order, as np.unique(keys,return_index=True))
        byk = np.argsort(keys, kind="stable")
        firstpos = byk[head]
        chosen = winners[np.lexsort((firstpos, -vals[winners]))[:beam_width]]
        src = flat[chosen] // 6
        kk = flat[chosen] % 6
        if verify:
            best = {}
            rows = value.tolist()
            for s_ in range(S):
                stack = tuple(St[s_, : Ln[s_]].tolist())
                for k in range(int(cut[s_]) + 1):
                    ns = stack[:k] + (i,)
                    v = rows[s_][k]
                    old = best.get(ns)
                    if old is None or v > old[0]:
                        best[ns] = (v, ns, s_, k)
            ref = sorted(best.values(), key=lambda t: t[0], reverse=True)[:beam_width]
            if [(t[2], t[3]) for t in ref] != list(zip(src.tolist(), kk.tolist())):
                raise AssertionError(
                    "heading_span beam selection differs at step %d" % i
                )
        St = St[src].copy()
        St[np.arange(len(src)), kk] = i
        St[cols[None, :] > kk[:, None]] = -1
        Ln = kk + 1
        scores = vals[chosen]
        paths = paths[src]
        paths[:, i] = kk + 1
        LP[starts] = 0.0
        CL[starts] = 0.0
        if TIMING:
            t0 = _tick("beam", t0)
    final = np.unique(St[St >= 0]).tolist()
    closing = {}
    if final:
        sa, sb, sc_, stail = spans.job_pairs(
            np.asarray(final), np.full(len(final), n - 1)
        )
        closing = dict(
            zip(
                final,
                logits(gather.run([(span_model, "booster", sa, sb, sc_, 1, stail)])[0]),
            )
        )
    keys = [
        scores[s_] + sum(closing[j] for j in St[s_, : Ln[s_]].tolist())
        for s_ in range(len(St))
    ]
    top = max(range(len(St)), key=lambda s_: keys[s_])
    return [int(v) for v in paths[top].tolist()]
