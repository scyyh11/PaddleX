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

"""Reference copy of the v44 module (verification only). Learned subtree boundaries add following-context evidence to outline search."""
import math

import numpy as np

from . import runtime_context as ctx
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
    from .heading_ml_features import pair_features
    from .heading_parent_decode import parent_feature

    n = len(meta)
    if not n:
        return []
    from .span_features_fast import prepare

    fast_features = prepare(meta, x)
    unary = np.full((n, 6), 1e-5)
    for j, c in enumerate(classes):
        if 1 <= c <= 6:
            unary[:, c - 1] = probs[:, j]
    pairp = (
        pair_model.predict_proba(
            np.asarray(
                [
                    pair_features(meta[i - 1], meta[i], x[i - 1], x[i])
                    for i in range(1, n)
                ]
            )
        )
        if n > 1
        else []
    )
    delta_idx = {int(c): i for i, c in enumerate(pair_model.classes_)}
    states = [(0.0, (), ())]
    from . import heading_joint_relations_reference as heading_joint_relations

    refs = heading_joint_relations.factors(meta, x)
    cfg = ctx.get_config()
    rw = cfg.heading_joint_relation_weight
    grammar = None
    if cfg.heading_numbering_grammar:
        from .heading_numbering_grammar import induce, penalty

        grammar = induce(meta)

    def close_score(starts, end):
        if not starts:
            return {}
        xx = np.asarray([fast_features(k, end) for k in starts])
        p = span_model.booster_.predict(xx, num_threads=1)
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return dict(zip(starts, weight * np.log(p / (1 - p))))

    for i in range(n):
        starts = sorted(set(j for _, stack, _ in states for j in stack))
        closing = close_score(starts, i - 1) if i else {}
        parents = [None] + starts
        px = np.asarray(
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
        prob = dict(zip(parents, parent_model.predict_proba(px)[:, 1]))
        next_states = {}
        for score, stack, path in states:
            for cut in range(min(len(stack), 5) + 1):
                parent = stack[cut - 1] if cut else None
                level = cut + 1
                value = (
                    score
                    + math.log(max(float(prob[parent]), 1e-6))
                    + 0.3 * math.log(max(float(unary[i, level - 1]), 1e-5))
                )
                value += sum(closing[j] for j in stack[cut:])
                if grammar is not None:
                    value += penalty(grammar, i, level, parent, path)
                if i:
                    k = delta_idx.get(level - path[-1])
                    p = pairp[i - 1, k] if k is not None else 1e-5
                    value += 0.35 * math.log(max(float(p), 1e-5))
                if rw:
                    value += rw * heading_joint_relations.score(refs[i], path, level)
                ns = stack[:cut] + (i,)
                old = next_states.get(ns)
                if old is None or value > old[0]:
                    next_states[ns] = (value, ns, path + (level,))
        states = sorted(next_states.values(), key=lambda s: s[0], reverse=True)[
            :beam_width
        ]
    closing = close_score(
        sorted(set(j for _, stack, _ in states for j in stack)), n - 1
    )
    return list(max(states, key=lambda s: s[0] + sum(closing[j] for j in s[1]))[2])
