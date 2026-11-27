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

"""Search complete outline paths using a small learned parent-compatibility model."""
import math

import numpy as np

from .heading_ml_features import pair_features


def parent_feature(a, b, fa, fb, gap):
    pair = pair_features(a or b, b, fa if a else np.zeros_like(fb), fb)
    return np.concatenate(
        [pair, np.array([a is None, min(gap, 500), np.log1p(gap)], dtype=np.float32)]
    )


def decode(
    meta,
    x,
    probs,
    classes,
    pair_model,
    parent_model,
    beam_width=12,
    relation_model=None,
):
    if not meta:
        return []
    n = len(meta)
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
    for i in range(n):
        parents = list(
            dict.fromkeys([None] + [j for _, stack, _ in states for j in stack])
        )
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
        refs = []
        rp = []
        rc = {}
        if relation_model is not None:
            from .heading_structure_features import style_refs

            refs = [(j, w) for j, w in style_refs(meta, i) if j != i - 1]
            if refs:
                rp = relation_model.predict_proba(
                    np.asarray(
                        [pair_features(meta[j], meta[i], x[j], x[i]) for j, w in refs]
                    )
                )
                rc = {int(c): k for k, c in enumerate(relation_model.classes_)}
        for score, stack, path in states:
            for cut in range(min(len(stack), 5) + 1):
                parent = stack[cut - 1] if cut else None
                level = cut + 1
                value = (
                    score
                    + math.log(max(float(prob[parent]), 1e-6))
                    + 0.3 * math.log(max(float(unary[i, level - 1]), 1e-5))
                )
                if i:
                    delta = level - path[-1]
                    j = delta_idx.get(delta)
                    trans = pairp[i - 1, j] if j is not None else 1e-5
                    value += 0.35 * math.log(max(float(trans), 1e-5))
                if refs:
                    value += sum(
                        w
                        * math.log(
                            max(
                                (
                                    float(rp[k, rc[level - path[j]]])
                                    if level - path[j] in rc
                                    else 1e-5
                                ),
                                1e-5,
                            )
                        )
                        for k, (j, w) in enumerate(refs)
                    ) / max(1, sum(w for j, w in refs))
                newstack = stack[:cut] + (i,)
                newpath = path + (int(level),)
                previous = next_states.get(newstack)
                if previous is None or value > previous[0]:
                    next_states[newstack] = (value, newstack, newpath)
        states = sorted(next_states.values(), key=lambda s: s[0], reverse=True)[
            :beam_width
        ]
    return list(states[0][2])
