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

"""Admit a non-conflicting original OCR page stitch with optional OOF support."""
import collections

from . import text_features as f


def apply(nodes, visible, pages, fixed, model, threshold):
    groups = collections.defaultdict(list)
    out = {i for i, j in fixed}
    inc = {j for i, j in fixed}
    candidates = []
    for n in nodes:
        if n["block"].get("_old_group") is not None:
            groups[n["block"]["_old_group"]].append(n)
    for nn in groups.values():
        for a, b in zip(nn, nn[1:]):
            if b["page"] != a["page"] + 1 or a["id"] in out or b["id"] in inc:
                continue
            if any(
                n["transcript"]
                or n["block"].get("_code_region")
                or n["block"].get("_text_region")
                for n in [a, b]
            ):
                continue
            if any(
                l in {"paragraph_title", "doc_title"}
                for pi, bi, l in visible[a["vindex"] + 1 : b["vindex"]]
            ):
                continue
            candidates.append((a["id"], b["id"]))
    if not candidates:
        return [], []
    probabilities = {}
    if threshold:
        edges, xx = f.candidates(nodes, visible)
        need = set(candidates)
        ids = [i for i, e in enumerate(edges) if e in need]
        if ids:
            probabilities = dict(
                zip([edges[i] for i in ids], model.predict_proba(xx[ids])[:, 1])
            )
    chosen = []
    logs = []
    for i, j in candidates:
        probability = float(probabilities.get((i, j), 0)) if threshold else 1.0
        if probability < threshold or any(
            i < k < j < l or k < i < l < j for k, l in fixed + chosen
        ):
            continue
        if i in out or j in inc:
            continue
        out.add(i)
        inc.add(j)
        chosen.append((i, j))
        logs.append(
            {
                "kind": "supported_original_page_stitch",
                "from": [nodes[i]["page"] + 1, nodes[i]["index"]],
                "to": [nodes[j]["page"] + 1, nodes[j]["index"]],
                "probability": probability,
            }
        )
    return chosen, logs
