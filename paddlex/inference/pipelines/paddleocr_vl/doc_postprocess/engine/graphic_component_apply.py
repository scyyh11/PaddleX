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

"""Combine a small composite mark into one image region; text content is retained."""

import json

from . import doc_resources
from . import runtime_context as ctx
from .graphic_components import cover_mark, extract, group


def apply(raw, final, doc):
    root = doc_resources.path("/graphic-models")
    folds = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["folds"]
    fold = doc_resources.fold(folds, doc, root)
    package = ctx.load_model(root / f"fold-{fold}.joblib")
    training_docs = package.get("training_documents")
    if training_docs and doc in training_docs:
        raise AssertionError(f"Document {doc!r} is in training_documents")
    nodes, edges, x = extract(raw)
    if not len(x):
        return []
    p = package["model"].predict_proba(x)[:, 1]
    events = []
    for g in group(nodes, edges, p, 0.5):
        pi = g["page_id"] - 1
        if not cover_mark(raw[pi], g):
            continue
        originals = {
            tuple(round(v * 1000) for v in n["box"])
            for n in nodes
            if n["page_id"] == g["page_id"] and n["block"] in g["visual_members"]
        }
        page = final[pi]
        w, h = page["width"], page["height"]
        hits = []
        for i, b in enumerate(page["parsing_res_list"]):
            box = tuple(
                max(0, min(1000, round(v / scale * 1000)))
                for v, scale in zip(b["block_bbox"], [w, h, w, h])
            )
            if b["block_label"] in {"image", "chart", "seal"} and box in originals:
                hits.append(i)
        if not hits:
            continue
        keeper = page["parsing_res_list"][hits[0]]
        keeper["block_bbox"] = [
            v * scale / 1000 for v, scale in zip(g["bbox_norm"], [w, h, w, h])
        ]
        consumed = set(hits[1:])
        page["parsing_res_list"] = [
            b for i, b in enumerate(page["parsing_res_list"]) if i not in consumed
        ]
        events.append(dict(kind="composite_cover_mark", **g))
    return events
