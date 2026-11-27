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

"""Veto weak paragraph merge links using current fragments, list markers, and page relation."""

import json

import numpy as np

from . import doc_resources
from . import runtime_context as ctx
from .enumeration_features import features as enumeration
from .paragraph_runtime_features import extract


def filter_links(nodes, links, doc=None, config=None):
    """Score candidate links with the paragraph-runtime model and veto low-scoring merges."""
    if not links:
        return links, []
    cfg = config if config is not None else ctx.get_config()
    same = cfg.paragraph_same_page_threshold
    cross = cfg.paragraph_cross_page_threshold
    root = doc_resources.path("/runtime-paragraph-models")
    folds = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["folds"]
    doc_name = doc or ""
    package = ctx.load_model(
        root / f"fold-{doc_resources.fold(folds, doc_name, root)}.joblib"
    )
    training_docs = package.get("training_documents")
    if training_docs and doc_name in training_docs:
        raise AssertionError(f"Document {doc_name!r} is in training_documents")
    base = extract(nodes, ctx.document_pages(), links)
    x = np.asarray(
        [
            np.r_[
                base[k],
                enumeration(nodes[i]["text"], nodes[j]["text"]),
                float(nodes[i]["page"] == nodes[j]["page"]),
            ]
            for k, (i, j) in enumerate(links)
        ],
        dtype=np.float32,
    )
    prob = package["model"].predict_proba(x)[:, 1]
    kept = []
    events = []
    for (i, j), p in zip(links, prob):
        threshold = same if nodes[i]["page"] == nodes[j]["page"] else cross
        if p < threshold:
            events.append(
                {
                    "kind": "current_fragment_pair_veto",
                    "from": [nodes[i]["page"] + 1, nodes[i]["index"]],
                    "to": [nodes[j]["page"] + 1, nodes[j]["index"]],
                    "p_join": float(p),
                    "threshold": threshold,
                }
            )
        else:
            kept.append((i, j))
    return kept, events
