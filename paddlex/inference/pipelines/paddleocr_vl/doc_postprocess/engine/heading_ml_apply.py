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

"""Apply lightweight, document-held-out decision trees to prediction features."""

import collections
import json

import numpy as np

from . import doc_resources
from . import runtime_context as ctx
from .heading_ml_features import extract, pair_features


def decode_levels(meta, x, probs, classes, pair_model):
    n = len(meta)
    if not n:
        return []
    levels = np.arange(1, 7)
    unary = np.full((n, 6), 1e-5)
    for j, c in enumerate(classes):
        if 1 <= c <= 6:
            unary[:, c - 1] = probs[:, j]
    unary = np.log(np.maximum(unary, 1e-5)) * 0.65
    for i, r in enumerate(meta):
        if r["family"][0] not in {"plain", "caps", "major"} and r["level"]:
            unary[i] -= 0.15 * np.abs(levels - r["level"])
    pairx = np.asarray(
        [pair_features(meta[i - 1], meta[i], x[i - 1], x[i]) for i in range(1, n)]
    )
    pairp = pair_model.predict_proba(pairx) if n > 1 else []
    delta_idx = {int(c): i for i, c in enumerate(pair_model.classes_)}
    dp = np.full((n, 6), -1e12)
    back = np.zeros((n, 6), dtype=int)
    dp[0, 0] = unary[0, 0]
    for i in range(1, n):
        for level in range(6):
            values = []
            for prev in range(6):
                j = delta_idx.get(level - prev)
                p = pairp[i - 1, j] if j is not None else 1e-5
                values.append(dp[i - 1, prev] + np.log(max(float(p), 1e-5)))
            back[i, level] = int(np.argmax(values))
            dp[i, level] = max(values) + unary[i, level]
    level = int(np.argmax(dp[-1]))
    path = [level + 1]
    for i in range(n - 1, 0, -1):
        level = back[i, level]
        path.append(level + 1)
    return [int(v) for v in reversed(path)]


def _check_not_training_doc(model, doc):
    training_docs = model.get("training_documents")
    if training_docs and doc in training_docs:
        raise AssertionError(f"Document {doc!r} is in training_documents")


def apply(
    results, log, doc, records, mode="conservative", decoder="rules", root="/models"
):
    cfg = ctx.get_config()
    extractor = extract
    if cfg.heading_features_v3:
        from .heading_ml_features_v3 import extract as extractor

    x, meta, _texts = extractor(records)
    base = doc_resources.path(root)
    route = base / "oof" / (doc + ".json")
    oof = (
        json.loads(route.read_text(encoding="utf-8"))
        if route.exists()
        else doc_resources.heading_route()
    )
    model = ctx.load_model(base / f"fold-{oof['fold']}.joblib")
    _check_not_training_doc(model, doc)
    if len(meta) == 0:
        return {}

    binary_x = x
    if model.get("binary_features") == "layout_context_v4":
        from .heading_layout_context import augment, load_pages

        binary_x = augment(x, meta, load_pages(doc))
    p = (
        model["binary"].booster_.predict(binary_x, num_threads=1)
        if hasattr(model["binary"], "booster_")
        else model["binary"].predict_proba(binary_x)[:, 1]
    )
    probs = model["depth"].predict_proba(x)
    classes = model["depth"].classes_

    if cfg.heading_toc_page_anchors:
        from .heading_toc_page_anchors import apply as anchor_titles

        p = anchor_titles(meta, p, doc, log)

    keep_old, new_cut = {
        "conservative": (0.25, 0.9),
        "balanced": (0.5, 0.8),
        "uniform": (cfg.heading_keep_old, cfg.heading_new_cut),
    }[mode]
    selected = [
        i for i, r in enumerate(meta) if p[i] >= (keep_old if r["level"] else new_cut)
    ]

    contact_excluded = set()
    if cfg.heading_contact_roles:
        from .heading_contact_roles import load_and_exclude

        contact_excluded = load_and_exclude(meta, doc)
        selected = [i for i in selected if i not in contact_excluded]

    cover_anchors = []
    if cfg.heading_cover_segments:
        from .heading_cover_segments import load_and_detect

        cover_anchors = load_and_detect(meta, doc)
        selected = sorted(set(selected) | set(cover_anchors))
        if cover_anchors:
            log.append(
                {
                    "kind": "independent_cover_segments",
                    "pages": [meta[i]["page"] for i in cover_anchors],
                    "titles": [meta[i]["text"] for i in cover_anchors],
                }
            )

    selected_meta = [meta[i] for i in selected]
    if decoder == "parent":
        from . import heading_joint_relations
        from .heading_parent_decode import decode as parent_decode

        parent_model = ctx.load_model(
            doc_resources.path("/models-parent") / f"parent-fold-{oof['fold']}.joblib"
        )
        _check_not_training_doc(parent_model, doc)
        if cfg.heading_span_weight is not None:
            from .heading_span import decode as span_decode

            sm = ctx.load_model(
                doc_resources.path("/models-span") / f"span-fold-{oof['fold']}.joblib"
            )
            _check_not_training_doc(sm, doc)

            def one_segment(inds):
                return span_decode(
                    [meta[i] for i in inds],
                    x[inds],
                    probs[inds],
                    classes,
                    model["pair"],
                    parent_model["parent"],
                    sm["span"],
                    weight=float(cfg.heading_span_weight),
                )

        else:

            def one_segment(inds):
                return parent_decode(
                    [meta[i] for i in inds],
                    x[inds],
                    probs[inds],
                    classes,
                    model["pair"],
                    parent_model["parent"],
                )

        gm = ctx.load_model(
            doc_resources.path("/models-graph") / f"graph-fold-{oof['fold']}.joblib"
        )
        _check_not_training_doc(gm, doc)
        rel_tok = heading_joint_relations.MODEL.set(gm["relation"])
        try:
            boundaries = sorted(
                {0, len(selected)}
                | {selected.index(i) for i in cover_anchors if i in selected}
            )
            levels = []
            for lo, hi in zip(boundaries, boundaries[1:]):
                levels.extend(one_segment(selected[lo:hi]))
        finally:
            heading_joint_relations.MODEL.reset(rel_tok)
    elif decoder == "viterbi":
        levels = decode_levels(
            selected_meta, x[selected], probs[selected], classes, model["pair"]
        )
    else:
        levels = [
            r["level"] or int(classes[int(np.argmax(probs[i]))])
            for i, r in zip(selected, selected_meta)
        ]

    if cfg.heading_graph_weight is not None:
        from .heading_graph import decode as graph_decode

        gm = ctx.load_model(
            doc_resources.path("/models-graph") / f"graph-fold-{oof['fold']}.joblib"
        )
        _check_not_training_doc(gm, doc)
        levels, info = graph_decode(
            [meta[i] for i in selected],
            x[selected],
            probs[selected],
            classes,
            model["pair"],
            gm["relation"],
            weight=float(cfg.heading_graph_weight),
            initial=levels,
        )
        log.append(dict(kind="heading_global_relation_graph", **info))

    targets = dict(zip(selected, levels))
    changes = collections.Counter()
    if cfg.heading_index_entries:
        from .heading_index_entries import filter_targets

        targets = filter_targets(meta, targets, log)

    for i, r in enumerate(meta):
        b = results[r["page"] - 1]["parsing_res_list"][r["index"]]
        old = r["level"]
        new = targets.get(i, 0)
        if i in contact_excluded:
            if b.label in {"doc_title", "paragraph_title"}:
                log.append(
                    {
                        "kind": "contact_or_copyright_body",
                        "page": r["page"],
                        "text": r["text"],
                    }
                )
            b.label = "text"
            continue
        assert b.content == r["text"]
        if new == old:
            continue
        if new == 0:
            b.label = "text"
        elif new == 1:
            b.label = "doc_title"
            b.title_level = 0
        else:
            b.label = "paragraph_title"
            b.title_level = new - 1
        kind = (
            "ml_heading_recovered"
            if not old
            else "ml_nontitle_demoted" if not new else "ml_sequence_hierarchy"
        )
        log.append(
            {
                "kind": kind,
                "page": r["page"],
                "text": r["text"],
                "old": old,
                "new": new,
                "p_title": float(p[i]),
                "model_fold": oof["fold"],
                "fit": oof.get("fit", "document_out_of_fold"),
                "decoder": decoder,
            }
        )
        changes[kind] += 1
    return dict(changes)
