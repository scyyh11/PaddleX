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

"""Detect independently restarted manuals from repeated, sparse cover layouts.

Uses shared product identifiers and image geometry; no document IDs, labels from
the benchmark, dictionaries of product names, or translation models.
"""
import collections
import re

import numpy as np

from .heading_ml_features import norm


def identifiers(s):
    return {
        t.upper()
        for t in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9-]{2,23}\b", s)
        if any(c.isalpha() for c in t) and any(c.isdigit() for c in t)
    }


def detect(meta, pages):
    bypage = collections.defaultdict(list)
    for i, r in enumerate(meta):
        bypage[r["page"]].append(i)
    proposals = []
    for pi, p in enumerate(pages):
        w, h = p["width"], p["height"]
        bs = p["parsing_res_list"]
        visible = [
            b
            for b in bs
            if b["block_label"] not in {"header", "footer", "number", "aside_text"}
            and b["block_content"].strip()
        ]
        if sum(len(b["block_content"]) for b in visible) > 300 or len(visible) > 5:
            continue
        images = [
            b
            for b in bs
            if b["block_label"] == "image"
            and (b["block_bbox"][2] - b["block_bbox"][0])
            * (b["block_bbox"][3] - b["block_bbox"][1])
            / w
            / h
            >= 0.07
        ]
        if len(images) != 1:
            continue
        image = np.asarray(images[0]["block_bbox"]) / np.asarray([w, h, w, h])
        candidates = []
        for i in bypage[pi + 1]:
            r = meta[i]
            x, y, xx, yy = r["box"]
            tokens = identifiers(r["text"])
            if (
                r["original_label"] not in {"doc_title", "paragraph_title"}
                or not tokens
            ):
                continue
            if not 8 <= len(r["text"]) <= 160 or not 0.10 <= y / h <= 0.45:
                continue
            if abs((x + xx) / 2 / w - 0.5) > 0.12 or (xx - x) / w < 0.35:
                continue
            candidates.append((i, tokens))
        if len(candidates) == 1:
            proposals.append((candidates[0][0], candidates[0][1], image))
    accepted = set()
    for i, tokens, image in proposals:
        peers = [
            j
            for j, tt, im in proposals
            if tokens & tt and np.max(np.abs(image - im)) < 0.045
        ]
        if len(peers) < 3 or min(meta[j]["page"] for j in peers) > 3:
            continue
        if len({norm(meta[j]["text"]) for j in peers}) < 2:
            continue
        if any(
            b - a < 3
            for a, b in zip(
                sorted(meta[j]["page"] for j in peers),
                sorted(meta[j]["page"] for j in peers)[1:],
            )
        ):
            continue
        accepted.update(peers)
    return sorted(accepted)


def load_and_detect(meta, doc):
    from .runtime_context import document_pages

    pages = document_pages()
    return detect(meta, pages)
