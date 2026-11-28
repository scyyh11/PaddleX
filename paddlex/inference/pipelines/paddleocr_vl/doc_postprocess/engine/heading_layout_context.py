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

"""Full-page layout evidence omitted by the earlier text-only candidate export."""
import collections
import re

import numpy as np

from .heading_ml_features import norm


def load_pages(doc):
    from .runtime_context import input_pages

    held = input_pages.get()
    if held is None or held[1] is None:
        raise RuntimeError(
            f"Input pages are unavailable in runtime_context for {doc!r}"
        )
    return held[1]


def augment(x, meta, pages):
    return _augment_fast(x, meta, pages)


def _augment_fast(x, meta, pages):
    stats = []
    blocks = []
    header_texts = collections.defaultdict(set)
    family_pages = collections.defaultdict(set)
    for pi, p in enumerate(pages):
        w, h = p["width"], p["height"]
        bs = p["parsing_res_list"]
        features = []
        for label in [
            "table",
            "image",
            "header",
            "text",
            "paragraph_title",
            "doc_title",
            "figure_title",
            "content",
            "number",
        ]:
            chosen = [b for b in bs if b["block_label"] == label]
            areas = [
                max(0, b["block_bbox"][2] - b["block_bbox"][0])
                * max(0, b["block_bbox"][3] - b["block_bbox"][1])
                / w
                / h
                for b in chosen
            ]
            features.extend([len(chosen), sum(areas), max(areas, default=0)])
        stats.append(features)
        blocks.append(bs)
        for b in bs:
            if b["block_label"] == "header":
                key = norm(b["block_content"])
                header_texts[key].add(pi)
                family_pages[re.sub(r"\d+", "#", key)].add(pi)
    stats = np.asarray(stats, dtype=float)
    docmean = stats.mean(axis=0) if len(stats) else np.zeros(27)
    page_count = max(len(pages), 1)
    n = len(meta)
    geo = np.empty((n, 12))
    tail = np.empty((n, 9))
    bypage = collections.defaultdict(list)
    for k, r in enumerate(meta):
        bypage[r["page"] - 1].append(k)
    for pi, rows in bypage.items():
        p = pages[pi]
        h = p["height"]
        box = np.asarray([meta[k]["box"] for k in rows], dtype=float)
        a, b, c, d = (box[:, t][:, None] for t in range(4))
        for li, label in enumerate(["table", "image", "text", "paragraph_title"]):
            qs = [q["block_bbox"] for q in blocks[pi] if q["block_label"] == label]
            if not qs:
                geo[rows, 3 * li] = 1
                geo[rows, 3 * li + 1] = 1
                geo[rows, 3 * li + 2] = 0
                continue
            qb = np.asarray(qs, dtype=float)
            u, v, uu, vv = (qb[:, t][None, :] for t in range(4))
            hx = np.minimum(c, uu) - np.maximum(a, u)
            hx = np.where(hx > 0, hx, 0.0)
            den = np.minimum(c - a, uu - u)
            den = np.where(den > 1, den, 1.0)
            keep = ~(hx / den < 0.2)
            bel = v - d
            bel = np.where(bel > 0, bel, 0.0) / h
            bel = np.where(keep & (v >= d - 3), bel, np.inf)
            abv = b - vv
            abv = np.where(abv > 0, abv, 0.0) / h
            abv = np.where(keep & (vv <= b + 3), abv, np.inf)
            vy = np.minimum(d, vv) - np.maximum(b, v)
            vy = np.where(vy > 0, vy, 0.0)
            area = (c - a) * (d - b)
            area = np.where(area > 1, area, 1.0)
            ov = np.where(keep, hx * vy / area, -np.inf)
            mb = bel.min(axis=1)
            ma = abv.min(axis=1)
            mo = ov.max(axis=1)
            geo[rows, 3 * li] = np.where(np.isinf(mb), 1, mb)
            geo[rows, 3 * li + 1] = np.where(np.isinf(ma), 1, ma)
            geo[rows, 3 * li + 2] = np.where(np.isinf(mo), 0, mo)
    for k, r in enumerate(meta):
        key = norm(r["text"])
        ps = sorted(header_texts.get(key, set()))
        span = max(ps) - min(ps) + 1 if ps else 0
        tail[k] = [
            len(ps) / page_count,
            span / page_count,
            len(ps) / max(span, 1),
            len(header_texts) / page_count,
            len(family_pages.get(re.sub(r"\d+", "#", key), set())) / page_count,
            int(bool(_IDS.search(r["text"]))),
            int(bool(_CONT.search(r["text"]))),
            int(bool(_MARK.search(r["text"]))),
            int(bool(_INDEX.search(r["text"]))),
        ]
    pidx = np.asarray([r["page"] - 1 for r in meta], dtype=np.int64)
    context = (
        np.concatenate(
            [stats[pidx], np.broadcast_to(docmean, (n, len(docmean))), geo, tail],
            axis=1,
        )
        if n
        else np.zeros((0, 75))
    )
    return np.concatenate(
        [x, np.asarray(context, dtype=np.float32).reshape(n, 75)], axis=1
    )


_IDS = re.compile(
    r"\b(?:WBS|ISBN|ISSN|DOI|Report\s*(?:No|Number)|Document\s*(?:No|Number))\b", re.I
)
_CONT = re.compile(r"\b(?:continued|cont\.?d)\b|[续續]", re.I)
_MARK = re.compile(r"\\(?:underline|textbf)|__|\*\*")
_INDEX = re.compile(r"\b(?:index|contents|glossary|code)\b|[目索]引|目[录錄]", re.I)


def _augment_reference(x, meta, pages):
    stats = []
    blocks = []
    header_texts = collections.defaultdict(set)
    family_pages = collections.defaultdict(set)
    for pi, p in enumerate(pages):
        w, h = p["width"], p["height"]
        bs = p["parsing_res_list"]
        features = []
        for label in [
            "table",
            "image",
            "header",
            "text",
            "paragraph_title",
            "doc_title",
            "figure_title",
            "content",
            "number",
        ]:
            chosen = [b for b in bs if b["block_label"] == label]
            areas = [
                max(0, b["block_bbox"][2] - b["block_bbox"][0])
                * max(0, b["block_bbox"][3] - b["block_bbox"][1])
                / w
                / h
                for b in chosen
            ]
            features.extend([len(chosen), sum(areas), max(areas, default=0)])
        stats.append(features)
        blocks.append(bs)
        for b in bs:
            if b["block_label"] == "header":
                key = norm(b["block_content"])
                header_texts[key].add(pi)
                family_pages[re.sub(r"\d+", "#", key)].add(pi)
    stats = np.asarray(stats, dtype=float)
    docmean = stats.mean(axis=0) if len(stats) else np.zeros(27)
    context = []
    page_count = max(len(pages), 1)
    for r in meta:
        pi = r["page"] - 1
        p = pages[pi]
        w, h = p["width"], p["height"]
        a, b, c, d = r["box"]
        f = []
        for label in ["table", "image", "text", "paragraph_title"]:
            bs = [q for q in blocks[pi] if q["block_label"] == label]
            below = []
            above = []
            overlap = []
            for q in bs:
                u, v, uu, vv = q["block_bbox"]
                horizontal = max(0, min(c, uu) - max(a, u)) / max(1, min(c - a, uu - u))
                if horizontal < 0.2:
                    continue
                if v >= d - 3:
                    below.append(max(0, v - d) / h)
                if vv <= b + 3:
                    above.append(max(0, b - vv) / h)
                inter = max(0, min(c, uu) - max(a, u)) * max(0, min(d, vv) - max(b, v))
                overlap.append(inter / max(1, (c - a) * (d - b)))
            f.extend(
                [min(below, default=1), min(above, default=1), max(overlap, default=0)]
            )
        key = norm(r["text"])
        ps = sorted(header_texts.get(key, set()))
        span = max(ps) - min(ps) + 1 if ps else 0
        f.extend(
            [
                len(ps) / page_count,
                span / page_count,
                len(ps) / max(span, 1),
                len(header_texts) / page_count,
                len(family_pages.get(re.sub(r"\d+", "#", key), set())) / page_count,
                int(
                    bool(
                        re.search(
                            r"\b(?:WBS|ISBN|ISSN|DOI|Report\s*(?:No|Number)|Document\s*(?:No|Number))\b",
                            r["text"],
                            re.I,
                        )
                    )
                ),
                int(
                    bool(
                        re.search(r"\b(?:continued|cont\.?d)\b|[续續]", r["text"], re.I)
                    )
                ),
                int(bool(re.search(r"\\(?:underline|textbf)|__|\*\*", r["text"]))),
                int(
                    bool(
                        re.search(
                            r"\b(?:index|contents|glossary|code)\b|[目索]引|目[录錄]",
                            r["text"],
                            re.I,
                        )
                    )
                ),
            ]
        )
        context.append(np.concatenate([stats[pi], docmean, np.asarray(f)]))
    return np.concatenate(
        [x, np.asarray(context, dtype=np.float32).reshape(len(meta), 75)], axis=1
    )
