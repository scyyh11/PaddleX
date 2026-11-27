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

"""Candidate connected visual regions from prediction geometry only."""
import math
import re

import numpy as np

from . import charcount

VISUAL = {"image", "chart", "seal", "header_image", "footer_image"}
TEXT = {"text", "doc_title", "paragraph_title", "header", "footer", "figure_title"}
LABELS = sorted(VISUAL | TEXT)


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def area(a):
    return max(0, a[2] - a[0]) * max(0, a[3] - a[1])


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def extract(pages):
    nodes = []
    edges = []
    features = []
    for pi, p in enumerate(pages):
        blocks = p["parsing_res_list"]
        local = []
        w, h = p["width"], p["height"]
        for bi, b in enumerate(blocks):
            s = b["block_content"].strip()
            label = b["block_label"]
            if label not in VISUAL and not (label in TEXT and 1 <= len(s) <= 120):
                continue
            box = [float(v) / scale for v, scale in zip(b["block_bbox"], [w, h, w, h])]
            row = {
                "page_id": pi + 1,
                "block": bi,
                "box": box,
                "label": label,
                "text": s,
                "visual": label in VISUAL,
            }
            local.append(len(nodes))
            nodes.append(row)
        if not any(nodes[i]["visual"] for i in local):
            continue
        context = [
            len(blocks),
            sum(q["block_label"] in VISUAL for q in blocks),
            sum(q["block_label"] == "table" for q in blocks),
            sum(q["block_label"] == "figure_title" for q in blocks),
            sum(len(q["block_content"]) > 120 for q in blocks),
            pi == 0,
            pi < 3,
            math.log1p(len(pages)),
        ]

        def node(r):
            a, b, c, d = r["box"]
            s = r["text"]
            n = max(len(s), 1)
            return [
                a,
                b,
                c,
                d,
                c - a,
                d - b,
                area(r["box"]),
                math.log1p(n),
                s.count("\n"),
                sum(map(str.isdigit, s)) / n,
                sum(map(str.isupper, s)) / n,
                charcount.cjk(s) / n,
                int(
                    bool(
                        re.search(
                            r"[。;；!?！？]|\b(?:report|annual|year|chapter)\b|年度|报告|報告|第.{1,4}[章節节]",
                            s,
                            re.I,
                        )
                    )
                ),
            ] + [int(r["label"] == label) for label in LABELS]

        for off, i in enumerate(local):
            a = nodes[i]
            for j in local[off + 1 :]:
                b = nodes[j]
                if not (a["visual"] or b["visual"]):
                    continue
                aa, bb = a["box"], b["box"]
                u = union(aa, bb)
                xd = max(0, max(aa[0], bb[0]) - min(aa[2], bb[2]))
                yd = max(0, max(aa[1], bb[1]) - min(aa[3], bb[3]))
                if math.hypot(xd, yd) > 0.075 or area(u) > 0.55:
                    continue
                aw, ah = aa[2] - aa[0], aa[3] - aa[1]
                bw, bh = bb[2] - bb[0], bb[3] - bb[1]
                xover = max(0, min(aa[2], bb[2]) - max(aa[0], bb[0]))
                yover = max(0, min(aa[3], bb[3]) - max(aa[1], bb[1]))
                inside = [
                    q
                    for q in blocks
                    if q["block_label"] in {"table", "figure_title"}
                    and overlap(
                        u,
                        [v / scale for v, scale in zip(q["block_bbox"], [w, h, w, h])],
                    )
                    > 0
                ]
                feat = (
                    node(a)
                    + node(b)
                    + context
                    + [
                        xd,
                        yd,
                        xover / max(0.001, min(aw, bw)),
                        yover / max(0.001, min(ah, bh)),
                        aw / max(bw, 0.001),
                        ah / max(bh, 0.001),
                        area(u),
                        area(u) / max(0.0001, area(aa) + area(bb)),
                        len(inside),
                        abs(a["block"] - b["block"]),
                        abs((aa[0] + aa[2] - bb[0] - bb[2]) / 2),
                        abs((aa[1] + aa[3] - bb[1] - bb[3]) / 2),
                    ]
                )
                edges.append((i, j))
                features.append(feat)
    return nodes, edges, np.asarray(features, dtype=np.float32)


def group(nodes, edges, prob, threshold):
    parent = list(range(len(nodes)))
    groups = {i: [i] for i in range(len(nodes))}
    boxes = {i: r["box"] for i, r in enumerate(nodes)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in sorted(range(len(edges)), key=lambda k: -prob[k]):
        if prob[k] < threshold:
            break
        i, j = edges[k]
        a, b = find(i), find(j)
        if a == b:
            continue
        u = union(boxes[a], boxes[b])
        members = groups[a] + groups[b]
        if len(members) > 6 or area(u) > 0.55:
            continue
        # Do not bridge disconnected graphic panels through a long text line.
        if sum(area(nodes[q]["box"]) for q in members) < 0.25 * area(u):
            continue
        parent[b] = a
        groups[a] = members
        boxes[a] = u
        del groups[b]
        del boxes[b]
    return [
        {
            "page_id": nodes[inds[0]]["page_id"],
            "members": [nodes[i]["block"] for i in inds],
            "visual_members": [nodes[i]["block"] for i in inds if nodes[i]["visual"]],
            "bbox_norm": [round(v * 1000) for v in boxes[root]],
        }
        for root, inds in groups.items()
        if len(inds) > 1 and any(nodes[i]["visual"] for i in inds)
    ]


def cover_mark(page, proposal):
    """Small uncaptioned marks on sparse title pages, excluding body figures."""
    bs = [
        b
        for b in page["parsing_res_list"]
        if b["block_label"] not in {"header", "footer", "number"}
    ]
    if len(bs) > 12 or any(b["block_label"] in {"table", "figure_title"} for b in bs):
        return False
    if not any(b["block_label"] in {"doc_title", "paragraph_title"} for b in bs):
        return False
    if sum(len(b["block_content"]) > 100 for b in bs) > 1:
        return False
    for i in proposal["members"]:
        part = page["parsing_res_list"][i]
        if part["block_label"] in TEXT and re.search(
            r"(?:著|编著|編著|主编|主編|译|譯|编译|編譯|审校|審校)\s*$|^(?:by\s|written\s+by|edited\s+by|translated\s+by|author\s*:)",
            part["block_content"].strip(),
            re.I,
        ):
            return False
    a, b, c, d = proposal["bbox_norm"]
    return (c - a) * (d - b) <= 120000 and c - a < 650 and d - b < 240
