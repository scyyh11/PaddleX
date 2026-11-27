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

"""Prediction-only evidence that two OCR blocks belong to one paragraph."""
import collections
import math
import re
import statistics

import numpy as np

from . import charcount

LABELS = [
    "text",
    "abstract",
    "reference_content",
    "vertical_text",
    "content",
    "algorithm",
    "code",
    "code_txt",
    "doc_title",
    "paragraph_title",
]
# Exact C-level forms of the per-character counts used below:
#   sum('\u4e00'<=t<='\u9fff' for t in s) == len(_CJK.findall(s)); sum(t.isupper() for t in letters) == len(_UPPER.findall(s))
#   (letters are the [A-Za-z] characters); sum(t in P for t in s) == sum(map(P.__contains__, s)); sum(t.isdigit() ...) == sum(map(str.isdigit, s))
_CJK = re.compile("[\u4e00-\u9fff]")
_UPPER = re.compile("[A-Z]")
_PUNCT = charcount.counter(".，,。;；:：!！?？")


def describe(pages):
    nodes = []
    lookup = {}
    bypage = collections.defaultdict(list)
    for pi, p in enumerate(pages):
        refs = [
            b["block_bbox"]
            for b in p["parsing_res_list"]
            if b["block_label"] == "text"
            and len(b["block_content"]) >= 100
            and b["block_bbox"][2] - b["block_bbox"][0] > 0.25 * p["width"]
        ]
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] not in LABELS or not b["block_content"].strip():
                continue
            box = [
                float(v) / s
                for v, s in zip(
                    b["block_bbox"], [p["width"], p["height"], p["width"], p["height"]]
                )
            ]
            s = b["block_content"].strip()
            w = max(0.001, box[2] - box[0])
            h = max(0.001, box[3] - box[1])
            c = max(len(s), 1)
            words = re.findall(r"[A-Za-z]+", s)
            nletters = charcount.latin(s)
            widths = [
                q[2] - q[0]
                for q in refs
                if abs(q[0] - b["block_bbox"][0]) < 0.06 * p["width"]
            ]
            fraction = (
                (b["block_bbox"][2] - b["block_bbox"][0]) / statistics.median(widths)
                if widths
                else 1.0
            )
            phone = bool(re.search(r"(?:\+?\d[\d ()-]{7,}\d)", s))
            email = "@" in s and bool(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", s))
            features = (
                [
                    math.log1p(c),
                    c,
                    s.count("\n") + 1,
                    len(words),
                    sum(map(str.isdigit, s)) / c,
                    charcount.cjk(s) / c,
                    nletters / c,
                    len(_UPPER.findall(s)) / max(1, nletters),
                    _PUNCT(s) / c,
                    w,
                    h,
                    w * h,
                    w / max(h, 0.001),
                    fraction,
                    h / max(1, math.sqrt(c * h / w)),
                    phone,
                    email,
                    int(
                        bool(
                            re.match(
                                r"^\s*(?:\d+[.)、]|[•●▪\-]|[（(][一二三四五六七八九十\d]+[）)])",
                                s,
                            )
                        )
                    ),
                    int(bool(re.match(r"^[a-z]", s))),
                    int(bool(re.match(r"^[A-Z]", s))),
                    int(bool(re.match(r"^[\d]", s))),
                    int(bool(re.match(r"^[,;，；）)]", s))),
                    int(bool(re.search(r"[。.!?！？][”’\"\')）]*$", s))),
                    int(s.endswith((":", "："))),
                    int(s.endswith(("-", ",", "，", ";", "；"))),
                    s.count("(") + s.count("（") - s.count(")") - s.count("）"),
                    int(
                        bool(
                            re.search(
                                r"\b(?:Phone|Fax|Email|Tel|Address|Contact)\b|电话|传真|邮箱|地址|联系人|主编|总监",
                                s,
                                re.I,
                            )
                        )
                    ),
                ]
                + box
                + [int(b["block_label"] == k) for k in LABELS]
            )
            row = {
                "page": pi,
                "index": bi,
                "box": box,
                "source_box": tuple(b["block_bbox"]),
                "text": s,
                "features": features,
                "label": b["block_label"],
            }
            lookup[pi, tuple(b["block_bbox"])] = len(nodes)
            bypage[pi].append(len(nodes))
            nodes.append(row)
    return nodes, lookup, bypage


def pair(a, b):
    ax, ay, ar, ab = a["box"]
    bx, by, br, bb = b["box"]
    aa = np.asarray(a["features"])
    bc = np.asarray(b["features"])
    w1 = ar - ax
    w2 = br - bx
    return np.concatenate(
        [
            aa,
            bc,
            bc - aa,
            np.asarray(
                [
                    b["page"] - a["page"],
                    b["index"] - a["index"],
                    by - ab,
                    bx - ax,
                    br - ar,
                    min(ar, br) - max(ax, bx),
                    min(ab, bb) - max(ay, by),
                    abs(w2 - w1),
                    w2 / max(w1, 0.001),
                    int(a["label"] == b["label"]),
                    int(
                        a["text"].rstrip().endswith("-")
                        and re.match(r"^[a-z]", b["text"]) is not None
                    ),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def edges(nodes, bypage):
    out = []
    for pi, inds in bypage.items():
        for k, i in enumerate(inds):
            for j in inds[k + 1 : k + 5]:
                out.append((i, j))
        for gap in [1, 2]:
            for i in inds[-3:]:
                for j in bypage.get(pi + gap, [])[:3]:
                    out.append((i, j))
    return out
