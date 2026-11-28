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

"""Local connection features; no document IDs, annotation fields or reference text."""
import collections
import re

import numpy as np

KINDS = [
    "cross_page",
    "column",
    "around_float",
    "visual_fragment",
    "enumerated_region",
    "attribution",
    "cross_page_list",
    "contents_sequence",
    "quote_continuation",
    "floating_obstacle",
    "flush_fragment",
    "skip_illustration_pages",
    "page_filled_line",
    "running_heading_bridge",
    "open_parenthesis",
    "lettered_options",
    "quote_attribution",
    "figure_wrap_width_change",
    "geometric_column_corridor",
    "long_floating_region_bridge",
    "typographic_narrative_resume",
    "cross_page_direct_speech",
]


def features(nodes, visible, edges, logs):
    reasons = {}
    for l in logs:
        if "from" in l and "to" in l:
            reasons[tuple(l["from"]), tuple(l["to"])] = l["kind"]
    rows = []
    for i, j in edges:
        a, b = nodes[i], nodes[j]
        ax, ay, ar, ab = a["bbox"]
        bx, by, br, bb = b["bbox"]
        mid = visible[a["vindex"] + 1 : b["vindex"]]
        ct = collections.Counter(l for _, _, l in mid)
        kind = reasons.get(
            ((a["page"] + 1, a["index"]), (b["page"] + 1, b["index"])), ""
        )
        x, y = a["text"], b["text"]
        vals = (
            a["values"]
            + b["values"]
            + [
                b["page"] - a["page"],
                j - i - 1,
                by - ab,
                bx - ax,
                br - ar,
                abs(bx - ax),
                abs(br - ar),
                min(ar, br) - max(ax, bx),
                (br - bx) / max(ar - ax, 0.01),
                float(a["block"].get("_old_group") == b["block"].get("_old_group")),
                len(mid),
            ]
            + [
                ct[k]
                for k in [
                    "text",
                    "paragraph_title",
                    "doc_title",
                    "table",
                    "image",
                    "chart",
                    "figure_title",
                    "display_formula",
                    "vision_footnote",
                ]
            ]
            + [float(kind == k) for k in KINDS]
        )
        vals += [
            float(bool(re.search(p, t)))
            for t in [x, y]
            for p in [
                r"\n",
                r"^(?:\(?[ivxlc]+\)|\d+[.)]|[①-⑳]|[A-Z][.、])",
                r"\$|\\\(",
                r'[。.!?！？][”’"\u2019）)]*\s*$',
                r"(?:[①-⑳]|\$)\s*$",
                r"^(?:[•●·]|[-–—]\s+)",
                r"^(?:19|20)\d\d",
                r"[：:]\s*$",
                r"\b(?:Table|Figure)\s+\d",
            ]
        ]
        rows.append(vals)
    return np.asarray(rows, dtype=np.float32)


def protected(a, b):
    return any(
        n["transcript"]
        or n["block"].get("_text_region")
        or n["block"].get("_code_region")
        for n in [a, b]
    )
