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

"""All features derive only from predicted page blocks. No document identifiers."""
import collections
import copy
import math
import re

import numpy as np

from . import charcount
from .text_graph import END, LIST, SKIP, SPEAKER, TEXT, join, squash

LABELS = [
    "text",
    "abstract",
    "reference_content",
    "vertical_text",
    "content",
    "algorithm",
    "code",
    "code_txt",
]


def norm(s):
    return re.sub(r"\s+", "", s).lower()


def initialize(prepared, final, visual=None):
    visual = visual or {}
    nodes = []
    visible = []
    transcript = sum(
        bool(SPEAKER.search(b["block_content"]))
        for p in prepared
        for b in p["parsing_res_list"]
    ) >= max(5, len(prepared) * 0.3)
    for pi, (p, q) in enumerate(zip(prepared, final)):
        assert len(p["parsing_res_list"]) == len(q["parsing_res_list"])
        blocks = []
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            b["_old_group"] = b.get("global_group_id")
            b["_source_index"] = bi
            if b["block_label"] in TEXT and a["block_label"] in TEXT:
                b["block_content"] = (
                    a["block_content"]
                    if a.get("_preserve_source_lines")
                    else squash(a["block_content"])
                )
                b["global_group_id"] = b.get("global_block_id", 10000 * pi + bi)
            s = b["block_content"]
            m = SPEAKER.search(s)
            if (
                transcript
                and b["block_label"] in TEXT
                and m
                and m.start() > 2
                and re.match(r"^[a-z]", s.strip())
            ):
                c = copy.deepcopy(b)
                c["block_content"] = s[: m.start()].strip()
                c["_leading_continuation"] = True
                b["block_content"] = s[m.start() :].strip()
                blocks.append(c)
            blocks.append(b)
        q["parsing_res_list"] = blocks
        for bi, b in enumerate(blocks):
            if b["block_label"] in SKIP or (
                not b["block_content"].strip()
                and b["block_label"] not in {"image", "chart"}
            ):
                continue
            visible.append(
                (
                    pi,
                    bi,
                    "vision_footnote" if b.get("_isolated_note") else b["block_label"],
                )
            )
            if b["block_label"] not in TEXT or b.get("_isolated_note"):
                continue
            box = [
                v / z
                for v, z in zip(
                    b["block_bbox"], [q["width"], q["height"], q["width"], q["height"]]
                )
            ]
            node = {
                "page": pi,
                "index": bi,
                "source_index": b["_source_index"],
                "bbox": box,
                "block": b,
                "text": b["block_content"],
                "vindex": len(visible) - 1,
                "transcript": transcript,
                "visual": visual.get(b.get("_visual_key", f"{pi}:{bi}"), {}),
            }
            nodes.append(node)
    for i, n in enumerate(nodes):
        n["id"] = i
        n["first"] = i == 0 or nodes[i - 1]["page"] != n["page"]
        n["last"] = i == len(nodes) - 1 or nodes[i + 1]["page"] != n["page"]
        n["values"] = None
    return nodes, visible


# exact C-level forms of sum('a'<=t<='z' ...), sum('\u4e00'<=t<='\u9fff' ...), sum(t in '.!?。！？' ...)
_LOWER = charcount.counter("abcdefghijklmnopqrstuvwxyz")
_CJK = re.compile("[\u4e00-\u9fff]")
_ENDP = charcount.counter(".!?。！？")


def node_values(n):
    s = n["text"].strip()
    ss = norm(s)
    c = max(len(ss), 1)
    x, y, r, b = n["bbox"]
    lines = s.splitlines()
    lab = n["block"]["block_label"]
    v = n.get("visual", {})
    vv = [
        v.get(k, 0)
        for k in [
            "font",
            "lines",
            "first_left",
            "first_right",
            "last_left",
            "last_right",
            "median_left",
            "median_right",
            "extent",
            "rotated",
        ]
    ]
    extra = [
        float(s.endswith(("……", "…"))),
        float(s.startswith(("……", "…"))),
        float(s.startswith(("“", "「", "『", "‘", '"'))),
        float(s.count("（") > s.count("）") or s.count("(") > s.count(")")),
        float(s.count("“") > s.count("”") or s.count("「") > s.count("」")),
    ]
    return (
        vv
        + extra
        + [
            math.log1p(c),
            len(lines),
            sum(map(str.isdigit, ss)) / c,
            _LOWER(ss) / c,
            charcount.cjk(ss) / c,
            _ENDP(ss) / c,
            float(bool(END.search(s))),
            float(s.endswith((":", "："))),
            float(bool(LIST.match(s))),
            float(bool(re.match(r"^[a-z]", s))),
            float(bool(re.match(r"^[A-Z]", s))),
            float(bool(re.match(r"^\d", s))),
            float(bool(re.match(r"^[,;，；）)]", s))),
            float(s.endswith("-")),
            float(bool(SPEAKER.match(s))),
            float(bool(n["block"].get("_leading_continuation"))),
            x,
            y,
            r,
            b,
            r - x,
            b - y,
            float(n["first"]),
            float(n["last"]),
            float(n["transcript"]),
        ]
        + [float(lab == k) for k in LABELS]
    )


def candidates(nodes, visible):
    edges = []
    features = []
    for i, a in enumerate(nodes):
        count = collections.Counter()
        for j in range(i + 1, min(len(nodes), i + 110)):
            b = nodes[j]
            gap = b["page"] - a["page"]
            if gap > 6:
                break
            count[gap] += 1
            aligned = (
                abs(a["bbox"][0] - b["bbox"][0]) < 0.03
                and abs(a["bbox"][2] - b["bbox"][2]) < 0.08
            )
            if gap == 0 and count[gap] > 3 and (count[gap] > 16 or not aligned):
                continue
            if gap and count[gap] > 3 and (count[gap] > 8 or not aligned):
                continue
            # Only the final few body blocks can continue over a page boundary.
            if gap and any(
                n["page"] == a["page"]
                for n in nodes[min(i + 4, len(nodes) - 1) : min(i + 5, len(nodes))]
            ):
                continue
            middle = visible[a["vindex"] + 1 : b["vindex"]]
            ctr = collections.Counter(l for _, _, l in middle)
            ax, ay, ar, ab = a["bbox"]
            bx, by, br, bb = b["bbox"]
            vals = (
                a["values"]
                + b["values"]
                + [
                    gap,
                    j - i - 1,
                    by - ab,
                    bx - ax,
                    br - ar,
                    abs(bx - ax),
                    abs(br - ar),
                    min(ar, br) - max(ax, bx),
                    min(ab, bb) - max(ay, by),
                    (br - bx) / max(ar - ax, 0.01),
                    float(a["block"].get("_old_group") == b["block"].get("_old_group")),
                    float(
                        a["block"].get("group_id") == b["block"].get("group_id")
                        and a["page"] == b["page"]
                    ),
                    len(middle),
                    sum(ctr[k] for k in TEXT),
                    ctr["paragraph_title"] + ctr["doc_title"],
                    ctr["table"],
                    ctr["image"] + ctr["chart"],
                    ctr["figure_title"],
                    ctr["display_formula"],
                ]
            )
            edges.append((i, j))
            features.append(vals)
    return edges, np.asarray(features, dtype=np.float32)


def decode(nodes, edges, proba, threshold):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    n = len(nodes)
    if not edges:
        return []
    # Maximum-weight path cover: one predecessor and successor per fragment.
    # A private unmatched column lets each node choose to stop its paragraph.
    rows = list(range(n))
    cols = [n + i for i in range(n)]
    costs = [2.0] * n
    for (a, b), p in zip(edges, proba):
        if p <= threshold:
            continue
        gain = math.log(max(p, 1e-6) / (max(1 - p, 1e-6))) - math.log(
            threshold / (1 - threshold)
        )
        rows.append(a)
        cols.append(b)
        costs.append(2.0 - min(gain, 20))
    graph = coo_matrix(
        (np.asarray(costs) + 1e-8, (rows, cols)), shape=(n, 2 * n)
    ).tocsr()
    ii, jj = min_weight_full_bipartite_matching(graph)
    return [(int(i), int(j)) for i, j in zip(ii, jj) if j < n]


def merge_nodes(nodes, links):
    logs = []
    for i, j in sorted(links, reverse=True):
        a, b = nodes[i]["block"], nodes[j]["block"]
        a["block_content"] = join(a["block_content"], b["block_content"])
        b["block_content"] = ""
        b["global_group_id"] = a.get("global_group_id")
        logs.append(
            {
                "from": [nodes[i]["page"], nodes[i]["index"]],
                "to": [nodes[j]["page"], nodes[j]["index"]],
            }
        )
    return logs
