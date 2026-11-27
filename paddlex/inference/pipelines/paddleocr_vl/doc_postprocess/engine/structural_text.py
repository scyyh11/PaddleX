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

"""Paragraph continuations from page geometry and explicit list structure."""
import collections
import re

from . import layout_text
from . import runtime_context as ctx
from .text_graph import END, TEXT

FLOAT = {
    "table",
    "image",
    "chart",
    "figure_title",
    "display_formula",
    "formula_number",
    "vision_footnote",
}


def apply(nodes, visible, pages, fixed, mode):
    wide_paren = ctx.get_config().wide_parenthesis
    proposals = []
    logs = []
    out = {i for i, j in fixed}
    inc = {j for i, j in fixed}

    def add(a, b, why, weight=5):
        if a["id"] < b["id"] and a["id"] not in out and b["id"] not in inc:
            proposals.append((weight, a["id"], b["id"], why))

    repeat = collections.Counter()
    for pi, p in enumerate(pages):
        for b in p["parsing_res_list"]:
            if (
                b["block_label"] in {"paragraph_title", "doc_title"}
                and b["block_bbox"][1] / p["height"] < 0.16
            ):
                repeat[re.sub(r"\s+|\d+", "", b["block_content"])] += 1

    def barriers(a, b):
        middle = visible[a["vindex"] + 1 : b["vindex"]]
        hard = []
        for pi, bi, lab in middle:
            if lab in {"paragraph_title", "doc_title"}:
                p = pages[pi]
                b0 = p["parsing_res_list"][bi]
                s = b0["block_content"]
                y = b0["block_bbox"][1] / p["height"]
                if y < 0.16 and repeat[re.sub(r"\s+|\d+", "", s)] >= 3:
                    continue
                hard.append(lab)
            elif lab not in FLOAT:
                hard.append(lab)
        return hard

    def natural(a, b):
        x, y = layout_text.ending(a["text"]), b["text"].strip()
        return (
            x
            and y
            and not layout_text.reference_end(x)
            and not b["block"].get("_explicit_paragraph_start")
            and not re.match(r"^[~～—－]", x)
            and not re.search(r"\b(?:\d{1,2}mo|8vo|4to|folio)\.?$", x, re.I)
            and not layout_text.new_item(y)
            and not x.endswith((":", "："))
            and (not END.search(x) or re.match(r"^[a-z,;，；）)]", y))
        )

    by = collections.defaultdict(list)
    for n in nodes:
        by[n["page"]].append(n)
    for a, b in zip(nodes, nodes[1:]):
        if b["block"].get("_explicit_paragraph_start"):
            continue
        if (
            a["transcript"]
            or a["block"].get("_code_region")
            or b["block"].get("_code_region")
            or a["block"].get("_text_region")
            or b["block"].get("_text_region")
        ):
            continue
        gap = b["page"] - a["page"]
        av, bv = a["visual"], b["visual"]
        ax, ay, ar, ab = a["bbox"]
        bx, by0, br, bb = b["bbox"]
        x, y = layout_text.ending(a["text"]), b["text"].strip()
        if av.get("rotated") or bv.get("rotated") or barriers(a, b):
            continue
        font = av.get("font", 0)
        fontb = bv.get("font", 0)
        similar = font and fontb and 0.75 < font / fontb < 1.33
        flush = bv.get("first_left", 9) < 1.2
        filled = av.get("last_right", 9) < 1.5
        aligned = abs(ar - br) < 0.1 and (
            abs(ax - bx) < 0.09
            or min(ar, br) - max(ax, bx) > 0.65 * max(ar - ax, br - bx)
        )
        column = gap == 0 and bx > ax + 0.15 and by0 < ab - 0.12
        if mode in {"geometry", "structure", "all"}:
            if gap == 0 and (aligned and by0 > ab - 0.015 or column) and natural(a, b):
                mid = visible[a["vindex"] + 1 : b["vindex"]]
                if mid and all(l in FLOAT for _, _, l in mid) and flush:
                    add(a, b, "floating_obstacle", 7)
                elif (
                    not mid
                    and similar
                    and flush
                    and filled
                    and (column or by0 - ab < 0.04)
                ):
                    add(a, b, "flush_fragment", 5)
            if 0 < gap <= 4 and a["last"] and b["first"]:
                if gap > 1 and natural(a, b) and re.search(r"[-,，；;]$", x):
                    add(a, b, "skip_illustration_pages", 7)
                # A full final line plus an unindented following line can continue after a full stop.
                if (
                    gap == 1
                    and not layout_text.reference_end(x)
                    and filled
                    and flush
                    and similar
                    and len(x) > 30
                    and len(y) > 20
                    and not layout_text.new_item(y)
                    and not x.endswith((":", "："))
                ):
                    # Single-line destination boxes can hide a first-line indentation. Require broad body width.
                    if br - bx > 0.22 and (
                        bv.get("lines", 0) >= 2 or abs((br - bx) - (ar - ax)) < 0.05
                    ):
                        add(a, b, "page_filled_line", 6)
                if gap == 1 and natural(a, b):
                    add(a, b, "running_heading_bridge", 5)
                if (
                    gap == 1
                    and x.count("(") + x.count("（") > x.count(")") + x.count("）")
                    and (
                        ")" in y[:100] + y[-10:]
                        or wide_paren
                        and "）" in y[:100] + y[-10:]
                    )
                ):
                    add(a, b, "open_parenthesis", 6)
        if mode in {"structure", "all"}:
            # Short consecutive lettered answer choices form a list. Each source is emitted once.
            ma = re.match(r"^\s*([a-h])[.)]\s+", x, re.I)
            mb = re.match(r"^\s*([a-h])[.)]\s+", y, re.I)
            if (
                ma
                and mb
                and ma[1].islower()
                and mb[1].islower()
                and ord(mb[1].lower()) == ord(ma[1].lower()) + 1
                and len(x) < 160
                and len(y) < 160
                and gap <= 1
                and (gap or abs(ax - bx) < 0.06)
            ):
                add(a, b, "lettered_options", 7)
            if (
                gap <= 1
                and re.match(r"^[~～]", y)
                and len(y) < 180
                and len(x) > 40
                and not re.match(r"^[~～]", x)
                and (gap or by0 - ab < 0.04)
            ):
                add(a, b, "quote_attribution", 7)
    # Parallel narrative/quotation tracks: resume broad text around indented quotations.
    if mode in {"tracks", "all"}:
        for i, a in enumerate(nodes):
            if a["transcript"]:
                continue
            for j in range(i + 2, min(i + 10, len(nodes))):
                b = nodes[j]
                gap = b["page"] - a["page"]
                ax, ay, ar, ab = a["bbox"]
                bx, by0, br, bb = b["bbox"]
                if gap > 1:
                    break
                if gap == 0 and by0 < ab:
                    continue
                if abs(ax - bx) > 0.035 or abs(ar - br) > 0.055:
                    continue
                mid = nodes[i + 1 : j]
                if not all(
                    c["bbox"][0] > max(ax, bx) + 0.018
                    and c["bbox"][2] < max(ar, br) + 0.015
                    and c["bbox"][2] - c["bbox"][0] < (ar - ax) * 0.97
                    for c in mid
                ):
                    continue
                visible_between = visible[a["vindex"] + 1 : b["vindex"]]
                if any(l not in TEXT | FLOAT for _, _, l in visible_between):
                    continue
                if len(a["text"]) < 65 or len(b["text"]) < 40:
                    continue
                add(a, b, "narrative_around_quote", 8)
                break
    selected = []
    for w, i, j, why in sorted(proposals, key=lambda e: (-e[0], e[2] - e[1])):
        if i in out or j in inc:
            continue
        # Forbid crossing arcs across different text streams; nested quotation arcs are valid.
        if any(i < k < j < l or k < i < l < j for k, l in fixed + selected):
            continue
        out.add(i)
        inc.add(j)
        selected.append((i, j))
        logs.append(
            {
                "kind": why,
                "from": [nodes[i]["page"] + 1, nodes[i]["index"]],
                "to": [nodes[j]["page"] + 1, nodes[j]["index"]],
            }
        )
    return selected, logs
