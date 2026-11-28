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

"""Spatial corridor checks for column continuation and figure-wrapped paragraphs."""
import re

from . import charcount, layout_text
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


def apply(nodes, visible, pages, fixed, mode="width"):
    out = {i for i, j in fixed}
    inc = {j for i, j in fixed}
    proposals = []
    logs = []

    def proposal(a, b, why, priority):
        if a["id"] < b["id"] and a["id"] not in out and b["id"] not in inc:
            proposals.append((priority, a["id"], b["id"], why))

    def valid(n):
        return (
            not n["transcript"]
            and not (n["block"].get("_code_region") or n["block"].get("_text_region"))
            and not n["visual"].get("rotated")
        )

    for a, b in zip(nodes, nodes[1:]):
        if not valid(a) or not valid(b) or b["block"].get("_explicit_paragraph_start"):
            continue
        x, y = layout_text.ending(a["text"]), b["text"].strip()
        gap = b["page"] - a["page"]
        ax, ay, ar, ab = a["bbox"]
        bx, by, br, bb = b["bbox"]
        av, bv = a["visual"], b["visual"]
        af, bf = av.get("font", 0), bv.get("font", 0)
        if gap > 1 or not x or not y:
            continue
        natural = (
            len(x) > 30
            and len(y) > 15
            and a["block"]["block_label"]
            not in {"content", "reference_content", "algorithm", "code", "code_txt"}
            and not re.search(r"[;；\d]$", x)
            and not re.match(r"^\s*%", x)
            and av.get("last_right", 9) < 1.7
            and not END.search(x)
            and not x.endswith((":", "："))
            and not layout_text.new_item(y)
            and not re.match(r"^[~～—－]", x)
        )
        middle = visible[a["vindex"] + 1 : b["vindex"]]
        clear = all(l in FLOAT for _, _, l in middle)
        similar = af and bf and 0.7 < af / bf < 1.4
        if (
            gap == 0
            and natural
            and clear
            and similar
            and bv.get("first_left", 9) < 1.3
            and -0.008 < by - ab < 0.045
        ):
            overlap = min(ar, br) - max(ax, bx)
            if overlap > 0.5 * min(ar - ax, br - bx) and (
                abs(ax - bx) < 0.065 or abs(ar - br) < 0.065
            ):
                proposal(a, b, "figure_wrap_width_change", 8)
        if (
            mode in {"corridor", "quote", "track", "combined"}
            and gap == 0
            and natural
            and bx > ax + 0.15
            and by < ab - 0.10
            and similar
        ):
            # A misplaced section heading above the source or below the destination is not a barrier.
            obstruct = False
            for pi, bi, lab in middle:
                if lab in FLOAT:
                    continue
                q = pages[pi]
                block = q["parsing_res_list"][bi]
                hx, hy, hr, hb = [
                    v / z
                    for v, z in zip(
                        block["block_bbox"],
                        [q["width"], q["height"], q["width"], q["height"]],
                    )
                ]
                after_source = abs(hx - ax) < 0.12 and hy >= ab - 0.01
                before_dest = abs(hx - bx) < 0.12 and hb <= by + 0.01
                if after_source or before_dest:
                    obstruct = True
                    break
            if not obstruct:
                proposal(a, b, "geometric_column_corridor", 7)
        if (
            mode in {"quote", "combined"}
            and clear
            and x.endswith((":", "："))
            and not layout_text.new_item(y)
            and len(x) > 10
            and len(y) > 30
        ):
            # Indented first line, conventional paragraph typography, explicit quotation introducer.
            cjk = charcount.cjk(x) > len(x) * 0.4
            quoted = bool(
                re.search(
                    r"(?:曰|[说說][道過过]?|[记記载載]曰|云|奏[称稱]|[言谓謂]之?曰?)[:：]$",
                    x,
                )
            )
            if (
                cjk
                and quoted
                and similar
                and (gap == 1 or 0 <= by - ab < 0.04)
                and bv.get("first_left", 0) > 1.3
            ):
                proposal(a, b, "indented_direct_quotation", 9)
    if mode in {"track", "combined"}:
        for i, a in enumerate(nodes):
            if (
                not valid(a)
                or not layout_text.ending(a["text"]).endswith((":", "："))
                or len(a["text"]) < 25
            ):
                continue
            ax, ay, ar, ab = a["bbox"]
            for j in range(i + 2, min(i + 16, len(nodes))):
                b = nodes[j]
                bx, by, br, bb = b["bbox"]
                if b["page"] - a["page"] > 1:
                    break
                if b["visual"].get("first_left", 9) > 1.2:
                    continue
                if not valid(b) or abs(ax - bx) > 0.025 or abs(ar - br) > 0.03:
                    continue
                mid = nodes[i + 1 : j]
                if not all(
                    valid(n) and n["bbox"][0] > ax + 0.028 and n["bbox"][2] < ar + 0.015
                    for n in mid
                ):
                    break
                if mid[0]["visual"].get("first_left", 9) > 1.0:
                    break
                vv = visible[a["vindex"] + 1 : b["vindex"]]
                if any(l not in TEXT | FLOAT for _, _, l in vv):
                    break
                if layout_text.new_item(b["text"]):
                    break
                proposal(a, b, "narrative_track_across_indented_quote", 8)
                break
    chosen = []
    for weight, i, j, why in sorted(proposals, key=lambda e: (-e[0], e[2] - e[1])):
        if i in out or j in inc:
            continue
        if any(i < k < j < l or k < i < l < j for k, l in fixed + chosen):
            continue
        out.add(i)
        inc.add(j)
        chosen.append((i, j))
        logs.append(
            {
                "kind": why,
                "from": [nodes[i]["page"] + 1, nodes[i]["index"]],
                "to": [nodes[j]["page"] + 1, nodes[j]["index"]],
            }
        )
    return chosen, logs


def longgap(nodes, visible, pages, fixed):
    out = {i for i, j in fixed}
    inc = {j for i, j in fixed}
    chosen = []
    logs = []
    for a, b in zip(nodes, nodes[1:]):
        gap = b["page"] - a["page"]
        x, y = layout_text.ending(a["text"]), b["text"].strip()
        if not 1 < gap <= 12 or a["id"] in out or b["id"] in inc:
            continue
        if (
            a["transcript"]
            or a["block"].get("_text_region")
            or b["block"].get("_text_region")
            or a["block"].get("_code_region")
            or b["block"].get("_code_region")
        ):
            continue
        mid = visible[a["vindex"] + 1 : b["vindex"]]
        if not mid or any(l not in FLOAT for _, _, l in mid):
            continue
        if (
            not x
            or not y
            or layout_text.reference_end(x)
            or END.search(x)
            or layout_text.new_item(y)
            or x.endswith((":", "："))
        ):
            continue
        if a["visual"].get("rotated") or b["visual"].get("rotated"):
            continue
        af, bf = a["visual"].get("font", 0), b["visual"].get("font", 0)
        if af and bf and not 0.75 < af / bf < 1.33:
            continue
        chosen.append((a["id"], b["id"]))
        out.add(a["id"])
        inc.add(b["id"])
        logs.append(
            {
                "kind": "long_floating_region_bridge",
                "from": [a["page"] + 1, a["index"]],
                "to": [b["page"] + 1, b["index"]],
            }
        )
    return chosen, logs
