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

"""Separate indented quotations from the surrounding narrative stream."""
import re

from . import charcount, layout_text
from . import runtime_context as ctx
from .text_graph import TEXT

FLOAT = {
    "table",
    "image",
    "chart",
    "figure_title",
    "display_formula",
    "formula_number",
    "vision_footnote",
}


def cjk(s):
    return charcount.cjk(s) > len(s) * 0.35


def apply(nodes, visible, pages, fixed, mode):
    guarded = ctx.get_config().quote_paths_guarded
    logs = []
    out = {i: j for i, j in fixed}
    inc = {j: i for i, j in fixed}
    extra = []
    body = [
        n
        for n in nodes
        if len(n["text"]) > 100
        and n["bbox"][2] - n["bbox"][0] > 0.55
        and cjk(n["text"])
        and n["visual"].get("lines", 0) >= 3
    ]
    conventional = (
        len(body) > 20
        and sum(n["visual"].get("first_left", 0) > 1.5 for n in body) > len(body) * 0.3
    )

    def free(a, b):
        return a["id"] not in out and b["id"] not in inc

    def add(a, b, why):
        i, j = a["id"], b["id"]
        if not free(a, b) or any(
            i < k < j < l or k < i < l < j for k, l in fixed + extra
        ):
            return
        extra.append((i, j))
        out[i] = j
        inc[j] = i
        logs.append(
            {
                "kind": why,
                "from": [a["page"] + 1, a["index"]],
                "to": [b["page"] + 1, b["index"]],
            }
        )

    def normal(n):
        return (
            n["block"]["block_label"] in {"text", "abstract"}
            and not any(
                n["block"].get(k)
                for k in ["_code_region", "_text_region", "_isolated_note"]
            )
            and not n["visual"].get("rotated")
        )

    if conventional and mode in {"track", "both"}:
        for i, a in enumerate(nodes):
            if (
                not normal(a)
                or not layout_text.ending(a["text"]).endswith((":", "："))
                or not cjk(a["text"])
                or len(a["text"]) < 20
            ):
                continue
            ax, ay, ar, ab = a["bbox"]
            af = a["visual"].get("font", 0)
            for j in range(i + 2, min(i + 20, len(nodes))):
                b = nodes[j]
                bx, by, br, bb = b["bbox"]
                if b["page"] - a["page"] > 1:
                    break
                effective_ax = ax
                if (
                    guarded
                    and a["visual"].get("lines", 0) == 1
                    and 0.03 < ax - bx < 0.07
                    and br - bx > 0.55
                ):
                    effective_ax = bx
                if not normal(b) or abs(effective_ax - bx) > 0.012:
                    continue
                if b["visual"].get("first_left", 9) > 1.1:
                    break
                bf = b["visual"].get("font", 0)
                if not af or not bf or not 0.9 < af / bf < 1.12:
                    break
                mid = nodes[i + 1 : j]
                if guarded and any(
                    layout_text.new_item(n["text"])
                    or n["text"].rstrip().endswith((":", "："))
                    or (n["text"].count("+") + n["text"].count("="))
                    > max(2, len(n["text"]) * 0.03)
                    for n in mid
                ):
                    break
                if not all(
                    normal(n)
                    and n["bbox"][0] > effective_ax + 0.032
                    and n["bbox"][2] < max(ar, br) + 0.018
                    for n in mid
                ):
                    break
                if mid[0]["visual"].get("first_left", 9) > 1.2:
                    break
                if any(
                    l not in TEXT | FLOAT
                    for _, _, l in visible[a["vindex"] + 1 : b["vindex"]]
                ):
                    break
                add(a, b, "typographic_narrative_resume")
                break
    if mode in {"colon", "both"}:
        for a, b in zip(nodes, nodes[1:]):
            if (
                not normal(a)
                or not normal(b)
                or b["page"] != a["page"] + 1
                or not a["last"]
                or not b["first"]
            ):
                continue
            x = layout_text.ending(a["text"])
            y = b["text"]
            av, bv = a["visual"], b["visual"]
            if not x.endswith((":", "：")) or not cjk(x) or not cjk(y):
                continue
            if not re.search(
                r"(?:曰|[说說][道過过]?|[记記载載]曰|云|奏[称稱]|[言谓謂]之?曰?)[:：]$",
                x,
            ):
                continue
            if any(
                l not in FLOAT for _, _, l in visible[a["vindex"] + 1 : b["vindex"]]
            ):
                continue
            if (
                abs(a["bbox"][2] - b["bbox"][2]) > 0.05
                or abs(a["bbox"][0] - b["bbox"][0]) > 0.08
            ):
                continue
            af, bf = av.get("font", 0), bv.get("font", 0)
            if af and bf and 0.8 < af / bf < 1.25 and not layout_text.new_item(y):
                add(a, b, "cross_page_direct_speech")
    return fixed + extra, logs
