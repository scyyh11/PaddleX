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

"""Recover body continuations and caption obstacles from local figure geometry."""
import re

from . import charcount, layout_text
from .text_graph import END, TEXT

CAP = re.compile(
    r"^\s*(?:[图圖表]\s*\d|(?:Figure|Fig\.?|Table|Plate)\s*[\dIVX]+|[①-⑳]|[注註]\s*[：:])",
    re.I,
)
FLOAT = {
    "table",
    "image",
    "chart",
    "figure_title",
    "display_formula",
    "formula_number",
    "vision_footnote",
}


def prepare(pre, final, mode):
    logs = []
    if mode in {"captions", "all"}:
        for pi, (p, q) in enumerate(zip(pre, final)):
            bs = p["parsing_res_list"]
            images = [
                b["block_bbox"] for b in bs if b["block_label"] in {"image", "chart"}
            ]
            for bi, (a, b) in enumerate(zip(bs, q["parsing_res_list"])):
                s = a["block_content"].strip()
                if (
                    b["block_label"] not in TEXT | {"paragraph_title"}
                    or len(s) > 28
                    or not s
                    or re.search(r"[。:：;；!?！？]", s)
                ):
                    continue
                x, y, r, bot = a["block_bbox"]
                near = any(
                    r - x < (ir - ix) * 0.72
                    and ix - p["width"] * 0.01 <= x
                    and r <= ir + p["width"] * 0.01
                    and (
                        -0.008 < (y - ib) / p["height"] < 0.017
                        or -0.008 < (iy - bot) / p["height"] < 0.012
                    )
                    for ix, iy, ir, ib in images
                )
                if near:
                    for c in [a, b]:
                        c["_isolated_note"] = True
                    logs.append(
                        {"kind": "floating_image_label", "page": pi + 1, "index": bi}
                    )
    if mode in {"labels", "all"}:
        flat = [
            (pi, bi, a)
            for pi, p in enumerate(pre)
            for bi, a in enumerate(p["parsing_res_list"])
            if a["block_content"].strip()
            and a["block_label"] not in {"header", "footer", "number", "footnote"}
        ]
        for k, (pi, bi, a) in enumerate(flat):
            if a["block_label"] not in {
                "figure_title",
                "vision_footnote",
                "paragraph_title",
            }:
                continue
            s = a["block_content"].strip()
            p = pre[pi]
            if (
                len(s) < 8
                or len(s) > 180
                or CAP.match(s)
                or not re.search(r"[。.:：,，;；]", s)
            ):
                continue
            if layout_text.new_item(s):
                continue
            ax, ay, ar, ab = [
                v / z
                for v, z in zip(
                    a["block_bbox"], [p["width"], p["height"], p["width"], p["height"]]
                )
            ]
            good = False
            for t, source in [(k - 1, False), (k + 1, True)]:
                if not 0 <= t < len(flat):
                    continue
                pj, bj, b = flat[t]
                q = pre[pj]
                if (
                    b["block_label"] not in TEXT
                    or abs(pj - pi) > 1
                    or b.get("_isolated_note")
                ):
                    continue
                x = b["block_content"].strip()
                bx, by, br, bb = [
                    v / z
                    for v, z in zip(
                        b["block_bbox"],
                        [q["width"], q["height"], q["width"], q["height"]],
                    )
                ]
                sx, sy, sr, sb = (ax, ay, ar, ab) if source else (bx, by, br, bb)
                tx, ty, tr, tb = (bx, by, br, bb) if source else (ax, ay, ar, ab)
                gap = (pj - pi) if source else (pi - pj)
                prev = s if source else x
                nxt = x if source else s
                cross = gap == 1 or gap == 0 and tx > sx + 0.15 and ty < sb - 0.15
                if (
                    cross
                    and not END.search(layout_text.ending(prev))
                    and not prev.endswith((":", "："))
                    and not layout_text.new_item(nxt)
                ):
                    good = True
            if good:
                for pages in [pre, final]:
                    pages[pi]["parsing_res_list"][bi]["block_label"] = "text"
                logs.append(
                    {"kind": "column_body_label_recovery", "page": pi + 1, "index": bi}
                )
    return logs


def link(nodes, visible, pages, fixed):
    out = {i for i, j in fixed}
    inc = {j for i, j in fixed}
    extra = []
    logs = []
    for a, b in zip(nodes, nodes[1:]):
        if a["page"] != b["page"] or a["id"] in out or b["id"] in inc:
            continue
        if any(
            n["transcript"]
            or n["block"].get("_text_region")
            or n["block"].get("_code_region")
            for n in [a, b]
        ):
            continue
        ax, ay, ar, ab = a["bbox"]
        bx, by, br, bb = b["bbox"]
        av, bv = a["visual"], b["visual"]
        x, y = a["text"].strip(), b["text"].strip()
        if not (bx > ax + 0.15 and by < ab - 0.15 and len(x) > 40 and len(y) > 20):
            continue
        if (
            charcount.cjk(x) < len(x) * 0.35
            or layout_text.new_item(y)
            or x.endswith((":", "："))
        ):
            continue
        if (
            av.get("last_right", 9) > 1.0
            or bv.get("first_left", 9) > 1.2
            or av.get("lines", 0) < 2
            or bv.get("lines", 0) < 2
        ):
            continue
        if not 0.8 < av.get("font", 0) / max(bv.get("font", 0), 0.1) < 1.25:
            continue
        if any(l not in FLOAT for _, _, l in visible[a["vindex"] + 1 : b["vindex"]]):
            continue
        extra.append((a["id"], b["id"]))
        out.add(a["id"])
        inc.add(b["id"])
        logs.append(
            {
                "kind": "filled_column_continuation",
                "from": [a["page"] + 1, a["index"]],
                "to": [b["page"] + 1, b["index"]],
            }
        )
    return extra, logs
