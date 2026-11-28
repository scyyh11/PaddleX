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

"""Recover caption/note paragraph continuations without folding unrelated body text."""
import re

from . import layout_text
from . import runtime_context as ctx
from .text_graph import END, TEXT, join

LABELS = {"figure_title", "vision_footnote"}
SKIP = {
    "header",
    "footer",
    "number",
    "footnote",
    "aside_text",
    "header_image",
    "footer_image",
    "image",
    "chart",
    "display_formula",
    "formula_number",
}


def apply(pre, final, visual, mode):
    paired_columns = ctx.get_config().paired_caption_columns
    flat = [
        (pi, bi, a)
        for pi, p in enumerate(pre)
        for bi, a in enumerate(p["parsing_res_list"])
        if a["block_label"] not in SKIP and a["block_content"].strip()
    ]
    logs = []
    for (pi, bi, a), (pj, bj, b) in zip(flat, flat[1:]):
        if not ({a["block_label"], b["block_label"]} & LABELS):
            continue
        if (
            a["block_label"] not in TEXT | LABELS
            or b["block_label"] not in TEXT | LABELS
        ):
            continue
        x, y = a["block_content"].strip(), b["block_content"].strip()
        ax, ay, ar, ab = [
            z / t
            for z, t in zip(a["block_bbox"], [pre[pi]["width"], pre[pi]["height"]] * 2)
        ]
        bx, by, br, bb = [
            z / t
            for z, t in zip(b["block_bbox"], [pre[pj]["width"], pre[pj]["height"]] * 2)
        ]
        cross = pj == pi + 1
        column = (
            (
                mode == "columns"
                or paired_columns
                and a["block_label"] == "vision_footnote"
                and b["block_label"] == "vision_footnote"
            )
            and pj == pi
            and bx > ax + 0.15
            and by < ab - 0.15
        )
        paired = (
            column
            and a["block_label"] == "vision_footnote"
            and b["block_label"] == "vision_footnote"
            and ab > 0.72
            and by < 0.25
        )
        if (
            not (cross or column)
            or len(x) < (8 if paired else 15)
            or len(y) < 5
            or END.search(layout_text.ending(x))
            or x.endswith((":", "："))
        ):
            continue
        if (
            layout_text.new_item(y)
            or re.match(
                r"^(?:[图圖表]\s*\d|(?:Figure|Fig\.?|Table)\s*\d|[注註]\s*[:：]|Note\s*:)",
                y,
                re.I,
            )
            and not re.search(r"\b(?:see|in|of|on|and)\s*$", x, re.I)
        ):
            continue
        av, bv = visual.get(f"{pi}:{bi}", {}), visual.get(f"{pj}:{bj}", {})
        af, bf = av.get("font", 0), bv.get("font", 0)
        if (
            not af
            or not bf
            or not 0.72 < af / bf < 1.4
            or av.get("rotated")
            or bv.get("rotated")
        ):
            continue
        if av.get("last_right", 9) > 3 or bv.get("first_left", 9) > 1.5:
            continue
        if cross and (ab < 0.68 or by > 0.35):
            continue
        if cross and min(ar - ax, br - bx) / max(ar - ax, br - bx) < 0.55:
            continue
        # Retain each source fragment once; isolate the recovered caption from other body paths.
        for pages in [pre, final]:
            aa, bb0 = (
                pages[pi]["parsing_res_list"][bi],
                pages[pj]["parsing_res_list"][bj],
            )
            aa.update(
                block_label="text",
                block_content=join(x, y),
                _text_region="caption_note_continuation",
            )
            bb0["block_content"] = ""
            bb0["_text_region"] = "caption_note_continuation"
            for z in [aa, bb0]:
                if "global_block_id" in z:
                    z["global_group_id"] = z["global_block_id"]
        logs.append(
            {
                "kind": "caption_note_continuation",
                "from": [pi + 1, bi],
                "to": [pj + 1, bj],
            }
        )
    return logs
