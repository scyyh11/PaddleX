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

"""Restore a continuation mislabelled as an image note or a page footer."""
import collections
import re

from . import layout_text
from .text_graph import END

BODY = {"text", "abstract"}


def apply(pre, final, visual, labels):
    logs = []
    repeated = collections.Counter(
        re.sub(r"\d|\s", "", b["block_content"])
        for p in pre
        for b in p["parsing_res_list"]
        if b["block_label"] in {"header", "footer"}
    )

    def promote(pi, bi, why):
        for pp in [pre, final]:
            pp[pi]["parsing_res_list"][bi]["block_label"] = "text"
        key = f"{pi}:{bi}"
        if key in labels:
            visual[key] = labels[key]
        logs.append({"kind": why, "page": pi + 1, "index": bi})

    for pi, p in enumerate(pre):
        if pi == 0:
            continue
        prev = pre[pi - 1]
        src = [
            (i, b)
            for i, b in enumerate(prev["parsing_res_list"])
            if b["block_label"] in BODY and b["block_content"].strip()
        ]
        if not src:
            continue
        ai, a = src[-1]
        x = layout_text.ending(a["block_content"])
        av = visual.get(f"{pi-1}:{ai}", {})
        af = av.get("font", 0)
        if (
            not x
            or END.search(x)
            or x.endswith((":", "："))
            or a["block_bbox"][3] < prev["height"] * 0.65
        ):
            continue
        for bi, b in enumerate(p["parsing_res_list"]):
            y = b["block_content"].strip()
            lab = b["block_label"]
            if lab in BODY and y or lab in {"paragraph_title", "doc_title"}:
                break
            if (
                lab not in {"vision_footnote", "header", "figure_title"}
                or not y
                or b["block_bbox"][1] > p["height"] * 0.4
            ):
                continue
            if (
                not re.match(r"^[a-z」』）)]", y)
                or len(y) < 5
                or layout_text.new_item(y)
            ):
                continue
            if lab == "header" and repeated[re.sub(r"\d|\s", "", y)] > 1:
                continue
            bv = labels.get(f"{pi}:{bi}", {})
            bf = bv.get("font", 0)
            if (
                not af
                or not bf
                or not 0.75 < af / bf < 1.3
                or bv.get("first_left", 9) > 1.5
            ):
                continue
            if lab == "header" and not re.search(r"[。.!?！？]$", y):
                continue
            promote(pi, bi, "boundary_prose_label")
            break
    for pi, p in enumerate(pre[:-1]):
        nxt = pre[pi + 1]
        nn = [
            (bi, b)
            for bi, b in enumerate(nxt["parsing_res_list"])
            if b["block_label"]
            not in {"header", "number", "footer", "footnote", "aside_text"}
            and b["block_content"].strip()
        ]
        if not nn:
            continue
        bj, b = nn[0]
        y = b["block_content"].strip()
        if (
            b["block_label"] not in BODY
            or b["block_bbox"][1] / nxt["height"] > 0.3
            or layout_text.new_item(y)
        ):
            continue
        bv = visual.get(f"{pi+1}:{bj}", {})
        bf = bv.get("font", 0)
        for bi, a in enumerate(p["parsing_res_list"]):
            x = a["block_content"].strip()
            if (
                a["block_label"] != "footer"
                or len(x) < 30
                or len(x) > 350
                or repeated[re.sub(r"\d|\s", "", x)] > 1
                or END.search(layout_text.ending(x))
                or x.endswith((":", "："))
            ):
                continue
            av = labels.get(f"{pi}:{bi}", {})
            af = av.get("font", 0)
            if (
                af
                and bf
                and 0.8 < af / bf < 1.2
                and av.get("last_right", 9) < 1.5
                and bv.get("first_left", 9) < 1.5
            ):
                promote(pi, bi, "footer_prose_continuation")
    return logs
