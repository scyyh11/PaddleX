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

"""Split a title block that holds a CJK title line over its Latin translation.

Bilingual documents often set a heading as two lines, for example "化学系"
over "Department of Chemistry". PaddleOCR-VL returns both lines as one title
block and the Markdown writer joins them into one heading. When the layout
model detects the two lines as separate blocks, the heading models already
place the translation one level below the CJK heading; a merged block is
written the same way. Blocks with any other mix of lines are left unchanged.
"""
import re

from .line_breaks import join

CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
LATIN = re.compile(r"^[A-Za-z][A-Za-z0-9 ,.:;&()'’\-/]*$")
LABELS = ("paragraph_title", "doc_title")


def split(text):
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    if not 2 <= len(lines) <= 4:
        return None
    kinds = ["c" if CJK.search(l) else ("l" if LATIN.match(l) else "o") for l in lines]
    if "o" in kinds:
        return None
    groups = []
    for l, k in zip(lines, kinds):
        if groups and groups[-1][0] == k:
            groups[-1][1].append(l)
        else:
            groups.append((k, [l]))
    if len(groups) != 2 or groups[0][0] != "c":
        return None
    out = []
    for k, ls in groups:
        s = ls[0]
        for l in ls[1:]:
            s = join(s, l)
        out.append((s, len(ls)))
    return out


def apply(final):
    events = []
    for pi, page in enumerate(final):
        out = []
        for b in page["parsing_res_list"]:
            parts = (
                split(b.get("block_content"))
                if b.get("block_label") in LABELS
                else None
            )
            if not parts:
                out.append(b)
                continue
            x1, y1, x2, y2 = b["block_bbox"]
            total = sum(n for _, n in parts)
            top = y1
            for k, (text, n) in enumerate(parts):
                nb = dict(b)
                nb["block_content"] = text
                h = (y2 - y1) * n / total
                nb["block_bbox"] = [x1, int(round(top)), x2, int(round(top + h))]
                top += h
                nb.pop("block_polygon_points", None)
                if k == 1:
                    level = (
                        b.get("title_level")
                        if b.get("block_label") == "paragraph_title"
                        else 0
                    )
                    nb["block_label"] = "paragraph_title"
                    nb["title_level"] = (level if isinstance(level, int) else 0) + 1
                out.append(nb)
            events.append(
                {"kind": "bilingual_title_split", "page": pi + 1, "parts": len(parts)}
            )
        page["parsing_res_list"] = out
    return events
