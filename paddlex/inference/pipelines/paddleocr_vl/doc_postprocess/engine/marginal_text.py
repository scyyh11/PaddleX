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

"""Keep margin text that PaddleOCR-VL hides from Markdown.

PaddleOCR-VL drops aside_text, header and footer blocks from its Markdown. Side
notes and stamps (aside_text) are frequently real text of the page, and header
or footer blocks that sit away from the page edges are often body lines such as
dates or captions. Modes:
  aside      aside_text only
  aside_mid  aside_text, plus header/footer blocks whose box lies outside the
             outer 10% bands of the page
  all        aside_text, header and footer blocks everywhere
  all_plus   as all, plus page-number and footnote blocks"""


def apply(final, mode):
    events = []
    for pi, p in enumerate(final):
        h = float(p.get("height") or 1)
        for b in p["parsing_res_list"]:
            lab = b.get("block_label")
            txt = (b.get("block_content") or "").strip()
            if not txt or lab not in (
                "aside_text",
                "header",
                "footer",
                "number",
                "footnote",
            ):
                continue
            keep = False
            if lab in ("number", "footnote"):
                keep = mode == "all_plus"
            elif lab == "aside_text":
                keep = True
            elif mode in ("all", "all_plus"):
                keep = True
            elif mode == "aside_mid":
                y0, y1 = b["block_bbox"][1] / h, b["block_bbox"][3] / h
                keep = y0 > 0.10 and y1 < 0.90
            if keep:
                events.append(
                    {"kind": "margin_text_kept", "page": pi + 1, "label": lab}
                )
                b["block_label"] = "text"
                if txt.startswith("#"):
                    b["block_content"] = "\\" + txt
    return events
