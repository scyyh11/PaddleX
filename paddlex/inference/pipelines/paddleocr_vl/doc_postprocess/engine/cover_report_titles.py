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

"""Keep a cover-page report name inside the document title it belongs to.

Annual and interim reports usually open with the issuer's name over the
report name ("金科地产集团股份有限公司" over "2019年年度报告", or
"Alaska Air Group" over "2024 Annual Report"). The layout model returns the
two lines as separate title blocks, so the Markdown gets a document title
with the report name as its first child heading, although the cover holds a
single title. On the first page only, when a short single-line document title
is followed by a title block that is a report name alone, the report name is
written as the plain line under the title rather than as a new heading. Both
lines keep their own blocks and reading position. The decision uses only the
predicted page.
"""
import re

REPORT = re.compile(
    r"^(?:\d{4}\s*年?\s*(?:年度|半年度|第[一二三四1-4]季度|中期)?\s*(?:报告|報告|年报|年報)(?:摘要|全文)?"
    r"|(?:\d{4}\s+)?(?:annual|interim|half[- ]year(?:ly)?)\s+report(?:\s+\d{4})?)$",
    re.I,
)
PASS = {
    "image",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "number",
    "seal",
    "aside_text",
}


def apply(final):
    if not final:
        return []
    page = final[0]
    blocks = page["parsing_res_list"]
    events = []
    order = [i for i, b in enumerate(blocks) if b.get("block_label") not in PASS]
    drop = set()
    for i, j in zip(order, order[1:]):
        a, b = blocks[i], blocks[j]
        if (
            i in drop
            or a.get("block_label") != "doc_title"
            or b.get("block_label") not in ("doc_title", "paragraph_title")
        ):
            continue
        ta = (a.get("block_content") or "").strip()
        tb = (b.get("block_content") or "").strip()
        if not ta or "\n" in ta or len(ta) > 40 or not REPORT.match(tb):
            continue
        b["block_label"] = "text"
        b.pop("title_level", None)
        drop.add(j)
        events.append(
            {
                "kind": "cover_report_name_under_title",
                "page": 1,
                "title": ta,
                "report": tb,
            }
        )
    return events
