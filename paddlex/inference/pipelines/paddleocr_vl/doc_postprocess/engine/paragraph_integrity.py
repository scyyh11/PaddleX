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

"""Retain independent short labels instead of treating tight boxes as full lines."""

import re
import statistics


def additional_rejection(a, b):
    x, y = a["text"].strip(), b["text"].strip()
    if a["page"] != b["page"]:
        return None
    if a["block"].get("_standalone_demoted_heading") or b["block"].get(
        "_standalone_demoted_heading"
    ):
        return "independent_demoted_heading"
    phone = lambda s: bool(re.match(r"^\s*[+（(]?\d[\d ()（）-]{6,}", s))
    email = lambda s: bool(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", s))
    if len(x) < 60 and len(y) < 100 and phone(y) and not (phone(x) or email(x)):
        return "contact_method_after_name_or_label"
    fraction = a["block"].get("_source_column_fraction", 1.0)
    if (
        fraction < 0.65
        and len(x) <= 80
        and len(y) <= 80
        and not re.search(r"[，,;；:：\-]$", x)
        and not a["block"].get("_code_region")
        and not b["block"].get("_code_region")
    ):
        return "short_line_does_not_fill_its_column"
    return None


def annotate_source_flags(raw_pages, pre, final):
    """Annotate demoted short headings and source column width fractions."""
    raw = {
        (pi, tuple(b["block_bbox"])): b
        for pi, p in enumerate(raw_pages)
        for b in p["parsing_res_list"]
    }
    reference = {
        pi: [
            b["block_bbox"]
            for b in p["parsing_res_list"]
            if b["block_label"] == "text"
            and len(b["block_content"]) >= 100
            and b["block_bbox"][2] - b["block_bbox"][0] > 0.25 * p["width"]
        ]
        for pi, p in enumerate(raw_pages)
    }
    for pi, p in enumerate(final):
        for bi, b in enumerate(p["parsing_res_list"]):
            source = raw.get((pi, tuple(b["block_bbox"])))
            if not source or b["block_label"] != "text":
                continue
            s = source["block_content"].strip()
            box = source["block_bbox"]
            w = raw_pages[pi]["width"]
            if (
                source["block_label"] in {"doc_title", "paragraph_title"}
                and len(s) <= 160
                and not re.search(r"[。.!?！？][”’\"\']?$", s)
            ):
                b["_standalone_demoted_heading"] = True
                pre[pi]["parsing_res_list"][bi]["_standalone_demoted_heading"] = True
            widths = [
                q[2] - q[0] for q in reference[pi] if abs(q[0] - box[0]) < 0.06 * w
            ]
            if widths:
                ratio = (box[2] - box[0]) / max(1, statistics.median(widths))
                b["_source_column_fraction"] = ratio
                pre[pi]["parsing_res_list"][bi]["_source_column_fraction"] = ratio
