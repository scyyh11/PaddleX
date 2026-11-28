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

"""Retain list columns and explicitly separated exercise lines."""

import re

ENUM = re.compile(
    r"^\s*(?:\d+[.、．)]\s*|[（(]\d+[）)]\s*|[【\[]\d+[】\]]\s*|\d{1,4}\s+\S)"
)
MONTH = re.compile(
    r"^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d",
    re.I,
)


def describe(text):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    enumerated = sum(bool(ENUM.match(s) or MONTH.match(s)) for s in lines)
    dot_leaders = sum(bool(re.search(r"\.{3,}|…{2,}", s)) for s in lines)
    listlike = len(lines) >= 3 and (enumerated >= 2 or dot_leaders >= 2)
    exercise = enumerated >= 2 and bool(
        re.search(
            r"\\(?:frac|xlongequal|xrightarrow|rightarrow|sqrt)|\$[^\n]*[=+^_][^\n]*\$",
            text,
        )
    )
    return {"list": listlike, "exercise": exercise, "preserve": exercise}


def rejection(a, b):
    x, y = a["block"], b["block"]

    def contact_urls(s):
        if not re.search(
            r"\b(?:Fax|Workweek|Telephone|Tel)\b|传真|傳真|电话|電話", s, re.I
        ):
            return set()
        if not re.search(r"(?:\+?\d[\d ()-]{7,}\d)", s):
            return set()
        return set(
            v.lower().rstrip(".,;")
            for v in re.findall(
                r"(?:https?://|www\.)[^\s<>]+|[\w.-]+\.(?:gov|org|edu|com|net)(?:/[^\s<>]*)?",
                s,
                re.I,
            )
        )

    u, v = contact_urls(a["text"]), contact_urls(b["text"])
    if u and v and not (u & v):
        return "independent_contact_locations"
    if x.get("_source_exercise_lines") or y.get("_source_exercise_lines"):
        return "independent_exercise_block"
    if not (x.get("_source_list_column") or y.get("_source_list_column")):
        return None
    if a["page"] != b["page"]:
        return "independent_list_page"
    ax, ay, ar, ab = a["bbox"]
    bx, by, br, bb = b["bbox"]
    if bx >= ar - 0.025 and by < ab or ax >= br - 0.025 and ay < bb:
        return "independent_list_column"
    return None


def annotate_source_flags(raw_pages, pre, final):
    """Annotate source list columns and exercise line blocks."""
    flags = {
        (pi, tuple(b["block_bbox"])): describe(b["block_content"])
        for pi, p in enumerate(raw_pages)
        for b in p["parsing_res_list"]
        if b["block_label"] in {"text", "abstract", "reference_content", "content"}
    }
    for pi, (p, q) in enumerate(zip(pre, final)):
        for a, b in zip(p["parsing_res_list"], q["parsing_res_list"]):
            f = flags.get((pi, tuple(a["block_bbox"])))
            if f:
                for block in (a, b):
                    block["_source_list_column"] = f["list"]
                    block["_source_exercise_lines"] = f["exercise"]
                    block["_preserve_source_lines"] = f["preserve"]
