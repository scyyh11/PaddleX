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

"""Recover code-listing regions from rotated document pages, without recognition."""
import collections
import re

from . import charcount
from .runtime_context import readonly_soup as BeautifulSoup

CODE = re.compile(
    r"\b(?:subroutine|implicit|dimension|common|continue|endif|return|integer|double precision)\b",
    re.I,
)
START = re.compile(
    r"^\s*(?:recursive\s+)?(?:subroutine|(?:real|double precision|integer)\s+function)\b",
    re.I,
)
END = re.compile(r"\bend\s*$", re.I)
ALLOW = {
    "text",
    "paragraph_title",
    "doc_title",
    "abstract",
    "header",
    "footer",
    "vision_footnote",
    "figure_title",
    "algorithm",
    "code_txt",
}


def plain(s):
    return (
        BeautifulSoup(s, "html.parser").get_text(" ", strip=True).replace("\\n", " ")
        if "<table" in s
        else s
    )


def apply(pre, final):
    strong = []
    weak = []
    for p in pre:
        bs = [b for b in p["parsing_res_list"] if b["block_content"].strip()]
        text = " ".join(plain(b["block_content"]) for b in bs)
        count = len(CODE.findall(text))
        vertical = sum(
            b["block_bbox"][3] - b["block_bbox"][1]
            > 3 * (b["block_bbox"][2] - b["block_bbox"][0])
            for b in bs
            if b["block_label"] in ALLOW
        )
        n = sum(b["block_label"] in ALLOW for b in bs)
        strong.append(count >= 3 and vertical >= max(2, n * 0.3))
        weak.append(count >= 3 and charcount.cjk(text) < len(text) * 0.03)
    active = list(strong)
    for _ in range(8):
        active = [
            a
            or weak[i]
            and ((i > 0 and active[i - 1]) or (i + 1 < len(active) and active[i + 1]))
            for i, a in enumerate(active)
        ]
    # A converted source table must not remain duplicated inside a retained merged table.
    groups = collections.defaultdict(list)
    for pi, p in enumerate(final):
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] == "table":
                groups[b.get("global_group_id", ("self", pi, bi))].append((pi, bi))
    convertible = set()
    for keys in groups.values():
        if all(
            active[pi] and pre[pi]["parsing_res_list"][bi]["block_label"] == "table"
            for pi, bi in keys
        ):
            convertible.update(keys)
    logs = []
    pagechunks = {}
    for pi, (p, q) in enumerate(zip(pre, final)):
        if not active[pi]:
            continue
        selected = []
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            if a["block_label"] in ALLOW or (pi, bi) in convertible:
                if a["block_content"].strip():
                    selected.append((bi, plain(a["block_content"])))
        if len(selected) < 2:
            continue
        selected.sort(
            key=lambda item: (
                p["parsing_res_list"][item[0]]["block_bbox"][0],
                -p["parsing_res_list"][item[0]]["block_bbox"][3],
            )
        )
        chunks = []
        current = []
        for bi, s in selected:
            if current and (START.match(s) or END.search(current[-1][1])):
                chunks.append(current)
                current = []
            current.append((bi, s))
        if current:
            chunks.append(current)
        anchors = []
        for parts in chunks:
            anchor = parts[0][0]
            s = " ".join(t.strip() for _, t in parts)
            anchors.append(anchor)
            for bi, _ in parts:
                a, b = p["parsing_res_list"][bi], q["parsing_res_list"][bi]
                content = s if bi == anchor else ""
                for item in [a, b]:
                    item.update(
                        block_label="text", block_content=content, _code_region=True
                    )
                a["group_id"] = a.get("block_id", bi)
                b["global_group_id"] = b.get("global_block_id")
            logs.append(
                {
                    "kind": "rotated_code_region",
                    "page": pi + 1,
                    "blocks": [i for i, _ in parts],
                    "characters": len(s),
                }
            )
        pagechunks[pi] = anchors
    # Text graph will handle continuation between these paragraph-sized code regions.
    return logs
