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

"""Continue an author's catalogue entry across pages, volumes, and indented titles."""
import collections
import copy
import re

from .runtime_context import readonly_soup as BeautifulSoup
from .text_graph import TEXT

HEADER = re.compile(r"c[aog]talogue\s+of\s+authors", re.I)
AUTHOR = re.compile(
    r"^\s*[A-Z][A-Za-z\-\u2019\']{2,}\s*[,\.]\s+(?:[A-Z]|Mrs|Miss|Dr|Rev)"
)
BIB = re.compile(r"\b(?:\d+\s*vols?\.?|\d{1,2}mo|8vo|4to|folio)\b", re.I)


def apply(pre, final):
    prior = convert_tables(pre, final)
    active = False
    current = []
    groups = []
    for pi, p in enumerate(pre):
        text = " ".join(
            b["block_content"]
            for b in p["parsing_res_list"]
            if b["block_label"] != "table"
        )
        explicit = bool(HEADER.search(text[:500]))
        bib = len(BIB.findall(text))
        enabled = explicit or active and bib >= 3
        if not enabled:
            if len(current) > 1:
                groups.append(current)
            current = []
            active = False
            continue
        active = True
        for bi, b in enumerate(p["parsing_res_list"]):
            s = b["block_content"].strip()
            lab = b["block_label"]
            if (
                lab in {"header", "footer", "number", "footnote"}
                or HEADER.search(s)
                or re.fullmatch(r"\d{1,3}", s)
            ):
                continue
            if lab not in TEXT or not s:
                if s and lab in {"paragraph_title", "doc_title", "table"}:
                    if len(current) > 1:
                        groups.append(current)
                    current = []
                continue
            if AUTHOR.match(s):
                if len(current) > 1:
                    groups.append(current)
                current = [(pi, bi)]
            elif current:
                current.append((pi, bi))
    if len(current) > 1:
        groups.append(current)
    logs = prior
    for keys in groups:
        text = " ".join(
            pre[pi]["parsing_res_list"][bi]["block_content"] for pi, bi in keys
        )
        for pi, bi in keys:
            for pages in [pre, final]:
                b = pages[pi]["parsing_res_list"][bi]
                b.update(
                    block_content=text if (pi, bi) == keys[0] else "",
                    _text_region="catalogue_author",
                )
                if "global_block_id" in b:
                    b["global_group_id"] = b["global_block_id"]
        logs.append(
            {
                "kind": "catalogue_author_region",
                "blocks": [[pi + 1, bi] for pi, bi in keys],
                "characters": len(text),
            }
        )
    return logs


def convert_tables(pre, final):
    active = False
    can = set()
    for pi, p in enumerate(pre):
        text = " ".join(
            (
                BeautifulSoup(b["block_content"], "html.parser").get_text(
                    " ", strip=True
                )
                if "<table" in b["block_content"]
                else b["block_content"]
            )
            for b in p["parsing_res_list"]
        )
        explicit = bool(HEADER.search(text[:600]))
        active = explicit or active and len(BIB.findall(text)) >= 3
        if active:
            for bi, b in enumerate(p["parsing_res_list"]):
                if (
                    b["block_label"] == "table"
                    and len(BIB.findall(b["block_content"])) >= 3
                ):
                    can.add((pi, bi))
    groups = collections.defaultdict(list)
    for pi, p in enumerate(final):
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] == "table":
                groups[b.get("global_group_id", ("self", pi, bi))].append((pi, bi))
    safe = {key for keys in groups.values() if set(keys).issubset(can) for key in keys}
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        aa = []
        bb = []
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            if (pi, bi) not in safe:
                aa.append(a)
                bb.append(b)
                continue
            soup = BeautifulSoup(a["block_content"], "html.parser")
            rows = [
                " ".join(
                    c.get_text(" ", strip=True)
                    for c in row.find_all(["td", "th"], recursive=False)
                )
                for row in soup.find_all("tr")
            ]
            rows = [s for s in rows if s.strip()]
            x, y, r, bot = a["block_bbox"]
            for k, s in enumerate(rows):
                ident = 1000000000 + pi * 100000 + bi * 1000 + k
                box = [
                    x,
                    y + (bot - y) * k / len(rows),
                    r,
                    y + (bot - y) * (k + 1) / len(rows),
                ]
                for original, dest in [(a, aa), (b, bb)]:
                    c = copy.deepcopy(original)
                    c.update(
                        block_label="text",
                        block_content=s,
                        block_bbox=box,
                        block_id=ident,
                        group_id=ident,
                        global_block_id=ident,
                        global_group_id=ident,
                    )
                    dest.append(c)
            logs.append(
                {
                    "kind": "bibliographic_table_rows",
                    "page": pi + 1,
                    "index": bi,
                    "rows": len(rows),
                }
            )
        p["parsing_res_list"] = aa
        q["parsing_res_list"] = bb
    return logs
