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

"""Recover space-separated numeric cells using an adjacent printed schema."""
import re

from .table_graph import cells, fragments, norm

NUMBER = re.compile(r"[+−-]?(?:\d+(?:[.,]\d+)?|\.\d+)%?")


def expand_pair(a, b):
    from bs4 import BeautifulSoup

    ta, tb = a["tab"], b["tab"]
    ra, rb = ta["rows"], tb["rows"]
    if not (
        2 <= len(ra) == len(rb) <= 5 and ta["ncols"] == 2 and 5 <= tb["ncols"] <= 40
    ):
        return None
    if any(len(x) != 2 or len(y) != tb["ncols"] for x, y in zip(ra, rb)):
        return None
    if any(
        not x[0] or norm(x[0]) != norm(y[0]) or NUMBER.fullmatch(x[0])
        for x, y in zip(ra, rb)
    ):
        return None
    values = [x[1].split() for x in ra]
    if any(
        len(v) != tb["ncols"] - 1 or any(not NUMBER.fullmatch(s) for s in v)
        for v in values
    ):
        return None
    if any(not NUMBER.fullmatch(s.strip()) for row in rb for s in row[1:]):
        return None
    soup = BeautifulSoup(a["html"], "html.parser")
    for row, vv in zip(soup.find_all("tr"), values):
        cc = cells(row)
        if any(c.get("colspan") or c.get("rowspan") for c in cc):
            return None
        cc[1].extract()
        for value in vv:
            cell = soup.new_tag("td")
            cell.string = value
            row.append(cell)
    return str(soup)


def apply(pages, final):
    fs = fragments(pages, final)
    events = []
    for a, b in zip(fs, fs[1:]):
        if not (
            a["page"] + 1 == b["page"]
            and a["rank"] == a["tables_on_page"] - 1
            and b["rank"] == 0
        ):
            continue
        if a["body_after"] or b["body_before"]:
            continue
        aa, bb = a["bbox"], b["bbox"]
        wa = aa[2] - aa[0]
        wb = bb[2] - bb[0]
        if abs(wa - wb) / max(wa, wb) > 0.08 or abs(aa[0] - bb[0]) > 0.04:
            continue
        if aa[3] < a["doc_bottom"] - 0.08 or bb[1] > b["doc_top"] + 0.08:
            continue
        for src, neighbor in [(a, b), (b, a)]:
            recovered = expand_pair(src, neighbor)
            if recovered:
                pi, bi = src["page"], src["index"]
                pages[pi]["parsing_res_list"][bi]["block_content"] = recovered
                pages[pi]["parsing_res_list"][bi]["table_rebuild"] = True
                events.append(
                    {
                        "kind": "numeric_columns_recovered",
                        "page": pi + 1,
                        "block": bi,
                        "columns": neighbor["tab"]["ncols"],
                    }
                )
    return events
