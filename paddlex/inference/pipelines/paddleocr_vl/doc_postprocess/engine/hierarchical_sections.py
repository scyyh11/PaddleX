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

"""Infer parent table sections from explicit restarted child-unit numbering."""
import collections
import html
import re

from .table_graph import cells, parse, safe_int

UNIT = re.compile(r"(?:第?\s*(\d{1,3})\s*单元|\bunit\s+(\d{1,3})\b)", re.I)
GROUP = re.compile(r"(?:第\s*\d+\s*组|\bgroup\s+\d+\b)", re.I)


def unit(row):
    cc = cells(row)
    if len(cc) != 1 or safe_int(cc[0].get("colspan", 1)) < 3:
        return None
    s = cc[0].get_text(" ", strip=True)
    m = UNIT.search(s)
    if not m or not GROUP.search(s):
        return None
    return int(m[1] or m[2])


def apply(pages, final):
    all_tables = []
    counts = collections.Counter()
    resets = 0
    for pi, p in enumerate(pages):
        for bi, b in enumerate(p["parsing_res_list"]):
            if (
                b["block_label"] != "table"
                or "<table" not in b["block_content"]
                or b.get("logical_table_handled")
            ):
                continue
            t = parse(b["block_content"])
            numbers = [unit(row) for row in t["trs"]]
            hits = [n for n in numbers if n is not None]
            if hits:
                counts[t["ncols"]] += len(hits)
                resets += hits.count(1)
            all_tables.append(
                {"page": pi, "index": bi, "tab": t, "units": numbers, "hits": hits}
            )
    if resets < 4 or sum(counts.values()) < 20:
        return []
    width = counts.most_common(1)[0][0]
    candidates = [
        i
        for i, t in enumerate(all_tables)
        if t["hits"] and abs(t["tab"]["ncols"] - width) <= 2
    ]
    if not candidates:
        return []
    selected = []
    for i in range(candidates[0], candidates[-1] + 1):
        t = all_tables[i]
        if abs(t["tab"]["ncols"] - width) <= 2 and (
            t["hits"] or i - 1 in candidates and i + 1 in candidates
        ):
            selected.append(t)
    groups = []
    current = None
    previous = None
    output = collections.defaultdict(list)
    for t in selected:
        pi, bi = t["page"], t["index"]
        anchor = (pi, bi)
        rows = t["tab"]["trs"]
        numbers = t["units"]
        if previous is not None and pi - previous["page"] > 1:
            current = None
        pages[pi]["parsing_res_list"][bi]["logical_table_handled"] = True
        final[pi]["parsing_res_list"][bi]["block_content"] = ""
        for ri, row in enumerate(rows):
            cc = cells(row)
            parent_label = (
                len(cc) == 1
                and safe_int(cc[0].get("colspan", 1)) >= 3
                and numbers[ri] is None
                and ri + 1 < len(rows)
                and numbers[ri + 1] == 1
            )
            if parent_label:
                current = None
                output[anchor].append(
                    "<p>" + html.escape(cc[0].get_text(" ", strip=True)) + "</p>"
                )
                continue
            if numbers[ri] == 1:
                current = None
            if current is None:
                current = {"anchor": anchor, "rows": [], "members": []}
                groups.append(current)
                output[anchor].append(current)
            current["rows"].append(str(row))
            if [pi + 1, bi] not in current["members"]:
                current["members"].append([pi + 1, bi])
        previous = t
    for (pi, bi), parts in output.items():
        final[pi]["parsing_res_list"][bi]["block_content"] = "\n\n".join(
            p if isinstance(p, str) else "<table>" + "".join(p["rows"]) + "</table>"
            for p in parts
        )
    return [
        {
            "kind": "restarted_child_unit_section",
            "fragments": g["members"],
            "rows": len(g["rows"]),
        }
        for g in groups
    ]
