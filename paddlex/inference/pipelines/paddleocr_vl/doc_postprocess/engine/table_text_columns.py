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

"""Recover a captioned two-column table's prose misclassified outside its box."""
from .table_graph import CAPTION, CONT, SKIP, cells, norm, parse
from .text_table_recovery import html_rows


def caption(page):
    for i, b in enumerate(page["parsing_res_list"]):
        if b["block_label"] == "table":
            break
        s = b["block_content"].strip()
        m = CAPTION.match(s)
        if m and len(s) < 220:
            return (norm(m[1]), bool(CONT.search(s)), i)
    return None


def reset_text_groups(pages, final, positions):
    allblocks = {
        (pi, bi): b
        for pi, p in enumerate(final)
        for bi, b in enumerate(p["parsing_res_list"])
    }
    byid = {b.get("global_block_id"): b for b in allblocks.values()}

    def root(b):
        g = b.get("global_group_id")
        seen = set()
        while g in byid and g not in seen and byid[g].get("global_group_id") != g:
            seen.add(g)
            g = byid[g].get("global_group_id")
        return g

    touched = {root(allblocks[pos]) for pos in positions}
    restore = [pos for pos, b in allblocks.items() if root(b) in touched]
    for pi, bi in restore:
        b = allblocks[pi, bi]
        old = pages[pi]["parsing_res_list"][bi]
        if old["block_label"] != "table":
            b["block_content"] = old["block_content"]
            b["block_label"] = old["block_label"]
            b["global_group_id"] = b.get("global_block_id")


def apply(pages, final):
    caps = {pi: caption(p) for pi, p in enumerate(pages)}
    known = {}
    for pi, p in enumerate(pages):
        if not caps[pi]:
            continue
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] != "table":
                continue
            t = parse(b["block_content"])
            if t["ncols"] == 2 and len(t["rows"]) >= 2 and len(t["rows"][0]) == 2:
                known.setdefault(caps[pi][0], []).append((pi, bi, t))
    logs = []
    for pi, p in enumerate(pages):
        cap = caps[pi]
        if not cap or not cap[1] or cap[0] not in known:
            continue
        neighbors = [x for x in known[cap[0]] if 0 < abs(x[0] - pi) <= 1]
        if not neighbors:
            continue
        blocks = p["parsing_res_list"]
        tables = [i for i, b in enumerate(blocks) if b["block_label"] == "table"]
        if len(tables) > 1:
            continue
        np, nb, nt = neighbors[0]
        neighbor = pages[np]["parsing_res_list"][nb]
        left, right = [
            neighbor["block_bbox"][j] / pages[np]["width"] * p["width"] for j in [0, 2]
        ]
        width = right - left
        if tables:
            bi = tables[0]
            tb = blocks[bi]
            tab = parse(tb["block_content"])
            if tab["ncols"] != 2 or not tab["rows"] or len(tab["rows"][0]) != 2:
                continue
            if [norm(s) for s in tab["rows"][0]] != [norm(s) for s in nt["rows"][0]]:
                continue
            left, right = tb["block_bbox"][0], tb["block_bbox"][2]
            width = right - left
            start = bi + 1
        else:
            start = cap[2] + 1
        extras = [
            (i, b)
            for i, b in enumerate(blocks[start:], start)
            if b["block_label"] not in SKIP and b["block_content"].strip()
        ]
        if len(extras) < 3 or sum(len(b["block_content"]) for i, b in extras) < 300:
            continue
        if any(
            b["block_label"] not in {"text", "content", "paragraph_title"}
            for i, b in extras
        ):
            continue
        if any(
            b["block_bbox"][0] < left - 0.05 * width
            or b["block_bbox"][2] > right + 0.08 * width
            for i, b in extras
        ):
            continue
        rightside = [
            (i, b) for i, b in extras if b["block_bbox"][0] >= left + 0.18 * width
        ]
        if len(rightside) < 0.8 * len(extras):
            continue
        if tables:
            if (
                len(rightside) != len(extras)
                or min(b["block_bbox"][1] for i, b in extras)
                < tb["block_bbox"][3] - 0.015 * p["height"]
            ):
                continue
            # Every stray block must begin inside the same explanatory column.
            if (
                max(b["block_bbox"][0] for i, b in extras)
                - min(b["block_bbox"][0] for i, b in extras)
                > 0.15 * width
            ):
                continue
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(tb["block_content"], "html.parser")
            last = soup.find_all("tr")[-1]
            cc = cells(last)
            if not cc:
                continue
            cc[-1].append("\n" + "\n".join(b["block_content"] for i, b in extras))
            newhtml = str(soup)
        else:
            # Sort by geometry, retaining original order for overlapping blocks.
            ordered = sorted(
                extras,
                key=lambda ib: (
                    round(ib[1]["block_bbox"][1] / max(1, p["height"]) * 200),
                    ib[1]["block_bbox"][0],
                    ib[0],
                ),
            )
            rows = []
            key = ""
            parts = []
            for i, b in ordered:
                if b["block_bbox"][0] < left + 0.18 * width:
                    if len(b["block_content"]) > 70:
                        break
                    if parts:
                        rows.append([key, "\n".join(parts)])
                        parts = []
                    key = b["block_content"]
                else:
                    parts.append(b["block_content"])
            else:
                if parts:
                    rows.append([key, "\n".join(parts)])
                newhtml = html_rows([nt["rows"][0]] + rows)
                bi = extras[0][0]
                tb = blocks[bi]
                reset_text_groups(pages, final, [(pi, i) for i, b in extras])
                tb["block_bbox"] = [
                    left,
                    min(b["block_bbox"][1] for i, b in extras),
                    right,
                    max(b["block_bbox"][3] for i, b in extras),
                ]
                final[pi]["parsing_res_list"][bi]["global_group_id"] = final[pi][
                    "parsing_res_list"
                ][bi].get("global_block_id")
                final[pi]["parsing_res_list"][bi]["block_label"] = "table"
                final[pi]["parsing_res_list"][bi]["block_content"] = newhtml
                logs.append(
                    {
                        "kind": "captioned_prose_table_recovered",
                        "page": pi + 1,
                        "rows": len(rows),
                        "blocks": len(extras),
                    }
                )
                # The replacement table occupies the first original block.
                for i, b in extras:
                    if i == bi:
                        continue
                    b["block_content"] = ""
                    final[pi]["parsing_res_list"][i]["block_content"] = ""
                tb["block_content"] = newhtml
                tb["block_label"] = "table"
                tb["table_rebuild"] = True
                continue
            continue
        reset_text_groups(pages, final, [(pi, i) for i, b in extras])
        for i, b in extras:
            b["block_content"] = ""
            final[pi]["parsing_res_list"][i]["block_content"] = ""
        tb["block_content"] = newhtml
        tb["block_bbox"][3] = max(
            [tb["block_bbox"][3]] + [b["block_bbox"][3] for i, b in extras]
        )
        tb["table_rebuild"] = True
        logs.append(
            {
                "kind": "captioned_table_tail_recovered",
                "page": pi + 1,
                "blocks": len(extras),
            }
        )
    return logs
