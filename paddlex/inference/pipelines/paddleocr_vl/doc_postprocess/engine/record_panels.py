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

"""Recover logical sections of compact multi-column code dictionaries.

Only prediction text, row spans and page geometry are used. A weighted interval
DP identifies repeated record schemas and blank gutters. Sections then follow
column-major reading order; printed section labels interrupt continuation.
"""
import collections
import copy
import re

from .fast_html import _get_formatter
from .table_graph import cells, parse, safe_int

CODE = re.compile(r"[A-Za-z]{2,5}")
CJK = re.compile(r"[\u3400-\u9fff]")
IGNORE = {
    "header",
    "header_image",
    "footer",
    "footer_image",
    "number",
    "aside_text",
    "footnote",
}


def is_code(text):
    text = text.strip()
    return bool(
        CODE.fullmatch(text) and sum(c.isupper() for c in text) >= len(text) / 2
    )


def panel_intervals(tab):
    """Interval DP: explain all non-gutter columns by a repeated code schema."""
    width = tab["ncols"]
    if not 4 <= width <= 20 or tab["nrows"] < 1:
        return []
    scalar = collections.defaultdict(list)
    for row in tab["grid"]:
        for col, span, rowspan, text in row:
            if span == 1:
                scalar[col].append(text)
    blank = {
        c: bool(scalar[c])
        and sum(not s.strip() for s in scalar[c]) / len(scalar[c]) > 0.96
        for c in range(width)
    }
    candidates = {}
    for start in range(width - 3):
        aa, bb = scalar[start + 1], scalar[start + 3]
        if min(len(aa), len(bb)) < 1:
            continue
        rate = min(sum(map(is_code, aa)) / len(aa), sum(map(is_code, bb)) / len(bb))
        keys = [s for s in scalar[start] if s.strip()]
        if (
            rate >= 0.68
            and keys
            and sum(bool(CJK.search(s)) for s in keys) / len(keys) >= 0.65
        ):
            candidates[start] = rate
    dp = {0: (0.0, [])}
    for col in range(width):
        if col not in dp:
            continue
        score, path = dp[col]
        options = []
        if blank[col]:
            options.append((col + 1, score - 0.03, path))
        if col in candidates:
            options.append(
                (col + 4, score + 4 * candidates[col], path + [(col, col + 4)])
            )
        for end, value, p in options:
            if end not in dp or value > dp[end][0]:
                dp[end] = (value, p)
    return dp.get(width, (0.0, []))[1]


def _cell_html(cell, colspan):
    """str() of BeautifulSoup's copy of a fast_html cell after c['colspan']=colspan: the same
    serialization as fast_html.Node.__str__ (attributes in name order, minimal formatter).
    """
    if colspan is None:
        return str(cell)
    attrs = dict(cell.attrs)
    attrs["colspan"] = colspan
    f = _get_formatter()
    content = "".join(
        f.substitute(c) if isinstance(c, str) else str(c) for c in cell.children
    )
    return (
        "<"
        + cell.name
        + "".join(
            " " + k + "=" + f.quoted_attribute_value(f.attribute_value(str(v)))
            for k, v in sorted(attrs.items())
        )
        + ">"
        + content
        + "</"
        + cell.name
        + ">"
    )


def extract_panels(tab, intervals):
    """Same rows as _extract_panels_reference.  A copied fast_html cell is BeautifulSoup's parse of
    str(cell), and parsing a fast_html serialization gives back the same tree and serialization.  So
    for rows whose selected cells are all fast_html td/th nodes, the row markup is assembled from the
    cell serializations (with the clipped colspan) and the label/header tests read the cells' own
    text, without building BeautifulSoup copies.  Other rows use the per-cell copy path.
    """
    from bs4 import BeautifulSoup

    from .fast_html import Node

    out = []
    for start, end in intervals:
        rows = []
        for row, grid in zip(tab["trs"], tab["grid"]):
            picked = [
                (cell, str(end - col) if col + span > end else None)
                for cell, (col, span, rowspan, text) in zip(cells(row), grid)
                if start <= col < end
            ]
            if not picked:
                continue
            if all(
                type(cell) is Node and cell.name in ("td", "th") for cell, _ in picked
            ):
                nonempty = [
                    (cell, colspan)
                    for cell, colspan in picked
                    if cell.get_text(strip=True)
                ]
                label = (
                    nonempty[0][0].get_text(" ", strip=True)
                    if len(nonempty) == 1
                    else None
                )
                marker = (
                    label is not None
                    and len(label) <= 30
                    and safe_int(
                        nonempty[0][1]
                        if nonempty[0][1] is not None
                        else nonempty[0][0].get("colspan", 1)
                    )
                    >= 3
                )
                if len(nonempty) == 4:
                    texts = [cell.get_text() for cell, _ in nonempty]
                    if (
                        not is_code(texts[1])
                        and not is_code(texts[3])
                        and all(len(t) < 20 for t in texts)
                        and sum(bool(_HEADER_WORD.search(t)) for t in texts) >= 3
                    ):
                        continue
                rows.append(
                    ("label", label)
                    if marker
                    else (
                        "row",
                        "<tr>"
                        + "".join(_cell_html(cell, colspan) for cell, colspan in picked)
                        + "</tr>",
                    )
                )
                continue
            soup = BeautifulSoup("<tr></tr>", "html.parser")
            for cell, colspan in picked:
                c = copy.deepcopy(cell)
                if colspan is not None:
                    c["colspan"] = colspan
                soup.tr.append(c)
            if not cells(soup.tr):
                continue
            nonempty = [c for c in cells(soup.tr) if c.get_text(strip=True)]
            label = (
                nonempty[0].get_text(" ", strip=True) if len(nonempty) == 1 else None
            )
            marker = (
                label is not None
                and len(label) <= 30
                and safe_int(nonempty[0].get("colspan", 1)) >= 3
            )
            header = (
                len(nonempty) == 4
                and not is_code(nonempty[1].get_text())
                and not is_code(nonempty[3].get_text())
                and all(len(c.get_text()) < 20 for c in nonempty)
                and sum(
                    bool(re.search(r"版|code|字|拆分", c.get_text(), re.I))
                    for c in nonempty
                )
                >= 3
            )
            if header:
                continue
            rows.append(("label", label) if marker else ("row", str(soup.tr)))
        out.append(rows)
    if VERIFY and out != _extract_panels_reference(tab, intervals):
        raise AssertionError("record_panels: panel rows differ")
    return out


_HEADER_WORD = re.compile(r"版|code|字|拆分", re.I)
VERIFY = False


def _extract_panels_reference(tab, intervals):
    from bs4 import BeautifulSoup

    out = []
    for start, end in intervals:
        rows = []
        for row, grid in zip(tab["trs"], tab["grid"]):
            soup = BeautifulSoup("<tr></tr>", "html.parser")
            for cell, (col, span, rowspan, text) in zip(cells(row), grid):
                if col < start or col >= end:
                    continue
                c = copy.deepcopy(cell)
                if col + span > end:
                    c["colspan"] = str(end - col)
                soup.tr.append(c)
            if not cells(soup.tr):
                continue
            # Explicit all-column labels delimit sections. Preserve the label
            # as ordinary text instead of a spurious one-cell data row.
            nonempty = [c for c in cells(soup.tr) if c.get_text(strip=True)]
            label = (
                nonempty[0].get_text(" ", strip=True) if len(nonempty) == 1 else None
            )
            marker = (
                label is not None
                and len(label) <= 30
                and safe_int(nonempty[0].get("colspan", 1)) >= 3
            )
            header = (
                len(nonempty) == 4
                and not is_code(nonempty[1].get_text())
                and not is_code(nonempty[3].get_text())
                and all(len(c.get_text()) < 20 for c in nonempty)
                and sum(
                    bool(re.search(r"版|code|字|拆分", c.get_text(), re.I))
                    for c in nonempty
                )
                >= 3
            )
            if header:
                continue
            rows.append(("label", label) if marker else ("row", str(soup.tr)))
        out.append(rows)
    return out


def apply(pages, final):
    candidates = []
    for pi, p in enumerate(pages):
        for bi, b in enumerate(p["parsing_res_list"]):
            if (
                b["block_label"] != "table"
                or "<table" not in b["block_content"]
                or b.get("logical_table_handled")
            ):
                continue
            tab = parse(b["block_content"])
            intervals = panel_intervals(tab)
            if intervals:
                candidates.append((pi, bi, tab, intervals))
    # A document-level schema prevents an isolated incidental code table from
    # activating the dictionary reconstruction algorithm.
    if len(candidates) < 8 or sum(t["nrows"] for _, _, t, _ in candidates) < 80:
        return []
    units = []
    logs = []
    selected = {(pi, bi) for pi, bi, _, _ in candidates}
    for pi, bi, tab, intervals in candidates:
        b = pages[pi]["parsing_res_list"][bi]
        box = b["block_bbox"]
        width = box[2] - box[0]
        for panel, (interval, rows) in enumerate(
            zip(intervals, extract_panels(tab, intervals))
        ):
            x0 = box[0] + width * panel / len(intervals)
            x1 = box[0] + width * (panel + 1) / len(intervals)
            units.append(
                {
                    "page": pi,
                    "index": bi,
                    "panel": panel,
                    "box": [x0, box[1], x1, box[3]],
                    "rows": rows,
                }
            )
        b["logical_table_handled"] = True
        final[pi]["parsing_res_list"][bi]["block_content"] = ""
    # A narrow table fragment belongs to its physical column. Sort by column
    # before height, avoiding row-major interleaving of independent panels.
    ordered = []
    for pi in sorted({u["page"] for u in units}):
        here = sorted([u for u in units if u["page"] == pi], key=lambda u: u["box"][0])
        columns = []
        for u in here:
            col = next(
                (
                    c
                    for c in columns
                    if abs(c[0]["box"][0] - u["box"][0]) < pages[pi]["width"] * 0.07
                ),
                None,
            )
            if col is None:
                columns.append([u])
            else:
                col.append(u)
        for column in columns:
            ordered.extend(sorted(column, key=lambda u: u["box"][1]))
    groups = []
    active = None
    prev = None
    for u in ordered:
        pi = u["page"]
        box = u["box"]
        p = pages[pi]
        same_column = (
            prev is not None
            and pi == prev["page"]
            and abs(box[0] - prev["box"][0]) < p["width"] * 0.07
        )
        detached = same_column and box[1] - prev["box"][3] > 0.006 * p["height"]
        external = []
        for bi, b in enumerate(p["parsing_res_list"]):
            bb = b["block_bbox"]
            s = b["block_content"].strip()
            if (pi, bi) in selected or b["block_label"] in IGNORE or not s:
                continue
            if (
                box[0] - 0.02 * p["width"]
                <= (bb[0] + bb[2]) / 2
                <= box[2] + 0.02 * p["width"]
                and -0.015 * p["height"] <= box[1] - bb[3] < 0.035 * p["height"]
            ):
                external.append(s)
        discontinuity = prev is None or pi - prev["page"] > 1
        if prev is not None and not same_column:
            source = pages[prev["page"]]
            oldbox = prev["box"]
            for b in source["parsing_res_list"]:
                bb = b["block_bbox"]
                # An unreadable small graphic at a section-label position is
                # evidence to stop, never a source of invented OCR text.
                if (
                    b["block_label"] == "image"
                    and not b["block_content"].strip()
                    and oldbox[0] < (bb[0] + bb[2]) / 2 < oldbox[2]
                    and 0 <= bb[1] - oldbox[3] < 0.025 * source["height"]
                    and bb[3] - bb[1] < 0.035 * source["height"]
                    and bb[2] - bb[0] < 0.15 * source["width"]
                ):
                    discontinuity = True
        if prev is not None and (prev["page"], prev["index"]) != (pi, u["index"]):
            # Never bridge over a table whose schema could not be recovered.
            for check_page in range(prev["page"], pi + 1):
                for check_index, b in enumerate(pages[check_page]["parsing_res_list"]):
                    key = (check_page, check_index)
                    if (
                        (prev["page"], prev["index"]) < key < (pi, u["index"])
                        and b["block_label"] == "table"
                        and key not in selected
                    ):
                        discontinuity = True
        if detached or external or discontinuity:
            active = None
        for kind, value in u["rows"]:
            if kind == "label":
                active = None
                # Labels are already emitted by external caption blocks when
                # detected there; internal labels have their own text output.
                groups.append(
                    {
                        "anchor": (pi, u["index"]),
                        "label": value,
                        "rows": [],
                        "members": [],
                    }
                )
                continue
            if active is None:
                active = {"anchor": (pi, u["index"]), "rows": [], "members": []}
                groups.append(active)
            active["rows"].append(value)
            member = [pi + 1, u["index"], u["panel"]]
            if member not in active["members"]:
                active["members"].append(member)
        prev = u
    output = collections.defaultdict(list)
    for g in groups:
        if g.get("label"):
            import html

            output[g["anchor"]].append("<p>" + html.escape(g["label"]) + "</p>")
        if g["rows"]:
            output[g["anchor"]].append("<table>" + "".join(g["rows"]) + "</table>")
            logs.append(
                {
                    "kind": "record_panel_section",
                    "fragments": g["members"],
                    "rows": len(g["rows"]),
                }
            )
    for (pi, bi), parts in output.items():
        final[pi]["parsing_res_list"][bi]["block_content"] = "\n\n".join(parts)
    return logs
