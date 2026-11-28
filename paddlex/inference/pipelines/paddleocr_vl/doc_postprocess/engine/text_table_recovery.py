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

"""Recover strongly structured text blocks using observed rows and page context."""
import collections
import html
import re

from .table_graph import SKIP, norm, parse

KEY = re.compile(r"^[A-Z][A-Za-z0-9&/().-]{1,17}$")
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
UNIT = re.compile(r"\b(days|hours|minutes|months|years)\s*$", re.I)


def numbered_measurement_rows(text):
    """Printed numbered records with date ranges, units and summary rows."""
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    rows = []
    records = 0
    summaries = 0
    unit = None
    i = 0
    while i < len(lines):
        s = lines[i]
        m = UNIT.search(s)
        if (
            m
            and (
                i + 1 < len(lines)
                and re.match(r"^[A-Za-z][A-Za-z ]{0,18}\s+\d+\b", lines[i + 1])
            )
            and not DATE.search(s)
            and not re.match(r"^(?:Mean|Total)\b", s, re.I)
        ):
            label = s[: m.start()].strip()
            if not label and rows and len(rows[-1]) == 1:
                label = rows.pop()[0]
            if not label:
                return None
            if unit and unit != m[1].lower():
                return None
            unit = m[1].lower()
            rows.append([label, m[1]])
            i += 1
            continue
        dates = list(DATE.finditer(s))
        if len(dates) == 2 and re.match(r"^[A-Za-z][A-Za-z ]{0,18}\s+\d+\b", s):
            # The last standalone number following the dates is the observed
            # measurement; a parenthesized note belongs to that value.
            suffix = re.search(
                r"\s+:?\s*(\d[\d,]*(?:\.\d+)?(?:\s+\([^\d]*\))?)\s*$",
                s[dates[-1].end() :],
            )
            if not suffix:
                return None
            at = dates[-1].end() + suffix.start()
            rows.append([s[:at].rstrip(" :"), suffix[1]])
            records += 1
        elif re.match(r"^Total\s+[\d,]+\s*$", s, re.I):
            rows.append(["", s])
            summaries += 1
        elif re.match(r"^Mean\s*=", s, re.I):
            rows.append([s, ""])
            summaries += 1
        elif i + 1 < len(lines) and UNIT.fullmatch(lines[i + 1]):
            rows.append([s])
        else:
            return None
        i += 1
    return (
        (rows, unit)
        if records >= 5 and summaries >= 1 and unit and all(len(r) == 2 for r in rows)
        else None
    )


def circled_continuation_rows(text, previous):
    if previous["ncols"] != 1:
        return None
    old = [
        CIRCLED.index(s[0]) + 1
        for row in previous["rows"]
        for s in row
        if s and s[0] in CIRCLED
    ]
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    nums = [CIRCLED.index(s[0]) + 1 for s in lines if s[0] in CIRCLED]
    if (
        not old
        or len(nums) < 2
        or not lines
        or lines[0][0] not in CIRCLED
        or nums != list(range(old[-1] + 1, old[-1] + 1 + len(nums)))
    ):
        return None
    rows = []
    prose = []
    for s in lines:
        if s[0] in CIRCLED:
            if prose:
                rows.append([" ".join(prose)])
                prose = []
            rows.append([s])
        else:
            prose.append(s)
    if prose:
        rows.append([" ".join(prose)])
    return rows


def acronym_key(s):
    return bool(
        KEY.fullmatch(s.strip())
        and sum(c.isupper() for c in s) >= 2
        and sum(c.isupper() for c in s) >= sum(c.islower() for c in s) * 2
    )


def acronym_rows(text):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    rows = []
    unparsed = 0
    i = 0
    while i < len(lines):
        parts = lines[i].split(None, 1)
        if len(parts) == 2 and acronym_key(parts[0]) and len(parts[1]) > 3:
            rows.append(parts)
            i += 1
        elif (
            acronym_key(lines[i])
            and i + 1 < len(lines)
            and not acronym_key(lines[i + 1])
            and len(lines[i + 1]) > 4
        ):
            rows.append([lines[i], lines[i + 1]])
            i += 2
        else:
            unparsed += 1
            i += 1
    if len(rows) < 8 or unparsed > max(1, len(lines) * 0.08):
        return None
    # A glossary is approximately alphabetic, unlike prose containing acronyms.
    keys = [norm(r[0])[:2] for r in rows]
    if sum(a <= b for a, b in zip(keys, keys[1:])) < 0.75 * (len(keys) - 1):
        return None
    return rows


def scale_rows(text, header):
    labels = " ".join(header).lower()
    if not ("disagree" in labels and "agree" in labels) or not 4 <= len(header) <= 6:
        return None
    count = len(header) - 1
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    wanted = " ".join(str(i) for i in range(1, count + 1))
    rows = []
    question = []
    for line in lines:
        if re.fullmatch(r"\d+[.)]\s+.+", line):
            if question:
                return None
            question = [line]
        elif re.sub(r"\s+", " ", line) == wanted and question:
            rows.append([" ".join(question)] + wanted.split())
            question = []
        elif question:
            question.append(line)
    return [header] + rows if len(rows) >= 5 and not question else None


def html_rows(rows):
    return (
        "<table>"
        + "".join(
            "<tr>" + "".join("<td>" + html.escape(s) + "</td>" for s in row) + "</tr>"
            for row in rows
        )
        + "</table>"
    )


def apply(pages, final):
    tables = []
    candidates = []
    for pi, p in enumerate(pages):
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] == "table" and "<table" in b["block_content"]:
                tab = parse(b["block_content"])
                tables.append((pi, bi, tab))
            elif b["block_label"] in {"text", "content"}:
                rows = acronym_rows(b["block_content"])
                if rows:
                    candidates.append((pi, bi, rows, "acronym_table_recovered"))
    # Require neighboring document evidence, so an isolated acronym list stays
    # text. All text and labels used here are model predictions.
    accepted = []
    schemas = {}
    for pi, bi, rows, kind in candidates:
        neighbors = [r for p, j, r, k in candidates if 0 < abs(p - pi) <= 1]
        glossary = any(
            0 < abs(p - pi) <= 1
            and len(t["rows"]) >= 5
            and t["ncols"] == 2
            and sum(acronym_key(r[0]) for r in t["rows"] if len(r) == 2)
            >= 0.8 * len(t["rows"])
            for p, j, t in tables
        )
        if neighbors or glossary:
            accepted.append((pi, bi, rows, kind))
    # Tables on the neighbouring pages, in the order of `tables` (page order: page pi-1 before
    # pi+1); the same tables are tried in the same order as scanning the whole list.
    bypage = collections.defaultdict(list)
    for tp, tb, t in tables:
        bypage[tp].append((tp, tb, t))
    for pi, p in enumerate(pages):
        near = bypage.get(pi - 1, []) + bypage.get(pi + 1, [])
        if not near:
            continue
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] not in {"text", "content"}:
                continue
            for tp, tb, t in near:
                if abs(tp - pi) != 1 or not t["rows"]:
                    continue
                rows = scale_rows(b["block_content"], t["rows"][0])
                if rows:
                    accepted.append((pi, bi, rows, "rating_scale_table_recovered"))
                    break
    measurements = []
    for pi, p in enumerate(pages):
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] not in {"text", "content"}:
                continue
            found = numbered_measurement_rows(b["block_content"])
            if found:
                measurements.append((pi, bi, found))
    for pi, bi, (rows, unit) in measurements:
        if not any(abs(pi - pj) == 1 and unit == u for pj, bj, (rr, u) in measurements):
            continue
        accepted.append((pi, bi, rows, "numbered_measurements_recovered"))
        schemas[pi, bi] = "dated_records:" + unit
    for pi, p in enumerate(pages):
        if pi == 0:
            continue
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] != "algorithm" or any(
                q["block_label"] not in SKIP for q in p["parsing_res_list"][:bi]
            ):
                continue
            prev = pages[pi - 1]
            body = [
                (j, q)
                for j, q in enumerate(prev["parsing_res_list"])
                if q["block_label"] not in SKIP
            ]
            if not body or body[-1][1]["block_label"] != "table":
                continue
            pj, q = body[-1]
            box = b["block_bbox"]
            qb = q["block_bbox"]
            w = p["width"]
            pw = prev["width"]
            if (
                abs((box[2] - box[0]) / w - (qb[2] - qb[0]) / pw) > 0.08
                or box[1] / p["height"] > 0.2
                or qb[3] / prev["height"] < 0.8
            ):
                continue
            rows = circled_continuation_rows(
                b["block_content"], parse(q["block_content"])
            )
            if rows:
                accepted.append((pi, bi, rows, "numbered_single_column_recovered"))
                schemas[pi, bi] = "circled_single_column"
                schemas[pi - 1, pj] = "circled_single_column"
    if not accepted:
        return []
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

    touched = {root(allblocks[pi, bi]) for pi, bi, rows, kind in accepted}
    positions = [pos for pos, b in allblocks.items() if root(b) in touched]
    for pi, bi in positions:
        b = allblocks[pi, bi]
        old = pages[pi]["parsing_res_list"][bi]
        if old["block_label"] in {"text", "content", "algorithm"}:
            b["block_content"] = old["block_content"]
            b["block_label"] = old["block_label"]
            b["global_group_id"] = b.get("global_block_id")
    logs = []
    for pi, bi, rows, kind in accepted:
        for b in [pages[pi]["parsing_res_list"][bi], final[pi]["parsing_res_list"][bi]]:
            b["block_label"] = "table"
            b["block_content"] = html_rows(rows)
        logs.append({"kind": kind, "page": pi + 1, "block": bi, "rows": len(rows)})
    for (pi, bi), schema in schemas.items():
        pages[pi]["parsing_res_list"][bi]["recovered_table_schema"] = schema
    return logs
