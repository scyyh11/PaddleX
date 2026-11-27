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

"""Recover code/title/credit tables from predicted text, including two columns.

Line items are assigned to columns by a small beam dynamic program. Its state
tracks the year sequence in both columns; no reference text is supplied.
"""
import copy
import html
import re

CODE = re.compile(r"\b([A-Z]{2,4}\s?\d{3,4}[A-Za-z]?)\b")
YEAR = re.compile(r"\bYear\s*([1-6])\s*Semester\s*([1-3])\b", re.I)
HEADER = re.compile(r"Code\s+Course\s+Title\s+Units?", re.I)
PROGRAM = re.compile(
    r"\b(?:B\.?Sc|B\.?A|B\.?Eng|M\.?Sc|Diploma|Bachelor|Master)\s+(?:in|of)\b", re.I
)
MARKER = re.compile(
    r"\b[A-Z]{2,4}\s?\d{3,4}[A-Za-z]?\b|\bYear\s*[1-6]\s*Semester\s*[1-3]\b|Code\s+Course\s+Title\s+Units?|\bElectives?\s*(?:Courses?|[1-9])\b|\b(?:B\.?Sc|B\.?A|M\.?Sc)\s+in\b",
    re.I,
)


def item(text):
    text = re.sub(r"\s+", " ", text).strip()
    m = CODE.match(text)
    if m:
        code = m[1]
        rest = text[m.end() :].strip()
        units = re.search(r"\s+(\d{1,2}(?:\.\d)?)$", rest)
        title = rest[: units.start()].strip() if units else rest
        return {
            "kind": "row",
            "code": code,
            "title": title,
            "units": units[1] if units else "",
            "year": int(re.search(r"\d", code)[0]),
            "text": text,
        }
    m = YEAR.match(text)
    if m:
        return {"kind": "section", "year": int(m[1]), "text": m[0]}
    if HEADER.fullmatch(text):
        return {"kind": "header", "year": 0, "text": text}
    if PROGRAM.match(text):
        return {"kind": "program", "year": 0, "text": text}
    if re.match(r"^Electives?\b", text, re.I):
        return {"kind": "section", "year": 0, "text": text}
    return {"kind": "continuation", "year": 0, "text": text}


def split_line(line):
    # Do not accept case-insensitive prose as course codes.
    starts = sorted(
        set(m.start() for m in CODE.finditer(line))
        | set(m.start() for m in YEAR.finditer(line))
        | set(m.start() for m in HEADER.finditer(line))
        | set(m.start() for m in PROGRAM.finditer(line))
        | set(
            m.start()
            for m in re.finditer(r"\bElectives?\s+(?:[1-9]|Courses?|in\b)", line, re.I)
        )
    )
    if not starts:
        return [item(line)]
    if starts[0] > 0:
        starts = [0] + starts
    return [
        item(line[i:j])
        for i, j in zip(starts, starts[1:] + [len(line)])
        if line[i:j].strip()
    ]


def transition(last, it):
    year = it["year"]
    if it["kind"] == "program":
        return 0, 0
    if not year:
        return 0, last
    if not last:
        return 0, year
    if year < last:
        return 6 * (last - year), year
    return max(0, year - last - 1) * 1.5, year


def lanes(text):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    parts = [split_line(line) for line in lines]
    dual = sum(sum(it["kind"] == "row" for it in row) > 1 for row in parts) >= 3 or any(
        len(HEADER.findall(s)) >= 2 for s in lines
    )
    if not dual:
        return [[it for row in parts for it in row]]
    # Each state stores the last academic year on each side and prior assignment.
    beam = [(0.0, 0, 0, 0, [], [])]
    for pieces in parts:
        options = []
        if len(pieces) >= 2:
            # At most one vertical cut, retaining within-line reading order.
            options = [(pieces[:cut], pieces[cut:]) for cut in range(1, len(pieces))]
        else:
            options = [(pieces, []), ([], pieces)]
        new = []
        for cost, yl, yr, prev, left, right in beam:
            for ll, rr in options:
                c = cost
                nl, nr = yl, yr
                for it in ll:
                    delta, nl = transition(nl, it)
                    c += delta
                for it in rr:
                    delta, nr = transition(nr, it)
                    c += delta
                # A later column normally starts at an equal/later academic year.
                if nl and nr and nl > nr:
                    c += 2 * (nl - nr)
                if ll and rr:
                    c += 0.05 * abs(len(ll) - len(rr))
                    side = 2
                else:
                    side = 0 if ll else 1
                    if (
                        pieces[0]["kind"] == "continuation"
                        and prev < 2
                        and side != prev
                    ):
                        c += 0.3
                new.append((c, nl, nr, side, left + ll, right + rr))
        best = {}
        for state in sorted(new, key=lambda s: s[0]):
            best.setdefault(state[1:4], state)
        beam = sorted(best.values(), key=lambda s: s[0])[:24]
    best = min(beam, key=lambda s: s[0])
    return [best[4], best[5]]


def convert(text):
    chunks = []
    current = []
    lastyear = 0
    for lane in lanes(text):
        for it in lane:
            kind = it["kind"]
            if kind == "program":
                if current:
                    chunks.append((current, True))
                    current = []
                lastyear = 0
                continue
            if kind == "section" and it["year"] == 1 and lastyear > 1:
                if current:
                    chunks.append((current, True))
                    current = []
            if it["year"]:
                lastyear = it["year"]
            if kind == "continuation":
                s = it["text"]
                if (
                    current
                    and current[-1]["kind"] == "row"
                    and s
                    and not re.fullmatch(r"[\d\s.]+", s)
                ):
                    if current[-1]["units"]:
                        s = re.sub(
                            r"\s+" + re.escape(current[-1]["units"]) + r"$", "", s
                        )
                    current[-1]["title"] += " " + s
                continue
            current.append(copy.deepcopy(it))
    if current:
        chunks.append((current, False))
    out = []
    for items, closed in chunks:
        rows = []
        for it in items:
            if it["kind"] == "row":
                rows.append(
                    "<tr>"
                    + "".join(
                        "<td>" + html.escape(it[k]) + "</td>"
                        for k in ["code", "title", "units"]
                    )
                    + "</tr>"
                )
            elif it["kind"] == "header":
                rows.append("<tr><td>Code</td><td>Course Title</td><td>Units</td></tr>")
            else:
                rows.append(
                    '<tr><td colspan="3">' + html.escape(it["text"]) + "</td></tr>"
                )
        if sum(it["kind"] == "row" for it in items):
            out.append(
                {
                    "html": "<table>" + "".join(rows) + "</table>",
                    "items": items,
                    "closed": closed,
                }
            )
    return out


def apply(prepared, final):
    strong = []
    for pi, p in enumerate(prepared):
        for bi, b in enumerate(p["parsing_res_list"]):
            s = b["block_content"]
            if (
                b["block_label"] in {"content", "text"}
                and len(CODE.findall(s)) >= 4
                and (HEADER.search(s) or YEAR.search(s))
            ):
                strong.append((pi, bi))
    if len(strong) < 3:
        return []
    strongpages = {pi for pi, bi in strong}
    eligible = []
    for pi, p in enumerate(prepared):
        for bi, b in enumerate(p["parsing_res_list"]):
            s = b["block_content"]
            if (
                b["block_label"] in {"content", "text"}
                and CODE.search(s)
                and (pi, bi) in strong
            ):
                eligible.append((pi, bi))
            elif (
                b["block_label"] in {"content", "text"}
                and len(CODE.findall(s)) >= 1
                and (HEADER.search(s) or YEAR.search(s))
                and any(abs(pi - j) <= 1 for j in strongpages)
            ):
                eligible.append((pi, bi))
    # Undo text concatenation only for the groups containing recovered tables.
    allblocks = {
        (pi, bi): b
        for pi, p in enumerate(final)
        for bi, b in enumerate(p["parsing_res_list"])
    }
    byid = {b.get("global_block_id"): b for b in allblocks.values()}

    def group(b):
        seen = set()
        gid = b.get("global_group_id")
        while (
            gid in byid and gid not in seen and byid[gid].get("global_group_id") != gid
        ):
            seen.add(gid)
            gid = byid[gid].get("global_group_id")
        return gid

    groups = {group(allblocks[pos]) for pos in eligible}
    for (pi, bi), b in allblocks.items():
        old = prepared[pi]["parsing_res_list"][bi]
        if group(b) in groups and old["block_label"] in {"content", "text"}:
            b["block_content"] = old["block_content"]
            b["block_label"] = old["block_label"]
            b["global_group_id"] = b.get("global_block_id")
    logs = []
    chains = []
    current = []
    previous = None
    lastyear = 0
    flat = [
        (pi, bi)
        for pi, p in enumerate(prepared)
        for bi in range(len(p["parsing_res_list"]))
    ]
    order = {pos: i for i, pos in enumerate(flat)}
    for pos in eligible:
        pi, bi = pos
        old = prepared[pi]["parsing_res_list"][bi]
        chunks = convert(old["block_content"])
        if not chunks:
            continue
        newprogram = False
        if previous is not None:
            between = flat[order[previous] + 1 : order[pos]]
            newprogram = pi - previous[0] > 1 or any(
                prepared[x]["parsing_res_list"][y]["block_label"]
                in {"paragraph_title", "doc_title"}
                and prepared[x]["parsing_res_list"][y]["block_content"].strip()
                for x, y in between
            )
        b = allblocks[pos]
        b["block_content"] = ""
        b["block_label"] = "table"
        for chunk in chunks:
            firstyear = next((it["year"] for it in chunk["items"] if it["year"]), 0)
            reset = firstyear == 1 and lastyear > 1
            if current and (newprogram or reset):
                chains.append(current)
                current = []
            current.append((pos, chunk["html"]))
            newprogram = False
            lastyear = next(
                (it["year"] for it in reversed(chunk["items"]) if it["year"]), lastyear
            )
            if chunk["closed"]:
                chains.append(current)
                current = []
                lastyear = 0
        previous = pos
        logs.append(
            {
                "kind": "course_table_recovered",
                "page": pi + 1,
                "block": bi,
                "chunks": len(chunks),
            }
        )
    if current:
        chains.append(current)
    for chain in chains:
        pos = chain[0][0]
        text = (
            "<table>"
            + "".join(re.sub(r"^<table>|</table>$", "", s) for _, s in chain)
            + "</table>"
        )
        allblocks[pos]["block_content"] += "\n\n" + text
    return logs
