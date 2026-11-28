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

"""Recover long prose columns that layout parsing has wrapped in an HTML table."""
import collections
import copy
import re

from .runtime_context import readonly_soup as BeautifulSoup

CAPTION = re.compile(r"^\s*(?:table|tab\.|表)\s*[\d一二三四五六七八九十]", re.I)


def proposal(block, page):
    if block["block_label"] != "table":
        return None
    x, y, xx, yy = block["block_bbox"]
    w = page["width"]
    h = page["height"]
    if (xx - x) < w * 0.65 or (yy - y) < h * 0.42:
        return None
    soup = BeautifulSoup(block["block_content"], "html.parser")
    rows = soup.find_all("tr")
    if not 2 <= len(rows) <= 3:
        return None
    cells = [r.find_all(["td", "th"], recursive=False) for r in rows]
    cols = len(cells[0])
    if not 2 <= cols <= 4 or any(len(r) != cols for r in cells):
        return None
    if any(
        int(c.get("colspan", 1)) > 1 or int(c.get("rowspan", 1)) > 1
        for row in cells
        for c in row
    ):
        return None
    texts = [
        [c.get_text(" ", strip=True).replace("\\n", "\n") for c in row] for row in cells
    ]
    headers = texts[0]
    bodies = ["\n".join(row[i] for row in texts[1:]) for i in range(cols)]
    if not all(
        3 <= len(s) <= 90 and len(s.split()) <= 12 and not re.search(r"[。;；!?]", s)
        for s in headers
    ):
        return None
    if sum(len(s) >= 250 for s in bodies) < 2 or sum(map(len, bodies)) < 900:
        return None
    joined = " ".join(bodies)
    if sum(c.isdigit() for c in joined) / max(len(joined), 1) > 0.12:
        return None
    # Narrative columns contain complete clauses or independently numbered instructions.
    if (
        sum(
            bool(
                re.search(
                    r"[.;。；]|\b(?:the|and|shall|must|should|will|with|for)\b", s, re.I
                )
            )
            for s in bodies
        )
        < 2
    ):
        return None
    for b in page["parsing_res_list"]:
        if b is block:
            continue
        if CAPTION.match(b["block_content"]) and 0 <= y - b["block_bbox"][3] < h * 0.13:
            return None
    return headers, bodies


def template_proposal(block, page, vocabulary):
    if block["block_label"] != "table":
        return None
    x, y, xx, yy = block["block_bbox"]
    w, h = page["width"], page["height"]
    if xx - x < w * 0.55 or yy - y < h * 0.38:
        return None
    if any(
        CAPTION.match(b["block_content"]) and 0 <= y - b["block_bbox"][3] < h * 0.13
        for b in page["parsing_res_list"]
        if b is not block
    ):
        return None
    soup = BeautifulSoup(block["block_content"], "html.parser")
    rows = soup.find_all("tr")
    occupied = {}
    columns = collections.defaultdict(list)
    for ri, row in enumerate(rows):
        col = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (ri, col) in occupied:
                col += 1
            if int(cell.get("colspan", 1)) != 1:
                return None
            text = cell.get_text(" ", strip=True).replace("\\n", "\n")
            columns[col].append(text)
            for dy in range(int(cell.get("rowspan", 1))):
                occupied[(ri + dy, col)] = True
            col += 1
    if not 2 <= len(columns) <= 4:
        return None
    headers = []
    bodies = []
    for col, parts in sorted(columns.items()):
        text = "\n".join(p for p in parts if p).strip()
        if not text:
            headers.append("")
            bodies.append("")
            continue
        match = next(
            (
                h
                for h in sorted(vocabulary, key=len, reverse=True)
                if text.startswith(h)
            ),
            None,
        )
        if not match:
            return None
        headers.append(match)
        bodies.append(text[len(match) :].strip())
    if sum(bool(s) for s in headers) < 2 or sum(map(len, bodies)) < 400:
        return None
    if sum(c.isdigit() for c in "".join(bodies)) / max(sum(map(len, bodies)), 1) > 0.15:
        return None
    return headers, bodies


def apply(pages, log):
    votes = collections.Counter()
    for p in pages:
        for b in p["parsing_res_list"]:
            q = proposal(b, p)
            if q:
                votes[tuple(q[0])] += 1
    vocabulary = {h for hs, count in votes.items() if count >= 3 for h in hs}
    if not vocabulary:
        return
    for pi, page in enumerate(pages):
        blocks = []
        for bi, b in enumerate(page["parsing_res_list"]):
            q = template_proposal(b, page, vocabulary)
            if q is None:
                blocks.append(b)
                continue
            headers, bodies = q
            x, y, xx, yy = b["block_bbox"]
            cw = (xx - x) / len(headers)
            lineheight = min(36, (yy - y) * 0.04)
            group = f"{pi}:{bi}"
            for col, (head, body) in enumerate(zip(headers, bodies)):
                if not head and not body:
                    continue
                parts = [(head, "header")]
                # A leading lettered subsection ends where the numbered instructions begin.
                m = re.match(r"^\s*([A-Z][.)]\s+.{3,160}?)(?=\s+1[.)]\s+)", body, re.S)
                if m and not re.search(r"[.!?。]", m[1][2:]):
                    parts.append((m[1].strip(), "subheader"))
                    body = body[m.end() :].strip()
                parts.append((body, "body"))
                cy = y
                for text, kind in parts:
                    item = copy.deepcopy(b)
                    item["block_label"] = (
                        "text" if kind == "body" else "paragraph_title"
                    )
                    item["block_content"] = text
                    bottom = yy if kind == "body" else cy + lineheight
                    item["block_bbox"] = [
                        round(z) for z in [x + col * cw, cy, x + (col + 1) * cw, bottom]
                    ]
                    item["_narrative_column"] = {
                        "group": group,
                        "kind": kind,
                        "column": col,
                    }
                    blocks.append(item)
                    cy = bottom + 2
            log.append(
                {
                    "kind": "narrative_columns_recovered",
                    "page": pi + 1,
                    "headers": headers,
                    "column_count": len(headers),
                    "source_bbox": b["block_bbox"],
                    "body_lengths": list(map(len, bodies)),
                }
            )
        page["parsing_res_list"] = blocks


def attach(results, original):
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            meta = original.get((pi, tuple(b.bbox)), {}).get("_narrative_column")
            if meta:
                b._narrative_column = meta


def finalize(results, log):
    previous = 1
    groups = {}
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            meta = getattr(b, "_narrative_column", None)
            if not meta:
                if b.label == "doc_title":
                    previous = 1
                elif b.label == "paragraph_title":
                    previous = getattr(b, "title_level", 1) + 1
                continue
            groups.setdefault(meta["group"], min(previous + 1, 5))
            depth = groups[meta["group"]] + int(meta["kind"] == "subheader")
            if meta["kind"] == "body":
                b.label = "text"
                continue
            b.label = "paragraph_title"
            b.title_level = depth - 1
            log.append(
                {
                    "kind": "narrative_column_heading",
                    "page": pi + 1,
                    "text": b.content,
                    "new": depth,
                }
            )
