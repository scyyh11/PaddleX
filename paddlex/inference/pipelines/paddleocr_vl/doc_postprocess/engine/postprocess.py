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

from .runtime_context import memoize, readonly_soup
from .table_merge_bridge import (
    cached_table_merger,
    merge_to_html,
    readonly_soup_factory,
)

ML_MODE = "uniform"
ML_DECODER = "parent"
"""Document postprocessing from predicted blocks only. No annotations or model answers."""
import collections
import copy
import hashlib
import html
import itertools
import json
import math
import pathlib
import re
import statistics
import unicodedata

IGNORE = {
    "header",
    "header_image",
    "footer",
    "footer_image",
    "number",
    "footnote",
    "aside_text",
}
SKIP = IGNORE | {"vision_footnote", "formula_number"}
TEXT = {"text", "abstract", "reference_content", "vertical_text", "content"}
TERMINAL = re.compile(r"[。！？.!?][”’\"\'）)\]]*\s*$")
LIST_START = re.compile(
    r"^\s*(?:[•●▪■◆※]|\(?\d+[.)、]\s+|[（(][一二三四五六七八九十\d]+[）)]|[一二三四五六七八九十]+[、．])"
)
CONTINUED = re.compile(r"(?:续表|續表|接上表|continued|cont[’\x27]?d)", re.I)
MAJOR = {
    "abstract",
    "summary",
    "introduction",
    "background",
    "relatedwork",
    "materials and methods",
    "material and methods",
    "methods",
    "methodology",
    "experimental procedures",
    "experimental section",
    "results",
    "discussion",
    "results and discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgments",
    "acknowledgements",
    "acknowledgement",
    "funding",
    "figure legends",
    "supplementary information",
    "supplemental information",
    "appendix",
    "contents",
    "table of contents",
    "摘要",
    "前言",
    "引言",
    "绪论",
    "目录",
    "目錄",
    "序言",
    "丛书序",
    "参考文献",
    "致谢",
    "附录",
}


@memoize
def compact(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).lower()


MAJOR = {compact(s) for s in MAJOR}


def log_change(log, kind, page, **kw):
    log.append(dict(kind=kind, page=page + 1, **kw))


def cells(row):
    return row.find_all(["td", "th"], recursive=False)


def table_columns(soup):
    occupied = {}
    maximum = 0
    for ri, row in enumerate(soup.find_all("tr")):
        ci = 0
        for c in cells(row):
            while occupied.get((ri, ci)):
                ci += 1
            cs = max(1, int(c.get("colspan", 1)))
            rs = max(1, int(c.get("rowspan", 1)))
            for dy in range(rs):
                for dx in range(cs):
                    occupied[(ri + dy, ci + dx)] = True
            ci += cs
            maximum = max(maximum, ci)
    return maximum


def boundary_blocks(page):
    return [
        b
        for b in page["parsing_res_list"]
        if b["block_label"] not in SKIP
        and b["block_content"].strip()
        and not CONTINUED.search(b["block_content"][:80])
    ]


def recover_prose_tables(pages, log):
    speaker = re.compile(
        r"\b(?:DR\.|MR\.|MS\.|MRS\.|CHAIRMAN|CHAIRWOMAN)\s+[A-Z][A-Z .\-]{1,35}:"
    )
    votes = sum(
        any(
            b["block_label"] == "table"
            and len(speaker.findall(b["block_content"])) >= 2
            for b in p["parsing_res_list"]
        )
        for p in pages
    )
    if votes < 3 or votes < len(pages) * 0.15:
        return
    for pi, p in enumerate(pages):
        blocks = []
        for b in p["parsing_res_list"]:
            if b["block_label"] != "table":
                blocks.append(b)
                continue
            soup = readonly_soup(b["block_content"], "html.parser")
            ss = [
                c.get_text(" ", strip=True).replace("\\n", "\n")
                for row in soup.find_all("tr")
                for c in cells(row)
            ]
            if (
                table_columns(soup) > 2
                or sum(map(len, ss)) < 150
                or not ss
                or sum(bool(re.fullmatch(r"[\d.,%\s]+", s)) for s in ss) > len(ss) * 0.2
            ):
                blocks.append(b)
                continue
            parts = ["\n".join(ss)]
            if len(ss) > 1 and re.match(r"^[a-z]", ss[0]):
                parts = [ss[0], "\n".join(ss[1:])]
            for part in parts:
                c = copy.deepcopy(b)
                c["block_label"] = "text"
                c["block_content"] = part
                blocks.append(c)
            log_change(
                log, "prose_table_recovered", pi, cells=len(ss), text_blocks=len(parts)
            )
        p["parsing_res_list"] = blocks


def recover_glossary_tables(pages, log):
    active = False
    key_re = re.compile(r"^[A-Z][A-Z0-9/&() .\-]{0,23}$")
    title_re = re.compile(
        r"(?:acronyms?|initialisms?|abbreviations?|缩略语|縮略語|英文缩写)", re.I
    )
    for pi, p in enumerate(pages):
        for b in p["parsing_res_list"]:
            s = b["block_content"].strip()
            if b["block_label"] in {"doc_title", "paragraph_title"}:
                active = bool(title_re.search(s))
            if not active or b["block_label"] not in {"text", "content"}:
                continue
            lines = [line.strip() for line in s.splitlines() if line.strip()]
            if len(lines) < 8:
                continue
            pairs = []
            i = 0
            failed = False
            while i < len(lines):
                line = lines[i]
                if (
                    key_re.fullmatch(line)
                    and i + 1 < len(lines)
                    and not key_re.fullmatch(lines[i + 1])
                ):
                    pairs.append((line, lines[i + 1]))
                    i += 2
                    continue
                m = re.match(r"^([A-Z][A-Z0-9/&()\-]{1,20})\s+(.+)$", line)
                if m and len(m[2]) >= 3:
                    pairs.append((m[1], m[2]))
                    i += 1
                    continue
                # A small number of long-form glossary keys are proper names.
                if (
                    len(line) <= 24
                    and i + 1 < len(lines)
                    and len(lines[i + 1]) > len(line) + 8
                ):
                    pairs.append((line, lines[i + 1]))
                    i += 2
                    continue
                failed = True
                break
            if failed or len(pairs) < 8:
                continue
            b["block_label"] = "table"
            b["block_content"] = (
                "<table>"
                + "".join(
                    "<tr><td>"
                    + html.escape(k)
                    + "</td><td>"
                    + html.escape(v)
                    + "</td></tr>"
                    for k, v in pairs
                )
                + "</table>"
            )
            log_change(log, "glossary_table_recovered", pi, rows=len(pairs))


def separate_questionnaire_tables(results, log):
    def question(row):
        cc = cells(row)
        if len(cc) != 2:
            return None
        k, s = (c.get_text(" ", strip=True) for c in cc)
        if not re.fullmatch(r"\d+(?:\.[\da-z]+)+", k, re.I):
            return None
        if len(s) < 20 or not (
            "?" in s or re.match(r"^(?:Please|Describe|Indicate|Specify)\b", s, re.I)
        ):
            return None
        return k

    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if b.label != "table" or not b.content:
                continue
            soup = readonly_soup(b.content, "html.parser")
            rows = soup.find_all("tr")
            if not rows or not question(rows[0]):
                continue
            starts = [i for i, r in enumerate(rows) if question(r)]
            if len(starts) < 2 or len({question(rows[i]) for i in starts}) < 2:
                continue
            chunks = []
            for start, end in zip(starts, starts[1:] + [len(rows)]):
                chunks.append(
                    "<table>" + "".join(str(r) for r in rows[start:end]) + "</table>"
                )
            b.content = "\n\n".join(chunks)
            log_change(log, "independent_questionnaire_tables", pi, tables=len(chunks))


def repair_table_columns(pages, log):
    # A continuation row with only its final cell populated can lose empty cells.
    # Use matching table widths and the neighboring table's own column count.
    for pi in range(1, len(pages)):
        prev, curr = boundary_blocks(pages[pi - 1]), boundary_blocks(pages[pi])
        if not prev or not curr:
            continue
        a, b = prev[-1], curr[0]
        if a["block_label"] != "table" or b["block_label"] != "table":
            continue
        aw = (a["block_bbox"][2] - a["block_bbox"][0]) / pages[pi - 1]["width"]
        bw = (b["block_bbox"][2] - b["block_bbox"][0]) / pages[pi]["width"]
        if not min(aw, bw) or abs(aw - bw) / min(aw, bw) > 0.15:
            continue
        sa, sb = readonly_soup(a["block_content"], "html.parser"), readonly_soup(
            b["block_content"], "html.parser"
        )
        na, nb = table_columns(sa), table_columns(sb)
        if not (2 <= na <= 20 and 1 <= na - nb <= 2):
            continue
        rows = sb.find_all("tr")
        if not rows or any(len(cells(r)) != nb for r in rows):
            continue
        if not all(
            all(not c.get_text(strip=True) for c in cells(r)[:-1])
            and len(cells(r)[-1].get_text(strip=True)) > 30
            for r in rows
        ):
            continue
        from bs4 import BeautifulSoup

        sb = BeautifulSoup(b["block_content"], "html.parser")
        rows = sb.find_all("tr")
        for row in rows:
            for _ in range(na - nb):
                row.insert(0, sb.new_tag("td"))
        b["block_content"] = str(sb)
        log_change(log, "restore_empty_table_columns", pi, before=nb, after=na)


def align_sparse_table_columns(pages, log):
    def grid(soup):
        occupied = {}
        last = []
        maximum = table_columns(soup)
        for ri, row in enumerate(soup.find_all("tr")):
            values = [""] * maximum
            ci = 0
            for c in cells(row):
                while (ri, ci) in occupied:
                    values[ci] = occupied[(ri, ci)]
                    ci += 1
                cs = int(c.get("colspan", 1))
                rs = int(c.get("rowspan", 1))
                s = c.get_text(strip=True)
                for dx in range(cs):
                    if ci + dx < maximum:
                        values[ci + dx] = s
                    for dy in range(1, rs):
                        occupied[(ri + dy, ci + dx)] = s
                ci += cs
            last = values
        return last

    def typ(s):
        if not s:
            return "empty"
        if s in {"是", "否", "Yes", "No", "Y", "N"}:
            return "bool"
        if re.fullmatch(r"[\d.,%年月日/\-\s]+", s):
            return "number"
        return "long" if len(s) > 35 else "short"

    def cost(a, b):
        ta, tb = typ(a), typ(b)
        if ta == "empty":
            return 0.05
        if tb == "empty":
            return 0.8
        if ta != tb:
            return 1.4
        return min(0.5, abs(math.log((len(a) + 1) / (len(b) + 1))) * 0.15)

    for pi in range(1, len(pages)):
        aa, bb = boundary_blocks(pages[pi - 1]), boundary_blocks(pages[pi])
        if not aa or not bb:
            continue
        a, b = aa[-1], bb[0]
        if a["block_label"] != "table" or b["block_label"] != "table":
            continue
        aw = (a["block_bbox"][2] - a["block_bbox"][0]) / pages[pi - 1]["width"]
        bw = (b["block_bbox"][2] - b["block_bbox"][0]) / pages[pi]["width"]
        if not min(aw, bw) or abs(aw - bw) / min(aw, bw) > 0.12:
            continue
        from bs4 import BeautifulSoup

        sa, sb = BeautifulSoup(a["block_content"], "html.parser"), BeautifulSoup(
            b["block_content"], "html.parser"
        )
        n, m = table_columns(sa), table_columns(sb)
        if not (4 <= n <= 14 and 1 <= n - m <= 3):
            continue
        rows = sb.find_all("tr")
        if not rows or any(
            len(cells(r)) != m
            or any(
                int(c.get("colspan", 1)) != 1 or int(c.get("rowspan", 1)) != 1
                for c in cells(r)
            )
            for r in rows
        ):
            continue
        context = grid(sa)
        first = [c.get_text(strip=True) for c in cells(rows[0])]
        if not any(typ(s) == "long" for s in first):
            continue
        choices = []
        for positions in itertools.combinations(range(n), m):
            value = sum(
                cost(first[j], context[i]) for j, i in enumerate(positions)
            ) + 0.1 * (n - m)
            choices.append((value, positions))
        choices.sort()
        best, positions = choices[0]
        if (
            best / max(sum(bool(s) for s in first), 1) > 0.45
            or len(choices) > 1
            and choices[1][0] - best < 0.12
        ):
            continue
        for row in rows:
            old = list(cells(row))
            new = [None] * n
            for c, i in zip(old, positions):
                new[i] = c.extract()
            row.clear()
            for c in new:
                row.append(c if c is not None else sb.new_tag("td"))
        b["block_content"] = str(sb)
        log_change(
            log,
            "align_sparse_columns",
            pi,
            before=m,
            after=n,
            positions=list(positions),
            cost=best,
        )


def extra_merge_tables(results, log):
    # Works backwards so a multi-page chain is retained at its first fragment.
    for pi in range(len(results) - 1, 0, -1):
        pa, pb = results[pi - 1], results[pi]
        # An emptied table is a consumed continuation, and remains a boundary.
        # Skipping it would incorrectly attach the next independent table.
        visible = lambda p: [
            b
            for b in p["parsing_res_list"]
            if b.label not in SKIP
            and (b.content.strip() or b.label == "table")
            and not CONTINUED.search(b.content[:80])
        ]
        aa, bb = visible(pa), visible(pb)
        if not aa or not bb:
            continue
        a, b = aa[-1], bb[0]
        if (
            a.label != "table"
            or b.label != "table"
            or not a.content.strip()
            or not b.content.strip()
        ):
            continue
        aw = (a.bbox[2] - a.bbox[0]) / pa["width"]
        bw = (b.bbox[2] - b.bbox[0]) / pb["width"]
        if not min(aw, bw) or abs(aw - bw) / min(aw, bw) > 0.15:
            continue
        sa, sb = readonly_soup(a.content, "html.parser"), readonly_soup(
            b.content, "html.parser"
        )
        if table_columns(sa) != table_columns(sb):
            continue
        ra, rb = sa.find_all("tr"), sb.find_all("tr")
        if not ra or not rb:
            continue
        nh = 0
        for x, y in zip(ra[:5], rb[:5]):
            if compact(x.get_text()) == compact(y.get_text()) and len(cells(x)) == len(
                cells(y)
            ):
                nh += 1
            else:
                break
        a.content = merge_to_html(sa, sb, nh)
        b.content = ""
        b.global_group_id = a.global_group_id
        log_change(log, "extra_table_join", pi, removed_repeated_header_rows=nh)


def continuation(a, b, mode):
    x, y = a.content.strip(), b.content.strip()
    if not x or not y or a.label not in TEXT or b.label not in TEXT:
        return False
    if "content" in {a.label, b.label} and a.label != b.label:
        return False
    if LIST_START.match(y):
        return False
    if len(x) < 15 or len(y) < 2:
        return False
    if x.endswith((":", "：")):
        return False
    if mode == "boundary":
        return not TERMINAL.search(x) or bool(re.match(r"^[a-z,;，；）)]", y))
    if mode == "aggressive":
        return not (
            TERMINAL.search(x)
            and re.match(r"^(?:[A-Z][a-z]*\s|第[一二三四五六七八九十\d]+)", y)
        )
    return not TERMINAL.search(x) and (
        bool(re.search(r"[a-zA-Z\u4e00-\u9fff,，;；\-]$", x)) or x.endswith("$")
    )


def join_text(x, y):
    x, y = x.rstrip(), y.lstrip()
    if re.search(r"[a-zA-Z]-$", x) and re.match(r"^[a-z]", y):
        return x[:-1] + y
    sep = (
        ""
        if re.search(r"[\u4e00-\u9fff]$", x)
        or re.match(r"^[\u4e00-\u9fff，。；：！？]", y)
        else " "
    )
    return x + sep + y


def merge_text(results, log, mode):
    for pi, p in enumerate(results):
        visible = [
            b
            for b in p["parsing_res_list"]
            if b.label not in SKIP and b.content.strip()
        ]
        for i in range(len(visible) - 2, -1, -1):
            a, b = visible[i], visible[i + 1]
            if (
                b.bbox[1] < a.bbox[3] - p["height"] * 0.15
                and b.bbox[0] > a.bbox[0] + p["width"] * 0.2
                and continuation(a, b, mode)
            ):
                a.content = join_text(a.content, b.content)
                b.content = ""
                b.global_group_id = a.global_group_id
                log_change(log, "column_text_join", pi)
    for pi in range(len(results) - 1, 0, -1):
        a = [
            b
            for b in results[pi - 1]["parsing_res_list"]
            if b.label not in SKIP and b.content.strip()
        ]
        b = [
            b
            for b in results[pi]["parsing_res_list"]
            if b.label not in SKIP and b.content.strip()
        ]
        if not a or not b:
            continue
        left, right = a[-1], b[0]
        if continuation(left, right, mode):
            log_change(
                log,
                "text_join",
                pi,
                left_tail=left.content[-80:],
                right_head=right.content[:80],
            )
            left.content = join_text(left.content, right.content)
            right.content = ""
            right.global_group_id = left.global_group_id


def estimate_height(block, width):
    x1, y1, x2, y2 = block.bbox
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    text = block.content.strip()
    units = sum(
        1 if "\u4e00" <= c <= "\u9fff" else 0.5 for c in text if not c.isspace()
    )
    explicit = text.count("\n") + 1
    lines = max(explicit, round(math.sqrt(max(1, h * units / w))))
    return (h / lines) / width * 1000


def heading_kind(text):
    s = unicodedata.normalize("NFKC", text).strip()
    s = re.sub(r"^#+\s*", "", s)
    if re.match(r"^第[一二三四五六七八九十百零〇\d]+[章节篇部卷]", s):
        return "chapter", 1
    if re.match(r"^(?:CHAPTER|SECTION|PART)\s+[\dIVX]+", s, re.I):
        return "chapter", 1
    if re.match(r"^[一二三四五六七八九十百]+[、.．]", s):
        return "cn", 1
    if re.match(r"^\([一二三四五六七八九十百]+\)", s):
        return "cn_paren", 1
    m = re.match(r"^(\d+(?:\.\d+)*)(?:[.、]\s*|\s+|(?=[\u4e00-\u9fff]))", s)
    if m and len(m[1].split(".")[0]) < 3:
        ns = m[1].split(".")
        while len(ns) > 1 and ns[-1] == "0":
            ns.pop()
        return "num", len(ns)
    if re.match(r"^\(\d+\)", s):
        return "num_paren", 1
    if re.match(r"^[IVX]+[.)]\s", s):
        return "roman", 1
    if re.match(r"^[A-Z][.)]\s", s):
        return "letter", 1
    if compact(s.rstrip(":：")) in MAJOR:
        return "major", 1
    letters = re.sub("[^A-Za-z]", "", s)
    if len(letters) >= 5 and letters.isupper() and len(s) < 130:
        return "caps", 1
    return "unknown", 0


def relevel(results, log, mode):
    entries = []
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if b.label == "paragraph_title":
                entries.append(
                    (pi, p, b, heading_kind(b.content), estimate_height(b, p["width"]))
                )
    kinds = collections.Counter(e[3][0] for e in entries)
    normalized_titles = {compact(e[2].content) for e in entries}
    academic = (
        bool(
            normalized_titles
            & {
                "methods",
                "methodology",
                "materialandmethods",
                "materialsandmethods",
                "experimentalprocedures",
            }
        )
        and len(normalized_titles & MAJOR) >= 4
    )
    has_chapter = kinds["chapter"] > 1
    has_cn = kinds["cn"] > 2
    has_nums = kinds["num"] > 2
    cbase = 2 + int(has_chapter)
    nbase = cbase + 1 if has_cn else 2 + int(has_chapter)
    heights = [e[4] for e in entries]
    centers = []
    for h in sorted(heights, reverse=True):
        if not centers or (centers[-1] - h) / centers[-1] > 0.16:
            centers.append(h)
    anchors = []
    for pi, p, b, (kind, depth), height in entries:
        level = None
        if kind == "chapter":
            level = 2
        elif kind == "cn":
            level = cbase
        elif kind == "cn_paren":
            level = cbase + 1
        elif kind == "num":
            level = nbase + depth - 1
        elif kind == "num_paren":
            level = nbase + 1
        elif kind == "roman":
            level = 2 + int(has_chapter)
        elif kind == "letter":
            level = 3
        elif kind in {"major", "caps"}:
            level = 2
        if level:
            anchors.append((height, level))
        b._post_level = level
    for pi, p, b, (kind, depth), height in entries:
        old = getattr(b, "title_level", 1) + 1
        level = b._post_level
        if mode == "hybrid":
            if kind == "unknown":
                if academic:
                    # Nearby measured font heights should not create new semantic levels.
                    peers = [e[4] for e in entries if e[3][0] == "unknown"]
                    med = statistics.median(peers) if peers else height
                    level = 4 if height < med * 0.77 else 3
                else:
                    level = old
            elif kind in {"major", "caps"} and not academic:
                level = old
            elif kind == "num_paren" and has_cn and kinds["cn_paren"] > 2:
                level = nbase + 1
        if level is None:
            if mode == "semantic" and (kinds["major"] + kinds["caps"] >= 3):
                level = 3
            elif anchors:
                close = sorted(
                    anchors,
                    key=lambda v: abs(math.log(max(height, 0.01) / max(v[0], 0.01))),
                )[:5]
                level = round(statistics.median(v[1] for v in close))
                if height < statistics.median(v[0] for v in close) * 0.84:
                    level += 1
            else:
                rank = (
                    min(range(len(centers)), key=lambda i: abs(centers[i] - height))
                    if centers
                    else 0
                )
                level = min(4, rank + 2)
        level = max(2, min(6, level))
        if re.fullmatch(r"[\s✓✔□☐☑√是否适用不适用]+", b.content) or re.fullmatch(
            r"\d{1,4}", b.content.strip()
        ):
            b.label = "text"
            log_change(log, "demote_non_heading", pi, text=b.content)
            continue
        b.title_level = level - 1
        if old != level:
            log_change(log, "heading_level", pi, text=b.content, old=old, new=level)


def toc_relevel(results, log):
    # Read the document's predicted table of contents, never annotation titles.
    toc = []
    for p in results[: min(40, max(10, len(results) // 5))]:
        for b in p["parsing_res_list"]:
            if b.label != "content":
                continue
            for line in b.content.splitlines():
                m = re.match(
                    r"^\s*(.*?)\s*(?:\.{2,}|…+|·{2,})\s*([\divxlc]+)\s*$", line, re.I
                )
                if m and len(m[1]) > 2:
                    toc.append((m[1].strip(), heading_kind(m[1])))

    def title_key(s):
        s = re.sub(r"^#+\s*", "", s)
        s = re.sub(
            r"^\s*(?:\d+(?:\.\d+)*[.、]?\s+|[一二三四五六七八九十]+[、.])", "", s
        )
        return re.sub(
            r"[^\w\u4e00-\u9fff]", "", unicodedata.normalize("NFKC", s)
        ).lower()

    bykey = {title_key(s): (s, k) for s, k in toc}
    if len(bykey) < 12:
        return
    # Recover a cover title if the layout detector emitted it as ordinary text.
    first = results[0]
    primary = [b for b in first["parsing_res_list"] if b.label == "doc_title"]
    if not primary:
        candidates = [
            b
            for b in first["parsing_res_list"]
            if b.label in {"text", "paragraph_title"}
            and 8 <= len(b.content.strip()) <= 180
            and not re.search(r"copyright|all rights|\bversion\b", b.content, re.I)
        ]
        if candidates:
            b = max(candidates, key=lambda b: estimate_height(b, first["width"]))
            if estimate_height(b, first["width"]) > 23:
                b.label = "doc_title"
                b.title_level = 0
                primary = [b]
                log_change(log, "cover_title_recovered", 0, text=b.content)
    chapter = any(k[0] == "chapter" for _, k in toc)
    cn = any(k[0] == "cn" for _, k in toc)

    def toc_level(kind):
        k, d = kind
        if k == "chapter":
            return 2
        if k == "cn":
            return 3 if chapter else 2
        if k == "cn_paren":
            return 4 if chapter else 3
        if k == "num":
            return 1 + d + int(chapter or cn)
        return None

    anchors = []
    matches = []
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if b.label not in {"paragraph_title", "doc_title", "text"}:
                continue
            key = title_key(b.content)
            if key not in bykey:
                continue
            if len(b.content) > 200:
                continue
            level = toc_level(bykey[key][1])
            height = estimate_height(b, p["width"])
            matches.append((pi, p, b, level, height))
            if level is not None:
                anchors.append((height, level))
    if len(matches) < 8 or len(anchors) < 2:
        return
    plain_anchor_count = sum(
        level is not None and heading_kind(b.content)[0] == "unknown"
        for _, _, b, level, _ in matches
    )
    if chapter or cn or plain_anchor_count < len(anchors) * 0.7:
        return
    top = min(l for h, l in anchors)
    top_height = statistics.median(h for h, l in anchors if l == top)
    for pi, p, b, level, height in matches:
        old = getattr(b, "title_level", 0 if b.label == "doc_title" else 1) + 1
        if level is None:
            if chapter or cn:
                continue
            level = top + round(
                math.log(max(height, 0.01) / top_height) / math.log(0.76)
            )
        level = max(2, min(6, level))
        b.label = "paragraph_title"
        b.title_level = level - 1
        if old != level:
            log_change(log, "toc_heading_level", pi, text=b.content, old=old, new=level)
    # Captions misclassified as headings should remain captions, not tree nodes.
    matched_keys = {title_key(b.content) for _, _, b, _, _ in matches}
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if (
                b.label == "paragraph_title"
                and title_key(b.content) not in matched_keys
                and re.match(
                    r"^(?:Figure|Fig\.|Table|图|表)\s*[\d一二三四五六七八九十]+[\s.\-:：]",
                    b.content,
                    re.I,
                )
            ):
                b.label = "figure_title"
                log_change(log, "caption_demoted", pi, text=b.content)


def apply_heading_decisions(results, path, log):
    entries = json.loads(path.read_text())
    mapping = {r["key"]: r for r in entries["decisions"]}
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            s = re.sub(r"^#+\s*", "", b.content).strip()
            key = (
                str(pi)
                + ":"
                + ",".join(str(round(float(v), 1)) for v in b.bbox)
                + ":"
                + hashlib.sha256(s.encode()).hexdigest()[:16]
            )
            if key not in mapping:
                continue
            d = mapping[key]
            old = b.label
            oldlevel = getattr(b, "title_level", 0 if old == "doc_title" else 1) + 1
            # Individual tree-depth errors propagate to all descendants. Use
            # learned overrides only at high confidence; numbering wins later.
            if d["confidence"] < (0.85 if d["level"] == 0 else 0.75):
                continue
            if d["level"] == 1 and pi >= 3:
                continue
            if d["level"] == 0:
                if b.label in {"paragraph_title", "doc_title"}:
                    b.label = "text"
                    log_change(
                        log,
                        "learned_heading_demoted",
                        pi,
                        text=b.content,
                        confidence=d["confidence"],
                    )
            else:
                b.label = "paragraph_title"
                b.title_level = d["level"] - 1
                if oldlevel != d["level"] or old not in {
                    "paragraph_title",
                    "doc_title",
                }:
                    log_change(
                        log,
                        "learned_heading_level",
                        pi,
                        text=b.content,
                        old=oldlevel,
                        new=d["level"],
                        confidence=d["confidence"],
                    )


def structural_headings(results, log):
    """Use repeated numbering schemes as document-wide hierarchy constraints."""

    def signature(text):
        s = unicodedata.normalize("NFKC", text).strip()
        m = re.match(r"^第[一二三四五六七八九十百零〇\d]+([篇部卷章节])", s)
        if m:
            return ("cn_" + m[1], 1)
        m = re.match(r"^(PART|CHAPTER|SECTION)\s+[\dIVX]+", s, re.I)
        if m:
            return ("en_" + m[1].lower(), 1)
        m = re.match(r"^(\d+(?:[.\-]\d+)+)[.、]?\s+", s)
        if m:
            parts = re.split(r"[.\-]", m[1])
            while len(parts) > 1 and parts[-1] == "0":
                parts.pop()
            return ("num", len(parts))
        kind, depth = heading_kind(s)
        if kind == "unknown" and re.match(r"^[一二三四五六七八九十]+\s+\S", s):
            return "cn", 1
        if kind == "unknown" and re.match(r"^\[\d+\]\s*\S", s):
            return "num_paren", 1
        return kind, depth

    entries = []
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if b.label in {"doc_title", "paragraph_title"}:
                entries.append(
                    (pi, p, b, signature(b.content), estimate_height(b, p["width"]))
                )
    counts = collections.Counter(k for _, _, _, (k, d), _ in entries)
    chinese = [
        k for k in ["cn_篇", "cn_部", "cn_卷", "cn_章", "cn_节"] if counts[k] >= 2
    ]
    english = [k for k in ["en_part", "en_chapter", "en_section"] if counts[k] >= 2]
    chapter_levels = {k: 2 + i for i, k in enumerate(chinese or english)}
    has_chapter = bool(chapter_levels)
    base = 2 + len(chapter_levels)
    has_cn = counts["cn"] >= 3
    numbase = base + int(has_cn)

    def level_for(k, d):
        if k in chapter_levels:
            return chapter_levels[k]
        if k == "cn" and has_cn:
            return base
        if k == "cn_paren" and counts[k] >= 3:
            return base + 1
        if k == "num" and counts[k] >= 3:
            return numbase + d - 1
        if k == "num_paren" and counts[k] >= 3:
            return numbase + 1
        if k == "roman" and counts[k] >= 3:
            return base
        return None

    # A title assigned at the top of a page is not necessarily a new document.
    # This normalization only activates when repeated numbering establishes it.
    constrained = sum(level_for(*e[3]) is not None for e in entries) >= 6
    if not constrained:
        return
    anchors = []
    for pi, p, b, (k, d), h in entries:
        level = level_for(k, d)
        if level is not None:
            old = 1 if b.label == "doc_title" else getattr(b, "title_level", 1) + 1
            b.label = "paragraph_title"
            b.title_level = min(5, level - 1)
            anchors.append((h, level))
            if old != level:
                log_change(
                    log, "numbering_hierarchy", pi, text=b.content, old=old, new=level
                )
    # Recover short, large chapter labels that the layout model called body text.
    median_h = statistics.median(h for h, l in anchors)
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if b.label != "text" or not 3 <= len(b.content.strip()) <= 90:
                continue
            k, d = signature(b.content)
            level = level_for(k, d)
            if k not in chapter_levels or level is None or TERMINAL.search(b.content):
                continue
            if estimate_height(b, p["width"]) < median_h * 0.85:
                continue
            b.label = "paragraph_title"
            b.title_level = level - 1
            log_change(log, "chapter_heading_recovered", pi, text=b.content, new=level)


def clean_heading_markup(results, log):
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if b.label not in {"doc_title", "paragraph_title"}:
                continue
            old = b.content
            b.content = re.sub(
                r"\$\s*\\(?:underline|textbf|textit|mathrm|textrm|text)\s*\{([^{}]+)\}\s*\$",
                r"\1",
                old,
            )
            if old != b.content:
                log_change(log, "heading_format_normalized", pi)


from . import global_hierarchy, heading_ml_apply, narrative_columns
from . import recover_headings as heading_recovery


def process_in_memory(name, pages, visual_features):
    from types import SimpleNamespace

    from ...pipeline import _basic_restructure_pages
    from . import fast_copy

    args = SimpleNamespace(
        tables=True, headings="toc", text="boundary", heading_decisions=None
    )
    log = []
    visual = heading_recovery.visual_map(pages, visual_features)
    if args.tables:
        narrative_columns.apply(pages, log)
        recover_prose_tables(pages, log)
        recover_glossary_tables(pages, log)
        repair_table_columns(pages, log)
        for pi, p in enumerate(pages):
            for b in p["parsing_res_list"]:
                if b["block_label"] == "table":
                    old = b["block_content"]
                    b["block_content"] = re.sub(
                        r"\\n(?=\s|\d|[A-Z\u4e00-\u9fff]|$)", " ", old
                    )
                    if old != b["block_content"]:
                        log_change(log, "table_linebreak_normalized", pi)
    prepared = fast_copy.copy_pages(
        [
            {k: p[k] for k in ["width", "height", "model_settings", "parsing_res_list"]}
            for p in pages
        ],
        page_keys=(),
    )
    original_blocks = {
        (pi, tuple(b["block_bbox"])): dict(b)
        for pi, p in enumerate(pages)
        for b in p["parsing_res_list"]
    }
    results = list(
        _basic_restructure_pages(
            [{"res": p} for p in pages],
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=False,
            soup_factory=readonly_soup_factory,
            table_merger=cached_table_merger,
        )
    )
    narrative_columns.attach(results, original_blocks)
    heading_recovery.attach_visual(results, visual)
    if args.tables:
        extra_merge_tables(results, log)
        separate_questionnaire_tables(results, log)
    recovered = heading_recovery.recover(results, log) if args.headings != "off" else []
    if args.text != "off":
        merge_text(results, log, args.text)
    original = heading_recovery.original_only(results)
    if args.headings != "off":
        relevel(original, log, "hybrid" if args.headings == "toc" else args.headings)
    if args.headings == "toc":
        toc_relevel(original, log)
    if args.heading_decisions:
        apply_heading_decisions(
            results, pathlib.Path(args.heading_decisions) / (name + ".json"), log
        )
    if args.headings != "off":
        structural_headings(original, log)
    heading_recovery.assign_levels(results, recovered, log)
    heading_recovery.refine_hierarchy(results, log)
    heading_recovery.apply_toc_levels(results, log)
    if args.headings != "off":
        clean_heading_markup(results, log)
        heading_recovery.finalize_headings(results, log)
    features = global_hierarchy.export_records(results, original_blocks, name, None)
    heading_ml_apply.apply(
        results, log, name, features, mode=ML_MODE, decoder=ML_DECODER
    )
    narrative_columns.finalize(results, log)

    final = []
    for page in results:
        item = {k: page[k] for k in ["width", "height", "model_settings"]}
        blocks = []
        for b in page["parsing_res_list"]:
            d = {
                "block_label": b.label,
                "block_content": b.content,
                "block_bbox": list(b.bbox),
            }
            for field in ["global_block_id", "global_group_id", "title_level"]:
                if hasattr(b, field):
                    d[field] = getattr(b, field)
            if getattr(b, "image", None):
                d["image"] = {"path": b.image["path"], "img": None}
            blocks.append(d)
        item["parsing_res_list"] = blocks
        final.append(item)
    return prepared, final, log
