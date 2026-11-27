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

from .native_similarity import ratio as native_ratio
from .runtime_context import memoize

"""Link evidence from page columns, repeated headers and printed continuation cues."""
import re

MULTILINGUAL_CONT = re.compile(
    r"(?:continued\s*(?:\.*\)?$|on\s+(?:the\s+)?next\s+page)|cont[’\x27]?d|续表|續表|下[页頁](?:[续續]|[继繼][续續])|接下[页頁]|下接|次ページに続|続き|\(続\)|계속|Продолжение|Fortsetzung|suite\s+(?:à|a|de|sur|du)|contin[uú]a\s+(?:en|na)|continua\s+a)",
    re.I,
)
SUBSECTION = re.compile(r"^\s*[（(][a-z0-9一二三四五六七八九十]+[）)]", re.I)
SERIAL = re.compile(r"(?<![\d.])(\d{2,5})\.0(?=\D|$)")


@memoize
def clean(s):
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf|overline|underline)", "", s)
    s = re.sub(r"[\s${}_^\\()[\]{}+*·•，,:：；;]+", "", s).lower()
    return s


@memoize
def sim(a, b):
    return 1.0 if clean(a) == clean(b) else native_ratio(clean(a), clean(b))


def header_match(a, b):
    aa = [s for s in a if clean(s)]
    bb = [s for s in b if clean(s)]
    if min(len(aa), len(bb)) < 2:
        return 0.0
    # Monotone header-cell alignment permits one duplicated or omitted column.
    n, m = len(aa), len(bb)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            dp[i + 1][j + 1] = max(
                dp[i][j + 1],
                dp[i + 1][j],
                dp[i][j] + (sim(aa[i], bb[j]) if sim(aa[i], bb[j]) > 0.72 else 0),
            )
    return dp[n][m] / min(n, m)


def is_header(row):
    ss = [s for s in row if s.strip()]
    if len(ss) < 2 or max(map(len, ss)) > 140:
        return False
    letters = sum(sum(c.isalpha() for c in s) > 1 for s in ss)
    numeric = sum(bool(re.fullmatch(r"[\s\d.,%()（）/\-]+", s)) for s in ss)
    return letters >= 2 and numeric / len(ss) < 0.4


def serial_range(tab):
    vv = [int(m[1]) for row in tab["rows"] for s in row for m in SERIAL.finditer(s)]
    if len(vv) < 12:
        return None
    steps = [b - a for a, b in zip(vv, vv[1:])]
    if sum(0 < x <= 5 for x in steps) < len(steps) * 0.75:
        return None
    return vv[0], vv[-1]


def context(f, side):
    # Printed continuation cues remain evidence even when repeated on many
    # pages and consequently classified as running page-margin content.
    body = f.get("body_" + side, f[side])
    return [
        q for q in f[side] if q in body or MULTILINGUAL_CONT.search(q["block_content"])
    ]


HEADER_WORD = re.compile(
    r"\b(?:name|address|telephone|description|code|date|number|item|category|amount|balance|total|price|quantity|unit|type|field|format|component|document)\b|名称|项目|类别|期末|期初|发生额|金额|数量|单价|科目|编码|日期|说明|描述",
    re.I,
)


def label_header(row):
    return (
        sum(bool(HEADER_WORD.search(s)) for s in row) >= 2
        and max(map(len, row), default=0) < 120
    )


def header_change(a, b):
    ra, rb = a["tab"]["rows"], b["tab"]["rows"]
    if not ra or not rb or len(ra[0]) != len(rb[0]) or len(ra[0]) < 2:
        return False
    if not (label_header(ra[0]) and label_header(rb[0])):
        return False
    ss = [sim(x, y) for x, y in zip(ra[0], rb[0])]
    return sum(ss) / len(ss) < 0.5 or len(ss) == 2 and min(ss) < 0.45 and max(ss) < 0.75


def titled_number_reset(a, b):
    """A new full-width title plus restarted row identifiers starts a table."""
    ra, rb = a["tab"]["rows"], b["tab"]["rows"]
    if min(len(ra), len(rb)) < 5 or len(ra[0]) != 1 or len(rb[0]) != 1:
        return False
    sa, sb = clean(ra[0][0]), clean(rb[0][0])
    if not sa or not sb or sa in sb or sb in sa or sim(sa, sb) > 0.55:
        return False
    if header_match(ra[1], rb[1]) < 0.9 or not is_header(ra[1]):
        return False

    def sequence(rows):
        vv = [
            int(r[0].strip())
            for r in rows[2:]
            if r and re.fullmatch(r"\s*\d{1,5}\s*", r[0])
        ]
        return (
            vv
            if len(vv) >= 3
            and sum(y > x for x, y in zip(vv, vv[1:])) >= 0.9 * (len(vv) - 1)
            else []
        )

    va, vb = sequence(ra), sequence(rb)
    return bool(va and vb and vb[0] <= 2 and vb[0] < va[-1])


def proposal(a, b, p):
    ta, tb = a["tab"], b["tab"]
    ra, rb = ta["rows"], tb["rows"]
    aa, bb = a["bbox"], b["bbox"]
    wa = aa[2] - aa[0]
    wb = bb[2] - bb[0]
    if not ra or not rb:
        return p, None
    same_page = a["page"] == b["page"]
    adjacent = (
        a["rank"] + 1 == b["rank"]
        if same_page
        else (
            b["page"] == a["page"] + 1
            and a["rank"] == a["tables_on_page"] - 1
            and b["rank"] == 0
        )
    )
    if not adjacent:
        return p, None
    head = header_match(ra[0], rb[0])
    width = abs(wa - wb) / max(wa, wb, 0.01) < 0.16
    if (
        a.get("parallel_prose_schema")
        and b.get("parallel_prose_schema")
        and abs(wa - wb) / max(wa, wb, 0.01) < 0.28
        and head > 0.995
        and ta["ncols"] == tb["ncols"]
    ):
        if same_page:
            overlap = max(0, min(aa[3], bb[3]) - max(aa[1], bb[1])) / max(
                0.01, min(aa[3] - aa[1], bb[3] - bb[1])
            )
            if aa[2] < bb[0] + 0.03 and overlap > 0.65:
                return 0.998, "repeated_parallel_prose_columns"
        else:
            # Look above the destination column, rather than letting a scan's
            # side labels and bottom page furniture obstruct a column wrap.
            preceding = []
            for q in b["before"]:
                box = q["block_bbox"]
                w = b.get("page_width", 1e9)
                h = b.get("page_height", 1e9)
                x = (box[0] + box[2]) / (2 * w)
                bottom = box[3] / h
                if (
                    bb[0] <= x <= bb[2]
                    and 0 <= bb[1] - bottom < 0.15
                    and q["block_content"].strip()
                ):
                    preceding.append(q)
            if not preceding and abs(aa[1] - bb[1]) < 0.1:
                return 0.995, "repeated_prose_column_wrap"
    if (
        not same_page
        and a.get("recovered_table_schema")
        and a.get("recovered_table_schema") == b.get("recovered_table_schema")
        and width
        and not context(a, "after")
        and not context(b, "before")
    ):
        return 0.995, "recovered_record_table_continuation"
    if same_page:
        blocks = a["page_blocks"][a["index"] + 1 : b["index"]]
        subparts = [
            q
            for q in blocks
            if q["block_content"].strip()
            and q["block_label"]
            not in {"number", "header", "footer", "footnote", "aside_text"}
        ]
        if (
            subparts
            and all(SUBSECTION.match(q["block_content"]) for q in subparts)
            and width
            and min(a["summary"][6], b["summary"][6]) > 0.65
            and len(ra) == len(rb) == 2
            and clean(ra[-1][0]) == clean(rb[-1][0])
        ):
            return 0.998, "numbered_table_panels"
        if subparts:
            return p, None
        vertical_overlap = max(0, min(aa[3], bb[3]) - max(aa[1], bb[1])) / max(
            0.01, min(aa[3] - aa[1], bb[3] - bb[1])
        )
        if (
            aa[2] < bb[0] + 0.03
            and vertical_overlap > 0.75
            and width
            and head > 0.86
            and ta["ncols"] == tb["ncols"]
            and min(a["summary"][6], b["summary"][6]) > 0.2
        ):
            return 0.998, "parallel_table_columns"
        gap = bb[1] - aa[3]
        if (
            -0.015 <= gap <= 0.035
            and abs(aa[0] - bb[0]) < 0.04
            and width
            and len(ra) <= 3
            and label_header(ra[0])
            and not label_header(rb[0])
            and ta["ncols"] == tb["ncols"]
        ):
            return 0.998, "detached_table_header"
        if -0.015 <= gap <= 0.035 and abs(aa[0] - bb[0]) < 0.04 and width:
            if (
                min(a["summary"][6], b["summary"][6]) > 0.65
                and not is_header(rb[0])
                and not is_header(ra[0])
                and ta["ncols"] == tb["ncols"]
            ):
                return 0.995, "stacked_table_fragments"
        return p, None
    if (
        len(ra[0]) == len(rb[0]) == 1
        and re.match(r"^\s*(?:Table|表)\s*\d+", ra[0][0], re.I)
        and clean(ra[0][0]) == clean(rb[0][0])
        and width
    ):
        return 0.998, "repeated_internal_caption"
    before = context(b, "before")
    after = context(a, "after")
    # Repeated marginal text may be classified as a normal title on one page.
    source_margin = {
        clean(q["block_content"])
        for q in a["page_blocks"]
        if q["block_label"] in {"header", "header_image"}
    }
    before = [q for q in before if clean(q["block_content"]) not in source_margin]

    def margin_code(q):
        box = q["block_bbox"]
        w = b.get("page_width", 1e9)
        h = b.get("page_height", 1e9)
        return bool(
            re.fullmatch(
                r"[A-Z]{1,4}\s*\d+(?:\.\d+){1,3}[a-z]?", q["block_content"].strip()
            )
            and box[0] > 0.65 * w
            and box[2] - box[0] < 0.15 * w
            and box[3] < 0.25 * h
        )

    before = [q for q in before if not margin_code(q)]

    def page_marker(q):
        s = q["block_content"].strip()
        box = q["block_bbox"]
        h = b.get("page_height", 1e9)
        small = box[3] - box[1] < 0.04 * h
        return (
            small
            and box[3] / h <= max(0.16, b["doc_top"])
            and bool(
                re.fullmatch(r"[A-Z]{1,3}[-–—]\d{1,4}|.{0,45}\bpage\s+\d{1,4}", s, re.I)
            )
        )

    before = [q for q in before if not page_marker(q)]
    titles = {
        clean(q["block_content"])
        for q in context(a, "before")
        if q["block_label"]
        in {"text", "figure_title", "paragraph_title", "table_title"}
        and 4 <= len(q["block_content"]) <= 100
    }
    if (
        min(a["summary"][6], b["summary"][6]) > 0.35
        and before
        and not after
        and all(
            4 <= len(q["block_content"]) <= 100 and clean(q["block_content"]) in titles
            for q in before
        )
        and width
        and head > 0.85
        and abs(ta["ncols"] - tb["ncols"]) <= 3
    ):
        return 0.998, "repeated_table_title"
    repeated_title = bool(before) and all(
        4 <= len(q["block_content"]) <= 100 and clean(q["block_content"]) in titles
        for q in before
    )
    if repeated_title and width and not after:
        long_prose = (
            lambda rows: sum(len(s) for row in rows[1:] for s in row)
            / max(1, len(rows) - 1)
            > 100
        )
        header_evidence = (
            head > 0.98
            and is_header(ra[0])
            and is_header(rb[0])
            and ta["ncols"] == tb["ncols"]
        )
        glossary_evidence = (
            ta["ncols"] == tb["ncols"] == 2
            and min(len(ra), len(rb)) >= 8
            and all(len(row) == 2 and len(row[0]) < 40 for row in ra + rb)
            and clean(ra[-1][0])[:2] <= clean(rb[0][0])[:2]
        )
        glossary_prose = glossary_evidence and all(
            sum(len(row[1]) for row in rows) / len(rows) > 30 for rows in [ra, rb]
        )
        if long_prose(ra) and long_prose(rb) and header_evidence or glossary_prose:
            return 0.985, "repeated_prose_table_title"
    cues = [q["block_content"] for q in after + before if len(q["block_content"]) < 180]
    continued = any(MULTILINGUAL_CONT.search(s) for s in cues)
    ordinary = lambda qq: [
        q
        for q in qq
        if q["block_content"].strip()
        and not MULTILINGUAL_CONT.search(q["block_content"])
        and not SUBSECTION.match(q["block_content"])
    ]
    clean_boundary = not ordinary(after) and not ordinary(before)
    if continued and width and abs(ta["ncols"] - tb["ncols"]) <= 2 and clean_boundary:
        return 0.998, "multilingual_continuation"
    if not clean_boundary:
        return p, None
    if not width:
        # Partial-width scans can retain the same columns and identifier grammar.
        from .column_alignment import align

        def codekeys(rows):
            keys = []
            for row in rows:
                if row and re.match(r"^[A-Z]{1,4}\d{1,4}[/\-]\d+", row[0].strip()):
                    keys.append(re.sub(r"\d+", "#", row[0].strip().split()[0]))
            return keys

        ka, kb = codekeys(ra), codekeys(rb)
        if (
            ta["ncols"] == tb["ncols"]
            and ta["ncols"] >= 4
            and ka
            and kb
            and set(ka) & set(kb)
            and abs(aa[0] - bb[0]) < 0.07
            and min(wa, wb) / max(wa, wb) > 0.45
            and aa[3] >= a["doc_bottom"] - 0.05
            and bb[1] <= b["doc_top"] + 0.03
        ):
            cost, _ = align(ta, tb)
            if cost < 0.22 and (not label_header(rb[0]) or head > 0.8):
                return 0.96, "partial_width_identifier_table"
        return p, None
    boundary = aa[3] >= a["doc_bottom"] - 0.12 and bb[1] <= b["doc_top"] + 0.10
    if not boundary:
        return p, None
    # Narrative cells can occupy almost an entire page. Their continuation
    # has no new header and inherits blank or row-spanning columns, so a
    # numeric-column test alone misses them.
    if (
        4 <= min(ta["ncols"], tb["ncols"])
        and abs(ta["ncols"] - tb["ncols"]) <= 3
        and len(rb) <= 3
    ):
        last, first = ra[-1], rb[0]
        long_a = sum(len(s) > 100 for s in last)
        long_b = sum(len(s) > 100 for s in first)
        narrow_schema = abs(wa - wb) / max(wa, wb) < 0.05 and abs(aa[0] - bb[0]) < 0.03
        page_boundary = aa[3] >= a["doc_bottom"] - 0.06 and bb[1] <= b["doc_top"] + 0.08
        if (
            narrow_schema
            and page_boundary
            and long_a >= 1
            and long_b >= 1
            and (long_b >= 2 or sum(not clean(s) for s in first) >= len(first) * 0.5)
            and not is_header(first)
        ):
            from .column_alignment import align

            cost, _ = align(ta, tb)
            if cost < 0.65:
                return 0.97, "large_prose_cell_continuation"
    if 2 <= len(ra) == len(rb) <= 5 and ta["ncols"] == tb["ncols"] and ta["ncols"] >= 5:
        from .numeric_grid_recovery import NUMBER

        row_labels = all(
            len(x) == len(y) == ta["ncols"]
            and clean(x[0]) == clean(y[0])
            and x[0]
            and not NUMBER.fullmatch(x[0])
            for x, y in zip(ra, rb)
        )
        numeric = all(NUMBER.fullmatch(s.strip()) for row in ra + rb for s in row[1:])
        if row_labels and numeric:
            values = [
                float(s.replace(",", ".").strip("%")) for s in ra[0][1:] + rb[0][1:]
            ]
            if all(y > x for x, y in zip(values, values[1:])):
                return 0.995, "numeric_panel_continuation"
    # Prose tables have little numeric content. Exact repeated labels and a
    # substantial explanatory column provide evidence independently of digits.
    prose = lambda rows: any(
        sum(len(r[k]) for r in rows[1:] if len(r) > k) / max(1, len(rows) - 1) > 55
        for k in range(min(8, len(rows[0])))
    )
    if (
        head > 0.96
        and is_header(ra[0])
        and is_header(rb[0])
        and abs(ta["ncols"] - tb["ncols"]) <= 1
        and prose(ra)
        and prose(rb)
    ):
        return 0.985, "repeated_prose_table_header"
    if (
        head > 0.92
        and len(rb) <= 3
        and bb[3] - bb[1] < 0.10
        and label_header(rb[0])
        and ta["ncols"] == tb["ncols"]
    ):
        return 0.998, "repeated_detached_header"
    sub = any(SUBSECTION.match(q["block_content"]) for q in before)
    if (
        sub
        and min(a["summary"][6], b["summary"][6]) > 0.65
        and len(ra) == len(rb) == 2
        and clean(ra[-1][0]) == clean(rb[-1][0])
    ):
        return 0.998, "numbered_table_panels"
    if (
        is_header(ra[0])
        and is_header(rb[0])
        and head > (0.78 if sub else 0.86)
        and abs(ta["ncols"] - tb["ncols"]) <= 2
        and (sub or min(a["summary"][6], b["summary"][6]) > 0.15)
    ):
        return 0.997, "aligned_repeated_header"
    sa, sb = serial_range(ta), serial_range(tb)
    if sa and sb and 0 < sb[0] - sa[1] <= 8:
        return 0.997, "printed_serial_continuation"
    from .text_table_recovery import acronym_key

    if ta["ncols"] == tb["ncols"] == 2 and min(len(ra), len(rb)) >= 5:
        valid = lambda rows: sum(
            acronym_key(r[0]) for r in rows if len(r) == 2
        ) >= 0.8 * len(rows)
        if valid(ra) and valid(rb) and clean(ra[-1][0])[:2] <= clean(rb[0][0])[:2]:
            return 0.97, "alphabetical_glossary_continuation"
    if ta["ncols"] == tb["ncols"] == 2 and min(len(ra), len(rb)) >= 3:
        pairs = [row for row in ra[-6:] + rb[:6] if len(row) == 2]
        if len(pairs) >= 4:
            left = sum(len(row[0]) for row in pairs) / len(pairs)
            right = sum(len(row[1]) for row in pairs) / len(pairs)
            if (
                left < 32
                and right > max(25, 2 * left)
                and not label_header(rb[0])
                and not re.fullmatch(r"\d+(?:\.[\da-z]+)+", rb[0][0], re.I)
            ):
                return 0.93, "definition_list_continuation"
    return p, None
