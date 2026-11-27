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

"""Recognize explicit lexical appendices and navigation lists from source layout."""
import collections
import re

from .runtime_context import readonly_soup as BeautifulSoup
from .text_graph import TEXT

LEX = re.compile(
    r"字串|字格|熟[语語]|引[号號]抽取|[词詞][汇彙]表|[词詞]表|[词詞][条條]|[简簡][称稱]|V\s*[+＋]\s*V",
    re.I,
)
APP = re.compile(r"^\s*(?:附[录錄]|appendix)\s*([A-Z0-9一二三四五六七八九十]+)", re.I)
STOP = re.compile(
    r"^\s*(?:[参參]考文[献獻]|[后後]记|致[谢謝]|[结結][语語论論]|references|bibliography|acknowledg)",
    re.I,
)


def cells(s):
    soup = BeautifulSoup(s, "html.parser")
    return [
        c.get_text(" ", strip=True)
        for row in soup.find_all("tr")
        for c in row.find_all(["td", "th"], recursive=False)
    ]


def collapse(pre, final, regions, reason):
    groups = collections.defaultdict(list)
    logs = []
    for pi, p in enumerate(final):
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] == "table":
                groups[b.get("global_group_id", ("self", pi, bi))].append((pi, bi))
    for keys in regions:
        keys = set(keys)
        if not keys:
            continue
        # An original table converted to text must not also survive in a merged table.
        if any(not set(g).issubset(keys) for g in groups.values() if set(g) & keys):
            continue
        keys = sorted(keys)
        pieces = []
        for pi, bi in keys:
            b = pre[pi]["parsing_res_list"][bi]
            s = b["block_content"]
            pieces.append(" ".join(cells(s)) if "<table" in s else s)
        text = " ".join(pieces)
        for pi, bi in keys:
            for p in [pre, final]:
                b = p[pi]["parsing_res_list"][bi]
                b.update(
                    block_label="text",
                    block_content=text if (pi, bi) == keys[0] else "",
                    _text_region=reason,
                )
                if "global_block_id" in b:
                    b["global_group_id"] = b["global_block_id"]
        logs.append(
            {
                "kind": reason,
                "blocks": [[pi + 1, bi] for pi, bi in keys],
                "characters": len(text),
            }
        )
    return logs


def lexical(pre, final):
    repeated = collections.Counter(
        re.sub(r"\s+", "", b["block_content"])
        for p in pre
        for b in p["parsing_res_list"]
        if b["block_label"]
        in {"paragraph_title", "doc_title", "figure_title", "header"}
    )
    active = None
    startpage = -1
    enabled = False
    regions = collections.defaultdict(list)
    for pi, p in enumerate(pre):
        for bi, b in enumerate(p["parsing_res_list"]):
            lab = b["block_label"]
            s = b["block_content"].strip()
            am = APP.match(s)
            if lab in {"paragraph_title", "doc_title", "figure_title"} and len(s) < 180:
                if am:
                    key = am[1].lower()
                    if key != active:
                        active = key
                        startpage = pi
                        enabled = bool(LEX.search(s))
                    continue
                if STOP.match(s):
                    active = None
                    enabled = False
                if (
                    active is not None
                    and pi - startpage <= 1
                    and LEX.search(s)
                    and repeated[re.sub(r"\s+", "", s)] < 3
                ):
                    enabled = True
                    continue
            if enabled and lab in TEXT | {"table"} and s:
                regions[active].append((pi, bi))
    accepted = []
    for keys in regions.values():
        vals = []
        rich = 0
        for pi, bi in keys:
            b = pre[pi]["parsing_res_list"][bi]
            s = b["block_content"]
            if "<table" in s:
                cc = cells(s)
                vals += cc
                if len(cc) > 10 and sum(map(len, cc)) / len(cc) < 45:
                    rich += 1
            else:
                vals += s.splitlines()
                if len(s.splitlines()) > 10:
                    rich += 1
        if (
            rich >= 2
            and len(vals) > 20
            and sum(len(s) < 60 for s in vals) > 0.75 * len(vals)
        ):
            accepted.append(keys)
    return collapse(pre, final, accepted, "lexical_appendix_region")


SECTION_NAV = False
NAV = re.compile(
    r"^(?:(?:section\s+)?table\s+of\s+contents|(?:section\s+)?contents|content|目录|目錄|目\s*次|list\s+of\s+(?:figures|tables|illustrations)|figures|tables|插图目录|表格目录)$",
    re.I,
)
INDEX = re.compile(r"^(?:general\s+|subject\s+|alphabetical\s+)?index$", re.I)


def textlines(b):
    s = b["block_content"]
    return cells(s) if "<table" in s else s.splitlines()


def navstrength(b):
    ll = [l.strip() for l in textlines(b) if l.strip()]
    if not ll:
        return False
    if "<table" in b["block_content"]:
        rr = BeautifulSoup(b["block_content"], "html.parser").find_all("tr")
        rows = [
            [
                c.get_text(" ", strip=True)
                for c in row.find_all(["td", "th"], recursive=False)
            ]
            for row in rr
        ]
        if len(rows) >= 4 and sum(
            bool(row and re.fullmatch(r"[\dixvlcIVXLC .–—/-]+", row[-1]))
            for row in rows
        ) > 0.7 * len(rows):
            return True
    from .navigation_fast import page_number_suffix

    n = sum(page_number_suffix(l) for l in ll)
    return b["block_label"] == "content" or len(ll) >= 3 and n / len(ll) > 0.65


def navigation(pre, final):
    regions = []
    active = []
    kind = None
    miss = 0
    for pi, p in enumerate(pre):
        bs = p["parsing_res_list"]
        lab = [(i, b) for i, b in enumerate(bs) if b["block_content"].strip()]
        heads = [
            (i, re.sub(r"\s+", " ", b["block_content"]).strip())
            for i, b in lab
            if b["block_label"]
            in {"paragraph_title", "doc_title", "header", "figure_title"}
        ]
        ix = any(INDEX.fullmatch(s) for i, s in heads)
        nav = [s.lower() for i, s in heads if NAV.fullmatch(s)]
        content = [(i, b) for i, b in lab if b["block_label"] in TEXT | {"table"}]
        strong = sum(navstrength(b) for i, b in content)
        letters = sum(bool(re.fullmatch(r"[A-Z]", s)) for i, s in heads)
        index_page = False
        if ix or letters >= 2:
            if active:
                regions.append(active)
            active = []
            kind = None
            continue
        if kind == "index" and not index_page:
            index_page = letters >= 2 and strong > 0
        nav_page = (
            strong > 0
            and (
                strong >= len(content) * 0.5
                or any(b["block_label"] == "content" for i, b in content)
            )
        ) and (pi < len(pre) * 0.2 or kind is not None or bool(nav))
        candidate_kind = (
            "index"
            if index_page
            else (
                (
                    "nav:" + nav[-1]
                    if nav
                    else kind if kind and kind.startswith("nav:") else "nav:contents"
                )
                if nav_page
                else None
            )
        )
        if not candidate_kind:
            if active:
                regions.append(active)
            active = []
            kind = None
            continue
        if kind is not None and candidate_kind != kind:
            if active:
                regions.append(active)
            active = []
        kind = candidate_kind
        for bi, b in lab:
            s = b["block_content"].strip()
            l = b["block_label"]
            norm = re.sub(r"\s+", " ", s)
            if l in TEXT | {"table"}:
                if candidate_kind == "index" or navstrength(b) or len(s) < 100:
                    active.append((pi, bi))
            elif l in {"paragraph_title", "doc_title"}:
                if candidate_kind == "index" and re.fullmatch(
                    r"[A-Z]|Special characters", s, re.I
                ):
                    active.append((pi, bi))
                elif (
                    kind.startswith("nav:")
                    and not NAV.fullmatch(norm)
                    and len(s) < 160
                    and (
                        re.search(r"\d\s*$", s)
                        or re.match(
                            r"^(?:Chapter|第.+[章节節篇]|附[图表录]|[一二三四五六七八九十]+、)",
                            s,
                            re.I,
                        )
                    )
                ):
                    active.append((pi, bi))
    if active:
        regions.append(active)
    # Require multiple source blocks or pages; no automatic whole-book concatenation.
    regions = [
        [
            key
            for key in keys
            if pre[key[0]]["parsing_res_list"][key[1]]["block_label"] != "table"
            and final[key[0]]["parsing_res_list"][key[1]]["block_label"] != "table"
        ]
        for keys in regions
    ]
    regions = [
        keys
        for keys in regions
        if len(keys) >= 2
        and (
            len({p for p, b in keys}) >= 2
            or SECTION_NAV
            and all(
                pre[p]["parsing_res_list"][b]["block_label"] == "content"
                for p, b in keys
            )
        )
    ]
    return collapse(pre, final, regions, "navigation_region")
