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

"""Recover equipment/reagent inventories under explicit laboratory subheadings."""
import re

from .regions_text import collapse
from .runtime_context import readonly_soup as BeautifulSoup
from .text_graph import TEXT

SECTION = re.compile(
    r"(?:[仪儀]器[与與和及]?[试試][剂劑药藥]|[仪儀]器[设設][备備]|[仪儀]器和[药藥]品)"
)
SUB = re.compile(r"^\s*[12一二][.、．]?\s*([仪儀]器|[试試][剂劑药藥]|[药藥]品)")
UNIT = re.compile(r"\d+\s*(?:台|支|套|个|個|块|根|只|把|张|瓶|本)")
PURITY = re.compile(r"(?:分析|化学|色谱|优级|光谱|化學|色譜|優級)[纯純]")


def plain(s):
    return (
        BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
        if "<table" in s
        else s
    )


def apply(pre, final):
    active = False
    current = []
    regions = []
    lastpage = -1

    def flush():
        nonlocal current
        if len(current) > 1:
            regions.append(current)
        current = []

    for pi, p in enumerate(pre):
        for bi, b in enumerate(p["parsing_res_list"]):
            s = b["block_content"].strip()
            lab = b["block_label"]
            if not s or lab in {"header", "footer", "number", "footnote"}:
                continue
            sub = SUB.match(s)
            if lab in {"paragraph_title", "doc_title"}:
                if SECTION.search(s):
                    flush()
                    active = True
                    lastpage = pi
                    continue
                if sub and active:
                    flush()
                    lastpage = pi
                    continue
                flush()
                active = False
                continue
            if not active:
                continue
            if pi - lastpage > 1:
                flush()
                active = False
                continue
            if sub:
                flush()
            text = plain(s)
            listlike = (
                len(UNIT.findall(text)) >= 2
                or len(PURITY.findall(text)) >= 1
                or text.count("；") + text.count(";") >= 2
            )
            # A source split can leave one short final inventory line.
            shorttail = (
                bool(current)
                and len(text) < 90
                and bool(UNIT.search(text) or PURITY.search(text))
            )
            if lab in TEXT | {"table"} and (listlike or shorttail):
                current.append((pi, bi))
                lastpage = pi
            else:
                flush()
                active = False
    flush()
    return collapse(pre, final, regions, "laboratory_inventory")
