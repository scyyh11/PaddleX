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

"""Bound inventory regions by document section structure, not per-line punctuation."""
from .material_regions import PURITY, SECTION, SUB, UNIT, plain
from .regions_text import collapse
from .text_graph import TEXT


def apply(pre, final):
    active = False
    current = []
    regions = []
    startpage = -1

    def flush():
        nonlocal current
        if len(current) > 1:
            ss = " ".join(
                plain(pre[p]["parsing_res_list"][b]["block_content"])
                for p, b in current
            )
            if len(UNIT.findall(ss)) + len(PURITY.findall(ss)) >= 3:
                regions.append(current)
        current = []

    for pi, p in enumerate(pre):
        for bi, b in enumerate(p["parsing_res_list"]):
            s = b["block_content"].strip()
            lab = b["block_label"]
            if not s or lab in {"header", "footer", "number", "footnote"}:
                continue
            if len(s) < 90 and SECTION.search(s):
                flush()
                active = True
                startpage = pi
                continue
            if active and len(s) < 45 and SUB.match(s):
                flush()
                startpage = pi
                continue
            if lab in {"paragraph_title", "doc_title"}:
                flush()
                active = False
                continue
            if not active:
                continue
            if pi - startpage > 2:
                flush()
                active = False
                continue
            # Inventory names, quantities, and explanatory concentration lines stay together.
            if lab in TEXT | {"table"} and len(plain(s)) < 1800:
                current.append((pi, bi))
            else:
                flush()
                active = False
    flush()
    return collapse(pre, final, regions, "structured_inventory")
