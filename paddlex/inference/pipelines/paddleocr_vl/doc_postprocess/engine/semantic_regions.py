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

"""Whole, explicitly named indexes and complete source-code listings."""
import re

from .regions_text import collapse
from .runtime_context import readonly_soup as BeautifulSoup
from .text_graph import TEXT

INDEX = re.compile(
    r"^(?:(?:general|subject|personal author|author|alphabetical)\s+)?i(?:nd|d)ex$",
    re.I,
)
SYNTAX = re.compile(
    r"\bmdb\.|\bfrom\s+\w+\s+import\s|\*\s*(?:End\s+(?:Part|Assembly|Instance|Step)|Nset\b|Elset\b|Instance\s*,|Element\s*,)|\b(?:def|class)\s+\w+\(",
    re.I,
)


def plain(s):
    return (
        BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
        if "<table" in s
        else s
    )


def apply(pre, final, mode):
    logs = []
    if mode in {"index", "both"}:
        groups = []
        keys = []
        active = False
        for pi, p in enumerate(pre):
            bs = p["parsing_res_list"]
            heads = [
                (bi, re.sub(r"\s+", " ", b["block_content"]).strip())
                for bi, b in enumerate(bs)
                if b["block_label"]
                in {"paragraph_title", "doc_title", "header", "text"}
                and len(b["block_content"]) < 90
            ]
            explicit = any(INDEX.fullmatch(s) for bi, s in heads)
            ll = [
                s.strip()
                for b in bs
                if b["block_label"] in TEXT | {"table"}
                for s in plain(b["block_content"]).splitlines()
                if s.strip()
            ]
            refs = sum(bool(re.search(r"\d\s*$|\bAD-[AB]\d+", s)) for s in ll)
            listing = (
                len(ll) > 15
                and sum(len(s) < 110 for s in ll) > 0.85 * len(ll)
                and refs > len(ll) * 0.18
            )
            if explicit:
                if keys:
                    groups.append(keys)
                keys = []
                active = True
            if active and not (explicit or listing):
                if keys:
                    groups.append(keys)
                keys = []
                active = False
            if not active:
                continue
            for bi, b in enumerate(bs):
                if (
                    b["block_label"] in TEXT | {"table", "paragraph_title", "doc_title"}
                    and b["block_content"].strip()
                ):
                    keys.append((pi, bi))
        if keys:
            groups.append(keys)
        groups = [keys for keys in groups if len({pi for pi, bi in keys}) >= 2]
        logs += collapse(pre, final, groups, "explicit_index_section")
    if mode in {"scripts", "both"}:
        groups = []
        keys = []
        active = False
        for pi, p in enumerate(pre):
            bs = p["parsing_res_list"]
            s = "\n".join(
                plain(b["block_content"])
                for b in bs
                if b["block_label"] not in {"header", "footer", "number"}
            )
            count = len(SYNTAX.findall(s))
            # the line statistics are only consulted on an active page that is not already decided by count
            code = count >= 5
            if not code and active and count < 2:
                ll = [t.strip() for t in s.splitlines() if t.strip()]
                code = (
                    len(ll) >= 10
                    and sum(bool(re.match(r"^[-\d.,\s]+$", t)) for t in ll)
                    > len(ll) * 0.6
                )
            elif not code and active:
                code = True
            # A new deck has its own entry directive, independent of its preceding Python generator.
            if not code or bool(re.search(r"(?im)^\s*\*Heading\b", s)):
                if keys:
                    groups.append(keys)
                keys = []
            active = code
            if not active:
                continue
            for bi, b in enumerate(bs):
                if (
                    b["block_label"] in TEXT | {"table", "paragraph_title", "doc_title"}
                    and b["block_content"].strip()
                ):
                    keys.append((pi, bi))
        if keys:
            groups.append(keys)
        groups = [keys for keys in groups if len({pi for pi, bi in keys}) >= 2]
        logs += collapse(pre, final, groups, "source_listing_section")
    return logs
