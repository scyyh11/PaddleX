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

"""Distinguish contact fields and copyright sentences from outline headings."""
import collections
import re

_URL = re.compile(r"(?:https?://|www\.)", re.I)


def excluded(meta, pages):
    result = set()
    bypage = collections.defaultdict(list)
    for i, r in enumerate(meta):
        bypage[r["page"]].append(i)
        s = r["text"]
        if (
            len(s) > 45
            and re.search(r"copyright|©", s, re.I)
            and re.search(r"all rights reserved|版权所有|版權所有", s, re.I)
        ):
            result.add(i)
    for pi, p in enumerate(pages):
        text = "\n".join(
            b["block_content"]
            for b in p["parsing_res_list"]
            if b["block_label"] != "table"
        )
        # without an '@' the address alternative cannot match, and the other alternatives give the same matches
        links = (
            re.findall(
                r"(?:https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})", text, re.I
            )
            if "@" in text
            else _URL.findall(text)
        )
        if len(links) < 2:
            continue
        phones = [
            s
            for s in re.findall(r"\+?\d[\d() -]{5,}\d", text)
            if sum(c.isdigit() for c in s) >= 7
        ]
        if len(phones) < 2:
            continue
        inds = bypage[pi + 1]
        titles = [
            i
            for i in inds
            if meta[i]["original_label"] in {"doc_title", "paragraph_title"}
        ]
        if len(titles) < 3:
            continue
        # Preserve the page's top-level contact heading; nested organizations and
        # support-field labels remain text. Numbered sections are never removed.
        first = min(titles, key=lambda i: meta[i]["box"][1])
        if not re.search(
            r"contact|kontakt|контакт|coordonn|連絡|联[络系]|聯[絡繫]|联系",
            meta[first]["text"],
            re.I,
        ):
            continue
        for i in titles:
            if i == first or re.match(
                r"^\s*(?:\d+[.、)]|第[一二三四五六七八九十\d]+[章节節])",
                meta[i]["text"],
            ):
                continue
            if meta[i]["box"][1] >= meta[first]["box"][3]:
                result.add(i)
    return result


def load_and_exclude(meta, doc):
    from .runtime_context import document_pages

    pages = document_pages()
    return excluded(meta, pages)
