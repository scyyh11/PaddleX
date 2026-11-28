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

"""Keep independent tables separate by explicit captions and financial statement names."""

import re


def nearby_caption_context(fragment):
    nearby = []
    for block in reversed(fragment.get("before", [])):
        if block["block_label"] in {"table", "image", "chart"}:
            break
        text = block.get("block_content", "").strip()
        if len(text) > 200:
            break
        nearby.append(block)
        if len(nearby) == 3:
            break
    return list(reversed(nearby))


def financial_identity(fragment):
    rows = fragment["tab"]["rows"]
    titles = [str(r[0]) for r in rows[:2] if r]
    titles += [
        b["block_content"]
        for b in nearby_caption_context(fragment)
        if b["block_label"]
        in {"doc_title", "paragraph_title", "figure_title", "table_title"}
    ]
    patterns = [
        (
            "balance",
            r"^(?:合并|合併|母公司|公司)?资产负债表|^(?:合併)?資產負債表|^(?:consolidated\s+)?balance\s+sheets?",
        ),
        (
            "income",
            r"^(?:合并|合併|母公司|公司)?(?:利润表|利潤表|损益表|損益表)|^(?:consolidated\s+)?(?:income\s+statements?|statements?\s+of\s+(?:income|operations))",
        ),
        (
            "cashflow",
            r"^(?:合并|合併|母公司|公司)?(?:现金流量表|現金流量表)|^(?:consolidated\s+)?(?:cash\s+flow\s+statements?|statements?\s+of\s+cash\s+flows?)",
        ),
        (
            "equity",
            r"^(?:合并|合併|母公司|公司)?(?:所有者|股东|股東)权益变动表|^(?:consolidated\s+)?statements?\s+of\s+(?:changes\s+in\s+)?(?:stockholders|shareholders|equity)",
        ),
        (
            "valuation",
            r"^(?:业绩和估值指标|主要估值指标|估值指标|valuation\s+(?:metrics|ratios))",
        ),
        (
            "financial_metrics",
            r"^(?:主要|关键)?财务指标|^key\s+financial\s+(?:metrics|ratios)",
        ),
        (
            "accounting_data",
            r"^(?:主要|关键)?会计数据|^key\s+accounting\s+data",
        ),
    ]
    hits = set()
    for title in titles:
        title = re.sub(r"\s+", " ", title).strip()
        for identity, pattern in patterns:
            if re.search(pattern, title, re.I):
                hits.add(identity)
    return next(iter(hits)) if len(hits) == 1 else None


def reject_reason(a, b):
    ca, cb = a.get("caption_id"), b.get("caption_id")
    if ca and cb and ca != cb:
        return "different_explicit_table_numbers"
    ia, ib = financial_identity(a), financial_identity(b)
    if ia and ib and ia != ib:
        return "different_named_financial_tables"
    return None


def enrich_fragments(fragments, caption_re, cont_re, norm_fn):
    """Attach normalized nearby caption context to table fragments."""
    for f in fragments:
        nearby = nearby_caption_context(f)
        f["context"] = " ".join(q["block_content"] for q in nearby)
        cap = next(
            (
                caption_re.match(q["block_content"])
                for q in reversed(nearby)
                if caption_re.match(q["block_content"])
            ),
            None,
        )
        f["caption_id"] = norm_fn(cap[1]) if cap else ""
        f["continued"] = bool(cont_re.search(f["context"]))
    return fragments
