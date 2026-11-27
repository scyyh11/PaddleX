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

"""Index and dictionary headwords are entries, not document sections.

A document-level rule: when a document has many selected heading candidates
whose only content is one CJK character (optionally with an index code), those
candidates form an index. Ordinary section titles are left unchanged.
"""
import re
import unicodedata

MIN_ENTRIES = 40


def core(text):
    s = unicodedata.normalize("NFKC", text)
    s = re.sub(r"\$[^$]*\$|\\\(.*?\\\)|_\{[^}]*\}|\^\{[^}]*\}", "", s)
    return re.sub(r"[\W\d_a-zA-Z]", "", s)


def headword(text):
    raw = re.sub(r"\s", "", unicodedata.normalize("NFKC", text))
    return len(core(text)) == 1 and not re.search(r"[a-zA-Z]{2,}", raw)


def filter_selected(meta, selected, log):
    """Remove index entries before outline decoding, so remaining sections get a clean tree."""
    import statistics

    from .heading_ml_features import fsize

    entries = [i for i in selected if headword(meta[i]["text"])]
    if len(entries) < MIN_ENTRIES or len(entries) < 0.5 * len(selected):
        return list(selected)
    pages = sorted(meta[i]["page"] for i in entries)
    lo, hi = pages[0], pages[-1]
    font = statistics.median(fsize(meta[i]) for i in entries)
    entry_set = set(entries)
    kept = [
        i
        for i in selected
        if not (
            i in entry_set
            or (lo <= meta[i]["page"] <= hi and fsize(meta[i]) < 1.3 * font)
        )
    ]
    log.append(
        {
            "kind": "index_region_entries_removed",
            "count": len(selected) - len(kept),
            "headwords": len(entries),
            "pages": [lo, hi],
            "selected_before": len(selected),
        }
    )
    return kept


def filter_targets(meta, targets, log):
    import statistics

    from .heading_ml_features import fsize

    entries = [i for i, level in targets.items() if level and headword(meta[i]["text"])]
    selected = sum(1 for level in targets.values() if level)
    if len(entries) < MIN_ENTRIES or len(entries) < 0.5 * selected:
        return targets
    pages = sorted(meta[i]["page"] for i in entries)
    lo, hi = pages[0], pages[-1]
    font = statistics.median(fsize(meta[i]) for i in entries)
    out = dict(targets)
    demoted = 0
    for i, level in targets.items():
        if not level:
            continue
        r = meta[i]
        # Inside the index span, entries and sub-entries share the index
        # typography; a markedly larger title is a real section boundary.
        if i in entries or (lo <= r["page"] <= hi and fsize(r) < 1.3 * font):
            out[i] = 0
            demoted += 1
    log.append(
        {
            "kind": "index_region_entries_demoted",
            "count": demoted,
            "headwords": len(entries),
            "pages": [lo, hi],
            "selected_before": selected,
        }
    )
    return out
