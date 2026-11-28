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

"""Recover title existence by monotone, page-verified OCR contents alignment.

Running headers are eligible, but only at the page referred to by the contents.
Captions are excluded. No annotation or external model is read.
"""
import collections
import re
import unicodedata
from difflib import SequenceMatcher

from .heading_ml_features import norm


def parse(s):
    s = unicodedata.normalize("NFKC", s).strip()
    m = re.match(r"^(.*?)\s*(?:[._·…]{2,}\s*|\s+)(\d{1,4})\s*$", s)
    if not m:
        return None
    title = m[1].strip(" ._·…")
    if len(title) < 3 or re.match(r"^(?:Fig(?:ure)?\.?|Table|图|表)\s*\d", title, re.I):
        return None
    if re.search(r"[,;]\s*\d+\s*$", title):
        return None
    return title, int(m[2])


def entries(pages):
    output = []
    for pi, page in enumerate(pages, 1):
        blocks = page["parsing_res_list"]
        named = any(
            re.match(
                r"^(?:table\s+of\s+contents|contents|目\s*[录錄])[.。:\s]*$",
                b["block_content"].strip(),
                re.I,
            )
            for b in blocks
        )
        if pi > len(pages) * 0.3 + 5 and not named:
            continue
        found = []
        for b in blocks:
            if b["block_label"] != "content":
                continue
            raw = [s.strip() for s in b["block_content"].splitlines() if s.strip()]
            parsed = [parse(s) for s in raw]
            if len(raw) < 5 or sum(q is not None for q in parsed) < len(raw) * 0.6:
                continue
            found.extend(
                dict(text=q[0], printed=q[1], toc_page=pi) for q in parsed if q
            )
        output.extend(found)
    groups = []
    for entry in output:
        if not groups or entry["toc_page"] - groups[-1][-1]["toc_page"] > 2:
            groups.append([])
        groups[-1].append(entry)
    return groups


def monotone(nodes):
    """Maximum weight ordered matching, skipping OCR omissions on either side."""
    if not nodes:
        return []
    nodes = sorted(nodes, key=lambda n: (n["entry"], n["candidate"]))
    scores = []
    back = []
    for k, node in enumerate(nodes):
        previous = [
            j
            for j in range(k)
            if nodes[j]["entry"] < node["entry"]
            and nodes[j]["candidate"] < node["candidate"]
        ]
        j = max(previous, key=lambda j: scores[j]) if previous else -1
        scores.append(node["weight"] + (scores[j] if j >= 0 else 0))
        back.append(j)
    k = max(range(len(scores)), key=lambda k: scores[k])
    path = []
    while k >= 0:
        path.append(nodes[k])
        k = back[k]
    return path[::-1]


def align(meta, groups, prob=None):
    lookup = collections.defaultdict(list)
    keys = []
    for i, r in enumerate(meta):
        key = norm(r["text"])
        keys.append(key)
        if (
            r["label"] not in {"content", "footer", "figure_title"}
            and r["original_label"] not in {"footer", "figure_title"}
            and 3 <= len(key) <= 400
        ):
            lookup[key].append(i)
    from .toc_bound import admissible

    allowed = admissible(
        [
            norm(e["text"])
            for group in groups
            for e in group
            if len(norm(e["text"])) >= 10
        ],
        lookup,
    )
    all_hits = []
    summaries = []
    for gi, group in enumerate(groups):
        if len(group) < 6:
            continue
        # A later OCR "content" block may be an index, not a new manual.
        # Page-offset agreement and monotone matching determine the target range.
        begin = max(e["toc_page"] for e in group)
        end = 100000
        options = {}
        votes = collections.Counter()
        seen = set()
        for j, e in enumerate(group):
            key = norm(e["text"])
            opts = [
                (i, 1.0) for i in lookup.get(key, []) if begin < meta[i]["page"] < end
            ]
            if not opts and len(key) >= 10:
                for candidate in allowed.get(key, []):
                    inds = lookup[candidate]
                    if not 0.85 < len(candidate) / len(key) < 1.15:
                        continue
                    sim = SequenceMatcher(None, key, candidate, autojunk=False).ratio()
                    if sim >= 0.92:
                        opts.extend(
                            (i, sim) for i in inds if begin < meta[i]["page"] < end
                        )
            if not opts:
                continue
            options[j] = opts
            if key not in seen:
                votes.update({meta[i]["page"] - e["printed"] for i, sim in opts})
                seen.add(key)
        if not votes:
            continue
        offset, support = votes.most_common(1)[0]
        if support < 5 or support < 0.55 * len(seen):
            continue
        nodes = []
        for j, opts in options.items():
            for i, sim in opts:
                if meta[i]["page"] - group[j]["printed"] != offset:
                    continue
                old = meta[i]["original_label"] in {"doc_title", "paragraph_title"}
                # Equal text and page can refer to both a running header and
                # the body heading. Preserve the classifier's existing evidence
                # instead of silently choosing the earlier reading-order block.
                confidence = float(prob[i]) if prob is not None else 0.0
                nodes.append(
                    dict(
                        entry=j,
                        candidate=i,
                        weight=1 + sim + 0.1 * old + 0.35 * confidence,
                        text=group[j]["text"],
                        page=meta[i]["page"],
                        toc_page=group[j]["toc_page"],
                        offset=offset,
                    )
                )
        path = monotone(nodes)
        if len(path) < 5 or len(path) < support * 0.6:
            continue
        all_hits.extend(path)
        summaries.append(
            dict(
                toc_pages=sorted({e["toc_page"] for e in group}),
                offset=offset,
                support=support,
                anchors=len(path),
            )
        )
    unique = {}
    for hit in all_hits:
        unique.setdefault(hit["candidate"], hit)
    return list(unique.values()), summaries


def apply(meta, p, doc, log):
    from .runtime_context import document_pages

    pages = document_pages()
    hits, groups = align(meta, entries(pages), p)
    p = p.copy()
    for hit in hits:
        i = hit["candidate"]
        old = float(p[i])
        p[i] = max(p[i], 0.97)
        if old < 0.97:
            log.append(
                dict(
                    kind="toc_page_title_anchor",
                    page=meta[i]["page"],
                    text=meta[i]["text"],
                    source_label=meta[i]["original_label"],
                    p_before=old,
                    p_after=float(p[i]),
                    toc_page=hit["toc_page"],
                    printed_offset=hit["offset"],
                )
            )
    if hits:
        log.append(
            dict(kind="toc_monotone_alignment", groups=groups, anchors=len(hits))
        )
    return p
