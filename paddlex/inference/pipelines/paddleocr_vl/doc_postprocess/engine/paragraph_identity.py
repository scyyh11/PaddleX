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

"""Keep complete contact records, demoted headings, and source list columns out of paragraph chains."""

import re

EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?:\+?\d[\d ()-]{7,}\d)")


def record_emails(text):
    emails = set(EMAIL.findall(text))
    return emails if emails and PHONE.search(text) else set()


def _base_reject_reason(a, b):
    aa, bb = a["block"], b["block"]
    if aa.get("_standalone_source_label") or bb.get("_standalone_source_label"):
        return "standalone_source_label_boundary"
    if bb.get("_explicit_paragraph_start"):
        return "explicit_paragraph_start_boundary"
    ea, eb = record_emails(a["text"]), record_emails(b["text"])
    if ea and eb and ea != eb:
        return "distinct_complete_contact_records"
    return None


def reject_reason(a, b):
    from . import paragraph_integrity, source_text_structure

    return (
        _base_reject_reason(a, b)
        or paragraph_integrity.additional_rejection(a, b)
        or source_text_structure.rejection(a, b)
    )


def annotate_source_flags(raw_pages, pre, final):
    """Annotate demoted standalone labels onto ``pre`` and ``final`` blocks."""
    source = {
        (pi, tuple(b["block_bbox"])): b
        for pi, p in enumerate(raw_pages)
        for b in p["parsing_res_list"]
    }
    for pi, p in enumerate(final):
        for bi, b in enumerate(p["parsing_res_list"]):
            original = source.get((pi, tuple(b["block_bbox"])))
            if not original:
                continue
            s = original["block_content"].strip()
            if (
                original["block_label"] in {"doc_title", "paragraph_title"}
                and b["block_label"] == "text"
                and len(s) <= 80
                and s.endswith((":", "："))
            ):
                b["_standalone_source_label"] = True
                pre[pi]["parsing_res_list"][bi]["_standalone_source_label"] = True


def filter_links(nodes, links):
    """Filter candidate paragraph merge links using identity, integrity, and source-structure guards."""
    kept = []
    logs = []
    for i, j in links:
        reason = reject_reason(nodes[i], nodes[j])
        if reason:
            logs.append(
                {
                    "kind": reason,
                    "from": [nodes[i]["page"] + 1, nodes[i]["index"]],
                    "to": [nodes[j]["page"] + 1, nodes[j]["index"]],
                }
            )
        else:
            kept.append((i, j))
    return kept, logs
