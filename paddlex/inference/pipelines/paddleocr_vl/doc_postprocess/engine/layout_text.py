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

"""Predict-only paragraph graph from line geometry, reading tracks, and syntax."""
import collections
import re

import numpy as np

from . import panel_layout
from . import runtime_context as ctx
from . import text_features as f

PANEL_LAYOUT = True
from .text_graph import END, LIST, SKIP, TEXT

LISTNUM = re.compile(r"^\s*(?:\((\d+)\)|(\d+)[.)])\s*")


def ending(s):
    s = s.strip()
    s = re.sub(r"\s*\$\s*\^\s*\{[^{}]{1,20}\}\s*\$\s*$", "", s).rstrip()
    s = re.sub(r"(?<=[。.!?！？”」])\s*[①-⑳⁰¹²³⁴⁵⁶⁷⁸⁹]+$", "", s).rstrip()
    return s


def reference_start(s):
    return bool(
        re.match(
            r"^\s*[\u4e00-\u9fff·‧．、，,\s]{2,65}(?:[（(](?:[编編译譯著主修订訂]+)[）)])?\s*[（(]\d{4}[a-z]?[）)]\s*[:：]",
            s,
        )
        or re.match(
            r"^\s*[A-Z][A-Za-z’\x27-]+,\s*(?:[A-Z]\.\s*){1,3}.*[（(]\d{4}[a-z]?[）)]", s
        )
    )


def reference_end(s):
    ref_end = ctx.get_config().reference_end
    if ref_end == "off":
        return False
    pattern = r"(?:→|⇒)\s*(?:[①-⑳\d]+[\s·・.,、/～~–—－\-]*)+$"
    m = re.search(pattern, s)
    return bool(m and (ref_end == "all" or re.search(r"[①-⑳]", m[0])))


def new_item(s):
    cfg = ctx.get_config()
    if cfg.layout_numeric_items and re.match(
        r"^\s*\d{1,3}(?:[．、]\s*|[，,]\s*(?=[\u4e00-\u9fff]))", s
    ):
        return True
    if cfg.layout_letter_items and re.match(r"^\s*[a-zA-Z]\s+[\u4e00-\u9fff]", s):
        return True
    if cfg.layout_ref_mode != "base" and reference_start(s):
        return True
    if cfg.layout_item_mode in {"roman", "both"} and re.match(
        r"^\s*[(（](?:[ivxlcdmIVXLCDM]+|[a-zA-Z])[)）]\s*", s
    ):
        return True
    if cfg.layout_item_mode in {"brackets", "both"} and re.match(
        r"^\s*(?:[【〔〖][^】〕〗]{1,16}[】〕〗]|[①-⑳])", s
    ):
        return True
    if re.match(r"^\s*[\u4e00-\u9fff·‧．]{2,24}\s*[（(]\d{4}[a-z]?[）)]\s*[:：]", s):
        return True
    if re.match(
        r"^\s*[A-Z][A-Za-z’\x27-]+,\s*(?:[A-Z]\.\s*){1,3}.*[（(]\d{4}[a-z]?[）)]", s
    ):
        return True
    return bool(
        LIST.match(s)
        or re.match(r"^\s*(?:\[\d+\]|[-–—]\s+|[a-zA-Z][.)]\s+|[ivxlcdm]+[.)]\s+)", s)
    )


def _chars_reference(s):
    c = [0.0, 0.0, 0.0, 0.0]
    for x in s:
        if "\u4e00" <= x <= "\u9fff":
            c[0] += 1
        elif x.isdigit():
            c[1] += 1
        elif x.isalpha():
            c[2] += 1
        elif not x.isspace():
            c[3] += 1
    return np.array(c)


# chars(s) counts the same four classes with C-level passes: CJK-range characters, digits (no
# CJK-range character is a digit), letters outside the CJK range (CJK-range letters are subtracted),
# and the remaining non-space characters.  The CJK-range code points that are not letters in this
# Python's Unicode data are counted separately so the subtraction is exact.
from . import charcount as _cc

_CJK_NONALPHA = [cp for cp in range(0x4E00, 0xA000) if not chr(cp).isalpha()]
assert not any(chr(cp).isdigit() or chr(cp).isspace() for cp in range(0x4E00, 0xA000))
_CJK_NONALPHA_RE = (
    re.compile("[" + "".join(re.escape(chr(cp)) for cp in _CJK_NONALPHA) + "]")
    if _CJK_NONALPHA
    else None
)


def chars(s):
    c0 = _cc.cjk(s)
    if not c0 and s.isascii():
        c1 = sum(map(str.isdigit, s))
        c2 = sum(map(str.isalpha, s))
    else:
        c1 = sum(map(str.isdigit, s))
        c2 = sum(map(str.isalpha, s)) - (
            c0
            - (len(_CJK_NONALPHA_RE.findall(s)) if _CJK_NONALPHA_RE is not None else 0)
        )
    c3 = len(s) - sum(map(str.isspace, s)) - c0 - c1 - c2
    out = np.array([float(c0), float(c1), float(c2), float(c3)])
    return out


def repair_source(pre, final, visual):
    from scipy.optimize import nnls

    cfg = ctx.get_config()
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            a["_visual_key"] = b["_visual_key"] = f"{pi}:{bi}"
        panels = panel_layout.detect(p) if PANEL_LAYOUT else None
        groups = collections.defaultdict(list)
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] in TEXT and "group_id" in b:
                groups[b["group_id"]].append(bi)
        # Fit character widths from this page; fall back to document-level typography.
        design = []
        target = []
        for bi, b in enumerate(p["parsing_res_list"]):
            v = visual.get(f"{pi}:{bi}")
            s = b["block_content"]
            if (
                b["block_label"] in TEXT
                and v
                and len(s) > 25
                and "$" not in s
                and len(groups.get(b.get("group_id"), [])) <= 1
            ):
                design.append(chars(s))
                target.append(v["extent"])
        coeff = np.array([1.25, 0.65, 0.58, 0.6])
        if len(design) >= 4:
            # Weak ridge prior stabilizes scarce numeric/Latin glyph classes.
            X = np.vstack([design, np.eye(4) * 6])
            Y = np.r_[target, coeff * 6]
            coeff = nnls(X, Y)[0]
            coeff = np.clip(coeff, 0.3, 2.0)
        moved = []
        for _, idxs in groups.items():
            if not 2 <= len(idxs) <= 4:
                continue
            bb = [p["parsing_res_list"][i] for i in idxs]
            non = [k for k, b in enumerate(bb) if b["block_content"].strip()]
            if non != [0] or "$" in bb[0]["block_content"]:
                continue
            if any(q["parsing_res_list"][i]["block_label"] not in TEXT for i in idxs):
                continue
            # Repair only a demonstrably inverted column jump; preserve valid source groups.
            a = bb[0]
            bad = bool(
                panels
                and any(
                    panels["bands"][i] != panels["bands"][idxs[0]] for i in idxs[1:]
                )
            )
            for b in bb[1:]:
                ax, ay, ar, ab = a["block_bbox"]
                bx, by, br, bt = b["block_bbox"]
                if bx > ax + p["width"] * 0.15:
                    bad |= any(
                        c["block_label"] in TEXT
                        and c["block_content"]
                        and abs(c["block_bbox"][0] - bx) < p["width"] * 0.06
                        and c["block_bbox"][1] < by - p["height"] * 0.05
                        for c in p["parsing_res_list"]
                    )
            vv = [visual.get(f"{pi}:{i}") for i in idxs]
            if not bad or not all(vv):
                continue
            s = bb[0]["block_content"]
            weights = np.array([v["extent"] for v in vv])
            total = chars(s) @ coeff
            weights *= total / weights.sum()
            prefix = [0.0]
            for c in s:
                prefix.append(prefix[-1] + chars(c) @ coeff)
            cuts = [0]
            for expected in np.cumsum(weights)[:-1]:
                choices = [
                    i
                    for i in range(cuts[-1] + 1, len(s))
                    if not (
                        s[i - 1].isascii()
                        and s[i].isascii()
                        and s[i - 1].isalnum()
                        and s[i].isalnum()
                    )
                ]
                cut = min(choices, key=lambda i: abs(prefix[i] - expected))
                cuts.append(cut)
            cuts.append(len(s))
            for k, bi in enumerate(idxs):
                b = p["parsing_res_list"][bi]
                b["block_content"] = s[cuts[k] : cuts[k + 1]]
                b["group_id"] = b.get("block_id", bi)
                fb = q["parsing_res_list"][bi]
                fb["block_content"] = b["block_content"]
                fb["global_group_id"] = fb.get("global_block_id")
            for bi in idxs[1:]:
                moved.append(
                    (p["parsing_res_list"][bi]["_visual_key"], bb[0]["_visual_key"])
                )
            logs.append(
                {
                    "kind": "source_column_unjoin",
                    "page": pi + 1,
                    "blocks": idxs,
                    "cuts": cuts,
                    "coefficients": coeff.tolist(),
                }
            )
        # Move repaired fragments to their geometric place within the destination column.
        order = list(range(len(p["parsing_res_list"])))
        for key, sourcekey in moved:
            bi = next(
                i
                for i, b in enumerate(p["parsing_res_list"])
                if b["_visual_key"] == key
            )
            b = p["parsing_res_list"][bi]
            x, y, r, bt = b["block_bbox"]
            order.remove(bi)
            same = [
                i
                for i in order
                if abs(p["parsing_res_list"][i]["block_bbox"][0] - x)
                < p["width"] * 0.08
                and p["parsing_res_list"][i]["block_bbox"][1] < y
                and p["parsing_res_list"][i]["block_label"] not in SKIP
            ]
            if same:
                pos = max(order.index(i) for i in same) + 1
            else:
                pos = min(bi, len(order))
            if cfg.keep_source_order:
                origin = next(
                    i
                    for i, b in enumerate(p["parsing_res_list"])
                    if b["_visual_key"] == sourcekey
                )
                pos = max(pos, order.index(origin) + 1)
            order.insert(pos, bi)
        if panels:
            order = panel_layout.order(p, panels)
            logs.append(
                {
                    "kind": "horizontal_panel_order",
                    "page": pi + 1,
                    "bands": len(panels["cuts"]) + 1,
                    "order": order,
                }
            )
        p["parsing_res_list"] = [p["parsing_res_list"][i] for i in order]
        q["parsing_res_list"] = [q["parsing_res_list"][i] for i in order]
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            x, y, r, bt = [
                v / z
                for v, z in zip(
                    b["block_bbox"], [p["width"], p["height"], p["width"], p["height"]]
                )
            ]
            if (
                b["block_label"] in TEXT
                and len(b["block_content"].strip()) <= 2
                and y < 0.15
                and r - x < 0.06
                and 0.3 < (x + r) / 2 < 0.7
            ):
                b["block_label"] = a["block_label"] = "header"
                logs.append(
                    {
                        "kind": "isolated_running_marker",
                        "page": pi + 1,
                        "text": b["block_content"],
                    }
                )
    return logs


def apply(pre, final, visual, extras=True, graph_only=False, repair_hook=None):
    logs = (repair_hook or repair_source)(pre, final, visual)
    nodes, visible = f.initialize(pre, final, visual)
    by = collections.defaultdict(list)
    for n in nodes:
        n["visual"] = visual.get(n["block"].get("_visual_key", ""), {})
        by[n["page"]].append(n)
        if n["transcript"] and n["first"] and re.match(r"^[a-z]", n["text"].strip()):
            n["block"]["_leading_continuation"] = True
    proposals = {}

    def add(a, b, reason, weight):
        if a["id"] < b["id"] and weight > proposals.get((a["id"], b["id"]), (0, ""))[0]:
            proposals[a["id"], b["id"]] = (weight, reason)

    def cont(a, b, cross=False):
        x, y = ending(a["text"]), b["text"].strip()
        if not x or not y or b["block"].get("_explicit_paragraph_start"):
            return False
        if a["block"].get("_code_region") and b["block"].get("_code_region"):
            return (
                not b["block"].get("_code_start")
                and not re.search(r"\bend\s*$", x, re.I)
                and not re.match(
                    r"^\s*(?:recursive\s+)?(?:subroutine|(?:real|double precision|integer)\s+function)\b",
                    y,
                    re.I,
                )
            )
        if re.match(r"^[~～—－]", x):
            return False
        if a["block"].get("_text_region") or b["block"].get("_text_region"):
            return False
        if (
            "content" in {a["block"]["block_label"], b["block"]["block_label"]}
            and a["block"]["block_label"] != b["block"]["block_label"]
        ):
            return False
        if a["transcript"]:
            return (
                cross
                and bool(b["block"].get("_leading_continuation"))
                and not END.search(x)
            )
        if reference_end(x):
            return False
        if len(x) < 2 or len(y) < 2 or new_item(y) or x.endswith((":", "：")):
            return False
        return not END.search(x) or bool(re.match(r"^[a-z,;，；）)]", y))

    def headings(a, b):
        return any(
            l in {"paragraph_title", "doc_title"}
            for _, _, l in visible[a["vindex"] + 1 : b["vindex"]]
        )

    for pi, nn in by.items():
        for a, b in zip(nn, nn[1:]):
            if headings(a, b):
                continue
            ax, ay, ar, ab = a["bbox"]
            bx, by0, br, bb = b["bbox"]
            av, bv = a["visual"], b["visual"]
            x, y = a["text"].strip(), b["text"].strip()
            mid = visible[a["vindex"] + 1 : b["vindex"]]
            same = abs(ax - bx) < 0.065 and abs(ar - br) < 0.13
            column = by0 < ab - 0.15 and bx > ax + 0.2
            if column and not mid and cont(a, b):
                add(a, b, "column", 5)
            if (
                a["transcript"]
                and not b["block"].get("_leading_continuation")
                and not a["block"].get("_leading_continuation")
            ):
                add(a, b, "transcript_region", 8)
                continue
            if not extras:
                continue
            # Continuous text resumes after a floating table/figure in the same column.
            floats = mid and all(
                l
                in {
                    "table",
                    "image",
                    "chart",
                    "figure_title",
                    "display_formula",
                    "formula_number",
                    "vision_footnote",
                }
                for _, _, l in mid
            )
            if (
                same
                and by0 > ab - 0.005
                and cont(a, b)
                and floats
                and (bv.get("first_left", 3) < 1.5 or re.match(r"^[a-z]", y))
            ):
                add(a, b, "around_float", 7)
            # Adjacent line fragments: flush start, similar font, no new paragraph indent.
            font = av.get("font", 0)
            bfont = bv.get("font", 0)
            similar = font and bfont and 0.8 < font / bfont < 1.25
            if (
                not mid
                and same
                and -0.005 <= by0 - ab <= 0.025
                and similar
                and bv.get("first_left", 3) < 1.5
                and av.get("last_right", 9) < 2.0
                and cont(a, b)
            ):
                add(a, b, "visual_fragment", 6)
            # Sequential list items are a single list region. Restart/style changes terminate it.
            ma, mb = LISTNUM.match(x), LISTNUM.match(y)
            if (
                ma
                and mb
                and ma[1]
                and same
                and by0 >= ab - 0.01
                and re.search(r";(?:\s*and)?$", x)
                and sum("a" <= c.lower() <= "z" for c in x) > len(x) * 0.4
            ):
                na = int(ma[1] or ma[2])
                nb = int(mb[1] or mb[2])
                if nb == na + 1 and bool(ma[1]) == bool(mb[1]):
                    add(a, b, "enumerated_region", 7)
            # Quoted paragraphs and their attribution form one block.
            if False:
                add(a, b, "quote_region", 7)
            if (
                re.match(r"^[—～~－-]\s*[^\d]", y)
                and len(y) < 90
                and by0 >= ab - 0.005
                and by0 - ab < 0.025
                and bx > ax + 0.15
            ):
                add(a, b, "attribution", 6)
    for pi in range(1, len(final)):
        aa, bb = by.get(pi - 1, []), by.get(pi, [])
        if not aa or not bb:
            continue
        a, b = aa[-1], bb[0]
        # Keep section boundaries, including headings before the first/after the last text.
        between = visible[a["vindex"] + 1 : b["vindex"]]
        if any(l in {"paragraph_title", "doc_title"} for _, _, l in between):
            continue
        if cont(a, b, True):
            add(a, b, "cross_page", 5)
        if extras:
            ma, mb = LISTNUM.match(a["text"]), LISTNUM.match(b["text"])
            if (
                ma
                and mb
                and ma[1]
                and re.search(r";(?:\s*and)?$", a["text"].strip())
                and sum("a" <= c.lower() <= "z" for c in a["text"])
                > len(a["text"]) * 0.4
                and int(mb[1] or mb[2]) == int(ma[1] or ma[2]) + 1
                and bool(ma[1]) == bool(mb[1])
            ):
                add(a, b, "cross_page_list", 7)
            av, bv = a["visual"], b["visual"]
            if (
                a["block"]["block_label"] == "content"
                and b["block"]["block_label"] == "content"
            ):
                add(a, b, "contents_sequence", 7)
            if a["text"].rstrip().endswith(("……", "…")) and b[
                "text"
            ].lstrip().startswith(("……", "…")):
                add(a, b, "quote_continuation", 6)
    # Strongest non-conflicting edges form disjoint paths. No fragment is duplicated.
    outgoing = {}
    incoming = {}
    selected = []
    for (i, j), (w, reason) in sorted(
        proposals.items(), key=lambda kv: (-kv[1][0], kv[0][1] - kv[0][0])
    ):
        if i in outgoing or j in incoming:
            continue
        outgoing[i] = j
        incoming[j] = i
        selected.append((i, j))
        logs.append(
            {
                "kind": reason,
                "from": [nodes[i]["page"] + 1, nodes[i]["index"]],
                "to": [nodes[j]["page"] + 1, nodes[j]["index"]],
            }
        )
    if graph_only:
        return logs, nodes, visible, selected
    f.merge_nodes(nodes, selected)
    return logs
