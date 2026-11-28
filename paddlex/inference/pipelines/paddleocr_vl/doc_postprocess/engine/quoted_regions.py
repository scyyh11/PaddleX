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

"""Restore attributed epigraphs and aligned bilingual verse from page layout."""
import re

from . import charcount
from .text_graph import LIST, TEXT

ATTR = re.compile(r"^(?:[~～]|--|—|－|-\s*[\u4e00-\u9fff])")


def latin(s):
    return charcount.latin(s) / max(len(s), 1) > 0.5


def cjk(s):
    return charcount.cjk(s) / max(len(s), 1) > 0.4


def pairlines(a, b):
    aa = [x.strip() for x in a.splitlines() if x.strip()]
    bb = [x.strip() for x in b.splitlines() if x.strip()]
    if len(aa) == len(bb) and len(aa) > 1 and latin(a) and cjk(b):
        return " ".join(x + " " + y for x, y in zip(aa, bb))
    return a + " " + b


def apply(pre, final):
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        bs = p["parsing_res_list"]
        used = set()
        body = [
            b["block_bbox"][0]
            for b in bs
            if b["block_label"] == "text" and len(b["block_content"]) > 220
        ]
        margin = min(body, default=0) / p["width"]
        for bi, b in enumerate(bs):
            s = b["block_content"].strip()
            if b["block_label"] not in TEXT or not ATTR.match(s) or len(s) > 180:
                continue
            candidates = []
            below = b
            for j in range(bi - 1, max(-1, bi - 7), -1):
                a = bs[j]
                if a["block_label"] in {"number", "footnote", "header", "footer"}:
                    continue
                if a["block_label"] not in TEXT or j in used:
                    break
                x, y, r, bot = a["block_bbox"]
                gap = (below["block_bbox"][1] - bot) / p["height"]
                t = a["block_content"].strip()
                if not t:
                    continue
                if (
                    ATTR.match(t)
                    or len(t) > 650
                    or x / p["width"] < margin + 0.025
                    or gap > 0.065
                ):
                    break
                candidates.append(j)
                below = a
                if LIST.match(t):
                    break
            if not candidates:
                continue
            candidates = sorted(candidates) + [bi]
            pieces = []
            k = 0
            while k < len(candidates):
                j = candidates[k]
                a = bs[j]
                t = a["block_content"]
                if k + 1 < len(candidates) - 1:
                    j2 = candidates[k + 1]
                    b2 = bs[j2]
                    x, y, r, bot = a["block_bbox"]
                    xx, yy, rr, bb = b2["block_bbox"]
                    if (
                        xx > r
                        and min(bot, bb) - max(y, yy) > 0.65 * min(bot - y, bb - yy)
                        and latin(t)
                        and cjk(b2["block_content"])
                    ):
                        pieces.append(pairlines(t, b2["block_content"]))
                        k += 2
                        continue
                # Source OCR sometimes joins the two columns into a single string.
                ll = [x.strip() for x in t.splitlines() if x.strip()]
                n = len(ll) // 2
                if (
                    n >= 2
                    and len(ll) == 2 * n
                    and all(latin(z) for z in ll[:n])
                    and all(cjk(z) for z in ll[n:])
                ):
                    t = " ".join(x + " " + y for x, y in zip(ll[:n], ll[n:]))
                pieces.append(t)
                k += 1
            text = " ".join(pieces)
            anchor = candidates[0]
            for j in candidates:
                used.add(j)
                for page in [p, q]:
                    a = page["parsing_res_list"][j]
                    a.update(
                        block_content=text if j == anchor else "",
                        _text_region="attributed_quote",
                    )
                    if "global_block_id" in a:
                        a["global_group_id"] = a["global_block_id"]
            logs.append(
                {
                    "kind": "attributed_quote_region",
                    "page": pi + 1,
                    "blocks": candidates,
                    "characters": len(text),
                }
            )
    return logs


def tracks(nodes, visible, fixed):
    out = {i for i, j in fixed}
    inc = {j for i, j in fixed}
    extra = []
    logs = []
    for i, a in enumerate(nodes):
        if (
            not a["text"].rstrip().endswith(("：", ":"))
            or len(a["text"]) < 65
            or a["block"].get("_text_region")
        ):
            continue
        for j in range(i + 2, min(i + 8, len(nodes))):
            b = nodes[j]
            ax, ay, ar, ab = a["bbox"]
            bx, by, br, bb = b["bbox"]
            gap = b["page"] - a["page"]
            if gap > 1:
                break
            if (
                abs(ax - bx) > 0.025
                or abs(ar - br) > 0.045
                or b["block"].get("_text_region")
                or len(b["text"]) < 65
            ):
                continue
            mid = nodes[i + 1 : j]
            if not all(
                c["bbox"][0] > ax + 0.035
                and c["bbox"][2] <= ar + 0.015
                and c["block"]["block_label"] in TEXT
                for c in mid
            ):
                continue
            if any(l not in TEXT for _, _, l in visible[a["vindex"] + 1 : b["vindex"]]):
                continue
            av, bv = a["visual"].get("font", 0), b["visual"].get("font", 0)
            if av and bv and not 0.9 < av / bv < 1.1:
                continue
            if i in out or j in inc:
                break
            if any(i < k < j < l or k < i < l < j for k, l in fixed + extra):
                break
            extra.append((i, j))
            out.add(i)
            inc.add(j)
            logs.append(
                {
                    "kind": "narrative_parent_after_quote",
                    "from": [a["page"] + 1, a["index"]],
                    "to": [b["page"] + 1, b["index"]],
                }
            )
            break
    return extra, logs
