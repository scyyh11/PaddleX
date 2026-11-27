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

"""Preserve explicit paragraph/list boundaries already present in OCR text."""
import copy
import re

from . import layout_text
from . import runtime_context as ctx
from .paragraph_lines import glyph
from .text_graph import SPEAKER

MARK = re.compile(r"^(?:[A-Z][、.]\s*|[①-⑳]|【[^】]{1,12}】)")


def apply(pre, final, visual, mode):
    cfg = ctx.get_config()
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        aa = []
        bb = []
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            if (
                a["block_label"] not in {"text", "abstract", "reference_content"}
                or b["block_label"] not in {"text", "abstract", "reference_content"}
                or a.get("_code_region")
                or a.get("_text_region")
                or a.get("_leading_continuation")
            ):
                aa.append(a)
                bb.append(b)
                continue
            s = a["block_content"]
            if re.fullmatch(r"公元[〇一二三四五六七八九十百千零0-9]+年", s.strip()):
                a["_isolated_note"] = b["_isolated_note"] = True
            key = a.get("_visual_key", f"{pi}:{bi}")
            v = visual.get(key, {})
            line_width = v.get("crop_width", 0) / max(v.get("font", 0), 1)
            ll = list(re.finditer(r"[^\n]+", s))
            cuts = []
            marked = sum(bool(MARK.match(m[0].strip())) for m in ll)
            for left, right in zip(ll, ll[1:]):
                x, y = left[0].strip(), right[0].strip()
                reason = None
                if not x or not y:
                    continue
                if (
                    mode in {"markers", "both", "wideblank"}
                    and marked >= 2
                    and MARK.match(y)
                ):
                    reason = "explicit_list_item"
                if (
                    mode in {"long", "both", "wideblank"}
                    and re.search(r'[。.!?！？][”’"\u2019）)]*$', layout_text.ending(x))
                    and not SPEAKER.search(s)
                ):
                    blank = (
                        s[left.end() : right.start()].count("\n") >= 2
                        and len(x) > 80
                        and len(y) > 80
                    )
                    long = (
                        line_width > 8
                        and sum(glyph(c) for c in x) > line_width * 1.35
                        and sum(glyph(c) for c in y) > line_width * 0.9
                    )
                    if (blank or long) and re.match(r"^[\u4e00-\u9fffA-Z“「『]", y):
                        reason = "source_semantic_newline"
                if (
                    mode == "wideblank"
                    and s[left.end() : right.start()].count("\n") >= 2
                    and len(x) > 80
                    and len(y) > 80
                    and re.match(r"^[A-Z]", y)
                    and not re.search(r"[-,;:]$", x)
                    and not SPEAKER.search(s)
                ):
                    reason = "explicit_blank_paragraph"
                if a["block_label"] == "reference_content":
                    reason = (
                        "bibliography_entry"
                        if cfg.layout_ref_mode == "split"
                        and layout_text.reference_start(y)
                        and not re.search(r"(?:[,、&]|\band)\s*$", x)
                        else None
                    )
                if (
                    cfg.date_split
                    and left == ll[0]
                    and re.match(
                        r"^公元[〇一二三四五六七八九十百千零0-9]+年(?:\s|$)", x
                    )
                    and len(x) < 70
                    and not re.search(r"[。！？.!?]", x)
                    and len(y) > 15
                ):
                    reason = "calendar_heading_boundary"
                if reason:
                    cuts.append((right.start(), reason))
            if cfg.section_split:
                pattern = r"(?:^|\n)\s*(第[一二三四五六七八九十百0-9]+(?:部分|部|章|[节節]|篇)[^\n。！？.!?：:]{1,65}[：:])(?=\S)"
                for section in re.finditer(pattern, s):
                    title = section[1]
                    if len(s) - section.end() < 15 or not re.search(
                        r"[（(][A-Za-z][^）)]{2,55}[）)]", title
                    ):
                        continue
                    if section.start(1) > 0:
                        cuts.append((section.start(1), "inline_section_begin"))
                    cuts.append((section.end(1), "inline_section_body"))
                cuts = sorted(dict(cuts).items())
            if not cuts:
                aa.append(a)
                bb.append(b)
                continue
            bounds = [0] + [at for at, why in cuts] + [len(s)]
            rowcuts = None
            if cfg.keep_visual and len(v.get("bounds", [])) >= len(bounds) - 1:
                import numpy as np

                weights = np.r_[0.0, np.cumsum([glyph(c) for c in s])]
                rowweights = np.r_[0.0, np.cumsum([z[1] - z[0] for z in v["bounds"]])]
                rowcuts = [0]
                for index, at in enumerate(bounds[1:-1]):
                    target = weights[at] / max(weights[-1], 1) * rowweights[-1]
                    lo = rowcuts[-1] + 1
                    hi = len(rowweights) - len(bounds) + index + 1
                    rowcuts.append(
                        min(
                            range(lo, hi + 1), key=lambda n: abs(rowweights[n] - target)
                        )
                    )
                rowcuts.append(len(v["bounds"]))
            total = max(sum(glyph(c) for c in s), 1)
            offset = 0
            for k, (start, end) in enumerate(zip(bounds, bounds[1:])):
                content = s[start:end].strip()
                weight = sum(glyph(c) for c in s[start:end])
                vk = f"{key}:paragraph:{k}"
                visual[vk] = {}
                if rowcuts is not None:
                    vb = v["bounds"][rowcuts[k] : rowcuts[k + 1]]
                    vv = copy.deepcopy(v)
                    vv.update(
                        bounds=vb,
                        lines=len(vb),
                        first_left=vb[0][0] / v["font"],
                        last_right=(v["crop_width"] - vb[-1][1]) / v["font"],
                        extent=sum(z[1] - z[0] for z in vb) / v["font"],
                    )
                    visual[vk] = vv
                for old, dest in [(a, aa), (b, bb)]:
                    z = copy.deepcopy(old)
                    x, y, r, bot = z["block_bbox"]
                    ident = f"semantic-paragraph-{pi}-{bi}-{k}"
                    z.update(
                        block_content=content,
                        block_bbox=[
                            x,
                            y + (bot - y) * offset / total,
                            r,
                            y + (bot - y) * (offset + weight) / total,
                        ],
                        _visual_key=vk,
                        _explicit_paragraph_start=k > 0,
                        group_id=ident,
                        global_group_id=ident,
                        global_block_id=ident,
                    )
                    if k == 0 and cuts[0][1] == "calendar_heading_boundary":
                        z["_isolated_note"] = True
                    dest.append(z)
                offset += weight
            logs.append(
                {
                    "kind": "source_paragraph_boundaries",
                    "page": pi + 1,
                    "index": bi,
                    "cuts": cuts,
                }
            )
        p["parsing_res_list"] = aa
        q["parsing_res_list"] = bb
    return logs
