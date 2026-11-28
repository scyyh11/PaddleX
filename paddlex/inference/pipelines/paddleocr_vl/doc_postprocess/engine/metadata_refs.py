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

"""Keep a biographical date record with its printed cross-reference across columns."""
import re

from .text_graph import join

DATE = re.compile(r"^\d{3,4}\s*[-~～–—－]\s*\d{0,4}\s*[A-Za-z\u4e00-\u9fff]{1,20}\s*$")
REF = re.compile(r"^(?:→|⇒)\s*(?:[①-⑳\d]+[\s·・.,、/～~–—－\-]*)+$")


def apply(pre, final):
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        bs = p["parsing_res_list"]
        seq = [
            (i, b)
            for i, b in enumerate(bs)
            if b["block_content"].strip()
            and b["block_label"]
            not in {"number", "header", "footer", "footnote", "aside_text"}
        ]
        for (ai, a), (bi, b) in zip(seq, seq[1:]):
            x, y = a["block_content"].strip(), b["block_content"].strip()
            if (
                a["block_label"] not in {"text", "figure_title", "vision_footnote"}
                or b["block_label"] not in {"text", "figure_title", "vision_footnote"}
                or not DATE.fullmatch(x)
                or not REF.fullmatch(y)
            ):
                continue
            ax, ay, ar, ab = a["block_bbox"]
            bx, by, br, bb = b["block_bbox"]
            if bx < ax + p["width"] * 0.15 or by > ab - p["height"] * 0.15:
                continue
            for page in [p, q]:
                page["parsing_res_list"][ai].update(
                    block_content=join(x, y), _text_region="biographic_reference"
                )
                page["parsing_res_list"][bi].update(
                    block_content="", _text_region="biographic_reference"
                )
            logs.append(
                {"kind": "biographic_reference", "page": pi + 1, "from": ai, "to": bi}
            )
    return logs
