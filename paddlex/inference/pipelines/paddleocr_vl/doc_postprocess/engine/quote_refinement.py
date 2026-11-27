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

"""Direct speech introductions are attached when no narrative-resume path exists."""
import re

from . import layout_text
from .quote_paths import FLOAT, cjk

SAY = re.compile(
    r"(?:曰|[说說][道過过]?|[记記载載]曰|云|奏[称稱]|[言谓謂]之?曰?)[:：]$"
)


def apply(nodes, visible, pages, fixed):
    out = {i for i, j in fixed}
    inc = {j for i, j in fixed}
    extra = []
    logs = []
    for a, b in zip(nodes, nodes[1:]):
        i, j = a["id"], b["id"]
        x = layout_text.ending(a["text"])
        y = b["text"].strip()
        gap = b["page"] - a["page"]
        ax, ay, ar, ab = a["bbox"]
        bx, by, br, bb = b["bbox"]
        av, bv = a["visual"], b["visual"]
        if i in out or j in inc or gap not in {0, 1} or not cjk(x) or not cjk(y):
            continue
        if any(
            n["transcript"]
            or n["block"].get("_code_region")
            or n["block"].get("_text_region")
            for n in [a, b]
        ):
            continue
        if any(l not in FLOAT for _, _, l in visible[a["vindex"] + 1 : b["vindex"]]):
            continue
        if layout_text.new_item(y) or b["block"].get("_explicit_paragraph_start"):
            continue
        af, bf = av.get("font", 0), bv.get("font", 0)
        if not af or not bf or not 0.8 < af / bf < 1.25:
            continue
        speech = (
            SAY.search(x)
            and (gap == 1 or -0.005 < by - ab < 0.04)
            and (abs(ar - br) < 0.08 or av.get("lines", 0) == 1 and ar - ax < br - bx)
            and abs(ax - bx) < 0.15
        )
        ellipsis = (
            gap == 1
            and a["last"]
            and b["first"]
            and "…" in x
            and y.startswith("…")
            and av.get("last_right", 9) < 1
            and bv.get("first_left", 9) < 1
            and abs(ar - br) < 0.08
        )
        if not speech and not ellipsis:
            continue
        if any(i < k < j < l or k < i < l < j for k, l in fixed + extra):
            continue
        extra.append((i, j))
        out.add(i)
        inc.add(j)
        logs.append(
            {
                "kind": (
                    "speech_introduction" if speech else "ellipsis_quote_continuation"
                ),
                "from": [a["page"] + 1, a["index"]],
                "to": [b["page"] + 1, b["index"]],
            }
        )
    return extra, logs
