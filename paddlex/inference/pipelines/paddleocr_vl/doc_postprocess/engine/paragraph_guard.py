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

"""Preserve a full new English paragraph after a closed parenthetical sentence."""
import re

from . import edge_gate
from . import runtime_context as ctx

START = re.compile(
    r"^(?:The|This|These|Those|A|An|In|For|However|Moreover|Although|When|According|Each|Another)\s"
)


def apply(nodes, visible, fixed):
    cfg = ctx.get_config()
    kept = []
    logs = []
    for i, j in fixed:
        a, b = nodes[i], nodes[j]
        x, y = a["text"].strip(), b["text"].strip()
        gap = b["page"] - a["page"]
        mid = visible[a["vindex"] + 1 : b["vindex"]]
        cut = (
            not edge_gate.protected(a, b)
            and gap == 0
            and len(x) > 70
            and len(y) > 80
            and x.endswith(")")
            and x.count("(") == x.count(")")
            and START.match(y)
            and b["bbox"][1] - a["bbox"][3] > 0.06
            and mid
            and all(
                l in {"image", "chart", "figure_title", "vision_footnote"}
                for _, _, l in mid
            )
            and sum(c.isascii() and c.isalpha() for c in x) > len(x) * 0.5
        )
        af, bf = a["visual"].get("font", 0), b["visual"].get("font", 0)
        fontcut = (
            cfg.font_barrier
            and not edge_gate.protected(a, b)
            and gap == 0
            and af
            and bf
            and min(af, bf) / max(af, bf) < 0.65
            and a["visual"].get("lines", 0) >= 1
            and b["visual"].get("lines", 0) >= 3
            and re.search(r"[.!?][)\]\"\x27]*$", x)
            and re.match(r"^[a-z]", y)
            and b["bbox"][1] - a["bbox"][3] > 0.05
            and mid
            and all(
                l
                in {
                    "image",
                    "chart",
                    "figure_title",
                    "vision_footnote",
                    "table",
                    "display_formula",
                    "formula_number",
                }
                for _, _, l in mid
            )
        )
        indentcut = (
            cfg.indent_barrier
            and not edge_gate.protected(a, b)
            and gap == 0
            and re.search(r"[.!?][)\]\"\x27]*$", x)
            and re.match(r"^[a-z]", y)
            and b["visual"].get("first_left", 0) > 1.8
            and b["visual"].get("lines", 0) >= 3
            and b["bbox"][1] - a["bbox"][3] > 0.04
            and mid
            and all(
                l
                in {
                    "image",
                    "chart",
                    "figure_title",
                    "vision_footnote",
                    "table",
                    "display_formula",
                    "formula_number",
                }
                for _, _, l in mid
            )
        )
        cut = cut or fontcut or indentcut
        if cut:
            logs.append(
                {
                    "kind": "closed_parenthetical_paragraph",
                    "from": [a["page"] + 1, a["index"]],
                    "to": [b["page"] + 1, b["index"]],
                }
            )
        else:
            kept.append((i, j))
    return kept, logs
