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

"""Unclosed quotation delimiters constrain adjacent continuation paths."""
from . import edge_gate, layout_text
from .quote_paths import FLOAT, cjk


def apply(nodes, visible, fixed):
    edges = list(fixed)
    out = {i for i, j in edges}
    inc = {j for i, j in edges}
    logs = []
    for a, b in zip(nodes, nodes[1:]):
        i, j = a["id"], b["id"]
        x, y = a["text"].strip(), b["text"].strip()
        gap = b["page"] - a["page"]
        av, bv = a["visual"], b["visual"]
        if (
            i in out
            or j in inc
            or gap not in {0, 1}
            or edge_gate.protected(a, b)
            or not cjk(x)
            or not cjk(y)
        ):
            continue
        if (
            len(x) < 15
            or len(y) < 10
            or layout_text.new_item(y)
            or b["block"].get("_explicit_paragraph_start")
        ):
            continue
        if any(l not in FLOAT for _, _, l in visible[a["vindex"] + 1 : b["vindex"]]):
            continue
        af, bf = av.get("font", 0), bv.get("font", 0)
        if not af or not bf or not 0.75 < af / bf < 1.33:
            continue
        balanced = any(
            x.count(left) > x.count(right) and right in y[:150]
            for left, right in [("“", "”"), ("「", "」"), ("『", "』")]
        )
        same = abs(a["bbox"][0] - b["bbox"][0]) < 0.1 and (
            gap == 1 or -0.01 < b["bbox"][1] - a["bbox"][3] < 0.045
        )
        ellipsis = (
            gap == 1
            and a["last"]
            and b["first"]
            and "…" in x
            and y.startswith("…")
            and av.get("last_right", 9) < 1
            and bv.get("first_left", 9) < 1
            and abs(a["bbox"][2] - b["bbox"][2]) < 0.08
        )
        if not (balanced and same) and not ellipsis:
            continue
        if any(i < k < j < l or k < i < l < j for k, l in edges):
            continue
        edges.append((i, j))
        out.add(i)
        inc.add(j)
        logs.append(
            {
                "kind": (
                    "quotation_delimiter_continuation"
                    if balanced
                    else "ellipsis_quote_continuation"
                ),
                "from": [a["page"] + 1, a["index"]],
                "to": [b["page"] + 1, b["index"]],
            }
        )
    return edges, logs
