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

"""Semantic continuation prefixes and subordinate bullet items."""
import copy
import re

from . import edge_gate, layout_text
from .paragraph_lines import glyph
from .text_graph import END


def apply(pages, nodes, visible, fixed, mode):
    logs = []
    out = {i: j for i, j in fixed}
    inc = {j: i for i, j in fixed}
    edges = list(fixed)
    if mode in {"nested", "both"}:
        for a, b in zip(nodes, nodes[1:]):
            i, j = a["id"], b["id"]
            x, y = a["text"].strip(), b["text"].strip()
            if (
                i in out
                or j in inc
                or b["page"] != a["page"] + 1
                or not a["last"]
                or not b["first"]
            ):
                continue
            if (
                edge_gate.protected(a, b)
                or not re.match(r"^[•●]\s*", x)
                or not re.match(r"^[-–]\s+[A-Z]", y)
                or END.search(layout_text.ending(x))
            ):
                continue
            if any(
                l not in {"image", "chart", "figure_title", "vision_footnote", "table"}
                for _, _, l in visible[a["vindex"] + 1 : b["vindex"]]
            ):
                continue
            af, bf = a["visual"].get("font", 0), b["visual"].get("font", 0)
            if (
                not af
                or not bf
                or not 0.8 < af / bf < 1.25
                or abs(a["bbox"][0] - b["bbox"][0]) > 0.06
            ):
                continue
            edges.append((i, j))
            out[i] = j
            inc[j] = i
            logs.append(
                {
                    "kind": "nested_bullet_continuation",
                    "from": [a["page"] + 1, a["index"]],
                    "to": [b["page"] + 1, b["index"]],
                }
            )
    if mode in {"prefix", "both"}:
        for j, i in list(inc.items()):
            a, b = nodes[i], nodes[j]
            s = b["text"]
            vv = b["visual"]
            font = vv.get("font", 0)
            width = vv.get("crop_width", 0) / max(font, 1)
            ll = [l.strip() for l in s.splitlines() if l.strip()]
            if (
                j in out
                or edge_gate.protected(a, b)
                or b["block"]["block_label"] not in {"text", "abstract"}
                or len(ll) != 2
                or not font
                or width < 8
            ):
                continue
            cross = (
                b["page"] == a["page"] + 1
                or b["page"] == a["page"]
                and b["bbox"][0] > a["bbox"][0] + 0.15
                and b["bbox"][1] < a["bbox"][3] - 0.1
            )
            if (
                not cross
                or not re.search(r'[。！？.!?][”’"）)]*$', ll[0])
                or not re.match(r"^[\u4e00-\u9fffA-Z]", ll[1])
            ):
                continue
            n1, n2 = [sum(glyph(c) for c in l) for l in ll]
            if (
                not 0.5 * width < n1 < 1.3 * width
                or n2 < 2.2 * width
                or vv.get("lines", 0) < 4
            ):
                continue
            tail = copy.deepcopy(b["block"])
            tail.update(
                block_content=ll[1],
                global_block_id=f'continued-paragraph-tail-{b["page"]}-{b["index"]}',
                global_group_id=f'continued-paragraph-tail-{b["page"]}-{b["index"]}',
            )
            b["block"]["block_content"] = ll[0]
            page = pages[b["page"]]["parsing_res_list"]
            at = next(k for k, z in enumerate(page) if z is b["block"])
            page.insert(at + 1, tail)
            logs.append(
                {
                    "kind": "semantic_continuation_prefix",
                    "page": b["page"] + 1,
                    "index": b["index"],
                    "prefix_chars": len(ll[0]),
                    "tail_chars": len(ll[1]),
                }
            )
    return edges, logs
