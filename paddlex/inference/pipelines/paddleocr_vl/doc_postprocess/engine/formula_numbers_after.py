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

"""Render detected equation numbers as text placed right after their equation.

PaddleOCR-VL detects equation numbers as formula_number blocks, but its Markdown
writer drops them. Each number is moved behind the display formula it shares a
line with and kept as a short text line, so the equation itself stays the first
element of the pair."""


def _overlap(a, b):
    return max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(
        1, min(a[3] - a[1], b[3] - b[1])
    )


def apply(final):
    events = []
    for pi, p in enumerate(final):
        blocks = p["parsing_res_list"]
        nums = [
            b
            for b in blocks
            if b.get("block_label") == "formula_number"
            and (b.get("block_content") or "").strip()
        ]
        if not nums:
            continue
        rest = [b for b in blocks if b not in nums]
        for n in nums:
            best = None
            bv = 0.3
            for i, b in enumerate(rest):
                if b.get("block_label") in ("display_formula", "formula"):
                    v = _overlap(n["block_bbox"], b["block_bbox"])
                    if v > bv:
                        bv = v
                        best = i
            n["block_label"] = "text"
            n["block_content"] = n["block_content"].strip()
            if best is None:
                # keep original relative position: insert before the first block that followed it
                orig = blocks.index(n)
                after = next((b for b in blocks[orig + 1 :] if b in rest), None)
                rest.insert(rest.index(after) if after is not None else len(rest), n)
                events.append(
                    {
                        "kind": "formula_number_text",
                        "page": pi + 1,
                        "placed": "original",
                    }
                )
            else:
                rest.insert(best + 1, n)
                events.append(
                    {
                        "kind": "formula_number_text",
                        "page": pi + 1,
                        "placed": "after_formula",
                    }
                )
        p["parsing_res_list"] = rest
    return events
