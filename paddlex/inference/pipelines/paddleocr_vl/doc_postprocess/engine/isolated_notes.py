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

"""Retain notes in output while excluding them from narrative continuation tracks."""
import re
import statistics

from .text_graph import TEXT


def apply(pre, final, visual):
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        fonts = [
            visual.get(f"{pi}:{i}", {}).get("font", 0)
            for i, b in enumerate(p["parsing_res_list"])
            if b["block_label"] in TEXT and len(b["block_content"]) > 100
        ]
        fonts = [z for z in fonts if z]
        median = statistics.median(fonts) if fonts else 0
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            if a["block_label"] not in TEXT or b["block_label"] not in TEXT:
                continue
            s = a["block_content"].strip()
            v = visual.get(f"{pi}:{bi}", {})
            x, y, r, bot = a["block_bbox"]
            font = v.get("font", 0)
            explicit = bool(
                re.match(r"^(?:Source\s*:|Notes?\s*:|[注註]\s*[:：])", s, re.I)
            )
            numbered = bool(re.match(r"^\d{1,2}/\s+", s)) and y / p["height"] > 0.6
            small = (
                median
                and font
                and font < 0.79 * median
                and y / p["height"] > 0.68
                and (bot - y) / p["height"] < 0.16
            )
            if numbered or explicit and median and font and font < 0.9 * median:
                a["_isolated_note"] = b["_isolated_note"] = True
                logs.append(
                    {"kind": "isolated_note_track", "page": pi + 1, "index": bi}
                )
    return logs
