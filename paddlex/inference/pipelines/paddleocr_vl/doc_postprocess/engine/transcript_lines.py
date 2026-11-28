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

"""Recover the leading continuation using monospace line lengths and indent geometry."""
import copy
import re
import statistics

import numpy as np

from .text_graph import SPEAKER, TEXT


def split_prefix(text, v):
    s = re.sub(r"\s+", " ", text).strip()
    lines = v.get("bounds", [])
    n = len(lines)
    if n < 8 or len(s) < 250 or v.get("rotated") or not re.match(r"^[a-z]", s):
        return None
    font = v.get("font", 1)
    left = [a[0] for a in lines]
    median = statistics.median(left)
    indent = [i for i, a in enumerate(lines) if i > 0 and a[0] > median + 1.5 * font]
    if not indent:
        return None
    row = indent[0]
    if row >= n - 1:
        return None
    widths = np.array([a[1] - a[0] for a in lines], float)
    # A fixed-pitch transcript contains n-1 OCR line joins represented as spaces.
    pitch = widths.sum() / max(len(s) - (n - 1), 1)
    if not 0.4 * font < pitch < 1.6 * font:
        return None
    targets = widths / pitch
    positions = [0] + [m.end() for m in re.finditer(r"\s+", s)] + [len(s)]
    positions = sorted(set(positions))
    m = len(positions)
    cost = np.full((n + 1, m), np.inf)
    back = np.full((n + 1, m), -1, int)
    cost[0, 0] = 0
    from .runtime_context import backend

    if backend.get() == "cuda":
        from .gpu_transcript import solve

        cost, back = solve(
            positions, [int(p > 0 and s[p - 1].isspace()) for p in positions], targets
        )
    else:
        for k in range(1, n + 1):
            target = targets[k - 1]
            for end in range(1, m):
                for start in range(end - 1, -1, -1):
                    length = len(s[positions[start] : positions[end]].strip())
                    if length > target + 18:
                        break
                    if length < max(2, target - 18) or not np.isfinite(
                        cost[k - 1, start]
                    ):
                        continue
                    value = cost[k - 1, start] + (length - target) ** 2
                    if value < cost[k, end]:
                        cost[k, end] = value
                        back[k, end] = start
    if not np.isfinite(cost[n, -1]) or cost[n, -1] / n > 15:
        return None
    cuts = [m - 1]
    j = m - 1
    for k in range(n, 0, -1):
        j = int(back[k, j])
        cuts.append(j)
    cuts = cuts[::-1]
    cut = positions[cuts[row]]
    if cut < 30 or len(s) - cut < 50:
        return None
    # Indented transcript paragraphs normally begin a sentence or speaker turn.
    if not re.match(r'^[A-Z"“]', s[cut:]):
        return None
    return (
        s[:cut].rstrip(),
        s[cut:].lstrip(),
        {"line": row + 1, "fit_error": float(cost[n, -1] / n), "pitch": pitch},
    )


def apply(pre, final, visual):
    count = sum(
        bool(SPEAKER.search(b["block_content"]))
        for p in pre
        for b in p["parsing_res_list"]
    )
    if count < max(5, len(pre) * 0.3):
        return []
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        ids = [
            i
            for i, b in enumerate(p["parsing_res_list"])
            if b["block_label"] in TEXT
            and q["parsing_res_list"][i]["block_label"] in TEXT
            and b["block_content"].strip()
        ]
        if not ids:
            continue
        first = ids[0]
        box = p["parsing_res_list"][first]["block_bbox"]
        keys = [i for i in ids if p["parsing_res_list"][i]["block_bbox"] == box]
        text = " ".join(p["parsing_res_list"][i]["block_content"] for i in keys)
        v = visual.get(f"{pi}:{first}", {})
        cut = split_prefix(text, v)
        if cut is None:
            continue
        a, b, meta = cut
        for page in [p, q]:
            original = page["parsing_res_list"][first]
            prefix = copy.deepcopy(original)
            prefix.update(block_content=a, _leading_continuation=True)
            body = copy.deepcopy(original)
            body.update(block_content=b)
            body.pop("_leading_continuation", None)
            output = []
            for i, old in enumerate(page["parsing_res_list"]):
                if i == first:
                    output.extend([prefix, body])
                elif i not in keys:
                    output.append(old)
            page["parsing_res_list"] = output
        logs.append(
            {
                "kind": "transcript_indent_prefix",
                "page": pi + 1,
                "prefix_characters": len(a),
                **meta,
            }
        )
    return logs
