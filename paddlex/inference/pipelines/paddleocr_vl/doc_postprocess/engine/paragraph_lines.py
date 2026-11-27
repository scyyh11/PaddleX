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

"""Split OCR blocks at visible paragraph indents using glyph-width alignment DP."""
import collections
import copy
import re
import statistics

import numpy as np

from .text_graph import TEXT


def glyph(c):
    if "\u4e00" <= c <= "\u9fff":
        return 1.0
    if c.isspace():
        return 0.3
    if c in "ilIjtfr.,;:!'|":
        return 0.28
    if c in "mwMW@":
        return 0.8
    if c.isupper():
        return 0.65
    if c.isalpha() or c.isdigit():
        return 0.5
    return 0.6


def split(s, v, strict=True):
    if "$" in s or "<" in s or "\\" in s or len(s) < 150 or len(s) > 6000:
        return None
    bounds = v.get("bounds", [])
    font = v.get("font", 0)
    n = len(bounds)
    if not font or n < 5 or n > 65 or v.get("rotated"):
        return None
    # Both a real indent and a short previous line are required; wide leading whitespace alone is insufficient.
    left = statistics.median(a[0] for a in bounds)
    right = max(a[1] for a in bounds)
    candidates = [
        i
        for i in range(1, n - 1)
        if bounds[i][0] > left + font * 1.45 and bounds[i - 1][1] < right - font * 2.0
    ]
    if not candidates:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    prefix = np.r_[0.0, np.cumsum([glyph(c) for c in s])]
    widths = np.array([b[1] - b[0] for b in bounds], float)
    scale = widths.sum() / max(prefix[-1], 1)
    if not 0.6 * font < scale < 2.0 * font:
        return None
    targets = widths / scale
    positions = (
        [0]
        + [
            i
            for i in range(1, len(s))
            if s[i - 1].isspace()
            or "\u4e00" <= s[i] <= "\u9fff"
            or "\u4e00" <= s[i - 1] <= "\u9fff"
        ]
        + [len(s)]
    )
    pos = np.array(positions)
    pref = prefix[pos]
    m = len(pos)
    cost = np.full((n + 1, m), np.inf)
    back = np.full((n + 1, m), -1, int)
    cost[0, 0] = 0
    from .runtime_context import backend

    if backend.get() == "cuda":
        from .gpu_paragraph import solve

        cost, back = solve(pref, targets)
    else:
        for row in range(1, n + 1):
            target = targets[row - 1]
            for end in range(1, m):
                lo = np.searchsorted(pref, pref[end] - target - 4)
                hi = min(
                    end,
                    np.searchsorted(pref, pref[end] - max(1, target - 4), side="right"),
                )
                if hi <= lo:
                    continue
                vals = cost[row - 1, lo:hi] + (pref[end] - pref[lo:hi] - target) ** 2
                k = int(np.argmin(vals))
                value = vals[k]
                if np.isfinite(value):
                    cost[row, end] = value
                    back[row, end] = lo + k
    if not np.isfinite(cost[n, -1]) or cost[n, -1] / n > (2 if strict else 5):
        return None
    cuts = [m - 1]
    end = m - 1
    for row in range(n, 0, -1):
        end = int(back[row, end])
        cuts.append(end)
    cuts = cuts[::-1]
    chosen = []
    for row in candidates:
        at = int(pos[cuts[row]])
        before = s[:at].rstrip()
        after = s[at:].lstrip()
        if not re.search(r'[。.!?！？][”’"\'）)]*$', before):
            continue
        if not re.match(r'^[\u4e00-\u9fffA-Z“「『"(（]', after):
            continue
        if at < 35 or len(s) - at < 35:
            continue
        chosen.append((row, at))
    if not chosen:
        return None
    return s, chosen, {"fit": float(cost[n, -1] / n), "bounds": bounds}


def apply(pre, final, visual, strict=True):
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        groups = collections.Counter(
            b.get("group_id", b.get("block_id", i))
            for i, b in enumerate(p["parsing_res_list"])
        )
        aa = []
        bb = []
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            key = a.get("_visual_key", f"{pi}:{bi}")
            v = visual.get(key, {})
            result = None
            if (
                a["block_label"] in TEXT
                and b["block_label"] in TEXT
                and not a.get("_text_region")
                and not a.get("_code_region")
                and not b.get("_leading_continuation")
                and groups[a.get("group_id", a.get("block_id", bi))] == 1
            ):
                result = split(a["block_content"], v, strict)
            if result is None:
                aa.append(a)
                bb.append(b)
                continue
            s, cuts, meta = result
            boundaries = [(0, 0)] + cuts + [(len(meta["bounds"]), len(s))]
            for k, ((first, start), (end, stop)) in enumerate(
                zip(boundaries, boundaries[1:])
            ):
                segment = s[start:stop].strip()
                vv = copy.deepcopy(v)
                bounds = meta["bounds"][first:end]
                vv.update(
                    bounds=bounds,
                    lines=len(bounds),
                    first_left=(bounds[0][0] - min(x[0] for x in bounds)) / v["font"],
                    last_right=(max(x[1] for x in bounds) - bounds[-1][1]) / v["font"],
                    extent=sum(x[1] - x[0] for x in bounds) / v["font"],
                )
                vk = f"{key}:segment:{k}"
                visual[vk] = vv
                for old, dest in [(a, aa), (b, bb)]:
                    c = copy.deepcopy(old)
                    x, y, r, bot = c["block_bbox"]
                    c.update(
                        block_content=segment,
                        _visual_key=vk,
                        group_id=f"paragraph-{pi}-{bi}-{k}",
                        global_group_id=f"paragraph-{pi}-{bi}-{k}",
                        global_block_id=f"paragraph-{pi}-{bi}-{k}",
                        block_bbox=[
                            x,
                            y + (bot - y) * first / len(meta["bounds"]),
                            r,
                            y + (bot - y) * end / len(meta["bounds"]),
                        ],
                    )
                    dest.append(c)
            logs.append(
                {
                    "kind": "visual_paragraph_split",
                    "page": pi + 1,
                    "index": bi,
                    "segments": len(boundaries) - 1,
                    "fit_error": meta["fit"],
                }
            )
        p["parsing_res_list"] = aa
        q["parsing_res_list"] = bb
    return logs
