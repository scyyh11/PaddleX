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

"""Monotone column profile alignment for sparse continuation fragments."""
import math
import re

import numpy as np


def kind(s):
    s = re.sub(r"\s+", "", s)
    if not s:
        return 0
    if re.fullmatch(r"[\d.,，%％()（）+−\-—/]+", s):
        return 2 if "%" in s or "％" in s else 1
    if s.lower() in {"y", "n", "yes", "no", "是", "否"}:
        return 3
    return 4 if len(s) < 35 else 5


def profiles(tab):
    if "_column_profiles" in tab:
        return tab["_column_profiles"]
    width = tab["ncols"]
    out = np.zeros((width, 7), dtype=float)
    for row in tab["grid"][-25:]:
        for col, cs, rs, s in row:
            if cs != 1:
                continue
            out[col, kind(s)] += 1
            out[col, 6] += math.log1p(len(s))
    for row in out:
        n = max(1, row[:6].sum())
        row[:6] /= n
        row[6] /= n * 5
    tab["_column_profiles"] = out
    return out


def align(a, b):
    """Same DP as _align_reference.  The pair costs float(np.abs(x[i]-y[j]).sum())*.5 (times .4 for
    near-empty columns) are computed for all (i, j) at once: each sums the same 7 non-negative values
    left to right, so the doubles are identical; the DP itself runs on Python floats in the same order.
    """
    x, y = profiles(a), profiles(b)
    n, m = len(x), len(y)
    if n and m:
        c = np.abs(x[:, None, :] - y[None, :, :]).sum(axis=2) * 0.5
        c = np.where(
            (x[:, 0] > 0.9)[:, None] | (y[:, 0] > 0.9)[None, :], c * 0.4, c
        ).tolist()
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    back = {}
    for i in range(n + 1):
        row = dp[i]
        nxt = dp[i + 1] if i < n else None
        ci = c[i] if i < n and m else None
        for j in range(m + 1):
            d = row[j]
            if i < n and j < m:
                value = d + ci[j]
                if value < nxt[j + 1]:
                    nxt[j + 1] = value
                    back[i + 1, j + 1] = (i, j)
            if i < n and d + 0.65 < nxt[j]:
                nxt[j] = d + 0.65
                back[i + 1, j] = (i, j)
            if j < m and d + 0.65 < row[j + 1]:
                row[j + 1] = d + 0.65
                back[i, j + 1] = (i, j)
    pairs = []
    at = (n, m)
    while at != (0, 0):
        previous = back[at]
        if at[0] > previous[0] and at[1] > previous[1]:
            pairs.append(previous)
        at = previous
    out = (float(dp[n][m] / max(n, m, 1)), list(reversed(pairs)))
    if VERIFY and out != _align_reference(a, b):
        raise AssertionError("column_alignment: alignment differs")
    return out


VERIFY = False


def _align_reference(a, b):
    x, y = profiles(a), profiles(b)
    n, m = len(x), len(y)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0
    back = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and j < m:
                cost = float(np.abs(x[i] - y[j]).sum()) * 0.5
                # Empty-only columns carry little evidence; permit insertion.
                if x[i, 0] > 0.9 or y[j, 0] > 0.9:
                    cost *= 0.4
                value = dp[i, j] + cost
                if value < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = value
                    back[i + 1, j + 1] = (i, j)
            if i < n and dp[i, j] + 0.65 < dp[i + 1, j]:
                dp[i + 1, j] = dp[i, j] + 0.65
                back[i + 1, j] = (i, j)
            if j < m and dp[i, j] + 0.65 < dp[i, j + 1]:
                dp[i, j + 1] = dp[i, j] + 0.65
                back[i, j + 1] = (i, j)
    pairs = []
    at = (n, m)
    while at != (0, 0):
        previous = back[at]
        if at[0] > previous[0] and at[1] > previous[1]:
            pairs.append(previous)
        at = previous
    return float(dp[n, m] / max(n, m, 1)), list(reversed(pairs))


def continuation(a, b):
    ta, tb = a["tab"], b["tab"]
    n, m = ta["ncols"], tb["ncols"]
    if not (
        a["page"] + 1 == b["page"]
        and a["rank"] == a["tables_on_page"] - 1
        and b["rank"] == 0
    ):
        return False, None
    if not (3 <= min(n, m) <= max(n, m) <= 20 and abs(n - m) <= 3):
        return False, None
    if a.get("body_after", a["after"]) or any(
        q["block_label"] not in {"table_title"}
        for q in b.get("body_before", b["before"])
    ):
        return False, None
    ba, bb = a["bbox"], b["bbox"]
    wa = ba[2] - ba[0]
    wb = bb[2] - bb[0]
    if abs(wa - wb) / max(wa, wb, 0.01) > 0.13 or abs(ba[0] - bb[0]) > 0.07:
        return False, None
    if ba[3] < a["doc_bottom"] - 0.08 or bb[1] > b["doc_top"] + 0.08:
        return False, None
    first = tb["rows"][0]
    last = ta["rows"][-1]
    if last and re.search(
        r"^(?:合计|合計|总计|總計|total|grand total)\s*$", last[0], re.I
    ):
        return False, None
    # Continuation starts with data or a sparsely populated carry-over row.
    nonempty = sum(bool(s.strip()) for s in first)
    numeric = sum(kind(s) in {1, 2} for s in first)
    if not (numeric >= 2 or nonempty <= 2 and len(first) >= 3):
        return False, None
    cost, pairs = align(ta, tb)
    strict = (
        abs(wa - wb) / max(wa, wb, 0.01) < 0.025
        and abs(ba[0] - bb[0]) < 0.015
        and ba[3] >= a["doc_bottom"] - 0.025
        and bb[1] <= b["doc_top"] + 0.025
    )
    limit = 0.3
    if strict:
        limit = (
            0.7
            if nonempty <= 2 and len(first) >= 4
            else 0.45 if nonempty <= 2 else 0.36
        )
    return cost < limit and len(pairs) >= min(n, m) - 1, {
        "cost": cost,
        "mapping": pairs,
        "columns": [n, m],
    }
