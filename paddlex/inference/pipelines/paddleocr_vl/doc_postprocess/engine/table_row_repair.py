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

"""Conservative cell reconstruction at boundaries of selected table chains."""
import re

from .structural_links import clean, is_header, sim
from .table_graph import cells, norm


def aligned_header(a, b):
    """Monotone dynamic program; every nonempty label must be explained."""
    aa = [s for s in a if clean(s)]
    bb = [s for s in b if clean(s)]
    if min(len(aa), len(bb)) < 2 or not is_header(aa) or not is_header(bb):
        return False
    # Numeric labels identify different years/runs; visual similarity alone
    # must never erase those differences while removing repeated headers.
    if [re.findall(r"\d+(?:\.\d+)?", s) for s in aa] != [
        re.findall(r"\d+(?:\.\d+)?", s) for s in bb
    ]:
        return False
    n, m = len(aa), len(bb)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            s = sim(aa[i], bb[j])
            dp[i + 1][j + 1] = max(
                dp[i][j + 1], dp[i + 1][j], dp[i][j] + (s if s > 0.65 else 0)
            )
    return dp[n][m] / max(n, m) > (0.92 if n == m and n >= 4 else 0.955)


def duplicate_column(part, other):
    rows = part.find_all("tr")
    nr = other.find_all("tr")
    if len(rows) < 6 or not nr:
        return False
    cc = [cells(r) for r in rows]
    nn = cells(nr[0])
    n = len(cc[0])
    if n != len(nn) + 1 or any(
        len(r) != n
        or any(
            int(c.get("rowspan", 1)) != 1 or int(c.get("colspan", 1)) != 1 for c in r
        )
        for r in cc
    ):
        return False
    for i in range(n - 1):
        vals = [(norm(r[i].get_text()), norm(r[i + 1].get_text())) for r in cc]
        if not vals[0][0] or vals[0][0] != vals[0][1]:
            continue
        if len({a for a, b in vals[1:] if a}) < 3 or sum(
            a == b for a, b in vals
        ) < 0.98 * len(vals):
            continue
        head = [c.get_text(" ", strip=True) for j, c in enumerate(cc[0]) if j != i + 1]
        if not aligned_header(head, [c.get_text(" ", strip=True) for c in nn]):
            continue
        for row in cc:
            row[i + 1].extract()
        return True
    return False


def repair(chain):
    from bs4 import BeautifulSoup

    if len(chain) < 2:
        return chain, []
    soups = [BeautifulSoup(f["html"], "html.parser") for f in chain]
    events = []
    for i, soup in enumerate(soups):
        for j in [i - 1, i + 1]:
            if 0 <= j < len(soups) and duplicate_column(soup, soups[j]):
                events.append(
                    {"kind": "duplicated_column_removed", "page": chain[i]["page"] + 1}
                )
                break
    for i in range(1, len(soups)):
        prev = soups[i - 1].find_all("tr")
        rows = soups[i].find_all("tr")
        if not prev or not rows:
            continue
        first = soups[0].find_all("tr")[0]
        ha = [c.get_text(" ", strip=True) for c in cells(first)]
        hb = [c.get_text(" ", strip=True) for c in cells(rows[0])]
        skip = 0
        if [norm(s) for s in ha] == [norm(s) for s in hb]:
            skip = 1
        elif aligned_header(ha, hb):
            rows[0].extract()
            rows = rows[1:]
            events.append(
                {
                    "kind": "aligned_repeated_header_removed",
                    "page": chain[i]["page"] + 1,
                }
            )
        if len(rows) <= skip:
            continue
        a, b = cells(prev[-1]), cells(rows[skip])
        if (
            len(a) != 2
            or len(b) != 2
            or any(c.get("rowspan") or c.get("colspan") for c in a + b)
        ):
            continue
        aa = [c.get_text(" ", strip=True) for c in a]
        bb = [c.get_text(" ", strip=True) for c in b]
        if not aa[0] or len(aa[1]) < 50 or len(bb[1]) < 30:
            continue
        continued = re.search(
            r"\s*[（(]\s*(?:cont(?:inued)?\.?|续|續)\s*[）)]\s*$", bb[0], re.I
        )
        explicit = bool(continued and clean(bb[0][: continued.start()]) == clean(aa[0]))
        incomplete = (
            not bb[0]
            and bool(re.match(r"^[a-z]", bb[1]))
            and not re.search(r"[.!?。！？;；:]\s*$", aa[1])
        )
        if not (explicit or incomplete):
            continue
        a[1].append(" ")
        for child in list(b[1].contents):
            a[1].append(child.extract())
        rows[skip].extract()
        events.append(
            {
                "kind": "split_row_rejoined",
                "pages": [chain[i - 1]["page"] + 1, chain[i]["page"] + 1],
                "explicit_continuation": explicit,
            }
        )
    if not events:
        return chain, []
    revised = []
    for f, soup in zip(chain, soups):
        q = dict(f)
        q["html"] = str(soup)
        revised.append(q)
    return revised, events
