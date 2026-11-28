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

"""Structural guards for math-only lines and signed financial statement sections."""

import re

TEXT_LABELS = {"text", "abstract", "reference_content", "content", "vertical_text"}
ISOLATED_MATH = re.compile(
    r"^(?P<lead>[ \t]*)(?:\$\$(?P<double>[^\n]*?)\$\$|\$(?P<single>[^\n$]+)\$|\\\((?P<paren>[^\n]*?)\\\))(?P<trail>[ \t]*)$",
    re.M,
)


def closed_signed_statement(a, b):
    """Keep independently signed, closed financial statement sections separate."""
    if a["page"] == b["page"]:
        return False

    def clean(s):
        return re.sub(r"\s+", "", s).lower().rstrip(":：")

    ar = a["tab"]["rows"]
    br = b["tab"]["rows"]
    if not ar or not br:
        return False
    tail = clean(ar[-1][0]) if ar[-1] else ""
    head = " ".join(clean(row[0]) for row in br[:5] if row)
    signature = " ".join(q.get("block_content", "") for q in a.get("after", []))
    signed = bool(
        re.search(
            r"法定代表人|主管会计|会计机构负责人|authorized\s+signator|chief\s+financial\s+officer",
            signature,
            re.I,
        )
    )
    assets = tail in {"资产总计", "資產總計", "totalassets"}
    liabilities = bool(re.search(r"流动负债|流動負債|liabilit", head))
    return signed and assets and liabilities
