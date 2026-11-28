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

"""Keep recurring marginal strings out of the narrative continuation graph."""
import collections
import re


def apply(pre, final):
    groups = collections.defaultdict(list)
    for pi, p in enumerate(pre):
        for bi, b in enumerate(p["parsing_res_list"]):
            s = b["block_content"].strip()
            lab = b["block_label"]
            x, y, r, bot = [
                v / z
                for v, z in zip(
                    b["block_bbox"], [p["width"], p["height"], p["width"], p["height"]]
                )
            ]
            if (
                lab not in {"text", "paragraph_title", "doc_title", "figure_title"}
                or not 2 <= len(s) < 70
                or not (y < 0.16 or y > 0.82)
            ):
                continue
            key = re.sub(r"[\W\d_]+", "", s).lower()
            if len(key) < 2:
                continue
            groups[key].append((pi, bi, y, x))
    logs = []
    for key, items in groups.items():
        pages = {p for p, b, y, x in items}
        if len(pages) < 3:
            continue
        for pi, bi, y, x in items:
            # Repetition near the same page edge distinguishes a running title from a repeated body phrase.
            peers = {pp for pp, bb, yy, xx in items if abs(y - yy) < 0.055}
            if len(peers) < 3:
                continue
            for doc in (pre, final):
                doc[pi]["parsing_res_list"][bi]["_isolated_note"] = True
            logs.append(
                {
                    "kind": "repeated_marginal_marker",
                    "page": pi + 1,
                    "index": bi,
                    "repeated_pages": len(peers),
                }
            )
    return logs
