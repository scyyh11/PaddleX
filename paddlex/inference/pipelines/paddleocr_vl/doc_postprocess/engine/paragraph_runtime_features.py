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

"""Features of actual post-split paragraph candidates, including source provenance."""
import numpy as np

from .paragraph_pair_features import describe, pair


def extract(nodes, pages, links):
    synthetic = [
        {"width": p["width"], "height": p["height"], "parsing_res_list": []}
        for p in pages
    ]
    for n in nodes:
        b = n["block"]
        synthetic[n["page"]]["parsing_res_list"].append(
            {
                "block_label": b["block_label"],
                "block_bbox": b["block_bbox"],
                "block_content": n["text"],
            }
        )
    rows, _, _ = describe(synthetic)
    assert len(rows) == len(nodes)

    def source(n):
        return (
            str(n["block"].get("_visual_key", ""))
            .split(":segment:")[0]
            .split(":part:")[0]
        )

    features = []
    for i, j in links:
        a, b = nodes[i], nodes[j]
        av, bv = a.get("visual", {}), b.get("visual", {})
        aa, bb = a["block"], b["block"]
        extra = [
            int(bool(source(a)) and source(a) == source(b)),
            int(
                aa.get("_source_index", -1) == bb.get("_source_index", -2)
                and a["page"] == b["page"]
            ),
            int(aa.get("_source_list_column", False)),
            int(bb.get("_source_list_column", False)),
            int(aa.get("_source_exercise_lines", False)),
            int(bb.get("_source_exercise_lines", False)),
            int(bb.get("_explicit_paragraph_start", False)),
            int(aa.get("_standalone_demoted_heading", False)),
            int(bb.get("_standalone_demoted_heading", False)),
        ]
        for v in [av, bv]:
            extra.extend(
                float(v.get(k, 0))
                for k in [
                    "font",
                    "lines",
                    "first_left",
                    "first_right",
                    "last_left",
                    "last_right",
                    "median_left",
                    "median_right",
                    "extent",
                    "rotated",
                ]
            )
        features.append(np.r_[pair(rows[i], rows[j]), extra])
    return np.asarray(features, dtype=np.float32)
