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

"""Keep header/footer graphics when a document has no other graphics.

Raw PaddleOCR-VL hides header_image/footer_image blocks from Markdown. When a
document contains no body image, chart or seal at all, its marginal logos are
the only pictures in the file, so they are emitted as ordinary image blocks.
The decision uses only the predicted pages."""
VISUAL = {"image", "chart", "seal"}
MARGINAL = {"header_image", "footer_image"}


def apply(final):
    if any(
        b.get("block_label") in VISUAL for p in final for b in p["parsing_res_list"]
    ):
        return []
    events = []
    for pi, p in enumerate(final):
        for bi, b in enumerate(p["parsing_res_list"]):
            if b.get("block_label") in MARGINAL:
                events.append(
                    {
                        "kind": "marginal_figure_kept",
                        "page": pi + 1,
                        "index": bi,
                        "label": b["block_label"],
                    }
                )
                b["block_label"] = "image"
                if not isinstance(b.get("image"), dict):
                    b["image"] = {"path": f"page{pi+1}_marginal_{bi}.jpg", "img": None}
    return events
