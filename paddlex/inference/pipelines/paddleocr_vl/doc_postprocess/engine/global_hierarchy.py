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

"""Document-level heading reconstruction. Runtime inputs are predictions only."""
import json
import pathlib

from . import recover_headings as h


def export_records(results, original, doc, out):
    records = []
    for pi, p in enumerate(results):
        for bi, b in enumerate(p["parsing_res_list"]):
            if (
                b.label
                not in {
                    "doc_title",
                    "paragraph_title",
                    "text",
                    "header",
                    "aside_text",
                    "figure_title",
                    "content",
                    "footer",
                }
                or not b.content.strip()
            ):
                continue
            source = original.get((pi, tuple(b.bbox)), {})
            v = getattr(b, "_visual_font", {})
            records.append(
                {
                    "page": pi + 1,
                    "index": bi,
                    "label": b.label,
                    "level": (
                        1
                        if b.label == "doc_title"
                        else (
                            getattr(b, "title_level", 1) + 1
                            if b.label == "paragraph_title"
                            else 0
                        )
                    ),
                    "text": b.content,
                    "original_label": source.get("block_label"),
                    "original_text": source.get("block_content"),
                    "box": [float(x) for x in b.bbox],
                    "width": p["width"],
                    "height": p["height"],
                    "visual": v,
                    "estimated_font": h.font(b, p["width"]),
                    "family": h.family(b.content),
                    "toc_hint": getattr(b, "_toc_level_hint", None),
                }
            )
    if out is not None:
        path = pathlib.Path(out) / "features"
        path.mkdir(exist_ok=True)
        (path / (doc + ".json")).write_text(json.dumps(records, ensure_ascii=False))
    return records


def reconstruct(results, log):
    pass
