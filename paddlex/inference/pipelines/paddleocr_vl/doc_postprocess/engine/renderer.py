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

"""Render post-processed page dicts to document Markdown."""


def render(pages):
    from ...pipeline import _basic_restructure_pages

    for pi, p in enumerate(pages):
        p.update(page_index=pi, input_path=f"page_{pi + 1}.png")
    original = [
        {"parsing_res_list": [dict(b) for b in p["parsing_res_list"]]} for p in pages
    ]
    results = list(
        _basic_restructure_pages(
            [{"res": dict(p)} for p in pages],
            merge_tables=False,
            relevel_titles=False,
            concatenate_pages=False,
        )
    )
    parts = []
    for pi, (page, saved) in enumerate(zip(results, original), 1):
        for b, s in zip(page["parsing_res_list"], saved["parsing_res_list"]):
            b.content = s["block_content"]
            for key in ["global_block_id", "global_group_id", "title_level"]:
                if key in s:
                    setattr(b, key, s[key])
            if getattr(b, "image", None):
                box = [
                    max(0, min(1000, round(float(v) / size * 1000)))
                    for v, size in zip(
                        b.bbox,
                        [page["width"], page["height"], page["width"], page["height"]],
                    )
                ]
                b.image["path"] = (
                    f"page{pi}_" + ("_".join(str(v) for v in box)) + ".jpg"
                )
        parts.append(page._to_markdown(pretty=False)["markdown_texts"])
    return "\n\n".join(parts) + "\n"
