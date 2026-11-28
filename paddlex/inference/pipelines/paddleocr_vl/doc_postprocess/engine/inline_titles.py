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

"""Separate explicit short bilingual headings embedded in a prose OCR block."""
import copy
import re

from .text_graph import TEXT

TITLE = re.compile(
    r"^\s*(?:[①-⑳\d一二三四五六七八九十().、． ]*)?[\u4e00-\u9fff·・ ]{1,18}\s*[(（][A-Za-z][A-Za-z ,\-]{2,65}[)）]\s*$"
)


def apply(pre, final):
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        aa = []
        bb = []
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            parts = a["block_content"].strip().split("\n", 1)
            if (
                a["block_label"] not in TEXT
                or b["block_label"] not in TEXT
                or len(parts) != 2
                or not TITLE.fullmatch(parts[0])
                or len(parts[1]) < 30
                or a.get("_text_region")
                or a.get("_code_region")
            ):
                aa.append(a)
                bb.append(b)
                continue
            for k, s in enumerate(parts):
                for old, dest in [(a, aa), (b, bb)]:
                    c = copy.deepcopy(old)
                    c.update(
                        block_content=s.strip(),
                        block_label="paragraph_title" if k == 0 else "text",
                        group_id=f"inline-{pi}-{bi}-{k}",
                        global_group_id=f"inline-{pi}-{bi}-{k}",
                        global_block_id=f"inline-{pi}-{bi}-{k}",
                    )
                    dest.append(c)
            logs.append(
                {"kind": "embedded_bilingual_heading", "page": pi + 1, "index": bi}
            )
        p["parsing_res_list"] = aa
        q["parsing_res_list"] = bb
    return logs
