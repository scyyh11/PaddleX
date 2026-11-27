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

"""Only explicit standalone mathematical paragraphs become display math."""

import re

from .structure_guard import ISOLATED_MATH, TEXT_LABELS

MATH_EVIDENCE = re.compile(
    r"[=<>+*/^_]|\\(?:frac|dfrac|tfrac|sum|prod|int|sqrt|lim|xlongequal|xrightarrow|rightarrow|approx|equiv|leq|geq|neq)\b"
)


def apply(pages):
    logs = []
    for pi, page in enumerate(pages):
        for bi, b in enumerate(page["parsing_res_list"]):
            old = b["block_content"]
            if (
                b["block_label"] not in TEXT_LABELS
                or "<table" in old.lower()
                or "```" in old
            ):
                continue
            count = 0

            def replace(m):
                nonlocal count
                payload = next(
                    v for v in [m["double"], m["single"], m["paren"]] if v is not None
                )
                before, after = old[: m.start()], old[m.end() :]
                separate_before = not before.strip() or bool(
                    re.search(r"\n[ \t]*\n[ \t]*$", before)
                )
                separate_after = not after.strip() or bool(
                    re.match(r"^[ \t]*\n[ \t]*\n", after)
                )
                if not (
                    separate_before and separate_after and MATH_EVIDENCE.search(payload)
                ):
                    return m[0]
                count += 1
                return m["lead"] + r"\[" + payload + r"\]" + m["trail"]

            new = ISOLATED_MATH.sub(replace, old)
            if new != old:
                b["block_content"] = new
                logs.append(
                    {
                        "kind": "preserve_explicit_math_paragraph",
                        "page": pi + 1,
                        "block": bi,
                        "expressions": count,
                    }
                )
    return logs
