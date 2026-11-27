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

"""Keep math delimiters local to their paragraph; preserve literal currency signs."""
import re

PAIR = re.compile(
    r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<!\\)\$(.*?)(?<!\\)\$", re.S
)


def paragraph(s):
    def sub(m):
        k = next(i for i, v in enumerate(m.groups()) if v is not None)
        v = m.group(k + 1)
        if k < 2:
            # Spurious dollar inside a display formula is a mismatched OCR delimiter.
            v = v.replace("$", "")
            return r"\[" + v + r"\]"
        return r"\(" + v + r"\)"

    s = PAIR.sub(sub, s)
    return s.replace("\\$", r"\textdollar{}").replace("$", r"\textdollar{}")


def apply(md):
    # HTML tables are already structurally bounded and are left byte-identical.
    chunks = re.split(r"(<table\b.*?</table>)", md, flags=re.S)
    return "".join(
        (
            s
            if s.startswith("<table")
            else "\n\n".join(paragraph(p) for p in s.split("\n\n"))
        )
        for s in chunks
    )


if __name__ == "__main__":
    import hashlib
    import json
    import pathlib

    w = pathlib.Path(
        "/ssd4/cuicheng01-paddleocr-vl16-postprocess-20261102/texts-20261104"
    )
    source = w / "experiments/text-hybrid-0.999"
    dest = w / "experiments/text-markup"
    (dest / "predictions").mkdir(parents=True, exist_ok=True)
    (dest / "audit").mkdir(exist_ok=True)
    for p in sorted((source / "predictions").glob("*.md")):
        old = p.read_text()
        new = apply(old)
        (dest / "predictions" / p.name).write_text(new)
        (dest / "audit" / (p.stem + ".json")).write_text(
            json.dumps(
                {
                    "source": "text-hybrid-0.999",
                    "markup_repaired": new != old,
                    "sha256": hashlib.sha256(new.encode()).hexdigest(),
                }
            )
        )
    print("complete", len(list((dest / "predictions").glob("*.md"))))
