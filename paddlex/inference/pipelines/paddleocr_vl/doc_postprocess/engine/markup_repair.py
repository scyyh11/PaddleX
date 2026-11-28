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

DISPLAY_DOLLAR = re.compile(r"\$\$(.*?)\$\$", re.S)
DISPLAY_BRACKET = re.compile(r"\\\[(.*?)\\\]", re.S)
INLINE_PAREN = re.compile(r"\\\((.*?)\\\)", re.S)
CURRENCY = re.compile(
    r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s?(?:[kKmMbBtT]|bn|mn|million|billion|trillion|thousand|万|亿))?"
    r"(?=$|[,.;:!?)\]}>，。；：！？）】]|\s*[+\-–—/×=]\s*\$|\s+(?![+\-*/=<>≤≥≠±×÷^_\\|]))"
)


def _find_inline_math_end(s, start):
    """Find matching unescaped single closing ``$`` for inline math, or ``None``."""
    j = start + 1
    n = len(s)
    while j < n:
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "$":
            if s.startswith("$", j + 1):
                return None
            if not CURRENCY.match(s, j) and (j + 1 >= n or not s[j + 1].isdigit()):
                return j
            return None
        j += 1
    return None


def paragraph(s):
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s.startswith("$$", i):
            m = DISPLAY_DOLLAR.match(s, i)
            if m:
                v = m.group(1).replace("$", "")
                out.append(r"\[" + v + r"\]")
                i = m.end()
                continue
            out.append("$$")
            i += 2
            continue
        if s.startswith(r"\[", i):
            m = DISPLAY_BRACKET.match(s, i)
            if m:
                v = m.group(1).replace("$", "")
                out.append(r"\[" + v + r"\]")
                i = m.end()
                continue
        if s.startswith(r"\(", i):
            m = INLINE_PAREN.match(s, i)
            if m:
                out.append(r"\(" + m.group(1) + r"\)")
                i = m.end()
                continue
        if s[i] == "\\":
            if i + 1 < n:
                out.append(s[i : i + 2])
                i += 2
            else:
                out.append("\\")
                i += 1
            continue
        if s[i] == "$":
            cm = CURRENCY.match(s, i)
            if cm:
                out.append(cm.group(0))
                i = cm.end()
                continue
            end = _find_inline_math_end(s, i)
            if end is not None:
                body = s[i + 1 : end]
                if body.strip():
                    out.append(r"\(" + body + r"\)")
                    i = end + 1
                    continue
            out.append("$")
            i += 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


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
