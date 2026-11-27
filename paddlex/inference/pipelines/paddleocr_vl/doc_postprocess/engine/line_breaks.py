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

"""Render OCR line breaks: keep item boundaries as paragraphs, join soft wraps.

PaddleOCR-VL keeps a newline where a block contains list items, contents lines,
short records or standalone math. Flattening those lines makes one paragraph out
of several document items. Ordinary wrapped prose is still joined.
"""
import re

from .enumeration_features import marker

LEADER = re.compile(r"(?:\.{3,}|…{2,}|·{3,}|_{3,}|-{4,})\s*[\dIVXLCivxlc]+\s*$")
PAGE_END = re.compile(r"(?:\s|\.{2,}|…+)(?:\d{1,4}|[ivxlc]{1,6})\s*$", re.I)
MATH_LINE = re.compile(r"^\s*(?:\\\[.*\\\]|\$\$.*\$\$|\$[^$]+\$|\\\(.*\\\))\s*$")


def join(x, y):
    x, y = x.rstrip(), y.lstrip()
    if re.search(r"[a-zA-Z]-$", x) and re.match(r"^[a-z]", y):
        return x[:-1] + y
    return (
        x
        + (
            ""
            if re.search(r"[\u4e00-\u9fff]$", x)
            or re.match(r"^[\u4e00-\u9fff，。；：！？]", y)
            else " "
        )
        + y
    )


def hard_breaks(lines):
    n = len(lines)
    breaks = [False] * (n - 1)
    if n < 2:
        return breaks
    # A contents column is one annotated block; its entries are not paragraphs.
    contents = (
        sum(bool(LEADER.search(t) or PAGE_END.search(t)) for t in lines) >= 0.5 * n
    )
    for i in range(n - 1):
        a, b = lines[i], lines[i + 1]
        if MATH_LINE.match(a) or MATH_LINE.match(b):
            breaks[i] = True
        elif not contents and marker(b)[0] is not None:
            breaks[i] = True
    return breaks


def render(text):
    if "\n" not in text or "<table" in text.lower() or "```" in text:
        return re.sub(r"\s*\n\s*", " ", text)
    lines = [t.strip() for t in text.split("\n") if t.strip()]
    if len(lines) < 2:
        return " ".join(lines)
    out = lines[0]
    for brk, line in zip(hard_breaks(lines), lines[1:]):
        out = out.rstrip() + "\n\n" + line if brk else join(out, line)
    return out
