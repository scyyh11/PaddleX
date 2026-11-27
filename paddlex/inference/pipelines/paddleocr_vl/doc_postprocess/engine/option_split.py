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

"""Split multiple-choice options that PaddleOCR-VL wrote into one paragraph.

A paragraph such as "A. 正数 B. 负数 C. 非正数 D. 非负数" (or a question stem
followed by its options) becomes one paragraph per option. Splitting happens
only when the letters appear in order starting from A, outside inline math,
so ordinary prose and initials are left alone."""
import re

MARK = re.compile(
    r"(?:(?<=^)|(?<=[\s\u3000]))(?:([A-H])\s*[\.．、]|[（(]([A-H])[)）])(?=\s*[^\s\d,，;；.．、:：)）\]])"
)
LEADER = re.compile(r"\.{4,}|…{2,}|·{4,}")
CAPTION = re.compile(r"^(?:fig(?:ure)?\.?|table|图|表)\s*\d", re.I)
REFERENCE = re.compile(
    r"et al\.|\b(?:19|20)\d\d[a-z]?[\.,;)]|^\s*\[\d+\]|[A-Z][a-z]+,\s+(?:[A-Z]\.\s*){1,3}[,;]?"
)
SKIP = ("#", "|", "<", "$$", "\\[", "![", "```")


def _math_spans(p):
    spans = []
    i = 0
    for m in re.finditer(r"\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)", p, re.S):
        spans.append(m.span())
    return spans


def split_paragraph(p):
    s = p.strip()
    if (
        not s
        or s.startswith(SKIP)
        or s.count("\n") > 8
        or LEADER.search(s)
        or CAPTION.match(s)
        or REFERENCE.search(s)
    ):
        return [p]
    spans = _math_spans(s)
    inmath = lambda k: any(a <= k < b for a, b in spans)
    marks = [
        (m.start(), m.end(), (m.group(1) or m.group(2)))
        for m in MARK.finditer(s)
        if not inmath(m.start())
    ]
    chain = []
    want = "A"
    last_end = None
    for pos, end, letter in marks:
        has_content = last_end is not None and s[last_end:pos].strip() != ""
        if letter == want and (not chain or has_content):
            chain.append(pos)
            want = chr(ord(want) + 1)
            last_end = end
        elif letter == "A" and len(chain) < 2:
            chain = [pos]
            want = "B"
            last_end = end
        elif chain and letter != want and not has_content:
            if len(chain) < 2:
                chain = []
                want = "A"
                last_end = None
    if len(chain) < 2 or s[last_end:].strip() == "":
        return [p]
    cuts = [0] + [c for c in chain if c > 0] + [len(s)]
    parts = [s[a:b].strip() for a, b in zip(cuts, cuts[1:]) if s[a:b].strip()]
    return parts if len(parts) > 1 else [p]


def apply(text):
    out = []
    n = 0
    for para in re.split(r"(\n\s*\n)", text):
        if re.fullmatch(r"\n\s*\n", para or ""):
            out.append(para)
            continue
        parts = split_paragraph(para)
        if len(parts) > 1:
            n += 1
            out.append("\n\n".join(parts))
        else:
            out.append(para)
    return "".join(out), n


if __name__ == "__main__":
    t = "4、在数轴上表示的数是（D）\n\nA. 正数 B. 负数 C. 非正数 D. 非负数\n\nThe author J. A. Smith wrote A. B. C. text.\n\n$$ A. x $$"
    o, n = apply(t)
    print(o)
    print(n)
    assert "A. 正数\n\nB. 负数\n\nC. 非正数\n\nD. 非负数" in o
    assert "J. A. Smith wrote A. B. C. text." in o
    assert (
        len(
            split_paragraph(
                "Pickernell, D. G., Christie, M. J., Rowe A., Thomas, B. C., Putterill"
            )
        )
        == 1
    )
    assert (
        len(split_paragraph("前言 ..... V A.3 电灼伤 ..... 39 B.1 刚性试验指 ..... 40"))
        == 1
    )
    assert (
        len(split_paragraph("Figure 1.2. Diversity A. diversity and B. disparity")) == 1
    )
    assert (
        len(split_paragraph("Balesdent, J., A. Mariotti and B. Guillet. 1987. Natural"))
        == 1
    )
    assert (
        len(
            split_paragraph("[93] Q. Yang, A. Aitouche, and B. O. Bouamama, Structural")
        )
        == 1
    )
    assert len(split_paragraph("A. why he had crashed B. how he had crashed")) == 2
    o2, _ = apply(
        "1. 下列说法正确的是 (C) A. 所有的整数都是正数 B. 不是正数的数一定是负数"
    )
    print(o2)
    o3, _ = apply("Choose: (A) one (B) two (C) three")
    print(o3)
