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

"""List-item evidence for paragraph links; uses only predicted text."""
import re
import unicodedata

import numpy as np

CN = "一二三四五六七八九十百零〇两"
ROMAN = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
    "xi": 11,
    "xii": 12,
}
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
BULLETS = "•●▪■□◆◇►▶➢✓✔·○◦▫"


def cnint(s):
    v = 0
    cur = 0
    for ch in s:
        if ch == "十":
            cur = (cur or 1) * 10
            v += cur
            cur = 0
        elif ch in "百":
            cur = (cur or 1) * 100
            v += cur
            cur = 0
        else:
            cur = "零一二三四五六七八九".find(ch) if ch in "零一二三四五六七八九" else 0
    return v + cur


def marker(text):
    """Return (family, ordinal) for a leading enumeration marker, or (None, 0)."""
    raw = text.lstrip()
    if raw and raw[0] in CIRCLED:
        return "circled", CIRCLED.index(raw[0]) + 1
    s = unicodedata.normalize("NFKC", raw)
    if not s:
        return None, 0
    if s[0] in BULLETS or re.match(r"^[-–—*]\s", s):
        return "bullet", 0
    m = re.match(r"^第([" + CN + r"\d]+)[条條款项項章节節]", s)
    if m:
        return "cn_article", int(m[1]) if m[1].isdigit() else cnint(m[1])
    m = re.match(r"^[(（]([" + CN + r"]{1,3})[)）]", s)
    if m:
        return "cn_paren", cnint(m[1])
    m = re.match(r"^([" + CN + r"]{1,3})[、.]", s)
    if m:
        return "cn_list", cnint(m[1])
    m = re.match(r"^[(（](\d{1,3})[)）]", s)
    if m:
        return "num_paren", int(m[1])
    m = re.match(r"^(\d{1,3})[.)、](?!\d)", s)
    if m:
        return "num", int(m[1])
    m = re.match(r"^[(（]([ivx]{1,4})[)）]\s", s, re.I)
    if m and m[1].lower() in ROMAN:
        return "roman_paren", ROMAN[m[1].lower()]
    m = re.match(r"^([ivx]{1,4})[.)]\s", s, re.I)
    if m and m[1].lower() in ROMAN:
        return "roman", ROMAN[m[1].lower()]
    m = re.match(r"^[(（]([a-zA-Z])[)）]\s?", s)
    if m:
        return "letter_paren", ord(m[1].lower()) - 96
    m = re.match(r"^([a-zA-Z])[.)]\s", s)
    if m:
        return "letter", ord(m[1].lower()) - 96
    return None, 0


FAMILIES = [
    "circled",
    "bullet",
    "cn_article",
    "cn_paren",
    "cn_list",
    "num_paren",
    "num",
    "roman_paren",
    "roman",
    "letter_paren",
    "letter",
]


def features(a, b):
    fa, oa = marker(a)
    fb, ob = marker(b)
    x = [
        int(fa is not None),
        int(fb is not None),
        int(fa is not None and fa == fb),
        int(fa is not None and fa == fb and ob == oa + 1),
        int(fb is not None and ob == 1),
        int(fa is not None and fa == fb and fa == "bullet"),
    ]
    x += [int(fb == f) for f in FAMILIES]
    lines = [t for t in a.splitlines() if t.strip()]
    last = marker(lines[-1])[0] if lines else None
    # A previous block whose final line is itself an item is a list, even when
    # its first line is introductory text.
    x += [
        int(last is not None),
        int(last is not None and last == fb),
        sum(marker(t)[0] is not None for t in lines),
    ]
    return np.asarray(x, dtype=np.float32)
