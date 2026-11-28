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

"""Drop image references that are layout noise rather than figures.

PaddleOCR-VL's layout model sometimes reports rules, borders, bullets and
small page decorations as image blocks. Boxes are on the renderer's 0-1000
page grid. Three kinds are removed:
- tiny regions under 400 grid units (about 2% x 2% of the page),
- slivers more than 15 times longer than they are wide (lines and rules),
- small marks under 2500 units repeated at the same place on at least five
  other pages of the document (running decorations).
Text, tables and formulas are untouched; only the image reference is removed.
"""
import re

REF = re.compile(r"!\[[^\]]*\]\([^)]*?page(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.\w+\)")


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def noise(refs):
    out = set()
    for i, (page, b) in enumerate(refs):
        w, h = max(1, b[2] - b[0]), max(1, b[3] - b[1])
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < 400 or max(w / h, h / w) > 15:
            out.add(i)
            continue
        if (
            area < 2500
            and len({p for p, c in refs if p != page and _iou(b, c) > 0.8}) >= 5
        ):
            out.add(i)
    return out


def apply(text):
    ms = list(REF.finditer(text))
    refs = [(int(m.group(1)), [int(m.group(k)) for k in range(2, 6)]) for m in ms]
    drop = noise(refs)
    if not drop:
        return text, 0
    out = []
    last = 0
    for i, m in enumerate(ms):
        if i in drop:
            out.append(text[last : m.start()])
            last = m.end()
    out.append(text[last:])
    return "".join(out), len(drop)
