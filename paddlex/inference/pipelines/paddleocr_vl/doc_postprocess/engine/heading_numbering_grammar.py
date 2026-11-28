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

"""Induce numbering runs from a document; use them as soft tree constraints.

No annotation, document ID, or prescribed absolute title depth is used.
Repeated ordinals start a new run. A run needs three distinct members.
"""
import collections
import math
import re
import unicodedata

from . import runtime_context as ctx
from .heading_ml_features import fsize
from .numbered_outline import cnint

CN = "零〇一二三四五六七八九十百千两兩"


def roman(s):
    values = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)
    v = [values[c] for c in s.upper()]
    return sum(-x if i + 1 < len(v) and x < v[i + 1] else x for i, x in enumerate(v))


def token(text):
    s = unicodedata.normalize("NFKC", text).strip()
    m = re.match(r"^第([" + CN + r"\d]+)([部篇卷章节節])", s)
    if m:
        kind = "part" if m[2] in "部篇卷" else "chapter" if m[2] == "章" else "section"
        return "cn_" + kind, (), cnint(m[1])
    m = re.match(r"^(part|chapter|section)\s+(\d+|[IVXLCDM]+)\b", s, re.I)
    if m:
        return "en_" + m[1].lower(), (), int(m[2]) if m[2].isdigit() else roman(m[2])
    m = re.match(r"^(\d{1,3}(?:\.\d{1,3})*)\.([A-Za-z])(?:[.)、:：\s]|$)", s)
    if m:
        return (
            "decimal_letter",
            tuple(map(int, m[1].split("."))),
            ord(m[2].lower()) - 96,
        )
    m = re.match(r"^(\d{1,3}(?:\.\d{1,3})*)(?:[.)、:：\s]|$)", s)
    if m:
        ns = tuple(map(int, m[1].split(".")))
        # 1.0 is a chapter convention, not a sibling of 1.1 and 1.2.
        while len(ns) > 1 and ns[-1] == 0:
            ns = ns[:-1]
        return "decimal", ns[:-1], ns[-1]
    m = re.match(r"^\(([" + CN + r"]+|\d{1,3})\)", s)
    if m:
        return "cn_paren" if not m[1].isdigit() else "number_paren", (), cnint(m[1])
    m = re.match(r"^([" + CN + r"]{1,5})[、.．\s]", s)
    if m:
        return "cn_list", (), cnint(m[1])
    m = re.match(r"^([IVXLCDM]+|[ivxlcdm]+)[.)、\s]", s)
    if m:
        return "roman_upper" if m[1].isupper() else "roman_lower", (), roman(m[1])
    m = re.match(r"^(\(?)([A-Za-z])[.)、]", s)
    if m:
        return (
            "letter_"
            + ("upper" if m[2].isupper() else "lower")
            + ("_paren" if m[1] else ""),
            (),
            ord(m[2].lower()) - 96,
        )
    return None


def induce(meta):
    reset_nesting = ctx.get_config().heading_reset_nesting
    if reset_nesting:
        from .heading_reset_nesting import tokens as _corrected

        tokens = _corrected(meta, token)
    else:
        tokens = [token(r["text"]) for r in meta]
    segments = []
    segment = 0
    for r in meta:
        if re.match(r"^\s*(?:附[录錄]|appendix\b|annex\b)", r["text"], re.I):
            segment += 1
        segments.append(segment)
    # In a run such as Part 1, Part 2, Chapter 3, Part 4, the isolated
    # Chapter is an ordinal continuation, not a new nesting convention.
    majors = [
        i for i, t in enumerate(tokens) if t and t[0] in {"cn_part", "cn_chapter"}
    ]
    for k in range(1, len(majors) - 1):
        a, i, b = majors[k - 1 : k + 2]
        ta, t, tb = tokens[a], tokens[i], tokens[b]
        if (
            segments[a] == segments[i] == segments[b]
            and ta[0] == tb[0] == "cn_part"
            and t[0] == "cn_chapter"
            and ta[2] + 1 == t[2] == tb[2] - 1
        ):
            if (
                max(abs(math.log(fsize(meta[j]) / fsize(meta[i]))) for j in [a, b])
                < 0.25
            ):
                tokens[i] = ("cn_part", (), t[2])
    groups = collections.defaultdict(list)
    for i, t in enumerate(tokens):
        if t:
            groups[(segments[i], t[:2])].append(i)
    runs = []
    for key, inds in groups.items():
        current = []
        for i in inds:
            if current:
                j = current[-1]
                delta = tokens[i][2] - tokens[j][2]
                # Consecutive numbering plus compatible typography is evidence.
                # A reset or a large jump must not link independent sections.
                if (
                    not 1 <= delta <= 3
                    or abs(math.log(fsize(meta[i]) / fsize(meta[j]))) > 0.4
                ):
                    if len(current) >= 3:
                        runs.append((key, current))
                    current = []
            current.append(i)
        if len(current) >= 3:
            runs.append((key, current))
    hints = [[] for _ in meta]
    supported = set()
    for key, inds in runs:
        if sum(tokens[b][2] == tokens[a][2] + 1 for a, b in zip(inds, inds[1:])) < 2:
            continue
        for a, b in zip(inds, inds[1:]):
            hints[b].append(("equal", a, 1.0))
        supported.update(inds)
    # Literal decimal prefixes provide parent evidence when both nodes exist.
    active = {}
    for i, t in enumerate(tokens):
        if i and segments[i] != segments[i - 1]:
            active = {}
        if not t or t[0] not in {"decimal", "decimal_letter"}:
            continue
        if t[0] == "decimal_letter":
            if t[1] in active and i in supported:
                hints[i].append(("parent", active[t[1]], 1.0))
            continue
        number = t[1] + (t[2],)
        for p in list(active):
            if len(p) >= len(number) and p[: len(number) - 1] == number[:-1]:
                del active[p]
        if number[:-1] in active and (i in supported or len(number) >= 3):
            hints[i].append(("parent", active[number[:-1]], 1.0))
        active[number] = i
    # Chapter/section words state relative roles; their document base level is
    # left to the model. Only established runs may act as structural anchors.
    active = {}
    for i, t in enumerate(tokens):
        if i and segments[i] != segments[i - 1]:
            active = {}
        if not t:
            continue
        fam = t[0]
        for lang in ["cn", "en"]:
            if fam == lang + "_part" and i in supported:
                active[lang + "_part"] = i
                active.pop(lang + "_chapter", None)
            elif fam == lang + "_chapter" and i in supported:
                if lang + "_part" in active:
                    hints[i].append(("parent", active[lang + "_part"], 1.0))
                active[lang + "_chapter"] = i
            elif fam == lang + "_section" and i in supported:
                p = active.get(lang + "_chapter", active.get(lang + "_part"))
                if p is not None:
                    hints[i].append(("parent", p, 1.0))
    if reset_nesting:
        from .heading_reset_nesting import augment

        hints, _ = augment(meta, hints, token)
    return hints


def penalty(hints, i, level, parent, path, weight=3.0):
    value = 0.0
    for kind, j, w in hints[i]:
        if kind == "equal":
            value -= weight * w * min(2, abs(level - path[j]))
        elif kind == "parent":
            value -= weight * w * int(parent != j)
    return value
