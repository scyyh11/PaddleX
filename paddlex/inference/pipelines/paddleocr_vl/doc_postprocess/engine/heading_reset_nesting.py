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

"""Infer cross-family nesting from numbering resets.

When a numbering family restarts every time another family advances, the
restarting family is subordinate: I/II/III with A/B/C, 一、二、 with (一)(二),
1/2/3 with (1)(2)(3). Single letters that are also Roman numerals are resolved
from their neighbours, so "C." after "B." is a letter. The rule uses heading
text only and adds soft parent hints to the existing outline search.
"""
import collections
import re
import unicodedata

ROMAN = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)


def roman_value(s):
    v = [ROMAN[c] for c in s.upper()]
    return sum(-x if i + 1 < len(v) and x < v[i + 1] else x for i, x in enumerate(v))


def tokens(meta, base_token):
    out = [base_token(r["text"]) for r in meta]
    letters = {}
    for i, r in enumerate(meta):
        s = unicodedata.normalize("NFKC", r["text"]).strip()
        m = re.match(r"^([A-Z])[.)]\s", s)
        if not m:
            continue
        ch = m[1]
        # A one-letter Roman numeral is a letter when the letter sequence
        # reaches it (B then C, H then I, U then V, W then X).
        prev = chr(ord(ch) - 1) if ch > "A" else None
        is_letter_context = prev is not None and letters.get("last") == prev
        if ch in "CDLM" or is_letter_context or ch == "A":
            out[i] = ("letter_upper", (), ord(ch) - 64)
        if out[i] and out[i][0] == "letter_upper":
            letters["last"] = ch
    return out


def nesting_hints(meta, base_token, weight=1.0):
    tok = tokens(meta, base_token)
    fams = [(t[0], t[1]) if t else None for t in tok]
    order = [
        (i, fams[i], tok[i][2])
        for i in range(len(meta))
        if fams[i] and isinstance(tok[i][2], int)
    ]
    # Run starts per family: every ordinal 1, including the first run when
    # another numbered family already precedes it.
    starts = collections.defaultdict(list)
    seen = set()
    for i, f, n in order:
        if n == 1 and (f in seen or any(g != f for g in seen)):
            starts[f].append(i)
        seen.add(f)
    hints = collections.defaultdict(list)
    relations = {}
    for child, restarts in starts.items():
        if len(restarts) < 2:
            continue
        counts = collections.Counter()
        for s in restarts:
            prev = [(i, f) for i, f, n in order if i < s and f != child]
            last_child = max(
                (i for i, f, n in order if i < s and f == child), default=-1
            )
            between = {f for i, f in prev if i > last_child}
            counts.update(between)
        if not counts:
            continue
        parent, c = counts.most_common(1)[0]
        if c < 0.8 * len(restarts):
            continue
        # A parent family must not itself restart under the child family.
        if relations.get(parent) == child:
            continue
        relations[child] = parent
    for i, f, n in order:
        p = relations.get(f)
        if p is None:
            continue
        cand = [j for j, g, m in order if j < i and g == p]
        if cand:
            hints[i].append(("parent", cand[-1], weight))
    return hints, relations


def augment(meta, hints, base_token):
    extra, relations = nesting_hints(meta, base_token)
    for i, h in extra.items():
        # Explicit decimal prefixes and chapter words already govern parents.
        if not any(kind == "parent" for kind, _, _ in hints[i]):
            hints[i].extend(h)
    return hints, relations
