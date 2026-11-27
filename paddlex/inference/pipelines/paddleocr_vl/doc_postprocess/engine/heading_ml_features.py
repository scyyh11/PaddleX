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

from .runtime_context import memoize

"""Prediction-only numeric and pairwise evidence for small decision-tree models."""
import collections
import hashlib
import math
import re
import statistics
import unicodedata

import numpy as np

LABELS = [
    "doc_title",
    "paragraph_title",
    "text",
    "header",
    "aside_text",
    "figure_title",
    "content",
    "footer",
]
FAMILIES = [
    "plain",
    "caps",
    "major",
    "num",
    "num_paren",
    "cn",
    "cn_paren",
    "cn_upper",
    "cn_章",
    "cn_节",
    "cn_部",
    "cn_篇",
    "cn_卷",
    "en_chapter",
    "en_section",
    "en_part",
    "roman",
    "letter",
    "stem",
]


@memoize
def norm(s):
    s = unicodedata.normalize("NFKC", s)
    raw = s
    if "\\" in s:  # the unwrap pattern needs a literal backslash
        for _ in range(6):
            s = _UNWRAP.sub(r"\1", s)
    out = _NONWORD.sub("", s).lower()
    if _NORM_VERIFY:
        t = raw
        for _ in range(6):
            t = re.sub(
                r"\\+(?:text|mathrm|textrm|textbf|textit|underline)\s*\{([^{}]*)\}",
                r"\1",
                t,
            )
        if out != re.sub(r"[^\w]", "", t).lower():
            raise AssertionError("heading_ml_features: norm differs")
    return out


_NORM_VERIFY = False
_UNWRAP = re.compile(
    r"\\+(?:text|mathrm|textrm|textbf|textit|underline)\s*\{([^{}]*)\}"
)
_NONWORD = re.compile(r"[^\w]")


def rid(r):
    return (
        str(r["page"])
        + ":"
        + str(r["index"])
        + ":"
        + hashlib.sha256(r["text"].encode()).hexdigest()[:12]
    )


def fsize(r):
    a = r["visual"].get("font", r["estimated_font"])
    b = r["estimated_font"]
    return max(0.1, b if a > b * 1.8 else a)


def med(xs, default):
    return statistics.median(xs) if xs else default


@memoize
def number(s):
    m = _NUMBER.match(unicodedata.normalize("NFKC", s))
    return tuple(map(int, _NUMSEP.split(m[1]))) if m else ()


_NUMBER = re.compile(r"^\s*(\d+(?:[.\-]\d+)*)")
_NUMSEP = re.compile("[.\-]")


def extract(records):
    pages = collections.defaultdict(list)
    keys = collections.defaultdict(list)
    for r in records:
        pages[r["page"]].append(r)
        keys[norm(r["text"])].append(r)
    bodies = [r for r in records if r["label"] == "text" and len(r["text"]) > 120]
    body = med([fsize(r) for r in bodies], 20)
    bd = med([r["visual"].get("density", 0.25) for r in bodies], 0.25)
    heads = [r for r in records if r["level"]]
    headfonts = sorted(fsize(r) for r in heads)
    maxp = max(pages, default=1)
    families = collections.Counter(r["family"][0] for r in heads)
    style = collections.Counter(
        (round(fsize(r) / body, 1), r["family"][0]) for r in heads
    )
    candidates = [
        r
        for r in records
        if r["level"]
        or (r["label"] not in {"content", "footer"} and 2 <= len(r["text"]) <= 320)
    ]
    page_stats = {}
    for page, p in pages.items():
        pb = [a for a in p if a["label"] == "text" and len(a["text"]) > 120]
        page_stats[page] = (
            med([fsize(a) for a in pb], body),
            med([a["visual"].get("density", bd) for a in pb], bd),
            sum(bool(a["level"]) for a in p),
            sum(len(a["text"]) > 120 for a in p),
        )
    indices = {id(r): i for p in pages.values() for i, r in enumerate(p)}
    values = []
    texts = []
    for r in candidates:
        p = pages[r["page"]]
        i = indices[id(r)]
        prev = p[max(0, i - 1)]
        nxt = p[min(len(p) - 1, i + 1)]
        s = unicodedata.normalize("NFKC", r["text"])
        key = norm(s)
        f = fsize(r)
        v = r["visual"]
        w = r["width"]
        h = r["height"]
        x, y, xx, yy = r["box"]
        n = max(len(s), 1)
        letters = re.findall("[a-zA-Z]", s)
        words = re.findall("[a-zA-Z]+", s)
        same = keys[key]
        fam, depth = r["family"]
        nums = number(s)
        pfont, pd, pheads, plong = page_stats[r["page"]]
        feat = [
            x / w,
            y / h,
            xx / w,
            yy / h,
            (xx - x) / w,
            (yy - y) / h,
            abs((x + xx) / 2 / w - 0.5),
            f,
            f / body,
            f / pfont,
            r["estimated_font"] / body,
            v.get("font", f) / max(r["estimated_font"], 0.1),
            v.get("density", bd),
            v.get("density", bd) / max(bd, 0.01),
            v.get("density", pd) / max(pd, 0.01),
            v.get("stroke", 0),
            v.get("lines", s.count("\n") + 1),
            math.log1p(n),
            len(words),
            s.count("\n"),
            sum(c.isupper() for c in letters) / max(len(letters), 1),
            sum(a[0].isupper() for a in words) / max(len(words), 1),
            sum("\u4e00" <= c <= "\u9fff" for c in s) / n,
            sum(c.isdigit() for c in s) / n,
            sum(c in ".。，,;；:：?!！？" for c in s) / n,
            int(s.rstrip().endswith((".", "。", ";", "；"))),
            int(s.rstrip().endswith((":", "："))),
            len(re.findall("_{2,}", s)),
            depth,
            nums[0] if nums else -1,
            len(nums),
            r["level"],
            r.get("toc_hint") or 0,
            min(r["page"], 15),
            r["page"] / maxp,
            int(r["page"] < 4),
            len(p),
            pheads,
            plong,
            len(same),
            len({a["page"] for a in same}),
            min(a["page"] for a in same) == r["page"],
            max(a["page"] for a in same) == r["page"],
            med([a["box"][1] / a["height"] for a in same], 0),
            max((fsize(a) for a in same), default=f) / f,
            len(set(a["label"] for a in same)),
            style[(round(f / body, 1), fam)],
            sum(a <= f for a in headfonts) / max(len(headfonts), 1),
            f / fsize(prev),
            f / fsize(nxt),
            len(prev["text"]),
            len(nxt["text"]),
            int(prev["page"] == r["page"]),
            int(nxt["page"] == r["page"]),
            int(bool(re.match(r"^(?:Figure|Fig\.|Table|图|表)\s*\d", s, re.I))),
            int(
                bool(
                    re.search(
                        r"copyright|all rights reserved|ISBN|EDRS|DOCUMENT RESUME",
                        s,
                        re.I,
                    )
                )
            ),
            int(
                bool(
                    re.fullmatch(
                        r"\s*(?:note[s]?|warning[s]?|caution|remark[s]?|tip[s]?|注|註|注意|说明|說明)[:：.\s]*",
                        s,
                        re.I,
                    )
                )
            ),
            int(
                bool(
                    re.search(
                        r"\b(?:Problem|Action|Solution|Recommendation|Definition|Qualifications)\b",
                        s,
                        re.I,
                    )
                )
            ),
            int(
                bool(
                    re.search(
                        r"\b(?:Appendix|Annex|Abstract|Summary|Introduction|Preface|Contents|Conclusion|References)\b",
                        s,
                        re.I,
                    )
                )
            ),
        ]
        feat += (
            [int(r["label"] == a) for a in LABELS]
            + [int(r.get("original_label") == a) for a in LABELS]
            + [int(prev["label"] == a) for a in LABELS]
            + [int(nxt["label"] == a) for a in LABELS]
        )
        feat += [int(fam == a) for a in FAMILIES] + [
            families[a] / max(len(heads), 1) for a in FAMILIES
        ]
        values.append(feat)
        texts.append(s)
    return np.asarray(values, dtype=np.float32), candidates, texts


def pair_features(a, b, fa, fb):
    na, nb = number(a["text"]), number(b["text"])
    common = 0
    for x, y in zip(na, nb):
        if x != y:
            break
        common += 1
    features = [
        fsize(b) / fsize(a),
        b["visual"].get("density", 0.25) / max(a["visual"].get("density", 0.25), 0.01),
        b["box"][0] / b["width"] - a["box"][0] / a["width"],
        (b["box"][0] + b["box"][2]) / b["width"] / 2
        - (a["box"][0] + a["box"][2]) / a["width"] / 2,
        b["page"] - a["page"],
        int(b["page"] == a["page"]),
        b["level"] - a["level"],
        int(a["family"] == b["family"]),
        int(a["label"] == b["label"]),
        int(norm(a["text"]) == norm(b["text"])),
        common,
        len(na),
        len(nb),
        int(
            bool(
                na
                and nb
                and len(na) == len(nb)
                and na[:-1] == nb[:-1]
                and nb[-1] == na[-1] + 1
            )
        ),
        int(bool(na and nb and len(nb) > len(na) and nb[: len(na)] == na)),
        int(bool(nb and nb[-1] == 1)),
        int(bool(na and nb and nb[0] > na[0])),
    ]
    return np.concatenate([fa, fb, fb - fa, np.asarray(features, dtype=np.float32)])
