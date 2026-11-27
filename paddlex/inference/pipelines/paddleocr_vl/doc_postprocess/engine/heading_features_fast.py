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

"""Exact, faster heading candidate features (same values as heading_ml_features.extract and
heading_ml_features_v3.extract).

The originals recompute document-level summaries for every candidate (records sharing a
normalized text, style groups, sorted heading fonts).  Here each summary is computed once
per key with the same expression, and per-character counts use equivalent C-level calls:
  * sum(a<=f for a in sorted_fonts)          -> bisect_right(sorted_fonts, f)
  * sum(q<t for q in sorted_ints)            -> bisect_left(sorted_ints, t)
  * sum(c.isupper() for c in [a-zA-Z] chars) -> count of [A-Z]
  * sum('\\u4e00'<=c<='\\u9fff' for c in s)  -> count of [\\u4e00-\\u9fff]
  * sum(c.isdigit() for c in s)              -> sum(map(str.isdigit, s))
The v3 features reuse the base call made for the empty-input guard instead of computing it
twice.  HEADING_FEATURES_VERIFY=1 runs the original functions and requires identical arrays,
candidate lists and texts.
"""
import bisect
import collections
import math
import re
import unicodedata

import numpy as np

from . import charcount
from .heading_ml_features import FAMILIES, LABELS, fsize, med, norm, number
from .heading_structure_features import sequence

VERIFY = False
_LETTER = re.compile("[a-zA-Z]")
_UPPER = re.compile("[A-Z]")
_WORD = re.compile("[a-zA-Z]+")
_CJK = re.compile("[\u4e00-\u9fff]")
_UNDERSCORES = re.compile("_{2,}")
_PUNCT = charcount.counter(".。，,;；:：?!！？")
_FIGURE = re.compile(r"^(?:Figure|Fig\.|Table|图|表)\s*\d", re.I)
_COPYRIGHT = re.compile(
    r"copyright|all rights reserved|ISBN|EDRS|DOCUMENT RESUME", re.I
)
_NOTE = re.compile(
    r"\s*(?:note[s]?|warning[s]?|caution|remark[s]?|tip[s]?|注|註|注意|说明|說明)[:：.\s]*",
    re.I,
)
_ROLE = re.compile(
    r"\b(?:Problem|Action|Solution|Recommendation|Definition|Qualifications)\b", re.I
)
_SECTION = re.compile(
    r"\b(?:Appendix|Annex|Abstract|Summary|Introduction|Preface|Contents|Conclusion|References)\b",
    re.I,
)
_V3_PATTERNS = [
    re.compile(p, f)
    for p, f in [
        (r"^(?:附[录錄]|Appendix|Annex)\s*[^\s]*", re.I),
        (r"^(?:第[一二三四五六七八九十\d]+[部篇卷]|Part\s+[IVXLCDM\d]+)", re.I),
        (r"^(?:第[一二三四五六七八九十\d]+章|Chapter\s+[IVXLCDM\d]+)", re.I),
        (r"^(?:第[一二三四五六七八九十\d]+[节節]|Section\s+[IVXLCDM\d]+)", re.I),
        (r"^\([A-Z]\)", 0),
        (r"^[A-Z][.)]", 0),
        (r"^\([a-z]\)", 0),
        (r"^[a-z][.)]", 0),
        (r"^[IVXLCDM]+[.)]", 0),
        (r"^[ivxlcdm]+[.)]", 0),
        (r"^[一二三四五六七八九十]", 0),
    ]
]


def base_extract(records):
    out = _base(records)
    if out is None:
        from . import heading_ml_features

        out = heading_ml_features.extract(records)
    if VERIFY:
        from . import heading_ml_features

        _same(out, heading_ml_features.extract(records), "base heading features")
    return out


def _base(records):
    pages = collections.defaultdict(list)
    keys = collections.defaultdict(list)
    fonts = {}
    for r in records:
        pages[r["page"]].append(r)
        keys[norm(r["text"])].append(r)
        fonts[id(r)] = fsize(r)
    bodies = [r for r in records if r["label"] == "text" and len(r["text"]) > 120]
    body = med([fonts[id(r)] for r in bodies], 20)
    bd = med([r["visual"].get("density", 0.25) for r in bodies], 0.25)
    heads = [r for r in records if r["level"]]
    headfonts = sorted(fonts[id(r)] for r in heads)
    maxp = max(pages, default=1)
    families = collections.Counter(r["family"][0] for r in heads)
    style = collections.Counter(
        (round(fonts[id(r)] / body, 1), r["family"][0]) for r in heads
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
            med([fonts[id(a)] for a in pb], body),
            med([a["visual"].get("density", bd) for a in pb], bd),
            sum(bool(a["level"]) for a in p),
            sum(len(a["text"]) > 120 for a in p),
        )
    indices = {id(r): i for p in pages.values() for i, r in enumerate(p)}
    same_stats = {}
    nheads = max(len(heads), 1)
    nheadfonts = max(len(headfonts), 1)
    family_share = [families[a] / max(len(heads), 1) for a in FAMILIES]
    label_id = {a: k for k, a in enumerate(LABELS)}
    family_id = {a: k for k, a in enumerate(FAMILIES)}
    raw = []
    texts = []
    log1p = math.log1p
    for r in candidates:
        p = pages[r["page"]]
        i = indices[id(r)]
        prev = p[max(0, i - 1)]
        nxt = p[min(len(p) - 1, i + 1)]
        s = unicodedata.normalize("NFKC", r["text"])
        key = norm(s)
        f = fonts[id(r)]
        v = r["visual"]
        x, y, xx, yy = r["box"]
        n = max(len(s), 1)
        page = r["page"]
        nletters = len(_LETTER.findall(s))
        words = _WORD.findall(s)
        fam, depth = r["family"]
        nums = number(s)
        st = same_stats.get(key)
        if st is None:
            same = keys[key]
            st = (
                len(same),
                len({a["page"] for a in same}),
                min(a["page"] for a in same),
                max(a["page"] for a in same),
                med([a["box"][1] / a["height"] for a in same], 0),
                (
                    max(fonts[id(a)] if id(a) in fonts else fsize(a) for a in same)
                    if same
                    else None
                ),
                len(set(a["label"] for a in same)),
            )
            same_stats[key] = st
        nsame, npages, minpage, maxpage, medy, maxfont, nlabels = st
        pfont, pd, pheads, plong = page_stats[page]
        nl = s.count("\n")
        tail = s.rstrip()
        tup = (
            x,
            y,
            xx,
            yy,
            r["width"],
            r["height"],
            f,
            pfont,
            r["estimated_font"],
            v.get("font", f),
            v.get("density", bd),
            v.get("density", pd),
            pd,
            v.get("stroke", 0),
            v.get("lines", nl + 1),
            log1p(n),
            len(words),
            nl,
            len(_UPPER.findall(s)),
            nletters,
            sum(1 for a in words if a[0] <= "Z"),
            charcount.cjk(s),
            sum(map(str.isdigit, s)),
            _PUNCT(s),
            n,
            int(tail.endswith((".", "。", ";", "；"))),
            int(tail.endswith((":", "："))),
            len(_UNDERSCORES.findall(s)),
            depth,
            nums[0] if nums else -1,
            len(nums),
            r["level"],
            r.get("toc_hint") or 0,
            page,
            len(p),
            pheads,
            plong,
            nsame,
            npages,
            minpage == page,
            maxpage == page,
            medy,
            maxfont if maxfont is not None else f,
            nlabels,
            style[(round(f / body, 1), fam)],
            bisect.bisect_right(headfonts, f),
            fonts[id(prev)],
            fonts[id(nxt)],
            len(prev["text"]),
            len(nxt["text"]),
            int(prev["page"] == page),
            int(nxt["page"] == page),
            int(bool(_FIGURE.match(s))),
            int(bool(_COPYRIGHT.search(s))),
            int(bool(_NOTE.fullmatch(s))),
            int(bool(_ROLE.search(s))),
            int(bool(_SECTION.search(s))),
            label_id.get(r["label"], -1),
            label_id.get(r.get("original_label"), -1),
            label_id.get(prev["label"], -1),
            label_id.get(nxt["label"], -1),
            family_id.get(fam, -1),
        )
        if None in tup or not all(type(t) in _NUMERIC for t in tup[:56]):
            return None  # unusual value types: use the row-by-row construction
        raw.append(tup)
        texts.append(s)
    if not raw:
        return np.asarray([], dtype=np.float32), candidates, texts
    R = np.asarray(raw, dtype=np.float64)
    c = lambda k: R[:, k]
    x, y, xx, yy, w, h, f, pfont, est = (c(k) for k in range(9))
    cols = [
        x / w,
        y / h,
        xx / w,
        yy / h,
        (xx - x) / w,
        (yy - y) / h,
        np.abs((x + xx) / 2 / w - 0.5),
        f,
        f / body,
        f / pfont,
        est / body,
        c(9) / np.where(0.1 > est, 0.1, est),
        c(10),
        c(10) / max(bd, 0.01),
        c(11) / np.where(0.01 > c(12), 0.01, c(12)),
        c(13),
        c(14),
        c(15),
        c(16),
        c(17),
        c(18) / np.maximum(c(19), 1),
        c(20) / np.maximum(c(16), 1),
        c(21) / c(24),
        c(22) / c(24),
        c(23) / c(24),
        c(25),
        c(26),
        c(27),
        c(28),
        c(29),
        c(30),
        c(31),
        c(32),
        np.minimum(c(33), 15),
        c(33) / maxp,
        c(33) < 4,
        c(34),
        c(35),
        c(36),
        c(37),
        c(38),
        c(39),
        c(40),
        c(41),
        c(42) / f,
        c(43),
        c(44),
        c(45) / nheadfonts,
        f / c(46),
        f / c(47),
        c(48),
        c(49),
        c(50),
        c(51),
        c(52),
        c(53),
        c(54),
        c(55),
        c(56),
    ]
    out = np.empty((len(raw), 129), dtype=np.float64)
    for k, col in enumerate(cols):
        out[:, k] = col
    ids = R[:, 57:62].astype(np.int64)
    lab = np.arange(len(LABELS))[None, :]
    for g in range(4):
        out[:, 59 + 8 * g : 67 + 8 * g] = ids[:, g : g + 1] == lab
    out[:, 91:110] = ids[:, 4:5] == np.arange(len(FAMILIES))[None, :]
    out[:, 110:129] = np.asarray(family_share, dtype=np.float64)[None, :]
    return out.astype(np.float32), candidates, texts


_NUMERIC = (int, float, bool)


def v3_extract(records, base=None):
    return _v3(records, base if base is not None else base_extract(records))


def _v3(records, base):
    x, meta, texts = base
    pages = collections.defaultdict(list)
    styles = collections.defaultdict(list)
    families = collections.defaultdict(list)
    seq = {}
    fonts = {}
    skey = {}
    for r in records:
        q = sequence(r)
        f = fsize(r)
        seq[id(r)] = q
        fonts[id(r)] = f
        k = (round(math.log(f) * 6), round(r["box"][0] / r["width"] * 20), q)
        skey[id(r)] = k
        pages[r["page"]].append(r)
        styles[k].append(r)
        families[q].append(r)
    body = med(
        [
            fonts[id(r)]
            for r in records
            if r["label"] == "text" and len(r["text"]) > 120
        ],
        20,
    )
    headingfonts = sorted(
        {
            round(math.log(fonts[id(r)] / body) * 6)
            for r in records
            if r["original_label"] in {"doc_title", "paragraph_title"}
        }
    )
    style_stats = {
        k: (
            sum(q["original_label"] in {"doc_title", "paragraph_title"} for q in v)
            / len(v),
            sum(q["label"] == "header" for q in v) / len(v),
            med([len(q["text"]) for q in v], 0),
            len({q["page"] for q in v}),
        )
        for k, v in styles.items()
    }
    family_stats = {
        k: (
            sum(q["original_label"] in {"doc_title", "paragraph_title"} for q in v)
            / len(v),
            med([fonts[id(q)] for q in v], 1),
            sum(q["level"] == 1 for q in v) / len(v),
        )
        for k, v in families.items()
    }
    page_stats = {}
    indices = {id(r): i for p in pages.values() for i, r in enumerate(p)}
    for page, rr in pages.items():
        pheads = [
            q for q in rr if q["original_label"] in {"doc_title", "paragraph_title"}
        ]
        pbody = [q for q in rr if q["label"] == "text" and len(q["text"]) > 120]
        page_stats[page] = (
            pheads,
            pbody,
            med([q["box"][0] / q["width"] for q in pbody], 0.1),
            med([q["box"][2] / q["width"] for q in pbody], 0.9),
            sum(len(q["text"]) for q in rr),
            max([fonts[id(q)] for q in rr], default=0.1),
        )
    level_stats = {}
    nfonts = max(len(headingfonts), 1)
    out = []
    for r, s in zip(meta, texts):
        page = pages[r["page"]]
        i = indices[id(r)]
        prev = page[max(i - 1, 0)]
        nxt = page[min(i + 1, len(page) - 1)]
        w, h = r["width"], r["height"]
        a, b, c, d = r["box"]
        f = fonts[id(r)]
        k = skey[id(r)]
        style = styles[k]
        numstyle = seq[id(r)]
        family = families[numstyle]
        ls = level_stats.get(k)
        if ls is None:
            lev = [q["level"] for q in style if q["level"]]
            ls = level_stats[k] = (
                med(lev, 0),
                min(lev, default=0),
                max(lev, default=0),
            )
        pheads, pbody, left, right, pagechars, pagemax = page_stats[r["page"]]
        ss = style_stats[k]
        fs = family_stats[numstyle]
        isbare = numstyle == "cn_bare"
        numbered = numstyle not in {"plain", "caps", "major"}
        feat = [
            (b - prev["box"][3]) / h,
            (nxt["box"][1] - d) / h,
            (a - prev["box"][0]) / w,
            (nxt["box"][0] - a) / w,
            int(i == 0),
            int(i == len(page) - 1),
            int(prev["level"] > 0),
            int(nxt["level"] > 0),
            a / w - left,
            c / w - right,
            (a + c) / w / 2 - (left + right) / 2,
            len(style),
            ss[0],
            ss[1],
            ls[0],
            ls[1],
            ls[2],
            ss[2],
            ss[3],
            len(family),
            fs[0],
            fs[1] / f,
            fs[2],
            len(pheads),
            len(pbody),
            pagechars,
            int(f >= pagemax * 0.9),
            int(isbare),
            int(numbered),
        ]
        feat += [int(bool(m.match(s))) for m in _V3_PATTERNS]
        feat += [
            len(headingfonts),
            bisect.bisect_left(headingfonts, round(math.log(f / body) * 6)) / nfonts,
        ]
        out.append(feat)
    return np.concatenate([x, np.asarray(out, dtype=np.float32)], axis=1), meta, texts


def _same(fast, ref, what):
    fx, fm, ft = fast
    rx, rm, rt = ref
    ok = (
        fx.dtype == rx.dtype
        and fx.shape == rx.shape
        and np.array_equal(fx, rx, equal_nan=True)
        and len(fm) == len(rm)
        and all(a is b for a, b in zip(fm, rm))
        and ft == rt
    )
    if ok:
        ok = np.array_equal(np.signbit(fx), np.signbit(rx))
    if not ok:
        raise AssertionError("heading_features_fast mismatch: " + what)
