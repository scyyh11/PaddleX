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

"""Document-relative geometry and row/column continuity features."""
import math
import re

import numpy as np

from .column_alignment import align, kind


def norm(s):
    return re.sub(r"\s+", "", s).lower()


def numeric_sequence(rows):
    vals = []
    for row in rows:
        if row and re.fullmatch(r"\s*\d{1,5}[.)、．]?\s*", row[0]):
            vals.append(float(re.sub(r"\D", "", row[0])))
    return vals


def row_stats(row):
    n = max(len(row), 1)
    k = [kind(s) for s in row]
    return [sum(v == c for v in k) / n for c in range(6)] + [
        math.log1p(sum(len(s) for s in row) / n)
    ]


_TOTAL = re.compile(r"(?:合计|合計|总计|總計|grand total|^total)\s*$", re.I)
_LOWER_START = re.compile(r"^[a-z，,）)]")
_STOP = re.compile(r"[。.!?！？]$")
_CN_ITEM = re.compile(r"^\s*[（(]?[一二三四五六七八九十]+[、．.）)]")
_TOKEN = re.compile(r"[a-z]{2,}|[\u4e00-\u9fff]{1,}")


def _fragment(f):
    """Values of extras() that depend on one fragment only, computed once per fragment."""
    from . import runtime_context

    store = runtime_context.analysis_cache.get(None)
    cache = (
        store.setdefault(("table_features_v2_fragment",), {})
        if store is not None
        else {}
    )
    hit = cache.get(id(f))
    if hit is not None and hit[0] is f:
        return hit[1]
    t = f["tab"]
    rows = t["rows"]
    first = rows[0] if rows else []
    last = rows[-1] if rows else []
    token = lambda rr: set(
        _TOKEN.findall((" ".join(" ".join(row) for row in rr)).lower())
    )
    v = {
        "first": first,
        "last": last,
        "span": max(0.2, f["doc_bottom"] - f["doc_top"]),
        "total_last": float(bool(last) and bool(_TOTAL.search(last[0]))),
        "lower_first": float(bool(first) and bool(_LOWER_START.match(first[0]))),
        "stop_last": float(bool(last) and bool(_STOP.search(last[0]))),
        "cn_item": float(any(_CN_ITEM.match(row[0]) for row in rows if row)),
        "numbers": numeric_sequence(rows),
        "head_tokens": token(rows[:2]),
        "tail_tokens": token(rows[-2:]),
        "norm_rows": [
            [norm(x) for x in rows[i]] if len(rows) > i else [] for i in range(3)
        ],
        "stats_first": row_stats(first),
        "stats_last": row_stats(last),
    }
    cache[id(f)] = (f, v)
    return v


def extras(a, b):
    """Same vector as _extras_reference; one-fragment terms come from _fragment()."""
    ta, tb = a["tab"], b["tab"]
    ra, rb = ta["rows"], tb["rows"]
    fa, fb = _fragment(a), _fragment(b)
    aa = fa["last"]
    bb = fb["first"]
    span = fa["span"]
    spb = fb["span"]
    geom = [
        (a["doc_bottom"] - a["bbox"][3]) / span,
        (b["bbox"][1] - b["doc_top"]) / spb,
        (a["bbox"][3] - a["bbox"][1]) / span,
        (b["bbox"][3] - b["bbox"][1]) / spb,
    ]
    n = max(len(aa), len(bb), 1)
    empty = [(not x.strip(), not y.strip()) for x, y in zip(aa, bb)]
    boundary = [
        sum(x and y for x, y in empty) / n,
        sum(x != y for x, y in empty) / n,
        fa["total_last"],
        fb["lower_first"],
        fa["stop_last"],
        fa["cn_item"],
        fb["cn_item"],
    ]
    va, vb = fa["numbers"], fb["numbers"]
    seq = [len(va) / max(len(ra), 1), len(vb) / max(len(rb), 1)]
    seq += [
        float(bool(va and vb)),
        float(bool(va and vb) and va[-1] + 1 == vb[0]),
        float(bool(va and vb) and vb[0] <= va[-1]),
    ]
    if va and vb:
        seq += [math.copysign(math.log1p(abs(vb[0] - va[-1])), vb[0] - va[-1])]
    else:
        seq += [0]
    if 1 <= min(ta["ncols"], tb["ncols"]) and max(ta["ncols"], tb["ncols"]) <= 25:
        cost, pairs = align(ta, tb)
        dp = [cost, len(pairs) / max(ta["ncols"], tb["ncols"])]
    else:
        dp = [2, 0]
    xa, xb = fa["head_tokens"], fb["head_tokens"]
    xt = fa["tail_tokens"]
    overlap = [len(xa & xb) / max(len(xa | xb), 1), len(xt & xb) / max(len(xt | xb), 1)]
    schema = []
    for i in range(3):
        x = fa["norm_rows"][i]
        y = fb["norm_rows"][i]
        schema += [
            float(len(x) == len(y)),
            sum(v == w for v, w in zip(x, y)) / max(len(x), len(y), 1),
        ]
    out = np.array(
        geom
        + boundary
        + seq
        + dp
        + overlap
        + schema
        + fa["stats_last"]
        + fb["stats_first"],
        dtype=np.float32,
    )
    if VERIFY:
        ref = _extras_reference(a, b)
        if not (
            out.shape == ref.shape
            and np.array_equal(out, ref)
            and np.array_equal(np.signbit(out), np.signbit(ref))
        ):
            raise AssertionError("table_features_v2: extras differ")
    return out


VERIFY = False


def _extras_reference(a, b):
    ta, tb = a["tab"], b["tab"]
    ra, rb = ta["rows"], tb["rows"]
    aa = ra[-1] if ra else []
    bb = rb[0] if rb else []
    span = max(0.2, a["doc_bottom"] - a["doc_top"])
    spb = max(0.2, b["doc_bottom"] - b["doc_top"])
    geom = [
        (a["doc_bottom"] - a["bbox"][3]) / span,
        (b["bbox"][1] - b["doc_top"]) / spb,
        (a["bbox"][3] - a["bbox"][1]) / span,
        (b["bbox"][3] - b["bbox"][1]) / spb,
    ]
    n = max(len(aa), len(bb), 1)
    empty = [(not x.strip(), not y.strip()) for x, y in zip(aa, bb)]
    boundary = [
        sum(x and y for x, y in empty) / n,
        sum(x != y for x, y in empty) / n,
        float(
            bool(aa)
            and bool(
                re.search(
                    r"(?:合计|合計|总计|總計|grand total|^total)\s*$", aa[0], re.I
                )
            )
        ),
        float(bool(bb) and bool(re.match(r"^[a-z，,）)]", bb[0]))),
        float(bool(aa) and bool(re.search(r"[。.!?！？]$", aa[0]))),
        float(
            any(
                re.match(r"^\s*[（(]?[一二三四五六七八九十]+[、．.）)]", row[0])
                for row in ra
                if row
            )
        ),
        float(
            any(
                re.match(r"^\s*[（(]?[一二三四五六七八九十]+[、．.）)]", row[0])
                for row in rb
                if row
            )
        ),
    ]
    va, vb = numeric_sequence(ra), numeric_sequence(rb)
    seq = [len(va) / max(len(ra), 1), len(vb) / max(len(rb), 1)]
    seq += [
        float(bool(va and vb)),
        float(bool(va and vb) and va[-1] + 1 == vb[0]),
        float(bool(va and vb) and vb[0] <= va[-1]),
    ]
    if va and vb:
        seq += [math.copysign(math.log1p(abs(vb[0] - va[-1])), vb[0] - va[-1])]
    else:
        seq += [0]
    # DP is limited to plausible table widths; wider malformed grids retain a sentinel.
    if 1 <= min(ta["ncols"], tb["ncols"]) and max(ta["ncols"], tb["ncols"]) <= 25:
        cost, pairs = align(ta, tb)
        dp = [cost, len(pairs) / max(ta["ncols"], tb["ncols"])]
    else:
        dp = [2, 0]
    token = lambda rows: set(
        re.findall(
            r"[a-z]{2,}|[\u4e00-\u9fff]{1,}",
            (" ".join(" ".join(row) for row in rows)).lower(),
        )
    )
    xa, xb = token(ra[:2]), token(rb[:2])
    xt = token(ra[-2:])
    overlap = [len(xa & xb) / max(len(xa | xb), 1), len(xt & xb) / max(len(xt | xb), 1)]
    schema = []
    for i in range(3):
        x = ra[i] if len(ra) > i else []
        y = rb[i] if len(rb) > i else []
        schema += [
            float(len(x) == len(y)),
            sum(norm(v) == norm(w) for v, w in zip(x, y)) / max(len(x), len(y), 1),
        ]
    return np.array(
        geom + boundary + seq + dp + overlap + schema + row_stats(aa) + row_stats(bb),
        dtype=np.float32,
    )
