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

"""Prediction-only fragment geometry and document-internal style references."""
import math
import re
import unicodedata

import numpy as np

from .heading_ml_features import fsize, norm, pair_features

CN = "一二三四五六七八九十百千零〇两兩"


def sequence(r):
    s = unicodedata.normalize("NFKC", r["text"]).strip()
    fam = r["family"][0]
    m = re.match(r"^([" + CN + r"]+)(?:[、.\s]+|(?=[\u4e00-\u9fff]))", s)
    if m and len(m[1]) <= 4:
        return "cn_bare" if fam == "plain" else fam
    if fam in {"plain", "caps", "major"}:
        return fam
    return fam + ":" + str(r["family"][1]) + (":paren" if s.startswith("(") else "")


def style_refs(meta, i):
    if not i:
        return []
    b = meta[i]
    s = sequence(b)
    f = fsize(b)
    refs = {i - 1: 1.0}
    # Revisit the most recent comparable style, even when intervening subheads exist.
    for mode in ["sequence", "font", "text"]:
        found = 0
        for j in range(i - 1, max(-1, i - 120), -1):
            a = meta[j]
            if mode == "sequence":
                ok = s not in {"plain", "caps", "major"} and sequence(a) == s
            elif mode == "font":
                ok = (
                    sequence(a) == s
                    and abs(math.log(fsize(a) / f)) < 0.12
                    and abs(a["box"][0] / a["width"] - b["box"][0] / b["width"]) < 0.04
                )
            else:
                ok = norm(a["text"]) == norm(b["text"])
            if ok:
                refs[j] = max(refs.get(j, 0), 1.0 if mode == "sequence" else 0.7)
                found += 1
                if found == 2:
                    break
    return sorted(refs.items())


def merge_feature(a, b, fa, fb):
    ax, ay, ar, ab = a["box"]
    bx, by, br, bb = b["box"]
    w = b["width"]
    h = b["height"]
    return np.concatenate(
        [
            pair_features(a, b, fa, fb),
            np.array(
                [
                    (by - ab) / h,
                    (bx - ar) / w,
                    max(0, min(ar, br) - max(ax, bx)) / max(1, min(ar - ax, br - bx)),
                    max(0, min(ab, bb) - max(ay, by)) / max(1, min(ab - ay, bb - by)),
                    b["index"] - a["index"],
                    len(a["text"]) + len(b["text"]),
                    int(sequence(a) == sequence(b)),
                    int(a["text"].rstrip().endswith((".", ":", "。", "："))),
                ],
                dtype=np.float32,
            ),
        ]
    )
