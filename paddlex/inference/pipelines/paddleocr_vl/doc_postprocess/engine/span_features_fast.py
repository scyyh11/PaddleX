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

"""Exact range features with shared document metadata and integer prefix counts."""
import math

import numpy as np

from .heading_ml_features import fsize, number


def prepare(meta, x):
    n = len(meta)
    fonts = np.array([fsize(r) for r in meta])
    nums = [number(r["text"]) for r in meta]
    families = {tuple(r["family"]) for r in meta}
    counts = {
        fam: np.r_[0, np.cumsum([tuple(r["family"]) == fam for r in meta])]
        for fam in families
    }
    prefixes = {}
    for num in set(nums):
        prefixes[num] = np.r_[
            0, np.cumsum([bool(num and q[: len(num)] == num) for q in nums])
        ]

    def features(start, end):
        a = meta[start]
        last = meta[end]
        nxt = meta[end + 1] if end + 1 < n else None
        first = start + 1 if start < end else start
        fs = fonts[start + 1 : end + 1]
        same = (
            counts[tuple(a["family"])][end + 1] - counts[tuple(a["family"])][start + 1]
        )
        pref = prefixes[nums[start]][end + 1] - prefixes[nums[start]][start + 1]
        extra = [
            end == n - 1,
            start == 0,
            end - start,
            math.log1p(end - start),
            (end - start) / max(n, 1),
            last["page"] - a["page"],
            float(np.max(fs)) / fonts[start] if len(fs) else 1.0,
            float(np.mean(fs)) / fonts[start] if len(fs) else 1.0,
            float(np.min(fs)) / fonts[start] if len(fs) else 1.0,
            same / max(1, end - start),
            pref / max(1, end - start),
            fonts[first] / fonts[start],
            fonts[end + 1] / fonts[start] if nxt else 1.0,
            int(nxt["family"] == a["family"]) if nxt else 0,
            int(nxt["page"] == last["page"]) if nxt else 0,
        ]
        return np.concatenate(
            [
                x[start],
                x[end],
                x[end + 1] if end + 1 < n else np.zeros_like(x[start]),
                np.asarray(extra, dtype=np.float32),
            ]
        )

    return features
