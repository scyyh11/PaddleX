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

"""Recover horizontal panels with independent left-to-right reading order."""
import collections


def detect(p):
    bs = p["parsing_res_list"]
    w, h = p["width"], p["height"]
    if any(b["block_label"] == "table" for b in bs):
        return None
    main = [
        i
        for i, b in enumerate(bs)
        if b["block_label"] in {"text", "abstract"}
        and 0.15 * w < b["block_bbox"][2] - b["block_bbox"][0] < 0.43 * w
        and 0.1 * h < b["block_bbox"][1]
        and b["block_bbox"][3] < 0.94 * h
    ]
    if len(main) < 12:
        return None
    xs = sorted(bs[i]["block_bbox"][0] for i in main)
    split = max(zip(xs, xs[1:]), key=lambda z: z[1] - z[0])
    gutter = sum(split) / 2
    if split[1] - split[0] < 0.16 * w:
        return None
    left = [i for i in main if bs[i]["block_bbox"][0] < gutter]
    right = [i for i in main if bs[i]["block_bbox"][0] >= gutter]
    if min(len(left), len(right)) < 4:
        return None
    intervals = sorted((bs[i]["block_bbox"][1], bs[i]["block_bbox"][3]) for i in main)
    union = []
    for y, b in intervals:
        if not union or y > union[-1][1]:
            union.append([y, b])
        else:
            union[-1][1] = max(union[-1][1], b)
    cuts = [
        (a[1] + b[0]) / 2 for a, b in zip(union, union[1:]) if b[0] - a[1] > 0.045 * h
    ]
    if len(cuts) < 2:
        return None

    def band(i):
        return sum(bs[i]["block_bbox"][1] > c for c in cuts)

    bands = collections.defaultdict(list)
    for i in main:
        bands[band(i)].append(i)
    if len(bands) != len(cuts) + 1 or any(
        len(ii) < 3 or not any(i in left for i in ii) or not any(i in right for i in ii)
        for ii in bands.values()
    ):
        return None
    return {
        "bands": {i: band(i) for i in range(len(bs))},
        "gutter": gutter,
        "cuts": cuts,
        "main": main,
    }


def order(p, info):
    bs = p["parsing_res_list"]
    w, h = p["width"], p["height"]
    main = info["main"]
    top = min(bs[i]["block_bbox"][1] for i in main)
    bottom = max(bs[i]["block_bbox"][3] for i in main)

    def key(i):
        b = bs[i]
        x, y, r, bot = b["block_bbox"]
        band = info["bands"][i]
        if bot < top:
            band = -1
        elif y > bottom:
            band = len(info["cuts"]) + 1
        col = 0 if x < info["gutter"] else 1
        if r - x < 0.14 * w and r < info["gutter"] - 0.15 * w:
            col = -1
        return band, col, y, i

    return sorted(range(len(bs)), key=key)
