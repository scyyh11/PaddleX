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

"""Trim long periodic OCR decoding tails using a linear-time Z algorithm."""


def clean(s):
    if len(s) < 150:
        return s
    # Generation loops accumulate at the end of an output. Bound work per block.
    tail = s[-24000:]
    r = tail[::-1]
    n = len(r)
    z = [0] * n
    l = rr = 0
    # Only positions i <= 100 can trigger the return below, and z[i] depends only on earlier
    # positions, so the scan stops after i = 100 with the same result as scanning to n.
    for i in range(1, min(n, 101)):
        if i < rr:
            z[i] = min(rr - i, z[i - l])
        while i + z[i] < n and r[z[i]] == r[i + z[i]]:
            z[i] += 1
        if i + z[i] > rr:
            l, rr = i, i + z[i]
        if (
            2 <= i <= 100
            and z[i] >= 5 * i
            and z[i] + i >= 120
            and any(c.isalpha() for c in r[:i])
        ):
            return s[: len(s) - z[i]].rstrip() + s[len(s.rstrip()) :]
    return s


def apply(pre, final):
    logs = []
    for pi, (p, q) in enumerate(zip(pre, final)):
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            if a["block_label"] not in {
                "text",
                "content",
                "abstract",
                "reference_content",
            }:
                continue
            s = a["block_content"]
            t = clean(s)
            if s != t:
                a["block_content"] = t
                if b["block_content"] == s:
                    b["block_content"] = t
                logs.append(
                    {
                        "kind": "decoding_loop",
                        "page": pi + 1,
                        "index": bi,
                        "characters_removed": len(s) - len(t),
                    }
                )
    return logs
