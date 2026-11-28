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

"""Fast recursive deepcopy for JSON-shaped OCR page dicts."""

import copy

_ATOMIC = frozenset([str, int, float, bool, type(None)])


def deepcopy(obj):
    return _copy(obj, {})


def _copy(obj, memo):
    t = type(obj)
    if t in _ATOMIC:
        return obj
    if t is dict:
        return _dict(obj, memo)
    if t is list:
        return _list(obj, memo)
    return copy.deepcopy(obj, memo)


def _dict(src, memo):
    found = memo.get(id(src))
    if found is not None:
        return found
    out = {}
    memo[id(src)] = out
    atomic = _ATOMIC
    for k, v in src.items():
        if type(k) not in atomic:
            k = _copy(k, memo)
        t = type(v)
        if t in atomic:
            out[k] = v
        elif t is dict:
            out[k] = _dict(v, memo)
        elif t is list:
            out[k] = _list(v, memo)
        else:
            out[k] = copy.deepcopy(v, memo)
    _keep(src, memo)
    return out


def _list(src, memo):
    found = memo.get(id(src))
    if found is not None:
        return found
    out = []
    memo[id(src)] = out
    append = out.append
    atomic = _ATOMIC
    for v in src:
        t = type(v)
        if t in atomic:
            append(v)
        elif t is dict:
            append(_dict(v, memo))
        elif t is list:
            append(_list(v, memo))
        else:
            append(copy.deepcopy(v, memo))
    _keep(src, memo)
    return out


def _keep(src, memo):
    try:
        memo[id(memo)].append(src)
    except KeyError:
        memo[id(memo)] = [src]


def copy_pages(
    pages, page_keys=("layout_det_res",), block_keys=("block_polygon_points",)
):
    """Deep-copy OCR page dicts while sharing immutable ``page_keys`` and ``block_keys``."""
    if type(pages) is not list or not all(type(p) is dict for p in pages):
        return deepcopy(pages)
    skeleton = []
    keep = []
    for p in pages:
        q = dict(p)
        saved = {}
        for k in page_keys:
            if k in q:
                saved[k] = q[k]
                q[k] = None
        blocks = q.get("parsing_res_list")
        kept_blocks = None
        if type(blocks) is list and block_keys:
            nb = []
            kept_blocks = []
            for b in blocks:
                if type(b) is dict and any(k in b for k in block_keys):
                    c = dict(b)
                    s = {}
                    for k in block_keys:
                        if k in c:
                            s[k] = c[k]
                            c[k] = None
                    nb.append(c)
                    kept_blocks.append(s)
                else:
                    nb.append(b)
                    kept_blocks.append(None)
            q["parsing_res_list"] = nb
        skeleton.append(q)
        keep.append((saved, kept_blocks))
    out = deepcopy(skeleton)
    for q, (saved, kept_blocks) in zip(out, keep):
        for k, v in saved.items():
            q[k] = v
        if kept_blocks is not None:
            for b, s in zip(q["parsing_res_list"], kept_blocks):
                if s:
                    for k, v in s.items():
                        b[k] = v
    return out
