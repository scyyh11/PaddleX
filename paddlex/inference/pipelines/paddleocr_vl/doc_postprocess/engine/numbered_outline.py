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

"""Restore explicit numbering ancestry using the current document's outline."""
import collections
import re
import statistics
import unicodedata

CN = "零一二三四五六七八九十百千〇两兩"


def cnint(s):
    values = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "兩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    total = 0
    num = 0
    if s.isdigit():
        return int(s)
    for c in s:
        if c in values:
            num = values[c]
        else:
            total += (num or 1) * {"十": 10, "百": 100, "千": 1000}.get(c, 1)
            num = 0
    return total + num


def token(s):
    s = unicodedata.normalize("NFKC", s).strip()
    m = re.match(r"^第([" + CN + r"\d]+)([章部篇卷节節])", s)
    if m:
        return (
            "part" if m[2] in "部篇卷" else "chapter" if m[2] == "章" else "section",
            (cnint(m[1]),),
        )
    m = re.match(r"^(PART|CHAPTER|SECTION)\s+(\d+)\b", s, re.I)
    if m:
        return (m[1].lower(), (int(m[2]),))
    m = re.match(r"^(\d{1,3}(?:\.\d{1,3})+)(?:[.\s]|(?=[A-Za-z\u4e00-\u9fff]))", s)
    if m:
        ns = tuple(map(int, m[1].split(".")))
        while len(ns) > 1 and ns[-1] == 0:
            ns = ns[:-1]
        return ("decimal", ns)
    m = re.match(r"^([" + CN + r"]{1,5})(?:[、.\s]+|(?=[\u4e00-\u9fff]))", s)
    if m:
        return ("cn", (cnint(m[1]),))
    m = re.match(r"^\(([" + CN + r"]{1,5})\)", s)
    if m:
        return ("cn_paren", (cnint(m[1]),))
    return None


def apply(results, log):
    heads = []
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if b.label not in {"doc_title", "paragraph_title"} or not b.content.strip():
                continue
            level = 1 if b.label == "doc_title" else getattr(b, "title_level", 1) + 1
            heads.append(
                {"b": b, "page": pi + 1, "old": level, "token": token(b.content)}
            )
    counts = collections.Counter(r["token"][0] for r in heads if r["token"])
    chapters = [r for r in heads if r["token"] and r["token"][0] == "chapter"]
    nums = {r["token"][1][0] for r in chapters}
    coherent = len(nums) >= 3 and len(nums & set(range(1, max(nums) + 1))) >= 3
    if not coherent and counts["decimal"] < 8:
        return
    base = round(statistics.median(r["old"] for r in chapters)) if chapters else None
    active = {}
    decimal = {}
    major = None
    for r in heads:
        t = r["token"]
        new = r["old"]
        if not t:
            continue
        fam, seq = t
        if fam == "part":
            active = {"part": r}
            major = r
            decimal = {}
        elif fam == "chapter" and coherent:
            new = active["part"]["new"] + 1 if "part" in active else base
            active = {k: v for k, v in active.items() if k == "part"}
            active["chapter"] = r
            major = r
            decimal = {}
        elif fam == "section" and coherent and "chapter" in active:
            new = active["chapter"]["new"] + 1
            active = {k: v for k, v in active.items() if k in {"part", "chapter"}}
            active["section"] = r
            major = r
        elif fam == "decimal":
            if seq[:-1] in decimal:
                new = decimal[seq[:-1]]["new"] + 1
            elif (
                coherent
                and "chapter" in active
                and seq[0] == active["chapter"]["token"][1][0]
            ):
                new = active["chapter"]["new"] + len(seq) - 1
            elif seq in decimal:
                new = decimal[seq]["new"]
            elif len(seq) > 1:
                siblings = [
                    v
                    for k, v in decimal.items()
                    if len(k) == len(seq) and k[:-1] == seq[:-1]
                ]
                if siblings:
                    new = siblings[-1]["new"]
            decimal[seq] = r
        elif coherent and fam == "cn" and major is not None and counts["cn"] >= 4:
            new = major["new"] + 1
            active["cn"] = r
            active.pop("cn_paren", None)
        elif coherent and fam == "cn_paren" and "cn" in active:
            new = active["cn"]["new"] + 1
            active["cn_paren"] = r
        r["new"] = min(6, max(1, new))
        b = r["b"]
        if r["new"] != r["old"]:
            b.label = "doc_title" if r["new"] == 1 else "paragraph_title"
            b.title_level = r["new"] - 1
            log.append(
                {
                    "kind": "explicit_numbering_ancestry",
                    "page": r["page"],
                    "text": b.content,
                    "old": r["old"],
                    "new": r["new"],
                    "family": fam,
                }
            )
