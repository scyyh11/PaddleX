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

"""Code-listing recovery from OCR labels and semantic comment boundaries only."""
import collections
import copy
import re

from . import charcount, code_regions
from .runtime_context import readonly_soup as BeautifulSoup

ATTACH_HEADERS = True
UTILITY_STREAM = False


def lines(s):
    if "<table" in s:
        soup = BeautifulSoup(s, "html.parser")
        return [
            line
            for row in soup.find_all("tr")
            for line in " ".join(
                c.get_text(" ", strip=True)
                for c in row.find_all(["td", "th"], recursive=False)
            )
            .replace("\\n", "\n")
            .splitlines()
        ]
    return s.splitlines()


def start(s):
    t = s.strip()
    return bool(
        re.match(
            r"^(?:[cC]\s+)?(?:recursive\s+)?(?:subroutine|(?:real|double precision|integer)\s+function)\b",
            t,
            re.I,
        )
    )


def banner(s):
    t = s.strip()
    # Decorative comments label a coherent code section. Ordinary inline comments do not.
    return (
        len(t) < 220
        and bool(re.search(r"[A-Za-z]{4}", t))
        and bool(
            re.match(r"^(?:[cC!]\s*)?(?:\*{3,}|★{2,})\s*\w", t)
            or re.search(r"\*{4,}\s*[A-Z][A-Z ]{8,}", t)
        )
    )


def apply(pre, final, mode="single"):
    active = []
    weak = []
    for p in pre:
        bs = [b for b in p["parsing_res_list"] if b["block_content"].strip()]
        text = " ".join(code_regions.plain(b["block_content"]) for b in bs)
        count = len(code_regions.CODE.findall(text))
        vertical = sum(
            b["block_bbox"][3] - b["block_bbox"][1]
            > 3 * (b["block_bbox"][2] - b["block_bbox"][0])
            for b in bs
            if b["block_label"] in code_regions.ALLOW
        )
        n = sum(b["block_label"] in code_regions.ALLOW for b in bs)
        active.append(count >= 3 and vertical >= max(2, n * 0.3))
        weak.append(
            (
                vertical >= max(2, n * 0.4)
                or count >= 3
                or len(
                    re.findall(
                        r"\bcall\s+\w|\b(?:print|read)\s*\*|[a-zA-Z]\w*\([^)]*\)\s*=",
                        text,
                    )
                )
                >= 3
                or any(b["block_label"] == "table" for b in bs)
            )
            and charcount.cjk(text) < len(text) * 0.03
        )
    if sum(active) < 5:
        return code_regions.apply(pre, final)
    for _ in range(len(pre)):
        active = [
            a
            or weak[i]
            and ((i > 0 and active[i - 1]) or (i + 1 < len(active) and active[i + 1]))
            for i, a in enumerate(active)
        ]
    groups = collections.defaultdict(list)
    for pi, p in enumerate(final):
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] == "table":
                groups[b.get("global_group_id", ("self", pi, bi))].append((pi, bi))
    convertible = {
        k
        for keys in groups.values()
        if all(
            active[pi] and pre[pi]["parsing_res_list"][bi]["block_label"] == "table"
            for pi, bi in keys
        )
        for k in keys
    }
    logs = []
    decl_count = 0
    decl_chars = 0
    for pi, (p, q) in enumerate(zip(pre, final)):
        if not active[pi]:
            continue
        selected = [
            (bi, b)
            for bi, b in enumerate(p["parsing_res_list"])
            if b["block_content"].strip()
            and (b["block_label"] in code_regions.ALLOW or (pi, bi) in convertible)
        ]
        if not selected:
            continue
        selected.sort(
            key=lambda item: (item[1]["block_bbox"][0], -item[1]["block_bbox"][3])
        )
        chunks = []
        current = []
        current_start = False
        for bi, b in selected:
            for s in lines(b["block_content"]):
                s = s.strip()
                if not s:
                    continue
                major = (
                    banner(s)
                    and (mode == "banners" or bool(re.search(r"\*{5,}|★{3,}", s)))
                    or len(s) < 180
                    and bool(re.match(r"^(?:[cC!]\s*)?[A-Za-z][^=]*\*{4,}\s*$", s))
                )
                if mode in {"capital", "phases", "comments"}:
                    major = major or bool(
                        re.match(r"^(?:[cC!]\s*)?(?:\*{3,}|★{2,})\s*[A-Z]", s)
                    )
                isstart = (
                    start(s)
                    or mode
                    in {
                        "banners",
                        "major",
                        "capital",
                        "canonical",
                        "phases",
                        "comments",
                    }
                    and (
                        major
                        or re.fullmatch(
                            r"(?:[cC!]\s*)?(?:Declarations?|General Comments|Populating the subsystem databases)",
                            s,
                            re.I,
                        )
                    )
                )
                if mode in {"phases", "comments"}:
                    if isstart:
                        decl_count = decl_chars = 0
                    statement = re.sub(r"^[cC]\s+", "", s).strip()
                    declaration = bool(
                        re.match(
                            r"^(?:implicit|dimension|common|parameter|integer|double precision|real|logical|character|external|save|equivalence|data)\b",
                            statement,
                            re.I,
                        )
                    )
                    executable = bool(
                        re.match(
                            r"^(?:[A-Za-z]\w*(?:\([^)]*\))?\s*(?:=|[-—]{2,})|call\s|do\s+\d|if\s*\(|print\s*\*|write\s*\()",
                            statement,
                            re.I,
                        )
                    )
                    if decl_count >= 6 and decl_chars > 500 and executable:
                        isstart = True
                        decl_count = decl_chars = 0
                    if declaration:
                        decl_count += 1
                        decl_chars += len(s)
                    if mode == "comments" and re.match(
                        r"^[!！]\s*(?:changing|converting)\s+(?:back|from|to)\b",
                        s,
                        re.I,
                    ):
                        isstart = True
                if mode == "canonical":
                    s = re.sub(r"^[cC]\s+(?=[A-Za-z*!])", "! ", s)
                if ATTACH_HEADERS and current and major:
                    recent = [
                        t for _, t in current[-4:] if not re.fullmatch(r"[cC!* ]+", t)
                    ]
                    if len(recent) == 1 and start(recent[0]):
                        isstart = False
                if UTILITY_STREAM and start(s):
                    isstart = False
                if current and (
                    isstart
                    or not UTILITY_STREAM
                    and re.fullmatch(r"end", current[-1][1], re.I)
                ):
                    chunks.append((current, current_start))
                    current = []
                if not current:
                    current_start = isstart
                current.append((bi, s))
        if current:
            chunks.append((current, current_start))
        selected_ids = {i for i, _ in selected}
        anchor = selected[0][0]
        new = []
        for ci, (parts, head) in enumerate(chunks):
            b = copy.deepcopy(p["parsing_res_list"][parts[0][0]])
            b.update(
                block_label="text",
                block_content=" ".join(t for _, t in parts),
                _code_region=True,
                _code_start=head,
                group_id=f"code-{pi}-{ci}",
                global_group_id=f"code-{pi}-{ci}",
                global_block_id=f"code-{pi}-{ci}",
            )
            new.append(b)
        for page in (p, q):
            result = []
            for bi, b in enumerate(page["parsing_res_list"]):
                if bi == anchor:
                    result.extend(copy.deepcopy(new))
                elif bi not in selected_ids:
                    result.append(b)
            page["parsing_res_list"] = result
        logs.append(
            {
                "kind": "code_semantic_recovery",
                "page": pi + 1,
                "chunks": len(chunks),
                "tables": sum((pi, i) in convertible for i in selected_ids),
                "mode": mode,
            }
        )
    return logs
