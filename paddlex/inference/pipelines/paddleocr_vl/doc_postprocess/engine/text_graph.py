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

"""Prediction-only paragraph repair. Each source fragment emitted once."""
import copy
import re

TEXT = {
    "text",
    "abstract",
    "reference_content",
    "vertical_text",
    "content",
    "algorithm",
    "code",
    "code_txt",
}
SKIP = {
    "header",
    "header_image",
    "footer",
    "footer_image",
    "number",
    "footnote",
    "aside_text",
    "vision_footnote",
    "formula_number",
}
END = re.compile(r'[。！？.!?][”’"\'）)\]]*\s*$')
LIST = re.compile(
    r"^\s*(?:[•●▪■◆※]|\(?\d+[.)、]\s+|[（(][一二三四五六七八九十\d]+[）)]|[一二三四五六七八九十]+[、．])"
)
SPEAKER = re.compile(
    r"\b(?:DR\.|MR\.|MS\.|MRS\.|CHAIRMAN|CHAIRWOMAN)\s+[A-Z][A-Z .\-]{1,35}:"
)


def squash(s):
    return re.sub(r"\n\s*\n+", "\n", s)


def join(x, y):
    x, y = x.rstrip(), y.lstrip()
    if re.search(r"[a-zA-Z]-$", x) and re.match(r"^[a-z]", y):
        return x[:-1] + y
    return (
        x
        + (
            ""
            if re.search(r"[\u4e00-\u9fff]$", x)
            or re.match(r"^[\u4e00-\u9fff，。；：！？]", y)
            else " "
        )
        + y
    )


def apply(prepared, final, mode):
    log = []
    if mode == "squash":
        for pi, p in enumerate(final):
            for b in p["parsing_res_list"]:
                if b["block_label"] in TEXT:
                    b["block_content"] = squash(b["block_content"])
        return log
    # Restore only text fragments. All table and heading decisions stay in place.
    for pi, (p, q) in enumerate(zip(prepared, final)):
        assert len(p["parsing_res_list"]) == len(q["parsing_res_list"]), (
            pi,
            len(p["parsing_res_list"]),
            len(q["parsing_res_list"]),
        )
        for bi, (a, b) in enumerate(zip(p["parsing_res_list"], q["parsing_res_list"])):
            if b["block_label"] in TEXT and a["block_label"] in TEXT:
                b["block_content"] = squash(a["block_content"])
                b["global_group_id"] = b.get("global_block_id", 10000 * pi + bi)
    transcript = sum(
        bool(SPEAKER.search(b["block_content"]))
        for p in prepared
        for b in p["parsing_res_list"]
    ) >= max(5, len(prepared) * 0.3)
    if transcript:
        # Whole-page transcript regions contain multiple turns, while a leading
        # continuation before the first speaker belongs to the previous page.
        for pi, p in enumerate(final):
            blocks = []
            for bi, b in enumerate(p["parsing_res_list"]):
                s = b["block_content"]
                m = SPEAKER.search(s)
                if (
                    b["block_label"] in TEXT
                    and m
                    and m.start() > 2
                    and re.match(r"^[a-z]", s.strip())
                ):
                    c = copy.deepcopy(b)
                    c["block_content"] = s[: m.start()].strip()
                    c["_leading_continuation"] = True
                    b["block_content"] = s[m.start() :].strip()
                    blocks.append(c)
                    log.append(
                        {
                            "kind": "transcript_prefix",
                            "page": pi + 1,
                            "chars": len(c["block_content"]),
                        }
                    )
                blocks.append(b)
            p["parsing_res_list"] = blocks

    # In-page adjacency is geometric; headings and independent paragraphs delimit it.
    def eligible(b):
        return b["block_label"] in TEXT and bool(b["block_content"].strip())

    def compatible(a, b, cross=False):
        x, y = a["block_content"].strip(), b["block_content"].strip()
        if not eligible(a) or not eligible(b):
            return False
        if (
            "content" in {a["block_label"], b["block_label"]}
            and a["block_label"] != b["block_label"]
        ):
            return False
        if transcript:
            if not cross:
                return False
            return bool(
                b.get("_leading_continuation")
                or re.match(r"^[a-z]", y)
                and not SPEAKER.match(y)
            ) and not END.search(x)
        if LIST.match(y) or x.endswith((":", "：")):
            return False
        if len(x) < 2 or len(y) < 2:
            return False
        return not END.search(x) or bool(re.match(r"^[a-z,;，；）)]", y))

    def merge(a, b, kind, pi):
        a["block_content"] = join(a["block_content"], b["block_content"])
        b["block_content"] = ""
        b["global_group_id"] = a.get("global_group_id", a.get("global_block_id"))
        log.append({"kind": kind, "page": pi + 1})

    for pi, p in enumerate(final):
        vis = [
            b
            for b in p["parsing_res_list"]
            if b["block_label"] not in SKIP and b["block_content"].strip()
        ]
        for a, b in reversed(list(zip(vis, vis[1:]))):
            ax, ay, ar, ab = a["block_bbox"]
            bx, by, br, bb = b["block_bbox"]
            w, h = p["width"], p["height"]
            col = by < ab - 0.15 * h and bx > ax + 0.2 * w
            line = (
                by >= ay
                and -h * 0.008 <= by - ab <= h * 0.025
                and abs(ax - bx) < w * 0.05
                and abs(ar - br) < w * 0.13
            )
            if compatible(a, b) and (col or mode == "region" and line):
                merge(a, b, "column" if col else "fragment", pi)
    for pi in range(len(final) - 1, 0, -1):
        p, q = final[pi - 1 : pi + 1]
        # Images/tables/captions do not interrupt a continuing narrative. A heading does.
        va = [
            b
            for b in p["parsing_res_list"]
            if b["block_label"] in TEXT | {"paragraph_title", "doc_title"}
            and b["block_content"].strip()
        ]
        vb = [
            b
            for b in q["parsing_res_list"]
            if b["block_label"] in TEXT | {"paragraph_title", "doc_title"}
            and b["block_content"].strip()
        ]
        if va and vb and compatible(va[-1], vb[0], True):
            merge(va[-1], vb[0], "cross", pi)
    return log
