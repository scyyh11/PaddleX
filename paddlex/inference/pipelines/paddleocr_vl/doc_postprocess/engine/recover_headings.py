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

from .runtime_context import memoize

"""Recover heading blocks from predicted text and document-local evidence only.

No benchmark IDs, annotations, pretrained classifier, or reference outputs.
Numbering alone is insufficient: sibling typography and following content are
required. Decisions are made before paragraph joining to preserve boundaries.
"""
import collections
import json
import math
import pathlib
import re
import statistics
import unicodedata

CN = "一二三四五六七八九十百千零〇两兩"
MAJOR = {
    "abstract",
    "summary",
    "introduction",
    "background",
    "relatedwork",
    "methods",
    "methodology",
    "materialsandmethods",
    "materialandmethods",
    "experimentalprocedures",
    "experimentalsection",
    "results",
    "discussion",
    "resultsanddiscussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgments",
    "acknowledgements",
    "acknowledgement",
    "funding",
    "appendix",
    "contents",
    "tableofcontents",
    "摘要",
    "前言",
    "引言",
    "绪论",
    "緒論",
    "目录",
    "目錄",
    "序言",
    "参考文献",
    "參考文獻",
    "致谢",
    "致謝",
    "附录",
    "附錄",
}
IGNORED = {
    "header",
    "footer",
    "header_image",
    "footer_image",
    "number",
    "footnote",
    "aside_text",
    "vision_footnote",
    "formula_number",
}


@memoize
def visible(s):
    raw = s
    s = unicodedata.normalize("NFKC", s).strip()
    # Unwrap only typographic commands. Preserve mathematical expressions.
    # (both patterns need a literal backslash / dollar sign, so they are skipped when there is none)
    if "\\" in s:
        for _ in range(3):
            s = _UNWRAP.sub(r"\1", s)
    if "$" in s:
        s = _DOLLAR.sub(r"\1", s)
    # ' '.join(s.split()) == re.sub(r'\s+',' ',s).strip(): both use the Unicode whitespace class
    out = " ".join(s.split()).strip("*").strip()
    if _NORM_VERIFY and out != _visible_reference(raw):
        raise AssertionError("recover_headings: visible differs")
    return out


def _visible_reference(s):
    s = unicodedata.normalize("NFKC", s).strip()
    for _ in range(3):
        s = re.sub(
            r"\\(?:underline|textbf|textit|textrm|mathrm|text)\s*\{([^{}]*)\}", r"\1", s
        )
    s = re.sub(r"\$\s*([^$\\{}^_]+?)\s*\$", r"\1", s)
    return re.sub(r"\s+", " ", s).strip().strip("*").strip()


_NORM_VERIFY = False
_UNWRAP = re.compile(r"\\(?:underline|textbf|textit|textrm|mathrm|text)\s*\{([^{}]*)\}")
_DOLLAR = re.compile(r"\$\s*([^$\\{}^_]+?)\s*\$")
_SPACES = re.compile(r"\s+")
_NONWORD = re.compile(r"[^\w]")


@memoize
def key(s):
    return _NONWORD.sub("", visible(s)).lower()


@memoize
def toc_key(s):
    s = visible(s)
    s = re.sub(r"\s*\(Rev\..*$", "", s, flags=re.I)
    s = re.sub(r"\s*(?:\.{2,}|…+|·{2,})\s*[\divxlc]+\s*$", "", s, flags=re.I)
    return key(s)


@memoize
def family(s):
    s = visible(s)
    m = re.match(r"^第[" + CN + r"\d]+([篇部卷章节節])\s*", s)
    if m:
        return ("cn_" + m[1].replace("節", "节"), 1)
    m = re.match(r"^(PART|CHAPTER|SECTION)[.\s]+[\dIVXLCDM]+\b", s, re.I)
    if m:
        return ("en_" + m[1].lower(), 1)
    if re.match("^[" + CN + r"]+[、.．]\s*", s):
        return ("cn", 1)
    if re.match(r"^[壹贰貳叁參肆伍陆陸柒捌玖拾佰]+[、.．]\s*", s):
        return ("cn_upper", 1)
    if re.match(r"^\([" + CN + r"]+\)", s):
        return ("cn_paren", 1)
    if re.match(r"^[甲乙丙丁戊己庚辛壬癸][、.]", s):
        return ("stem", 1)
    if re.match(r"^\(\d{1,3}\)[.、]?\s*\D", s):
        return ("num_paren", 1)
    m = re.match(
        r"^(\d{1,3}(?:[.\-]\d{1,3})+)[.、]?\s*(?:[-–—]\s*|(?=[A-Za-z\u4e00-\u9fff]))", s
    )
    if m:
        ns = re.split(r"[.\-]", m[1])
        while len(ns) > 1 and ns[-1] == "0":
            ns.pop()
        return ("num", len(ns))
    if re.match(r"^\d{1,3}(?:[.、]\s*|\s+)(?=[A-Za-z\u4e00-\u9fff])", s):
        return ("num", 1)
    if re.match(r"^[IVXLCDM]+[.)-]\s", s, re.I):
        return ("roman", 1)
    if re.match(r"^[A-Za-z][.)]\s|^\([A-Za-z]\)\s*", s):
        return ("letter", 1)
    if key(s.rstrip(":：")) in MAJOR:
        return ("major", 1)
    # letters=re.sub(r'[^A-Za-z]','',s); len(letters)>=5 and letters.isupper() <=> at least five
    # A-Z and no a-z in s
    caps = not _ASCII_LOWER.search(s) and len(_ASCII_UPPER.findall(s)) >= 5
    if _NORM_VERIFY:
        letters = re.sub(r"[^A-Za-z]", "", s)
        if caps != (len(letters) >= 5 and letters.isupper()):
            raise AssertionError("recover_headings: caps test differs")
    if caps:
        return ("caps", 1)
    return ("plain", 0)


_CJK_RE = re.compile("[\u4e00-\u9fff]")
_ASCII_LOWER = re.compile("[a-z]")
_ASCII_UPPER = re.compile("[A-Z]")


def font(b, width):
    return font_value(b.content, tuple(b.bbox), width)


@memoize
def font_value(content, bbox, width):
    s = visible(content)
    x, y, xx, yy = bbox
    h = max(1, yy - y)
    w = max(1, xx - x)
    # == sum(1 if '\u4e00'<=c<='\u9fff' else .5 for c in s if not c.isspace()): CJK characters are never
    # whitespace, and halves and integers below 2**52 add exactly in any order (int when there are no halves)
    cjk = len(_CJK_RE.findall(s))
    other = len(s) - sum(map(str.isspace, s)) - cjk
    units = cjk + 0.5 * other if other else cjk
    lines = max(content.count("\n") + 1, round(math.sqrt(max(1, h * units / w))))
    return h / lines / width * 1000


def visual_map(pages, path):
    features = (
        path["features"]
        if isinstance(path, dict)
        else json.loads(pathlib.Path(path).read_text())["features"]
    )
    out = {}
    for pi, p in enumerate(pages):
        for bi, b in enumerate(p["parsing_res_list"]):
            f = features.get(f"{pi}:{bi}")
            if f:
                out[(pi, tuple(b["block_bbox"]))] = f
    return out


def attach_visual(results, features):
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            f = features.get((pi, tuple(b.bbox)))
            if f:
                b._visual_font = f


def make_records(results, headings_only=False):
    """Same records as _make_records_reference (only the heading records when headings_only).  The derived values of a block (visible text, key,
    family, font, scaled geometry) are kept per document and reused while the block keeps the same
    content object, bounding box, page size and visual-feature object; every call still returns new
    record dicts (callers annotate them)."""
    from . import runtime_context

    if _STATS:
        t0 = _time.perf_counter()
        timing["calls"][0] += 1
    store = runtime_context.analysis_cache.get(None)
    cache = store.setdefault(("make_records_values",), {}) if store is not None else {}
    records = []
    append = records.append
    misses = 0
    for pi, p in enumerate(results):
        blocks = p["parsing_res_list"]
        width = p["width"]
        height = p["height"]
        for bi, b in enumerate(blocks):
            label = b.label
            content = b.content
            if headings_only and label not in {"paragraph_title", "doc_title"}:
                continue
            # `not content or content.isspace()` is `not content.strip()` without copying the text
            if (
                (label in IGNORED and label not in {"header", "aside_text"})
                or not content
                or content.isspace()
            ):
                continue
            bbox = tuple(b.bbox)
            visual = getattr(b, "_visual_font", None)
            hit = cache.get(id(b))
            if (
                hit is None
                or hit[0] is not b
                or hit[1] is not content
                or hit[2] != bbox
                or hit[3] != width
                or hit[4] != height
                or hit[5] is not visual
            ):
                x, y, xx, yy = bbox
                s = visible(content)
                v = visual if visual is not None else {}
                hit = (
                    b,
                    content,
                    bbox,
                    width,
                    height,
                    visual,
                    s,
                    key(s),
                    family(s),
                    v.get("font", font(b, width)),
                    x / width * 1000,
                    y / height,
                    xx / width * 1000,
                    yy / height,
                    (yy - y) / width * 1000,
                )
                cache[id(b)] = hit
                misses += 1
            append(
                {
                    "page": pi,
                    "index": bi,
                    "block": b,
                    "text": hit[6],
                    "key": hit[7],
                    "family": hit[8],
                    "font": hit[9],
                    "visual": visual if visual is not None else {},
                    "x": hit[10],
                    "y": hit[11],
                    "right": hit[12],
                    "bottom": hit[13],
                    "height": hit[14],
                    "width": width,
                    "is_heading": label in {"paragraph_title", "doc_title"},
                }
            )
    if _STATS:
        timing["calls"][1] += _time.perf_counter() - t0
        timing["records"][0] += len(records)
        timing["misses"][0] += misses
        timing["blocks"][0] += sum(len(p["parsing_res_list"]) for p in results)
        timing["headings_only" if headings_only else "full"][0] += 1
    if _VERIFY:
        ref = _make_records_reference(results)
        if headings_only:
            ref = [r for r in ref if r["is_heading"]]
        if len(ref) != len(records) or any(
            set(a) != set(b)
            or any((a[k] is not b[k]) if k == "block" else (a[k] != b[k]) for k in a)
            for a, b in zip(records, ref)
        ):
            raise AssertionError("recover_headings: records differ")
    return records


def _make_records_reference(results):
    records = []
    for pi, p in enumerate(results):
        blocks = p["parsing_res_list"]
        width = p["width"]
        height = p["height"]
        for bi, b in enumerate(blocks):
            if (
                b.label in IGNORED and b.label not in {"header", "aside_text"}
            ) or not b.content.strip():
                continue
            x, y, xx, yy = b.bbox
            s = visible(b.content)
            visual = getattr(b, "_visual_font", {})
            records.append(
                {
                    "page": pi,
                    "index": bi,
                    "block": b,
                    "text": s,
                    "key": key(s),
                    "family": family(s),
                    "font": visual.get("font", font(b, width)),
                    "visual": visual,
                    "x": x / width * 1000,
                    "y": y / height,
                    "right": xx / width * 1000,
                    "bottom": yy / height,
                    "height": (yy - y) / width * 1000,
                    "width": width,
                    "is_heading": b.label in {"paragraph_title", "doc_title"},
                }
            )
    return records


import time as _time

_VERIFY = False
_STATS = False
timing = {
    "calls": [0, 0.0],
    "records": [0, 0],
    "misses": [0, 0],
    "blocks": [0, 0],
    "full": [0, 0],
    "headings_only": [0, 0],
}


def _similar_close(cache, peers, fam, r):
    """Same lists as
      similar=[p for p in peers if abs(math.log(max(r['font'],1)/max(p['font'],1)))<.21 and abs(r['x']-p['x'])<45]
      close=sorted(similar,key=lambda p:(abs(p['page']-r['page']),abs(p['index']-r['index'])))[:8]
    with the peer scan vectorized. The logarithm is re-evaluated with math.log for any ratio whose
    numpy logarithm lies within 1e-9 of the .21 bound, so every comparison matches the original.
    """
    import numpy as np

    if not peers:
        return [], []
    arr = cache.get(fam)
    if arr is None:
        arr = cache[fam] = (
            np.asarray([max(p["font"], 1) for p in peers], dtype=float),
            np.asarray([p["x"] for p in peers], dtype=float),
            np.asarray([p["page"] for p in peers]),
            np.asarray([p["index"] for p in peers]),
            [max(p["font"], 1) for p in peers],
        )
    fonts, xs, pages, indices, raw = arr
    rf = max(r["font"], 1)
    with np.errstate(all="ignore"):
        lq = np.abs(np.log(rf / fonts))
    ok = lq < 0.21
    for j in np.flatnonzero(~(np.abs(lq - 0.21) >= 1e-9)).tolist():
        ok[j] = abs(math.log(rf / raw[j])) < 0.21
    ok &= np.abs(r["x"] - xs) < 45
    idx = np.flatnonzero(ok)
    if not len(idx):
        if _VERIFY and any(
            abs(math.log(max(r["font"], 1) / max(p["font"], 1))) < 0.21
            and abs(r["x"] - p["x"]) < 45
            for p in peers
        ):
            raise AssertionError("recover_headings: peer lists differ")
        return [], []
    order = np.lexsort(
        (np.abs(indices[idx] - r["index"]), np.abs(pages[idx] - r["page"]))
    )[:8]
    similar, close = [peers[j] for j in idx.tolist()], [
        peers[j] for j in idx[order].tolist()
    ]
    if _VERIFY:
        ref = [
            p
            for p in peers
            if abs(math.log(max(r["font"], 1) / max(p["font"], 1))) < 0.21
            and abs(r["x"] - p["x"]) < 45
        ]
        refc = sorted(
            ref,
            key=lambda p: (abs(p["page"] - r["page"]), abs(p["index"] - r["index"])),
        )[:8]
        if [id(p) for p in ref] != [id(p) for p in similar] or [
            id(p) for p in refc
        ] != [id(p) for p in close]:
            raise AssertionError("recover_headings: peer lists differ")
    return similar, close


def propose(results):
    records = make_records(results)
    heads = [r for r in records if r["is_heading"]]
    header_repeats = collections.Counter(
        r["key"] for r in records if r["block"].label == "header"
    )
    byfamily = collections.defaultdict(list)
    bykey = collections.defaultdict(list)
    pagebody = collections.defaultdict(list)
    bodydensity = collections.defaultdict(list)
    for r in heads:
        byfamily[r["family"]].append(r)
        bykey[r["key"]].append(r)
    for r in records:
        if r["block"].label == "text" and len(r["text"]) > 100:
            pagebody[r["page"]].append(r["font"])
            if r["visual"]:
                bodydensity[bool(re.search(r"[\u4e00-\u9fff]", r["text"]))].append(
                    r["visual"]["density"]
                )
    allbody = [f for v in pagebody.values() for f in v]
    bodymedian = statistics.median(allbody) if allbody else 20
    toc = {}
    tocpages = set()
    page_records = collections.defaultdict(list)
    for r in records:
        page_records[r["page"]].append(r)
    for r in records[: max(80, len(records) // 5)]:
        if r["block"].label != "content":
            continue
        for line in r["block"].content.splitlines():
            m = re.match(
                r"^\s*(.{3,180}?)\s*(?:\.{2,}|…+|·{2,})\s*[\divxlc]+\s*$", line, re.I
            )
            if m:
                toc[key(m[1])] = m[1]
                tocpages.add(r["page"])
    # Contents entries can themselves be emitted as ordinary text, including
    # number-only lists without dot leaders. They must not become outline nodes.
    for pi, rr in page_records.items():
        if pi > min(40, len(results) // 4 + 3):
            continue
        explicit = any(
            len(r["text"]) < 100
            and re.match(
                r"^(?:table\s+of\s+contents|contents|目\s*[录錄])(?:\b|\s|[(:：]|$)",
                r["text"],
                re.I,
            )
            for r in rr
        )
        dense = (
            sum(r["family"][0] not in {"plain", "caps", "major"} for r in rr) >= 6
            and sum(len(r["text"]) > 200 for r in rr) <= 1
        )
        if explicit or (pi - 1 in tocpages and dense):
            tocpages.add(pi)
    for pi, rr in page_records.items():
        numbered = [r for r in rr if r["family"][0] == "num" and len(r["text"]) < 180]
        repeated = sum(
            any(p["page"] != pi for p in bykey.get(r["key"], [])) for r in numbered
        )
        if (
            len(numbered) >= 8
            and repeated >= len(numbered) * 0.6
            and sum(len(r["text"]) > 200 for r in rr) == 0
        ):
            tocpages.add(pi)
    for pi in tocpages:
        rr = page_records[pi]
        numbered = sum(r["family"][0] not in {"plain", "caps", "major"} for r in rr)
        dotlines = sum(
            len(re.findall(r"(?:\.{2,}|…+|·{2,})\s*\d+", r["block"].content))
            for r in rr
        )
        body_paragraph = any(
            r["block"].label == "text"
            and len(r["text"]) > 400
            and "\n" not in r["block"].content
            and not re.search(r"\.{3,}", r["text"])
            for r in rr
        )
        if max(numbered, dotlines) >= 5 and not body_paragraph:
            for r in rr:
                if r["is_heading"] and r["family"][0] not in {"plain", "caps", "major"}:
                    r["block"]._toc_entry = True
        for r in page_records[pi]:
            if (
                r["family"][0] not in {"plain", "caps", "major"}
                and len(r["text"]) <= 220
            ):
                toc[toc_key(r["text"])] = r["text"]
    proposals = []
    page_body = {}
    body_density = {}
    peer_arrays = {}
    for i, r in enumerate(records):
        b = r["block"]
        s = r["text"]
        fam = r["family"]
        k, depth = fam
        if (
            b.label
            not in {"text", "header", "figure_title", "aside_text", "vertical_text"}
            or not 3 <= len(s) <= 260
            or b.content.count("\n") > 3
        ):
            continue
        if b.label == "header" and (header_repeats[r["key"]] > 3 or len(s) > 120):
            continue
        if r["page"] in tocpages and k != "major":
            continue
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", s):
            continue
        if re.match(
            r"^(?:Figure|Fig\.?|Table|Exhibit|Equation|图|圖|表)\s*[\d一二三四五六七八九十]+",
            s,
            re.I,
        ):
            continue
        if re.search(
            r"https?://|www\.|\bISBN\b|©|copyright|all rights reserved|版权所有|版權所有|不适用|不適用|□|☐|☑|√|✓|=",
            s,
            re.I,
        ):
            continue
        if re.match(
            r"^(?:[•●▪■◆]|[-–—]\s|see\s+also\b|related\s+(?:tasks?|concepts?|references?)\b|for\s+example\b|where\b|dear\b|by\s+[A-Z]|edited\s+by\b)",
            s,
            re.I,
        ):
            continue
        if re.search(
            r"\s[><]\s|\[.{1,20}\]|____|\b(?:REPORT BIBLIOGRAPHY|SEARCH CONTROL)\b", s
        ):
            continue
        if key(s.rstrip(":：")) in {
            "说明",
            "說明",
            "其他说明",
            "其他說明",
            "注意",
            "note",
            "notes",
            "example",
            "examples",
            "considerations",
        }:
            continue
        if s.endswith((";", "；", ",", "，")):
            continue
        if (
            len(re.findall("[。；;]", s.rstrip("。.;；")))
            or len(re.findall(r"[.!?]\s+[A-Z]", s)) > 1
        ):
            continue
        if re.search(r"\.{3,}\s*\d+$", s):
            continue
        following_records = records[i + 1 : i + 9]
        nextrec = None
        has_status_response = False
        for nr in following_records:
            if nr["page"] > r["page"] + 1:
                break
            if nr["block"].label in IGNORED:
                continue
            if len(nr["text"]) < 70 and re.search(
                r"适用|適用|不适用|不適用|单位[:：]|單位[:：]|币种[:：]|幣種[:：]",
                nr["text"],
            ):
                has_status_response = True
                continue
            nextrec = nr
            break
        prev = records[i - 1] if i else None
        following = has_status_response or bool(
            nextrec
            and nextrec["page"] <= r["page"] + 1
            and (
                nextrec["block"].label in {"table", "image", "chart", "figure"}
                or nextrec["is_heading"]
                or (
                    nextrec["block"].label in {"text", "abstract", "reference_content"}
                    and len(nextrec["text"]) >= 45
                )
            )
        )
        if not following:
            continue
        peers = byfamily[fam]
        if r["page"] not in page_body:
            page_body[r["page"]] = (
                statistics.median(pagebody[r["page"]])
                if pagebody[r["page"]]
                else bodymedian
            )
        body = page_body[r["page"]]
        if b.label == "header" and r["font"] < body * 1.08 and k != "major":
            continue
        # Small lexical families (e.g. dotted numbers) must already occur as headings.
        similar, close = _similar_close(peer_arrays, peers, fam, r)
        punctuation = bool(re.search(r"[。.!?]$", s))
        punct_ok = not punctuation or (
            len(close) >= 3
            and sum(bool(re.search(r"[。.!?]$", p["text"])) for p in close)
            >= len(close) * 0.4
        )
        reason = None
        cjk = bool(re.search(r"[\u4e00-\u9fff]", s))
        density = r["visual"].get("density")
        if cjk not in body_density:
            body_density[cjk] = (
                statistics.median(bodydensity[cjk]) if bodydensity[cjk] else None
            )
        body_d = body_density[cjk]
        peer_ds = [p["visual"]["density"] for p in close if p["visual"]]
        peer_d = statistics.median(peer_ds) if peer_ds else None
        if len(toc) >= 4 and toc_key(s) in toc and r["page"] not in tocpages:
            reason = "toc_exact"
        elif (
            k not in {"plain", "caps"}
            and r["key"] in bykey
            and r["font"] >= body * 1.2
            and any(
                abs(math.log(max(r["font"], 1) / max(p["font"], 1))) < 0.18
                for p in bykey[r["key"]]
            )
        ):
            reason = "repeated_heading"
        elif (
            (
                k == "major"
                or key(re.sub(r"^\d+(?:\.\d+)*[.)、]?\s*", "", s).rstrip(":："))
                in MAJOR
            )
            and len(s) <= 60
            and not punctuation
            and r["font"] >= body * 0.85
        ):
            reason = "common_section"
        elif (
            k.startswith(("cn_章", "cn_篇", "cn_部", "en_chapter", "en_part"))
            and len(s) <= 90
            and not punctuation
            and r["font"] >= body * 1.15
            and not re.search(
                r"介绍了|介紹了|讲解了|講解了|讨论了|討論了|\b(?:describes|discusses|provides|presents)\b",
                s,
                re.I,
            )
        ):
            reason = "explicit_chapter_title"
        elif (
            k not in {"plain", "caps", "major"}
            and len(peers) >= 3
            and len(close) >= 2
            and punct_ok
        ):
            # Avoid multi-sentence numbered paragraphs and compact enumeration lists.
            cjk = bool(re.search(r"[\u4e00-\u9fff]", s))
            english_ok = (
                cjk
                or (k == "num" and depth >= 2)
                or (r["font"] >= body * 1.2 and k not in {"letter", "num_paren"})
            )
            if not cjk and (
                punctuation
                or re.search(
                    r"\b(?:shall|will|should|must|may|is|are|was|were|have|has|can|cannot)\b",
                    s,
                    re.I,
                )
            ):
                english_ok = False
            runin = bool(re.search(r"[:：].{12}", s))
            statement = len(s) > 55 and bool(
                re.search(r"[,，].*[,，]|(?:如本|如果|我们|我們|本公司将|本人将)", s)
            )
            lines = r["visual"].get("lines", r["height"] / r["font"])
            if (
                english_ok
                and not runin
                and not statement
                and len(s) <= 220
                and (lines <= 2.4 or k.startswith(("cn_", "en_")))
            ):
                reason = "numbered_siblings"
        elif k in {"caps", "plain"} and not punctuation and len(s) <= 150:
            alpha = [
                t
                for t in re.findall(r"[A-Za-z][A-Za-z'-]*", s)
                if t.lower()
                not in {
                    "a",
                    "an",
                    "the",
                    "of",
                    "in",
                    "on",
                    "for",
                    "to",
                    "and",
                    "or",
                    "with",
                    "by",
                    "as",
                    "at",
                }
            ]
            titlecase = (
                len(alpha) >= 2
                and sum(t[0].isupper() for t in alpha) >= len(alpha) * 0.85
            )
            strong = (
                r["font"] >= body * 1.23
                and len(close) >= 3
                and len(alpha) >= 2
                and (k == "caps" or titlecase)
            )
            separated = (
                prev is None
                or prev["page"] != r["page"]
                or r["y"] - prev["bottom"] > 0.008
            )
            if strong and separated:
                reason = "typographic_siblings"
        # Raster evidence can distinguish same-size bold headings from normal body
        # text, using only peers in this document. Do not assume all titles are bold.
        if density is not None and body_d and peer_d and peer_d > body_d * 1.22:
            if (
                reason not in {"toc_exact", "common_section"}
                and density < (peer_d + body_d) / 2
            ):
                reason = None
            if (
                not reason
                and cjk
                and len(s) <= 90
                and punct_ok
                and len(close) >= 3
                and density >= peer_d * 0.9
                and density >= body_d * 1.25
                and r["font"] >= body * 0.9
            ):
                if k not in {"caps", "major"} and not re.search(
                    r"[:：].{12}|[。;；].+|[,，].*[,，]|(?:如本|如果|我们|我們|本公司将|本人将)",
                    s,
                ):
                    reason = "visual_bold_siblings"
        if reason:
            r = dict(r)
            r["reason"] = reason
            r["body_font"] = body
            r["body_density"] = body_d
            r["peer_density"] = peer_d
            r["original_label"] = b.label
            r["peers"] = close or bykey.get(r["key"], []) or peers
            proposals.append(r)
    return proposals


def recover(results, log):
    recover_cover_and_metadata(results, log)
    # Chapter number and chapter name may be separate detector blocks. Their
    # identity is explicit, unlike a generic heading followed by a short paragraph.
    for pi, p in enumerate(results):
        blocks = [
            b
            for b in p["parsing_res_list"]
            if (b.label not in IGNORED or b.label in {"header", "aside_text"})
            and b.content.strip()
        ]
        for a, b in zip(blocks, blocks[1:]):
            if a.label not in {
                "text",
                "figure_title",
                "paragraph_title",
                "doc_title",
                "header",
                "aside_text",
            } or b.label not in {
                "text",
                "figure_title",
                "paragraph_title",
                "doc_title",
                "header",
                "aside_text",
            }:
                continue
            prefix = visible(a.content)
            name = visible(b.content)
            if not re.fullmatch(
                r"(?:CHAPTER|SECTION|PART)[.\s]+[\dIVXLCDM]+(?:\.\d+)*[.:]?|第["
                + CN
                + r"\d]+[篇部卷章节節]",
                prefix,
                re.I,
            ):
                continue
            if not 4 <= len(name) <= 180 or re.search(r"[。;；!?]$", name):
                continue
            letters = re.sub("[^A-Za-z]", "", name)
            if not (letters and letters.isupper()) and not re.search(
                "[\u4e00-\u9fff]", name
            ):
                continue
            ah = font(a, p["width"])
            bh = font(b, p["width"])
            if (
                bh < ah * 0.7
                or bh > ah * 2.6
                or b.bbox[1] < a.bbox[3]
                or (b.bbox[1] - a.bbox[3]) / p["height"] > 0.08
            ):
                continue
            old = b.content
            a.content = a.content.strip() + "\n" + old.strip()
            b.content = ""
            b.label = "text"
            if a.label not in {"paragraph_title", "doc_title"}:
                a.label = "paragraph_title"
                a.title_level = 1
            log.append(
                {
                    "kind": "chapter_subtitle_joined",
                    "page": pi + 1,
                    "prefix": prefix,
                    "subtitle": old,
                }
            )
    prepare_toc_anchors(results, log)
    proposals = propose(results)
    for r in proposals:
        b = r["block"]
        b.label = "paragraph_title"
        b.title_level = 1
        b._recovered_heading = r
        log.append(
            {
                "kind": "body_heading_recovered",
                "page": r["page"] + 1,
                "block": r["index"],
                "text": b.content,
                "original_label": r["original_label"],
                "reason": r["reason"],
                "font": r["font"],
                "body_font": r["body_font"],
                "visual": r["visual"],
                "body_density": r["body_density"],
                "peer_density": r["peer_density"],
            }
        )
    return proposals


def original_only(results):
    # Keep original hierarchy calculations unchanged by the newly recovered nodes.
    return [
        {
            **p,
            "parsing_res_list": [
                b for b in p["parsing_res_list"] if not hasattr(b, "_recovered_heading")
            ],
        }
        for p in results
    ]


def assign_levels(results, proposals, log):
    original = [
        r
        for r in make_records(results, True)
        if r["is_heading"] and not hasattr(r["block"], "_recovered_heading")
    ]
    for r in proposals:
        b = r["block"]
        peers = [p for p in original if p["family"] == r["family"]]
        exact = [p for p in original if p["key"] == r["key"]]
        if exact:
            peers = exact
        scored = []
        for p in peers:
            style = (
                abs(math.log(max(r["font"], 1) / max(p["font"], 1)))
                + abs(r["x"] - p["x"]) / 400
            )
            score = style + min(abs(r["page"] - p["page"]), 50) * 0.015
            level = (
                1
                if p["block"].label == "doc_title"
                else getattr(p["block"], "title_level", 1) + 1
            )
            scored.append((score, level, p["page"]))
        if scored:
            top = sorted(scored)[:5]
            weights = collections.defaultdict(float)
            for distance, level, pi in top:
                weights[level] += 1 / (distance + 0.05)
            level = max(weights, key=weights.get)
        elif r["family"][0] == "major" or r["reason"] == "common_section":
            level = 2
        else:
            prev = [
                p
                for p in original
                if p["page"] < r["page"]
                or (p["page"] == r["page"] and p["index"] < r["index"])
            ]
            near = prev[-1] if prev else None
            level = (
                (
                    1
                    if near["block"].label == "doc_title"
                    else getattr(near["block"], "title_level", 1) + 1
                )
                + 1
                if near
                else 2
            )
        level = max(2, min(6, level))
        b.title_level = level - 1
        log.append(
            {
                "kind": "recovered_heading_level",
                "page": r["page"] + 1,
                "text": b.content,
                "new": level,
                "reason": r["reason"],
            }
        )


def refine_hierarchy(results, log):
    """Reconstruct explicit numbering paths within their current section scope.

    Families are not assigned one depth for the whole document: a numbered item
    can sit beneath a Chinese numbered section in one chapter and beneath its
    parenthesized subsection in another. Original unnumbered subheadings follow
    their active numbered parent when their typography supports that relation.
    """
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if getattr(b, "_toc_entry", False) and b.label in {
                "paragraph_title",
                "doc_title",
            }:
                b.label = "text"
                log.append(
                    {"kind": "toc_entry_demoted", "page": pi + 1, "text": b.content}
                )
    heads = [r for r in make_records(results, True) if r["is_heading"]]
    counts = collections.Counter(r["family"][0] for r in heads)

    def oldlevel(r):
        return (
            1
            if r["block"].label == "doc_title"
            else getattr(r["block"], "title_level", 1) + 1
        )

    for r in heads:
        r["old_level"] = oldlevel(r)

    def setlevel(r, level, reason):
        level = max(2, min(6, level))
        b = r["block"]
        old = oldlevel(r)
        b.label = "paragraph_title"
        b.title_level = level - 1
        r["new_level"] = level
        if old != level:
            log.append(
                {
                    "kind": "contextual_heading_hierarchy",
                    "page": r["page"] + 1,
                    "text": b.content,
                    "old": old,
                    "new": level,
                    "reason": reason,
                }
            )

    def number(s):
        s = visible(s)
        m = re.match(r"^(\d+(?:[.\-]\d+)*)", s)
        if not m:
            return None
        nums = tuple(map(int, re.split("[.\-]", m[1])))
        while len(nums) > 1 and nums[-1] == 0:
            nums = nums[:-1]
        return nums

    def chapter_number(s):
        m = re.match(r"^(?:CHAPTER|SECTION|PART)[.\s]+(\d+)", visible(s), re.I)
        if m:
            return int(m[1])
        m = re.match(r"^第([\d一二三四五六七八九十百]+)", visible(s))
        if not m:
            return None
        t = m[1]
        if t.isdigit():
            return int(t)
        digit = {c: i for i, c in enumerate("零一二三四五六七八九")}
        if "十" in t:
            a, b = t.split("十", 1)
            return digit.get(a, 1) * 10 + digit.get(b, 0)
        return digit.get(t)

    chinese = counts["cn"] >= 5 and (counts["cn_paren"] >= 3 or counts["cn_upper"] >= 3)
    macro_order = ["cn_篇", "cn_部", "cn_卷", "cn_章", "cn_节"]
    macros = []
    fonts = {
        k: statistics.median(r["font"] for r in heads if r["family"][0] == k)
        for k in counts
    }
    for k in macro_order:
        rr = [r for r in heads if r["family"][0] == k]
        if len(rr) < (1 if k in {"cn_章", "cn_篇", "cn_部"} else 2):
            continue
        if (
            k == "cn_部"
            and sum(bool(re.match(r"^第.+?部分", r["text"])) for r in rr)
            > len(rr) * 0.7
        ):
            continue
        if (
            k == "cn_节"
            and counts["cn_upper"] >= 3
            and fonts.get("cn", 0) > fonts[k] * 1.15
        ):
            continue
        macros.append(k)
    macro_base = 2
    cover = [
        r
        for r in heads
        if r["block"].label == "doc_title" and r["page"] < 3 and len(r["key"]) >= 6
    ]
    for r in heads:
        if r["page"] < 3 or r["page"] > 10 or r["family"][0] != "plain":
            continue
        if any(r["key"].startswith(c["key"]) for c in cover):
            setlevel(r, 2, "interior_book_title")
            macro_base = 3
    macro_levels = {k: i + macro_base for i, k in enumerate(macros)}
    # A compact list of "part one / part two / part three" on one page is an
    # inventory, not three book parts. Require both adjacency and small type.
    components = collections.defaultdict(list)
    for r in heads:
        if r["family"][0] == "cn_部" and re.match(r"^第.+?部分", r["text"]):
            components[r["page"]].append(r)
    demoted = set()
    for pi, rr in components.items():
        if len(rr) < 3 or any(
            r["font"] >= fonts.get("cn_章", r["font"]) * 0.9 for r in rr
        ):
            continue
        for r in rr:
            r["block"].label = "text"
            demoted.add(id(r["block"]))
            log.append(
                {"kind": "component_list_demoted", "page": pi + 1, "text": r["text"]}
            )
    # Explicitly numbered families can switch names at the same rank. For
    # example, part 2 / chapter 3 / part 4 with matching type is one series.
    macro_heads = [
        r
        for r in heads
        if r["family"][0] in macro_levels and not re.match(r"^第.+?部分", r["text"])
    ]
    serial_peer = {}
    for r in macro_heads:
        rank = macros.index(r["family"][0])
        peers = [x for x in macro_heads if macros.index(x["family"][0]) <= rank]
        i = next(i for i, x in enumerate(peers) if x is r)
        if i == 0 or i + 1 == len(peers):
            continue
        a, b = peers[i - 1], peers[i + 1]
        an, rn, bn = map(lambda x: chapter_number(x["text"]), [a, r, b])
        if (
            a["family"] == b["family"] != r["family"]
            and None not in (an, rn, bn)
            and rn == an + 1
            and bn == rn + 1
            and abs(math.log(r["font"] / a["font"])) < 0.18
        ):
            serial_peer[id(r["block"])] = a["family"][0]
    active = {}
    macro_active = {}
    paths = {}
    last_numbered = None
    last_chapter = None
    cn_order = ["cn_upper", "cn", "cn_paren", "num", "num_paren", "letter"]
    for r in heads:
        if id(r["block"]) in demoted:
            continue
        k, depth = r["family"]
        s = r["text"]
        nums = number(s) if k == "num" else None
        is_component = k == "cn_部" and bool(re.match(r"^第.+?部分", s))
        if k in macro_levels and not is_component:
            rank = macros.index(k)
            parents = [macro_active[p] for p in macros[:rank] if p in macro_active]
            level = parents[-1]["new_level"] + 1 if parents else macro_base
            peer = serial_peer.get(id(r["block"]))
            if peer in macro_active:
                level = macro_active[peer]["new_level"]
            setlevel(r, level, "explicit_chapter_family")
            active = {}
            paths = {}
            last_numbered = r
            for lower in macros[rank:]:
                macro_active.pop(lower, None)
            macro_active[k] = r
            last_chapter = (chapter_number(s), r["new_level"])
            continue
        if k.startswith("en_"):
            last_chapter = (chapter_number(s), r["old_level"])
            paths = {}
        if nums and len(nums) > 1:
            parent = next(
                (
                    paths[nums[:n]]
                    for n in range(len(nums) - 1, 0, -1)
                    if nums[:n] in paths
                ),
                None,
            )
            if parent:
                pn, pl = parent
                setlevel(r, pl + len(nums) - pn, "decimal_prefix_parent")
            elif last_chapter and last_chapter[0] == nums[0]:
                setlevel(r, last_chapter[1] + len(nums) - 1, "decimal_matches_chapter")
            paths[nums] = (len(nums), r.get("new_level", r["old_level"]))
            last_numbered = r
            continue
        if chinese and k in cn_order and counts[k] >= 3:
            rank = cn_order.index(k)
            if k in active:
                level = active[k]["new_level"]
            else:
                parents = [active[p] for p in cn_order[:rank] if p in active]
                if parents:
                    level = parents[-1]["new_level"] + 1
                else:
                    level = (
                        max(
                            (m["new_level"] for m in macro_active.values()),
                            default=macro_base - 1,
                        )
                        + 1
                    )
            setlevel(r, level, "active_numbered_parent")
            for lower in cn_order[rank:]:
                active.pop(lower, None)
            active[k] = r
            last_numbered = r
            if nums:
                paths[nums] = (len(nums), level)
            continue
        if nums:
            paths[nums] = (len(nums), r.get("new_level", r["old_level"]))
        if (
            chinese
            and k == "cn_节"
            and k not in macro_levels
            and last_numbered
            and r["font"] <= last_numbered["font"] * 1.1
        ):
            setlevel(
                r,
                last_numbered.get("new_level", last_numbered["old_level"]) + 1,
                "local_minor_section",
            )
        if (
            chinese
            and k == "plain"
            and last_numbered
            and r["old_level"] >= 3
            and re.search(r"[\u4e00-\u9fff]", s)
            and r["font"] <= last_numbered["font"] * 1.1
            and len(s) < 160
        ):
            parentlevel = last_numbered.get("new_level", last_numbered["old_level"])
            setlevel(
                r, max(parentlevel + 1, r["old_level"]), "unnumbered_subheading_context"
            )


def refine_unnumbered_styles(results, log):
    """Use clearly separated document-local numbered heading styles as anchors."""
    heads = [r for r in make_records(results, True) if r["is_heading"]]
    bylevel = collections.defaultdict(list)
    for r in heads:
        if not r["visual"] or r["family"][0] in {"plain", "caps"}:
            continue
        level = (
            1
            if r["block"].label == "doc_title"
            else getattr(r["block"], "title_level", 1) + 1
        )
        if 2 <= level <= 6:
            bylevel[level].append(r)
    prototypes = {
        level: (
            statistics.median(r["font"] for r in rr),
            statistics.median(r["visual"]["density"] for r in rr),
        )
        for level, rr in bylevel.items()
        if len(rr) >= 4
    }
    if len(prototypes) < 2:
        return
    for r in heads:
        if (
            r["family"][0] not in {"plain", "caps"}
            or not r["visual"]
            or re.search("[\u4e00-\u9fff]", r["text"])
        ):
            continue
        if r["block"].label == "doc_title" and r["page"] < 3:
            continue
        candidates = sorted(
            (
                abs(math.log(r["font"] / f))
                + 0.5 * abs(math.log(max(r["visual"]["density"], 0.001) / d)),
                level,
            )
            for level, (f, d) in prototypes.items()
        )
        (best, level), (second, _) = candidates[:2]
        if best > 0.12 or second - best < 0.12:
            continue
        b = r["block"]
        old = 1 if b.label == "doc_title" else getattr(b, "title_level", 1) + 1
        if old == level:
            continue
        b.label = "paragraph_title"
        b.title_level = level - 1
        log.append(
            {
                "kind": "anchored_heading_style",
                "page": r["page"] + 1,
                "text": b.content,
                "old": old,
                "new": level,
                "distance": best,
                "margin": second - best,
            }
        )


def recover_cover_and_metadata(results, log):
    metadata = re.compile(
        r"^(?:(?:report\s+)?prepared\s+by|conducted\s+by|submitted\s+by|edited\s+by|reproductions?\s+supplied\s+by|permission\s+to\s+reproduce|this\s+document\s+has\s+been\s+reproduced)",
        re.I,
    )
    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if b.label in {"doc_title", "paragraph_title"} and metadata.search(
                visible(b.content).lstrip("* ")
            ):
                b.label = "text"
                log.append(
                    {
                        "kind": "publication_metadata_demoted",
                        "page": pi + 1,
                        "text": b.content,
                    }
                )
    for pi, p in enumerate(results[:3]):
        blocks = [
            b
            for b in p["parsing_res_list"]
            if b.content.strip()
            and b.label in {"text", "header", "paragraph_title", "doc_title"}
        ]
        if len(blocks) > 16 or sum(len(b.content) for b in blocks) > 2500:
            continue
        for i, b in enumerate(blocks):
            s = visible(b.content)
            letters = re.sub("[^A-Za-z]", "", s)
            if (
                b.label not in {"text", "header"}
                or not 60 <= len(s) <= 220
                or b.content.count("\n") < 1
                or not letters.isupper()
            ):
                continue
            if (
                b.bbox[1] / p["height"] > 0.35
                or abs((b.bbox[0] + b.bbox[2]) / 2 / p["width"] - 0.5) > 0.18
            ):
                continue
            if re.match(
                r"^(?:TITLE|AUTHOR|INSTITUTION|DOCUMENT|DEPARTMENT|COPYRIGHT)\b", s
            ):
                continue
            f = getattr(b, "_visual_font", {}).get("font", font(b, p["width"]))
            if any(
                a.label == "doc_title"
                and a.bbox[1] < b.bbox[1]
                and getattr(a, "_visual_font", {}).get("font", font(a, p["width"]))
                > f * 1.2
                for a in blocks
            ):
                continue
            target = b
            if i:
                a = blocks[i - 1]
                prefix = visible(a.content)
                pl = re.sub("[^A-Za-z]", "", prefix)
                if (
                    len(prefix) < 80
                    and pl.isupper()
                    and prefix.endswith(":")
                    and 0 < (b.bbox[1] - a.bbox[3]) / p["height"] < 0.08
                ):
                    a.content = a.content.strip() + "\n" + b.content.strip()
                    b.content = ""
                    b.label = "text"
                    target = a
            target.label = "doc_title"
            target.title_level = 0
            log.append(
                {
                    "kind": "cover_heading_recovered",
                    "page": pi + 1,
                    "text": target.content,
                }
            )
            for a in blocks:
                if a is target:
                    continue
                if a.label in {"doc_title", "paragraph_title"} and re.match(
                    r"^(?:The\s+)?(?:Final|Technical|Annual)\s+Report\s+(?:of|on|for)\b",
                    visible(a.content),
                    re.I,
                ):
                    a.label = "text"
                    log.append(
                        {
                            "kind": "cover_metadata_demoted",
                            "page": pi + 1,
                            "text": a.content,
                        }
                    )
            break


def prepare_toc_anchors(results, log):
    entries = []
    caption = re.compile(
        r"^(?:fig(?:ure)?[.\s]*\d|table\s*\d|[图圖表]\s*[\d一二三四五六七八九十])", re.I
    )
    for pi, p in enumerate(results[: min(40, max(10, len(results) // 5))]):
        for b in p["parsing_res_list"]:
            if b.label != "content":
                continue
            for line in b.content.splitlines():
                m = re.match(
                    r"^\s*(.{3,220}?)\s*(?:\.{2,}|…+|·{2,})\s*([\divxlc]+)\s*$",
                    line,
                    re.I,
                )
                if m and not caption.match(m[1].strip()):
                    entries.append((m[1].strip(), pi))
    if len(entries) < 6:
        return
    kinds = collections.Counter(family(s)[0] for s, _ in entries)
    has_chapters = any(kinds[k] >= 2 for k in ["cn_章", "en_chapter", "en_section"])
    has_parts = kinds["en_part"] >= 2 or kinds["cn_篇"] >= 2
    aliases = collections.defaultdict(list)
    for s, pi in entries:
        k, d = family(s)
        level = None
        if k in {"en_part", "cn_篇", "cn_部"}:
            level = 2
        elif k in {"cn_章", "en_chapter", "en_section"}:
            level = 2 + int(has_parts)
        elif k == "cn_节":
            level = 2 + int(has_chapters) + int(has_parts)
        elif k == "cn":
            level = 2 + int(has_chapters) + int(has_parts)
        elif k == "cn_paren":
            level = 3 + int(has_chapters) + int(has_parts)
        elif k == "num":
            level = 1 + d + int(has_parts)
        elif k in {"caps", "major"}:
            level = 2
        if level is None:
            continue
        aliases[toc_key(s)].append((level, pi))
        short = re.sub(
            r"^(?:(?:CHAPTER|SECTION|PART)[.\s]+[\dIVXLCDM]+\s*[-:—]?\s*|第["
            + CN
            + r"\d]+[篇部卷章节節]\s*|\d+(?:\.\d+)*[.、]?\s+)",
            "",
            s,
            flags=re.I,
        )
        if len(key(short)) >= 5:
            aliases[toc_key(short)].append((level, pi))
    matches = []
    records = make_records(results)
    repeated_headers = collections.Counter(
        r["key"] for r in records if r["block"].label == "header"
    )
    toc_pages = {pi for s, pi in entries}
    for r in records:
        if r["block"].label not in {
            "paragraph_title",
            "doc_title",
            "text",
            "header",
            "figure_title",
            "aside_text",
        }:
            continue
        if r["page"] in toc_pages or caption.match(r["text"]):
            continue
        if r["block"].label == "header" and repeated_headers[r["key"]] > 1:
            continue
        choices = aliases.get(toc_key(r["text"]), [])
        levels = {l for l, pi in choices if pi < r["page"]}
        if len(levels) != 1:
            continue
        if (
            r["family"][0] == "major"
            and r["page"] < 2
            and r["block"].label == "doc_title"
        ):
            continue
        matches.append((r, min(levels)))
    if len(matches) < 6 or len({r["page"] for r, l in matches}) < 4:
        return
    for r, level in matches:
        b = r["block"]
        b._toc_level_hint = min(level, 6)
        if not r["is_heading"]:
            old = b.label
            b.label = "paragraph_title"
            b.title_level = min(level, 6) - 1
            log.append(
                {
                    "kind": "toc_named_heading_recovered",
                    "page": r["page"] + 1,
                    "text": b.content,
                    "old_label": old,
                    "new_level": level,
                }
            )


def apply_toc_levels(results, log):
    heads = [r for r in make_records(results, True) if r["is_heading"]]
    anchors = [
        i
        for i, r in enumerate(heads)
        if hasattr(r["block"], "_toc_level_hint")
        and not r["family"][0].startswith("cn")
    ]
    if len(anchors) < 6:
        return

    def level(r):
        return (
            1
            if r["block"].label == "doc_title"
            else getattr(r["block"], "title_level", 1) + 1
        )

    def setlevel(r, new, why):
        b = r["block"]
        old = level(r)
        new = max(2, min(6, new))
        b.label = "paragraph_title"
        b.title_level = new - 1
        if old != new:
            log.append(
                {
                    "kind": "toc_context_hierarchy",
                    "page": r["page"] + 1,
                    "text": b.content,
                    "old": old,
                    "new": new,
                    "reason": why,
                }
            )

    for n, i in enumerate(anchors):
        r = heads[i]
        parent = r["block"]._toc_level_hint
        end = anchors[n + 1] if n + 1 < len(anchors) else len(heads)
        setlevel(r, parent, "predicted_toc_anchor")
        group = heads[i + 1 : end]
        if not group:
            continue
        deeper = [level(x) for x in group if level(x) > parent]
        offset = parent + 1 - min(deeper) if deeper else 0
        for x in group:
            # A missing TOC match must not let the preceding section consume the next
            # explicitly numbered section or an appendix and all its descendants.
            if x["family"][0] not in {"plain", "caps", "major"} or re.match(
                r"^(?:Appendix|Annex)\b", x["text"], re.I
            ):
                break
            setlevel(x, max(parent + 1, level(x) + offset), "within_toc_section")


def finalize_headings(results, log):
    """Remove non-title nodes and pure style markup after all recovery passes.

    Keep every text character. A formula with operators, scripts or remaining
    LaTeX commands is deliberately left unchanged, including its delimiters.
    """
    metadata = re.compile(
        r"^(?:(?:report\s+)?prepared\s+by|conducted\s+by|submitted\s+by|edited\s+by|reproductions?\s+supplied\s+by|permission\s+to\s+reproduce|this\s+document\s+has\s+been\s+reproduced)",
        re.I,
    )
    style = re.compile(
        r"\\(?:underline|textbf|textit|textrm|mathrm|text)\s*\{([^{}]*)\}"
    )

    def clean_math(m):
        old = m[0]
        value = m[1]
        if not style.search(value):
            return old
        for _ in range(8):
            new = style.sub(r"\1", value)
            if new == value:
                break
            value = new
        if re.search(r"[\\{}^_=+<>]|\\(?:frac|sum|int)", value):
            return old
        if not re.search(r"[^\W_]", value):
            return old
        return value.strip()

    for pi, p in enumerate(results):
        for b in p["parsing_res_list"]:
            if b.label not in {"doc_title", "paragraph_title"}:
                continue
            plain = visible(b.content)
            if metadata.search(plain.lstrip("* ")) or not re.search(r"[^\W_]", plain):
                b.label = "text"
                log.append(
                    {
                        "kind": "final_nontitle_demoted",
                        "page": pi + 1,
                        "text": b.content,
                    }
                )
                continue
            old = b.content
            b.content = re.sub(r"\$([^$]+)\$", clean_math, old)
            if old != b.content:
                log.append(
                    {
                        "kind": "nested_heading_style_normalized",
                        "page": pi + 1,
                        "before": old,
                        "after": b.content,
                    }
                )
