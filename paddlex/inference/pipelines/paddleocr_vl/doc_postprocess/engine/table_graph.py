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

"""Prediction-only table fragment graph and maximum-weight path cover."""
import collections
import copy
import math
import re
import unicodedata

import numpy as np

from . import fast_copy, runtime_context, structure_guard, table_identity
from .native_similarity import ratio as native_ratio
from .runtime_context import memoize


# difflib ratio in every environment, as in the environment the table models were developed in
# (rapidfuzz not installed); an installed rapidfuzz must not switch to its Levenshtein ratio
@memoize
def similarity(a, b):
    return 1.0 if a == b else native_ratio(a, b) if a and b else 0.0


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
CAPTION = re.compile(
    r"^\s*(?:table|tab\.?|表)\s*([A-Za-z]?\d+(?:[.\-－—]\d+)*|[一二三四五六七八九十]+)",
    re.I,
)
CONT = re.compile(
    r"(?:续表|續表|接上表|[（(]续[）)]|\bcontinued\s*(?:[）)]|$)|\bcont[’\x27]?d\s*(?:[）)]|$))",
    re.I,
)


@memoize
def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).lower()


def cells(row):
    return row.find_all(["td", "th"], recursive=False)


def safe_int(s):
    try:
        return max(1, min(1000, int(s)))
    except (TypeError, ValueError):
        return 1


@memoize
def parse(text):
    from .runtime_context import readonly_soup

    soup = readonly_soup(text, "html.parser")
    trs = soup.find_all("tr")
    occupied = {}
    grid = []
    width = 0
    spans = []
    for ri, row in enumerate(trs):
        cc = []
        ci = 0
        for c in cells(row):
            while (ri, ci) in occupied:
                ci += 1
            cs = safe_int(c.get("colspan", 1))
            rs = safe_int(c.get("rowspan", 1))
            val = c.get_text(" ", strip=True)
            cc.append((ci, cs, rs, val))
            for dy in range(rs):
                for dx in range(cs):
                    occupied[(ri + dy, ci + dx)] = val
            ci += cs
            width = max(width, ci)
            spans.append((cs, rs))
        grid.append(cc)
    rows = [[v[3] for v in row] for row in grid]
    alltext = " ".join(v for row in rows for v in row)
    return {
        "soup": soup,
        "trs": trs,
        "grid": grid,
        "rows": rows,
        "ncols": width,
        "nrows": len(rows),
        "ncell": len(spans),
        "colspan": sum(cs > 1 for cs, rs in spans),
        "rowspan": sum(rs > 1 for cs, rs in spans),
        "text": norm(alltext),
        "head": norm(" ".join(v for row in rows[:2] for v in row))[:600],
        "tail": norm(" ".join(v for row in rows[-2:] for v in row))[:600],
    }


def scalar_stats(values):
    n = max(len(values), 1)
    joined = "".join(values)
    length = max(len(joined), 1)
    return [
        sum(not v for v in values) / n,
        sum(bool(re.fullmatch(r"[\d.,%()（）+−\-\s]+", v)) for v in values) / n,
        sum(map(str.isdigit, joined)) / length,
        sum(map(str.isalpha, joined)) / length,
        sum(len(v) > 30 for v in values) / n,
        math.log1p(len(joined) / n),
    ]


def fragment_features(f):
    t = f["tab"]
    rows = t["rows"]
    values = [v for row in rows for v in row]
    a = [
        t["ncols"],
        math.log1p(t["nrows"]),
        math.log1p(t["ncell"]),
        t["rowspan"] / max(t["ncell"], 1),
        t["colspan"] / max(t["ncell"], 1),
    ]
    return (
        a
        + scalar_stats(values)
        + scalar_stats(rows[0] if rows else [])
        + scalar_stats(rows[-1] if rows else [])
    )


def fragments(pages, final=None):
    out = []
    tops = []
    bottoms = []
    running = collections.defaultdict(set)
    for pi, page in enumerate(pages):
        for b in page["parsing_res_list"]:
            s = b["block_content"].strip()
            box = b["block_bbox"]
            if (
                b["block_label"] != "table"
                and 4 < len(s) < 180
                and (
                    (box[1] + box[3]) / 2 / page["height"] < 0.16
                    or box[1] / page["height"] > 0.9
                )
            ):
                running[norm(s)].add(pi)
        bb = [
            b["block_bbox"]
            for b in page["parsing_res_list"]
            if b["block_label"] not in SKIP and b["block_content"].strip()
        ]
        if bb:
            tops.append(min(b[1] for b in bb) / page["height"])
            bottoms.append(max(b[3] for b in bb) / page["height"])
    running = {
        s
        for s, seen in running.items()
        if len(seen) >= max(3, min(8, math.ceil(len(pages) * 0.1)))
    }
    doc_top = float(np.quantile(tops, 0.25)) if tops else 0.1
    doc_bottom = float(np.quantile(bottoms, 0.75)) if bottoms else 0.9
    for pi, p in enumerate(pages):
        blocks = p["parsing_res_list"]
        indices = [
            i
            for i, b in enumerate(blocks)
            if b["block_label"] == "table"
            and "<table" in b["block_content"]
            and not b.get("logical_table_handled")
        ]
        for rank, bi in enumerate(indices):
            b = blocks[bi]
            t = parse(b["block_content"])
            if not t["nrows"]:
                continue
            before = [
                q
                for q in blocks[:bi]
                if q["block_label"] not in SKIP and q["block_content"].strip()
            ]
            after = [
                q
                for q in blocks[bi + 1 :]
                if q["block_label"] not in SKIP and q["block_content"].strip()
            ]
            context = " ".join(q["block_content"] for q in before[-2:])[-600:]
            cap = next(
                (
                    CAPTION.match(q["block_content"])
                    for q in reversed(before[-3:])
                    if CAPTION.match(q["block_content"])
                ),
                None,
            )
            box = [
                v / size
                for v, size in zip(
                    b["block_bbox"], [p["width"], p["height"], p["width"], p["height"]]
                )
            ]
            f = {
                "page": pi,
                "index": bi,
                "bbox": box,
                "html": b["block_content"],
                "tab": t,
                "rank": rank,
                "tables_on_page": len(indices),
                "doc_top": doc_top,
                "doc_bottom": doc_bottom,
                "before": before,
                "after": after,
                "context": context,
                "caption_id": norm(cap[1]) if cap else "",
                "continued": bool(CONT.search(context)),
                "group": None,
            }
            f["page_blocks"] = blocks
            f["page_width"] = p["width"]
            f["page_height"] = p["height"]
            f["recovered_table_schema"] = b.get("recovered_table_schema")
            f["body_before"] = [
                q for q in before if norm(q["block_content"]) not in running
            ]
            f["body_after"] = [
                q for q in after if norm(q["block_content"]) not in running
            ]
            if final is not None:
                q = final[pi]["parsing_res_list"][bi]
                f["group"] = q.get("global_group_id")
                f["block_id"] = q.get("global_block_id")
            f["summary"] = fragment_features(f)
            out.append(f)
    if final is not None:
        parent = {f["block_id"]: f["group"] for f in out}
        for f in out:
            g = f["group"]
            seen = set()
            while g in parent and parent[g] != g and g not in seen:
                seen.add(g)
                g = parent[g]
            f["group"] = g
    parallel = collections.defaultdict(set)

    def prose_signature(f):
        rows = f["tab"]["rows"]
        if len(rows) < 2 or len(rows[0]) < 2 or max(map(len, rows[0])) > 30:
            return None
        if not any(
            sum(len(r[k]) for r in rows[1:] if len(r) > k) / max(1, len(rows) - 1) > 45
            for k in range(len(rows[0]))
        ):
            return None
        return tuple(norm(s) for s in rows[0])

    for a, b in zip(out, out[1:]):
        aa, bb = a["bbox"], b["bbox"]
        sa, sb = prose_signature(a), prose_signature(b)
        overlap = max(0, min(aa[3], bb[3]) - max(aa[1], bb[1])) / max(
            0.01, min(aa[3] - aa[1], bb[3] - bb[1])
        )
        if (
            sa
            and sa == sb
            and a["page"] == b["page"]
            and aa[2] < bb[0] + 0.03
            and overlap > 0.65
        ):
            parallel[sa].add(a["page"])
    for f in out:
        signature = (
            tuple(norm(s) for s in f["tab"]["rows"][0]) if f["tab"]["rows"] else None
        )
        f["parallel_prose_schema"] = len(parallel.get(signature, set())) >= 2
    table_identity.enrich_fragments(out, CAPTION, CONT, norm)
    return out


_DIGIT = re.compile(r"\d")


def _fragment_terms(f):
    """Normalized header cells, first-row normalizations and context of one fragment (per document)."""
    from .runtime_context import analysis_cache

    store = analysis_cache.get(None)
    cache = store.setdefault(("table_graph_fragment",), {}) if store is not None else {}
    hit = cache.get(id(f))
    if hit is not None and hit[0] is f:
        return hit[1]
    rows = f["tab"]["rows"]
    head = [norm(x) for x in rows[0]] if rows else []
    v = (
        head,
        [bool(_DIGIT.search(x)) for x in head],
        [[norm(x) for x in row] for row in rows[:5]],
        norm(f["context"]),
    )
    cache[id(f)] = (f, v)
    return v


def pair_features(a, b, intervening):
    """Same vector as _pair_features_reference; one-fragment terms come from _fragment_terms()."""
    ta, tb = a["tab"], b["tab"]
    ba, bb = a["bbox"], b["bbox"]
    wa = ba[2] - ba[0]
    wb = bb[2] - bb[0]
    (ha, da, na_rows, ca_ctx), (hb, db, nb_rows, cb_ctx) = _fragment_terms(
        a
    ), _fragment_terms(b)
    equal = sum(x == y for x, y in zip(ha, hb)) / max(len(ha), len(hb), 1)
    header_sim = sum(similarity(x, y) for x, y in zip(ha, hb)) / max(
        len(ha), len(hb), 1
    )
    numerical_conflict = sum(
        bool(p and q and x != y) for x, y, p, q in zip(ha, hb, da, db)
    ) / max(len(ha), len(hb), 1)
    repeat = 0
    for x, y in zip(na_rows, nb_rows):
        if x == y:
            repeat += 1
        else:
            break
    between = (
        (a["after"] + b["before"])
        if a["page"] != b["page"]
        else a["after"][: max(0, b["index"] - a["index"] - 1)]
    )
    titles = [
        q
        for q in between
        if q["block_label"] in {"doc_title", "paragraph_title", "figure_title"}
    ]
    bodies = [q for q in between if q["block_label"] in {"text", "content", "abstract"}]
    ca, cb = a["caption_id"], b["caption_id"]
    geometry = [
        b["page"] - a["page"],
        intervening,
        ba[1],
        ba[3],
        bb[1],
        bb[3],
        wa,
        wb,
        abs(wa - wb) / max(wa, wb, 0.01),
        abs(ba[0] - bb[0]),
        abs(ba[2] - bb[2]),
        min(ba[2], bb[2]) - max(ba[0], bb[0]),
        float(a["rank"] == a["tables_on_page"] - 1),
        float(b["rank"] == 0),
        a["tables_on_page"],
        b["tables_on_page"],
        a["rank"],
        b["rank"],
        len(titles),
        len(bodies),
        math.log1p(sum(len(q["block_content"]) for q in bodies)),
        float(a["group"] is not None and a["group"] == b["group"]),
        float(b["continued"]),
        float(bool(ca)),
        float(bool(cb)),
        float(bool(ca) and ca == cb),
        float(bool(ca and cb) and ca != cb),
    ]
    relations = [
        abs(ta["ncols"] - tb["ncols"]),
        min(ta["ncols"], tb["ncols"]) / max(ta["ncols"], tb["ncols"], 1),
        repeat,
        equal,
        header_sim,
        numerical_conflict,
        similarity(ta["head"], tb["head"]),
        similarity(ta["tail"], tb["head"]),
        similarity(ca_ctx, cb_ctx),
        abs(len(ha) - len(hb)),
    ]
    xa, xb = a["summary"], b["summary"]
    base = np.asarray(
        geometry + relations + xa + xb + [abs(x - y) for x, y in zip(xa, xb)],
        dtype=np.float32,
    )
    if runtime_context.table_features_v2.get():
        from .table_features_v2 import extras

        base = np.concatenate([base, extras(a, b)])
    return base


def _pair_features_reference(a, b, intervening):
    ta, tb = a["tab"], b["tab"]
    ba, bb = a["bbox"], b["bbox"]
    wa = ba[2] - ba[0]
    wb = bb[2] - bb[0]
    ra = ta["rows"]
    rb = tb["rows"]
    ha = [norm(x) for x in ra[0]] if ra else []
    hb = [norm(x) for x in rb[0]] if rb else []
    equal = sum(x == y for x, y in zip(ha, hb)) / max(len(ha), len(hb), 1)
    header_sim = sum(similarity(x, y) for x, y in zip(ha, hb)) / max(
        len(ha), len(hb), 1
    )
    numerical_conflict = sum(
        bool(re.search(r"\d", x) and re.search(r"\d", y) and x != y)
        for x, y in zip(ha, hb)
    ) / max(len(ha), len(hb), 1)
    repeat = 0
    for x, y in zip(ra[:5], rb[:5]):
        if [norm(v) for v in x] == [norm(v) for v in y]:
            repeat += 1
        else:
            break
    between = (
        (a["after"] + b["before"])
        if a["page"] != b["page"]
        else a["after"][: max(0, b["index"] - a["index"] - 1)]
    )
    titles = [
        q
        for q in between
        if q["block_label"] in {"doc_title", "paragraph_title", "figure_title"}
    ]
    bodies = [q for q in between if q["block_label"] in {"text", "content", "abstract"}]
    ca, cb = a["caption_id"], b["caption_id"]
    geometry = [
        b["page"] - a["page"],
        intervening,
        ba[1],
        ba[3],
        bb[1],
        bb[3],
        wa,
        wb,
        abs(wa - wb) / max(wa, wb, 0.01),
        abs(ba[0] - bb[0]),
        abs(ba[2] - bb[2]),
        min(ba[2], bb[2]) - max(ba[0], bb[0]),
        float(a["rank"] == a["tables_on_page"] - 1),
        float(b["rank"] == 0),
        a["tables_on_page"],
        b["tables_on_page"],
        a["rank"],
        b["rank"],
        len(titles),
        len(bodies),
        math.log1p(sum(len(q["block_content"]) for q in bodies)),
        float(a["group"] is not None and a["group"] == b["group"]),
        float(b["continued"]),
        float(bool(ca)),
        float(bool(cb)),
        float(bool(ca) and ca == cb),
        float(bool(ca and cb) and ca != cb),
    ]
    relations = [
        abs(ta["ncols"] - tb["ncols"]),
        min(ta["ncols"], tb["ncols"]) / max(ta["ncols"], tb["ncols"], 1),
        repeat,
        equal,
        header_sim,
        numerical_conflict,
        similarity(ta["head"], tb["head"]),
        similarity(ta["tail"], tb["head"]),
        similarity(norm(a["context"]), norm(b["context"])),
        abs(len(ha) - len(hb)),
    ]
    xa, xb = a["summary"], b["summary"]
    base = np.asarray(
        geometry + relations + xa + xb + [abs(x - y) for x, y in zip(xa, xb)],
        dtype=np.float32,
    )
    if runtime_context.table_features_v2.get():
        from .table_features_v2 import extras

        base = np.concatenate([base, extras(a, b)])
    return base


def candidates(fs, max_gap=2):
    edges = []
    xx = []
    for i, a in enumerate(fs):
        for j in range(i + 1, min(i + 8, len(fs))):
            b = fs[j]
            if b["page"] - a["page"] > max_gap:
                break
            edges.append((i, j))
            xx.append(pair_features(a, b, j - i - 1))
    return edges, np.asarray(xx, dtype=np.float32)


def path_cover(n, edges, prob, threshold=0.5):
    from scipy.optimize import linear_sum_assignment

    if not edges:
        return []
    # Every fragment can have one predecessor and one successor. Forward edges
    # prevent cycles; dummy columns allow every fragment to start a new table.
    value = np.full((n, 2 * n), -1e6, dtype=np.float64)
    value[:, n:] = 0
    offset = math.log(threshold / (1 - threshold))
    for (i, j), p in zip(edges, prob):
        p = min(1 - 1e-6, max(1e-6, float(p)))
        value[i, j] = math.log(p / (1 - p)) - offset
    rr, cc = linear_sum_assignment(value, maximize=True)
    return [(int(i), int(j)) for i, j in zip(rr, cc) if j < n and value[i, j] > 0]


def merge_chain(chain, header_mode="exact"):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup("<table></table>", "html.parser")
    table = soup.table
    first = None
    removed = 0
    for f in chain:
        part = BeautifulSoup(f["html"], "html.parser")
        rows = part.find_all("tr")
        skip = 0
        if first is not None:
            for x, y in zip(first[:5], rows[:5]):

                def clean(s):
                    if header_mode in {"identity", "residual", "schema", "structure"}:
                        s = re.sub(
                            r"[（(]?\s*(?:continued|cont[’\x27]?d|续表|續表)\s*[）)]?",
                            "",
                            s,
                            flags=re.I,
                        )
                    return norm(s)

                a = [clean(c.get_text(" ", strip=True)) for c in cells(x)]
                b = [clean(c.get_text(" ", strip=True)) for c in cells(y)]
                if len(a) == len(b) and a == b:
                    skip += 1
                elif (
                    header_mode == "fuzzy"
                    and len(a) == len(b)
                    and sum(similarity(v, w) for v, w in zip(a, b)) / max(len(a), 1)
                    > 0.94
                ):
                    skip += 1
                else:
                    break
        if first is None:
            first = copy.deepcopy(rows)
        for row in rows[skip:]:
            table.append(row.extract())
        removed += skip
    return str(soup), removed


def ordinal_range(f):
    values = []
    for row in f["tab"]["rows"]:
        # Nested sections restart local numbering within one financial table.
        if row and re.match(r"^\s*[（(]?[一二三四五六七八九十]+[、．.）)]", row[0]):
            return None
        m = re.match(r"^\s*(\d{1,4})[.)、．]\s*[^\d]", row[0]) if row else None
        if m:
            values.append(int(m[1]))
    if len(values) < 3 or sum(b > a for a, b in zip(values, values[1:])) < 0.8 * (
        len(values) - 1
    ):
        return None
    return values[0], values[-1]


def split_questionnaires(final):
    logs = []

    def question(row):
        cc = cells(row)
        if len(cc) != 2:
            return None
        code, text = (c.get_text(" ", strip=True) for c in cc)
        if (
            re.fullmatch(r"\d+(?:\.[\da-z]+)+", code, re.I)
            and len(text) > 20
            and (
                "?" in text
                or re.match(r"^(?:Please|Describe|Indicate|Specify)\b", text, re.I)
            )
        ):
            return code

    for pi, p in enumerate(final):
        for bi, b in enumerate(p["parsing_res_list"]):
            if b["block_label"] != "table" or not b["block_content"]:
                continue
            from .runtime_context import readonly_soup

            soup = readonly_soup(b["block_content"], "html.parser")
            rows = soup.find_all("tr")
            if not rows:
                continue

            def marker(row):
                cc = cells(row)
                if len(cc) != 2:
                    return None
                code, text = (c.get_text(" ", strip=True) for c in cc)
                return (
                    code
                    if re.fullmatch(r"\d+(?:\.[\da-z]+)+", code, re.I)
                    and len(text) > 20
                    else None
                )

            starts = [i for i, row in enumerate(rows) if marker(row)]
            cycles = []
            for start, end in zip(starts, starts[1:] + [len(rows)]):
                cycles.append(
                    {
                        norm(cells(r)[0].get_text(" ", strip=True))
                        for r in rows[start + 1 : end]
                        if len(cells(r)) == 2 and cells(r)[0].get_text(strip=True)
                    }
                )
            recurring = sum(
                len(a & b) >= 3 and len(a & b) / max(len(a | b), 1) > 0.3
                for a, b in zip(cycles, cycles[1:])
            )
            if (
                not starts
                or starts[0] != 0
                or recurring < max(1, 0.4 * (len(starts) - 1))
            ):
                if not question(rows[0]):
                    continue
                starts = [i for i, row in enumerate(rows) if question(row)]
            if len(set(marker(rows[i]) for i in starts)) < 2:
                continue
            b["block_content"] = "\n\n".join(
                "<table>" + "".join(str(r) for r in rows[start:end]) + "</table>"
                for start, end in zip(starts, starts[1:] + [len(rows)])
            )
            logs.append(
                {
                    "kind": "independent_questionnaire_tables",
                    "page": pi + 1,
                    "block": bi,
                    "tables": len(starts),
                }
            )
    return logs


def identity_probability(a, b, p):
    """Explicit table identity evidence takes priority over local appearance."""
    reason = table_identity.reject_reason(a, b)
    if reason:
        return 0.0, reason
    if structure_guard.closed_signed_statement(a, b):
        return 0.0, "closed_signed_statement_sections"
    cfg = runtime_context.get_config()
    ta, tb = a["tab"], b["tab"]
    ra, rb = ta["rows"], tb["rows"]
    ca, cb = a["caption_id"], b["caption_id"]
    if ca and cb and ca != cb and not b["continued"]:
        return 0.0, "new_caption"
    if cfg.table_new_caption_guard and cb and not ca and not b["continued"]:
        same_old = a.get("group") is not None and a.get("group") == b.get("group")
        if not same_old:
            return 0.0, "new_caption_after_uncaptioned_table"
    if cfg.table_strict_identity:
        from .structural_links import header_change, titled_number_reset

        if titled_number_reset(a, b):
            return 0.0, "new_internal_title_and_number_reset"
        if header_change(a, b) and not b["continued"]:
            return 0.0, "changed_column_labels"
    oa, ob = ordinal_range(a), ordinal_range(b)
    if oa and ob and ob[0] <= oa[1]:
        return 0.0, "row_number_reset"
    # Repeated outer headers can describe different subtopics, e.g. a matrix
    # has identical applications across independent question sections.
    if len(ra) > 1 and len(rb) > 1:
        ha, hb = ra[0], rb[0]
        sa, sb = ra[1], rb[1]
        if (
            len(ha) == len(hb)
            and len(ha) > 2
            and sum(norm(x) == norm(y) for x, y in zip(ha, hb)) / len(ha) > 0.85
        ):
            bool_header = lambda row: len(row) > 2 and sum(
                norm(v) in {"y", "n", "yes", "no", "是", "否"} for v in row[1:]
            ) >= 0.8 * (len(row) - 1)
            if bool_header(sa) and bool_header(sb):
                clean = lambda s: norm(
                    re.sub(
                        r"[（(]?\s*(?:continued|cont[’\x27]?d)\s*[）)]?",
                        "",
                        s,
                        flags=re.I,
                    )
                )
                if clean(sa[0]) != clean(sb[0]):
                    return 0.0, "new_subtopic"
    continued = any(CONT.search(v) for row in rb[:2] for v in row) or b["continued"]
    boundary = (
        a["page"] + 1 == b["page"]
        and a["rank"] == a["tables_on_page"] - 1
        and b["rank"] == 0
    ) or (a["page"] == b["page"] and a["rank"] + 1 == b["rank"])
    if continued and boundary and abs(a["bbox"][0] - b["bbox"][0]) < 0.15:
        if min(ta["ncols"], tb["ncols"]) / max(ta["ncols"], tb["ncols"], 1) > 0.6:
            return 0.999, "explicit_continuation"
    if (
        oa
        and ob
        and oa[1] + 1 == ob[0]
        and ta["ncols"] == tb["ncols"]
        and b["page"] == a["page"] + 1
    ):
        return 0.995, "row_number_continuation"
    if ra and rb:
        last = ra[-1]
        total = bool(
            last
            and re.fullmatch(
                r"\s*(?:grand\s+total|total|合计|合計|总计|總計)\s*[:：]?\s*",
                last[0],
                re.I,
            )
        )
        repeat = sum(norm(x) == norm(y) for x, y in zip(ra[0], rb[0])) / max(
            len(ra[0]), len(rb[0]), 1
        )
        if total and repeat > 0.7:
            return 0.01, "closed_total"
    return float(p), None


# A bare Chinese continuation label ("续表", "（续表）", "接上表") has no table
# number of its own: the page only resumes the previous table. Labels that
# restate a number ("续表2-7") or follow a repeated title ("资产负债表（续）")
# introduce a separately captioned table and keep the continued-caption veto.
BARE_CN = re.compile(
    r"(?:续表|續表|接上表|续上表|續上表)(?![\s\-—–:：]*[0-9０-９一二三四五六七八九十A-Za-z])"
)


def bare_cn_continuation(f):
    texts = [q["block_content"].strip() for q in f.get("before", [])[-2:]]
    marked = [t for t in texts if CONT.search(t)]
    return bool(marked) and all(len(t) <= 12 and BARE_CN.search(t) for t in marked)


def apply(pages, final, model, threshold=0.5, header_mode="exact", add_threshold=0.98):
    fs = fragments(pages, final)
    edges, x = candidates(fs)
    p = model.predict_proba(x)[:, 1] if len(edges) else []
    learned0 = [float(v) for v in p]
    decisions = []
    original = {
        i: fast_copy.deepcopy(final[f["page"]]["parsing_res_list"][f["index"]])
        for i, f in enumerate(fs)
    }
    oldgroups = collections.defaultdict(list)
    for i, f in enumerate(fs):
        oldgroups[f["group"]].append(i)
    oldedges = {(i, j) for inds in oldgroups.values() for i, j in zip(inds, inds[1:])}
    if header_mode in {"residual", "schema", "structure"}:
        for ei, (i, j) in enumerate(edges):
            required = add_threshold
            # Numeric column profiles are stronger continuation evidence than
            # short categorical cells (e.g. independent code lookup tables).
            if min(fs[i]["summary"][6], fs[j]["summary"][6]) < 0.15:
                required = max(required, 0.65)
            learned = float(p[ei])
            p[ei] = (
                0.99
                if (i, j) in oldedges
                else (0.8 if j == i + 1 and learned > required else 0.001)
            )
    if header_mode in {"schema", "structure"}:
        from .column_alignment import continuation

        for ei, (i, j) in enumerate(edges):
            if p[ei] > 0.5:
                continue
            ok, detail = continuation(fs[i], fs[j])
            if ok:
                p[ei] = 0.9
                decisions.append(
                    {
                        "kind": "schema_continuation",
                        "left": [fs[i]["page"] + 1, fs[i]["index"]],
                        "right": [fs[j]["page"] + 1, fs[j]["index"]],
                        **detail,
                    }
                )
    if header_mode in {"identity", "residual", "schema", "structure"}:
        for ei, (i, j) in enumerate(edges):
            old = float(p[ei])
            p[ei], reason = identity_probability(fs[i], fs[j], old)
            if not reason and header_mode == "structure":
                from .structural_links import proposal

                p[ei], reason = proposal(fs[i], fs[j], float(p[ei]))
            if reason:
                decisions.append(
                    {
                        "kind": reason,
                        "left": [fs[i]["page"] + 1, fs[i]["index"]],
                        "right": [fs[j]["page"] + 1, fs[j]["index"]],
                        "before": old,
                        "after": float(p[ei]),
                    }
                )
    cfg = runtime_context.get_config()
    _vc = cfg.table_veto_continued
    _vl = cfg.table_veto_learned
    if _vc or _vl:
        for ei, (i, j) in enumerate(edges):
            if fs[i]["page"] == fs[j]["page"] or p[ei] <= 0.5:
                continue
            why = (
                "continued_caption"
                if (_vc and fs[j]["continued"])
                else ("low_learned" if (_vl and learned0[ei] < _vl) else None)
            )
            if (
                why == "continued_caption"
                and cfg.table_veto_bare_cn
                and bare_cn_continuation(fs[j])
            ):
                decisions.append(
                    {
                        "kind": "bare_continuation_kept",
                        "left": [fs[i]["page"] + 1, fs[i]["index"]],
                        "right": [fs[j]["page"] + 1, fs[j]["index"]],
                        "learned": learned0[ei],
                        "before": float(p[ei]),
                    }
                )
                why = None
            if why:
                d = {
                    "kind": "table_floor_veto",
                    "reason": why,
                    "left": [fs[i]["page"] + 1, fs[i]["index"]],
                    "right": [fs[j]["page"] + 1, fs[j]["index"]],
                    "learned": learned0[ei],
                    "before": float(p[ei]),
                }
                decisions.append(d)
                p[ei] = 0.001
    selected = path_cover(len(fs), edges, p, threshold)
    succ = dict(selected)
    incoming = {j for i, j in selected}
    chains = []
    for i in range(len(fs)):
        if i in incoming:
            continue
        chain = [i]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        chains.append(chain)
    for f in fs:
        b = final[f["page"]]["parsing_res_list"][f["index"]]
        b["block_label"] = "table"
        b["block_content"] = ""
    logs = []
    for inds in chains:
        changed = any(
            pages[fs[i]["page"]]["parsing_res_list"][fs[i]["index"]].get(
                "table_rebuild"
            )
            for i in inds
        )
        if (
            header_mode in {"residual", "schema", "structure"}
            and inds == oldgroups[fs[inds[0]]["group"]]
            and not changed
            and not (cfg.table_rebuild_chains and len(inds) > 1)
        ):
            for i in inds:
                f = fs[i]
                final[f["page"]]["parsing_res_list"][f["index"]] = original[i]
            continue
        from .table_row_repair import repair

        chain, repairs = repair([fs[i] for i in inds])
        logs += repairs
        text, removed = merge_chain(chain, header_mode)
        f = chain[0]
        final[f["page"]]["parsing_res_list"][f["index"]]["block_content"] = text
        if len(chain) > 1:
            logs.append(
                {
                    "kind": "graph_table_chain",
                    "fragments": [(f["page"] + 1, f["index"]) for f in chain],
                    "removed_header_rows": removed,
                }
            )
    return logs + decisions, {
        "fragments": len(fs),
        "candidate_edges": len(edges),
        "selected_edges": len(selected),
        "chains": len(chains),
    }
