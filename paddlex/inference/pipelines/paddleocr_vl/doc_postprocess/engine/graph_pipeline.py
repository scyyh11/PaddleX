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

from . import (
    catalogue_regions,
    clean_repetition,
    code_recovery,
    figure_flow,
    geometric_links,
    inline_titles,
    inventory_text,
    isolated_notes,
    layout_text,
    paragraph_lines,
    quote_paths,
    quote_refinement,
    quoted_regions,
    regions_text,
    running_text,
    semantic_regions,
    source_paragraphs,
    structural_text,
    transcript_lines,
    upstream_links,
)


def graph(pre, saved, v):
    mode = "ordered"
    pp, final = pre, saved
    logs = figure_flow.prepare(pp, final, "captions") + figure_flow.prepare(
        pp, final, "labels"
    )
    logs += semantic_regions.apply(pp, final, "scripts")
    logs += (
        running_text.apply(pp, final)
        + inventory_text.apply(pp, final)
        + isolated_notes.apply(pp, final, v)
        + clean_repetition.apply(pp, final)
    )
    logs += (
        transcript_lines.apply(pp, final, v)
        + quoted_regions.apply(pp, final)
        + catalogue_regions.apply(pp, final)
    )
    logs += (
        regions_text.lexical(pp, final)
        + regions_text.navigation(pp, final)
        + code_recovery.apply(pp, final, "comments")
    )
    if mode == "titles":
        logs += inline_titles.apply(pp, final)

    def repair_with_paragraphs(p, q, vv):
        result = layout_text.repair_source(p, q, vv)
        result += source_paragraphs.apply(p, q, vv, "wideblank")
        result += paragraph_lines.apply(p, q, vv, True)
        return result

    ll, nodes, visible, fixed = layout_text.apply(
        pp, final, v, graph_only=True, repair_hook=repair_with_paragraphs
    )
    logs += ll
    extra, el = structural_text.apply(nodes, visible, final, fixed, "structure")
    fixed += extra
    logs += el
    if mode != "base":
        more, ml = geometric_links.apply(nodes, visible, final, fixed, "corridor")
        fixed += more
        logs += ml
    more, ml = geometric_links.longgap(nodes, visible, final, fixed)
    fixed += more
    logs += ml
    fixed, ml = quote_paths.apply(nodes, visible, final, fixed, "both")
    logs += ml
    if mode in {"quote", "both"}:
        more, ml = quote_refinement.apply(nodes, visible, final, fixed)
        fixed += more
        logs += ml
    if mode == "upstream":
        more, ml = upstream_links.apply(nodes, visible, final, fixed, None, 0)
        fixed += more
        logs += ml
    return final, nodes, visible, fixed, logs
