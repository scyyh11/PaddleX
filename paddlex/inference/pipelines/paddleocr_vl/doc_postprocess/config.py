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

"""Explicit configuration for the multi-page document post-processing pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DocPostprocessConfig:
    """Immutable configuration replacing internal environment toggles and JSON knobs."""

    # Heading recovery & hierarchy decoding
    heading_keep_old: float = 0.55
    heading_new_cut: float = 0.55
    heading_contact_roles: bool = True
    heading_cover_segments: bool = True
    heading_toc_page_anchors: bool = True
    heading_numbering_grammar: bool = True
    heading_span_weight: Optional[float] = 2.0
    heading_span_lookahead: int = 32
    heading_features_v3: bool = True
    heading_index_entries: bool = True
    heading_reset_nesting: bool = True
    heading_joint_relation_weight: float = 3.0
    heading_graph_weight: Optional[float] = None
    heading_text_model: bool = False
    heading_structure_mode: Optional[str] = None
    bilingual_titles: bool = True
    cover_report_titles: bool = True

    # Cross-page table graph & identity guards
    table_veto_bare_cn: bool = True
    table_veto_continued: bool = True
    table_veto_learned: float = 0.0
    table_strict_identity: bool = True
    table_new_caption_guard: bool = True
    table_rebuild_chains: bool = False

    # Text & paragraph graph
    paragraph_same_page_threshold: float = 0.35
    paragraph_cross_page_threshold: float = 0.5
    date_split: bool = True
    section_split: bool = True
    font_barrier: bool = True
    indent_barrier: bool = False
    wide_parenthesis: bool = False
    reference_end: str = "circled"
    paired_caption_columns: bool = True
    early_running_markers: bool = False
    metadata_references: bool = True
    tail_regions: Optional[str] = "both"
    quote_closure: bool = False
    margin_text: Optional[str] = "all_plus"
    layout_item_mode: str = "roman"
    layout_ref_mode: str = "split"
    layout_numeric_items: bool = True
    layout_letter_items: bool = False
    keep_visual: bool = True
    quote_paths_guarded: bool = True
    keep_source_order: bool = True

    # Formula, figure, and Markdown cleanup
    formula_repair: bool = True
    formula_number_after: bool = True
    formula_single_line_env: bool = True
    figure_noise: bool = True
    figure_marginal_if_none: bool = True
    option_split: bool = True
