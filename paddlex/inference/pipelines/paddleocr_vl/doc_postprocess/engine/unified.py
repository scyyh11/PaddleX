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

"""Single document postprocessor: OCR JSON -> headings -> tables -> text -> Markdown."""

import time

from . import (
    bilingual_titles,
    course_tables,
    cover_report_titles,
    doc_resources,
    fast_copy,
    figure_noise,
    formula_numbers_after,
    formula_repair,
    formula_wrappers,
    graphic_component_apply,
    hierarchical_sections,
    marginal_figures,
    marginal_text,
    markup_repair,
    math_boundaries,
    numeric_grid_recovery,
    option_split,
    paragraph_identity,
    paragraph_integrity,
    postprocess,
    record_panels,
    renderer,
)
from . import runtime_context as ctx
from . import (
    source_text_structure,
    table_graph,
    table_text_columns,
    text_stage,
    text_table_recovery,
    toc_bound,
)


def apply_markdown_repairs(
    text,
    cfg=None,
    *,
    apply_markup_repair=True,
    apply_figure_noise=True,
):
    """Apply post-render Markdown repairs and return ``(text, events)``."""
    if cfg is None:
        cfg = ctx.get_config()
    repair_events = {}
    if apply_markup_repair:
        text = markup_repair.apply(text)
    if cfg.option_split:
        text, count = option_split.apply(text)
        repair_events["option_split"] = (
            [{"kind": "options_split", "count": count}] if count else []
        )
    if cfg.formula_single_line_env:
        text, count = formula_wrappers.apply(text)
        repair_events["formula_wrappers"] = (
            [{"kind": "single_line_env_unwrapped", "count": count}] if count else []
        )
    if cfg.formula_repair:
        text, count = formula_repair.apply(text)
        repair_events["formula_repair"] = (
            [{"kind": "display_formula_repaired", "count": count}] if count else []
        )
    if apply_figure_noise and cfg.figure_noise:
        text, count = figure_noise.apply(text)
        repair_events["figure_noise"] = (
            [{"kind": "image_noise_removed", "count": count}] if count else []
        )
    return text, repair_events


class DocumentPostprocessor:
    def __init__(
        self,
        device="cuda",
        stages=("headings", "tables", "text"),
        table_models="tables",
    ):
        if device not in ("cpu", "cuda"):
            raise ValueError(device)
        self.device = device
        self.stages = stages
        self.table_models = doc_resources.path(table_models)

    def warmup(self):
        started = time.perf_counter()
        token = ctx.backend.set(self.device)
        try:
            paths = []
            if "headings" in self.stages:
                for family, pattern in [
                    ("headings", "fold-*"),
                    ("parents", "parent-fold-*"),
                    ("spans", "span-fold-*"),
                ]:
                    paths += sorted(
                        p
                        for p in doc_resources.path(family).glob(pattern)
                        if p.suffix in (".safetensors", ".joblib")
                    )
            if "tables" in self.stages:
                paths += sorted(
                    p
                    for p in self.table_models.glob("fold-*")
                    if p.suffix in (".safetensors", ".joblib")
                )
            for family, pattern in [
                ("graph", "graph-fold-*"),
                ("paragraph-runtime", "fold-*"),
                ("paragraph-pairs", "fold-*"),
                ("graphic", "fold-*"),
            ]:
                paths += sorted(
                    p
                    for p in doc_resources.path(family).glob(pattern)
                    if p.suffix in (".safetensors", ".joblib")
                )
            seen_stems = set()
            loaded = 0
            for path in paths:
                stem = str(path.with_suffix(""))
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)
                ctx.load_model(path)
                loaded += 1
            if self.device == "cuda" and "text" in self.stages:
                import numpy as np

                from .gpu_paragraph import solve
                from .gpu_transcript import solve as transcript_solve

                solve(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0]))
                transcript_solve([0, 2, 4], [0, 1, 0], np.array([1.0, 2.0]))
            if "headings" in self.stages:
                toc_bound.admissible(["warmup title"], ["warmup titles"])
            return {
                "model_packages": loaded,
                "warmup_ms": (time.perf_counter() - started) * 1000,
            }
        finally:
            ctx.backend.reset(token)

    def process(
        self,
        doc,
        pages=None,
        heading_visual=None,
        text_visual=None,
        label_visual=None,
        prepared=None,
        final=None,
        prepared_hook=None,
    ):
        started = time.perf_counter()
        timings = {}
        events = {}
        cfg = ctx.get_config()

        raw_pages = fast_copy.copy_pages(pages) if pages is not None else None
        source_events = (
            math_boundaries.apply(raw_pages) if raw_pages is not None else []
        )
        events["source_structure"] = source_events

        tokens = [
            (ctx.analysis_cache, ctx.analysis_cache.set({})),
            (ctx.backend, ctx.backend.set(self.device)),
            (ctx.html_cache, ctx.html_cache.set({})),
            (ctx.raw_pages, ctx.raw_pages.set(raw_pages)),
            (ctx.input_pages, ctx.input_pages.set((doc, raw_pages))),
        ]
        try:
            if "headings" in self.stages:
                t = time.perf_counter()
                prepared, final, events["headings"] = postprocess.process_in_memory(
                    doc, fast_copy.copy_pages(raw_pages), heading_visual
                )
                paragraph_identity.annotate_source_flags(raw_pages, prepared, final)
                paragraph_integrity.annotate_source_flags(raw_pages, prepared, final)
                source_text_structure.annotate_source_flags(raw_pages, prepared, final)
                timings["headings_ms"] = (time.perf_counter() - t) * 1000
            if prepared_hook is not None and prepared is not None:
                t = time.perf_counter()
                prepared_hook(prepared)
                timings["prepared_hook_ms"] = (time.perf_counter() - t) * 1000
            if prepared is None or final is None:
                raise ValueError(
                    "Supply OCR pages with headings enabled, or prepared/final stage inputs"
                )
            if "tables" in self.stages:
                t = time.perf_counter()
                fold = doc_resources.fold(self.table_models)
                package = ctx.load_model(self.table_models / f"fold-{fold}.joblib")
                v2_tok = ctx.table_features_v2.set(
                    package.get("feature_version", 1) == 2
                )
                try:
                    logs = []
                    for module in [
                        text_table_recovery,
                        table_text_columns,
                        numeric_grid_recovery,
                        hierarchical_sections,
                        record_panels,
                    ]:
                        logs += module.apply(prepared, final)
                    ll, _stats = table_graph.apply(
                        prepared, final, package["model"], 0.5, "structure", 0.5
                    )
                    logs += ll
                    logs += table_graph.split_questionnaires(final)
                    logs += course_tables.apply(prepared, final)
                finally:
                    ctx.table_features_v2.reset(v2_tok)
                events["tables"] = logs
                timings["tables_ms"] = (time.perf_counter() - t) * 1000
            if "text" in self.stages:
                t = time.perf_counter()
                final, events["text"], gate_events, identity_events = (
                    text_stage.process(
                        prepared, final, text_visual, label_visual, doc=doc
                    )
                )
                events["paragraph_runtime_gate"] = gate_events
                events["paragraph_identity"] = identity_events
                timings["text_ms"] = (time.perf_counter() - t) * 1000

            events["graphic_components"] = graphic_component_apply.apply(
                raw_pages, final, doc
            )
            if cfg.figure_marginal_if_none:
                events["marginal_figures"] = marginal_figures.apply(final)
            if cfg.formula_number_after:
                events["formula_numbers"] = formula_numbers_after.apply(final)
            if cfg.margin_text:
                events["margin_text"] = marginal_text.apply(final, cfg.margin_text)
            if cfg.bilingual_titles:
                events["bilingual_titles"] = bilingual_titles.apply(final)
            if cfg.cover_report_titles:
                events["cover_report_titles"] = cover_report_titles.apply(final)

            t = time.perf_counter()
            text = renderer.render(final)
            text, repair_events = apply_markdown_repairs(
                text,
                cfg,
                apply_markup_repair="text" in self.stages,
                apply_figure_noise=True,
            )
            events.update(repair_events)
            timings["render_ms"] = (time.perf_counter() - t) * 1000
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            return {
                "markdown": text,
                "pages": final,
                "events": events,
                "timings": timings,
            }
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
