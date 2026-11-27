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

"""Text and paragraph merging stage driven explicitly by DocPostprocessConfig."""

from . import (
    caption_streams,
    label_recovery,
    line_breaks,
    metadata_refs,
    paragraph_guard,
    paragraph_identity,
    paragraph_runtime_gate_v2,
    quote_closure,
    running_text,
)
from . import runtime_context as ctx
from . import tail_regions, text_features
from .graph_pipeline import graph


def process(pre, saved, v, labels, doc=None):
    cfg = ctx.get_config()
    logs = running_text.apply(pre, saved) if cfg.early_running_markers else []
    if cfg.metadata_references:
        logs += metadata_refs.apply(pre, saved)
    logs += label_recovery.apply(pre, saved, v, labels)
    logs += caption_streams.apply(pre, saved, {**v, **labels}, "crosspage")
    final, nodes, visible, fixed, gl = graph(pre, saved, v)
    logs += gl
    fixed, gl = paragraph_guard.apply(nodes, visible, fixed)
    logs += gl
    if cfg.quote_closure:
        fixed, gl = quote_closure.apply(nodes, visible, fixed)
        logs += gl
    if cfg.tail_regions:
        fixed, gl = tail_regions.apply(final, nodes, visible, fixed, cfg.tail_regions)
        logs += gl

    kept, gate_events = paragraph_runtime_gate_v2.filter_links(
        nodes, fixed, doc=doc, config=cfg
    )
    kept, identity_events = paragraph_identity.filter_links(nodes, kept)
    text_features.merge_nodes(nodes, kept)

    for page in final:
        for b in page["parsing_res_list"]:
            if b["block_label"] in text_features.TEXT:
                b["block_content"] = line_breaks.render(b["block_content"])
    return final, logs, gate_events, identity_events
