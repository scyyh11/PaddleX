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

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..infra.models import DataInfo
from ..infra.storage import Storage
from ..schemas.paddleocr_vl import (
    InferRequest,
    InferResult,
    RestructurePagesRequest,
    RestructurePagesResult,
)
from ..schemas.shared.export import normalize_output_formats
from .document_export import (
    build_pipeline_exports,
    refill_paddleocr_vl_images_from_markdown,
)
from .image_output import postprocess_images, prune_result


def get_inference_kwargs(request: InferRequest) -> Dict[str, Any]:
    return dict(
        use_doc_orientation_classify=request.useDocOrientationClassify,
        use_doc_unwarping=request.useDocUnwarping,
        use_layout_detection=request.useLayoutDetection,
        use_layout_deform=request.useLayoutDeform,
        layout_deform_conf_thresh=request.layoutDeformConfThresh,
        layout_deform_min_roll=request.layoutDeformMinRoll,
        layout_deform_min_shear=request.layoutDeformMinShear,
        layout_deform_min_keystone=request.layoutDeformMinKeystone,
        use_chart_recognition=request.useChartRecognition,
        use_seal_recognition=request.useSealRecognition,
        use_ocr_for_image_block=request.useOcrForImageBlock,
        layout_threshold=request.layoutThreshold,
        layout_nms=request.layoutNms,
        layout_unclip_ratio=request.layoutUnclipRatio,
        layout_merge_bboxes_mode=request.layoutMergeBboxesMode,
        layout_shape_mode=request.layoutShapeMode,
        prompt_label=request.promptLabel,
        format_block_content=request.formatBlockContent,
        repetition_penalty=request.repetitionPenalty,
        temperature=request.temperature,
        top_p=request.topP,
        min_pixels=request.minPixels,
        max_pixels=request.maxPixels,
        max_new_tokens=request.maxNewTokens,
        merge_layout_blocks=request.mergeLayoutBlocks,
        markdown_ignore_labels=request.markdownIgnoreLabels,
        vlm_extra_args=request.vlmExtraArgs,
    )


def build_infer_result(
    pipeline: Any,
    request: InferRequest,
    images: Sequence[np.ndarray],
    data_info: DataInfo,
    preds: list,
    *,
    log_id: str,
    visualize: bool,
    file_storage: Optional[Storage],
    return_urls: bool,
    url_expires_in: int,
    max_output_img_size: Tuple[int, int],
) -> InferResult:
    if request.restructurePages:
        preds = list(
            pipeline.restructure_pages(
                preds,
                merge_tables=request.mergeTables,
                relevel_titles=request.relevelTitles,
                concatenate_pages=False,
            )
        )

    output_options = dict(
        log_id=log_id,
        file_storage=file_storage,
        return_urls=return_urls,
        url_expires_in=url_expires_in,
    )
    output_formats = normalize_output_formats(request.outputFormats)
    layout_parsing_results = []
    for i, (img, item) in enumerate(zip(images, preds)):
        pruned_result = prune_result(item.json["res"])
        md_data = item._to_markdown(
            pretty=request.prettifyMarkdown,
            show_formula_number=request.showFormulaNumber,
        )
        md_images = None
        if request.returnMarkdownImages:
            md_images = postprocess_images(
                md_data["markdown_images"],
                filename_template=f"markdown_{i}/{{key}}",
                max_img_size=max_output_img_size,
                **output_options,
            )

        visualizations = {}
        if visualize:
            visualizations = postprocess_images(
                {"input_img": img, **item.img},
                filename_template=f"{{key}}_{i}.jpg",
                max_img_size=max_output_img_size,
                **output_options,
            )

        entry = dict(
            prunedResult=pruned_result,
            markdown=dict(text=md_data["markdown_texts"], images=md_images),
            outputImages=(
                {k: v for k, v in visualizations.items() if k != "input_img"}
                if visualizations
                else None
            ),
            inputImage=visualizations.get("input_img"),
        )
        if output_formats:
            entry["exports"] = build_pipeline_exports(
                output_formats, item, **output_options
            )
        layout_parsing_results.append(entry)

    return InferResult(
        layoutParsingResults=layout_parsing_results,
        dataInfo=data_info,
    )


def build_restructure_pages_result(
    pipeline: Any,
    request: RestructurePagesRequest,
    *,
    log_id: str,
    file_storage: Optional[Storage],
    return_urls: bool,
    url_expires_in: int,
) -> RestructurePagesResult:
    original_results = [
        {"res": {**page.prunedResult, "input_path": "", "page_index": i}}
        for i, page in enumerate(request.pages)
    ]
    restructured_results = list(
        pipeline.restructure_pages(
            original_results,
            merge_tables=request.mergeTables,
            relevel_titles=request.relevelTitles,
            concatenate_pages=request.concatenatePages,
        )
    )

    if request.concatenatePages:
        merged_images = {}
        for page in request.pages:
            if page.markdownImages:
                merged_images.update(page.markdownImages)
        pages_with_images = [(restructured_results[0], merged_images)]
    else:
        pages_with_images = zip(
            restructured_results, (page.markdownImages for page in request.pages)
        )

    output_formats = normalize_output_formats(request.outputFormats)
    layout_parsing_results = []
    for result, markdown_images in pages_with_images:
        pruned_result = prune_result(result.json["res"])
        md_data = result._to_markdown(
            pretty=request.prettifyMarkdown,
            show_formula_number=request.showFormulaNumber,
        )
        entry = dict(
            prunedResult=pruned_result,
            markdown=dict(
                text=md_data["markdown_texts"],
                images=markdown_images if request.returnMarkdownImages else None,
            ),
        )
        if output_formats:
            refill_paddleocr_vl_images_from_markdown(result, markdown_images)
            entry["exports"] = build_pipeline_exports(
                output_formats,
                result,
                log_id=log_id,
                file_storage=file_storage,
                return_urls=return_urls,
                url_expires_in=url_expires_in,
            )
        layout_parsing_results.append(entry)

    return RestructurePagesResult(layoutParsingResults=layout_parsing_results)
