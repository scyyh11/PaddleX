# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import warnings

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Final, List, Tuple

from paddlex_hps_server import (
    BaseTritonPythonModel,
    app_common,
    logging,
    protocol,
    schemas,
    utils,
)
from paddlex_hps_server.batching import group_request_indices
from paddlex_hps_server.storage import SupportsGetURL, create_storage

_DEFAULT_MAX_NUM_INPUT_IMGS: Final[int] = 10
_DEFAULT_MAX_OUTPUT_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)


class _SequentialExecutor(object):
    def map(self, fn, *iterables):
        return map(fn, *iterables)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class TritonPythonModel(BaseTritonPythonModel):
    def initialize(self, args):
        super().initialize(args)
        self.context = {}
        self.context["file_storage"] = None
        self.context["return_urls"] = self.app_config.return_urls
        self.context["url_expires_in"] = -1
        self.context["max_num_input_imgs"] = _DEFAULT_MAX_NUM_INPUT_IMGS
        self.context["max_output_img_size"] = _DEFAULT_MAX_OUTPUT_IMG_SIZE
        if self.app_config.extra:
            if "file_storage" in self.app_config.extra:
                self.context["file_storage"] = create_storage(
                    self.app_config.extra["file_storage"]
                )
            if "return_img_urls" in self.app_config.extra:
                warnings.warn(
                    "`Serving.extra.return_img_urls` is deprecated; use the "
                    "top-level `Serving.return_urls` field instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if self.context["return_urls"] is None:
                    self.context["return_urls"] = bool(
                        self.app_config.extra["return_img_urls"]
                    )
            if "url_expires_in" in self.app_config.extra:
                self.context["url_expires_in"] = self.app_config.extra["url_expires_in"]
            if "max_num_input_imgs" in self.app_config.extra:
                self.context["max_num_input_imgs"] = self.app_config.extra[
                    "max_num_input_imgs"
                ]
            if "max_output_img_size" in self.app_config.extra:
                self.context["max_output_img_size"] = self.app_config.extra[
                    "max_output_img_size"
                ]
        if self.context["return_urls"] is None:
            self.context["return_urls"] = False
        if self.context["return_urls"]:
            file_storage = self.context["file_storage"]
            if not file_storage:
                raise ValueError(
                    "The file storage must be properly configured when URLs need to be returned."
                )
            if not isinstance(file_storage, SupportsGetURL):
                raise TypeError(f"{type(file_storage)} does not support getting URLs.")

    def get_input_model_type(self):
        return schemas.pp_structurev3.InferRequest

    def get_result_model_type(self):
        return schemas.pp_structurev3.InferResult

    def run(self, input, log_id):
        return self.run_batch([input], [log_id], log_id)

    def run_batch(self, inputs, log_ids, batch_id):
        result_or_output_dic = {}

        inference_kwargs = [self._get_inference_kwargs(request) for request in inputs]
        input_groups = group_request_indices(inference_kwargs)

        max_group_size = max(map(len, input_groups))
        if max_group_size > 1:
            executor = ThreadPoolExecutor(max_workers=max_group_size)
        else:
            executor = _SequentialExecutor()

        with executor:
            for input_ids_g in input_groups:
                inputs_g = [inputs[i] for i in input_ids_g]

                log_ids_g = [log_ids[i] for i in input_ids_g]

                ret = executor.map(self._preprocess, inputs_g, log_ids_g)
                ind_img_lsts, ind_data_info_lst, ind_visualize_enabled_lst = [], [], []
                ind_input_ids_lst, ind_log_ids_lst, ind_inputs_lst = [], [], []
                for i, item in enumerate(ret):
                    if isinstance(item, tuple):
                        assert len(item) == 3, len(item)
                        ind_img_lsts.append(item[0])
                        ind_data_info_lst.append(item[1])
                        ind_visualize_enabled_lst.append(item[2])
                        ind_input_ids_lst.append(input_ids_g[i])
                        ind_log_ids_lst.append(log_ids_g[i])
                        ind_inputs_lst.append(inputs_g[i])
                    else:
                        input_id = input_ids_g[i]
                        result_or_output_dic[input_id] = item

                if len(ind_img_lsts):
                    images = [img for item in ind_img_lsts for img in item]
                    preds = list(
                        self.pipeline(images, **inference_kwargs[input_ids_g[0]])
                    )

                    if len(preds) != len(images):
                        raise RuntimeError(
                            f"The number of predictions ({len(preds)}) is not the same as the number of input images ({len(images)})."
                        )

                    start_idx = 0
                    ind_preds = []
                    for item in ind_img_lsts:
                        ind_preds.append(preds[start_idx : start_idx + len(item)])
                        start_idx += len(item)

                    for input_id, result in zip(
                        ind_input_ids_lst,
                        executor.map(
                            self._postprocess,
                            ind_img_lsts,
                            ind_data_info_lst,
                            ind_visualize_enabled_lst,
                            ind_preds,
                            ind_log_ids_lst,
                            ind_inputs_lst,
                        ),
                    ):
                        result_or_output_dic[input_id] = result

            assert len(result_or_output_dic) == len(
                inputs
            ), f"Expected {len(inputs)} results or outputs, but got {len(result_or_output_dic)}"

            return [result_or_output_dic[i] for i in range(len(inputs))]

    @staticmethod
    def _get_inference_kwargs(request):
        return dict(
            use_doc_orientation_classify=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_textline_orientation=request.useTextlineOrientation,
            use_seal_recognition=request.useSealRecognition,
            use_table_recognition=request.useTableRecognition,
            use_formula_recognition=request.useFormulaRecognition,
            use_chart_recognition=request.useChartRecognition,
            use_region_detection=request.useRegionDetection,
            format_block_content=request.formatBlockContent,
            layout_threshold=request.layoutThreshold,
            layout_nms=request.layoutNms,
            layout_unclip_ratio=request.layoutUnclipRatio,
            layout_merge_bboxes_mode=request.layoutMergeBboxesMode,
            text_det_limit_side_len=request.textDetLimitSideLen,
            text_det_limit_type=request.textDetLimitType,
            text_det_thresh=request.textDetThresh,
            text_det_box_thresh=request.textDetBoxThresh,
            text_det_unclip_ratio=request.textDetUnclipRatio,
            text_rec_score_thresh=request.textRecScoreThresh,
            seal_det_limit_side_len=request.sealDetLimitSideLen,
            seal_det_limit_type=request.sealDetLimitType,
            seal_det_thresh=request.sealDetThresh,
            seal_det_box_thresh=request.sealDetBoxThresh,
            seal_det_unclip_ratio=request.sealDetUnclipRatio,
            seal_rec_score_thresh=request.sealRecScoreThresh,
            use_wired_table_cells_trans_to_html=request.useWiredTableCellsTransToHtml,
            use_wireless_table_cells_trans_to_html=request.useWirelessTableCellsTransToHtml,
            use_table_orientation_classify=request.useTableOrientationClassify,
            use_ocr_results_with_table_cells=request.useOcrResultsWithTableCells,
            use_e2e_wired_table_rec_model=request.useE2eWiredTableRecModel,
            use_e2e_wireless_table_rec_model=request.useE2eWirelessTableRecModel,
            markdown_ignore_labels=request.markdownIgnoreLabels,
        )

    def _preprocess(self, input, log_id):
        if input.fileType is None:
            if utils.is_url(input.file):
                maybe_file_type = utils.infer_file_type(input.file)
                if maybe_file_type is None or not (
                    maybe_file_type == "PDF" or maybe_file_type == "IMAGE"
                ):
                    return protocol.create_aistudio_output_without_result(
                        422,
                        "Unsupported file type",
                        log_id=log_id,
                    )
                file_type = maybe_file_type
            else:
                return protocol.create_aistudio_output_without_result(
                    422,
                    "File type cannot be determined",
                    log_id=log_id,
                )
        else:
            file_type = "PDF" if input.fileType == 0 else "IMAGE"
        visualize_enabled = (
            input.visualize
            if input.visualize is not None
            else self.app_config.visualize
        )

        try:
            file_bytes = utils.get_raw_bytes(input.file)
            images, data_info = utils.file_to_images(
                file_bytes,
                file_type,
                max_num_imgs=self.context["max_num_input_imgs"],
            )
        except utils.ImageTooLargeError as e:
            logging.error("Input image or document page exceeds pixel limit: %s", e)
            return protocol.create_aistudio_output_without_result(
                422,
                str(e),
                log_id=log_id,
            )
        except Exception as e:
            logging.error("Failed to get input file bytes: %s", e)
            return protocol.create_aistudio_output_without_result(
                422,
                "Input file is invalid",
                log_id=log_id,
            )

        return images, data_info, visualize_enabled

    def _postprocess(self, images, data_info, visualize_enabled, preds, log_id, input):
        layout_parsing_results: List[Dict[str, Any]] = []
        for i, (img, item) in enumerate(zip(images, preds)):
            pruned_res = app_common.prune_result(item.json["res"])
            # XXX
            md_data = item._to_markdown(
                pretty=input.prettifyMarkdown,
                show_formula_number=input.showFormulaNumber,
            )
            md_text = md_data["markdown_texts"]
            if input.returnMarkdownImages:
                md_imgs = app_common.postprocess_images(
                    md_data["markdown_images"],
                    log_id,
                    filename_template=f"markdown_{i}/{{key}}",
                    file_storage=self.context["file_storage"],
                    return_urls=self.context["return_urls"],
                    url_expires_in=self.context["url_expires_in"],
                    max_img_size=self.context["max_output_img_size"],
                )
            else:
                md_imgs = None
            md_flags = md_data["page_continuation_flags"]
            if visualize_enabled:
                imgs = {
                    "input_img": img,
                    **item.img,
                }
                imgs = app_common.postprocess_images(
                    imgs,
                    log_id,
                    filename_template=f"{{key}}_{i}.jpg",
                    file_storage=self.context["file_storage"],
                    return_urls=self.context["return_urls"],
                    url_expires_in=self.context["url_expires_in"],
                    max_img_size=self.context["max_output_img_size"],
                )
            else:
                imgs = {}
            entry = dict(
                prunedResult=pruned_res,
                markdown=dict(
                    text=md_text,
                    images=md_imgs,
                    isStart=md_flags[0],
                    isEnd=md_flags[1],
                ),
                outputImages=(
                    {k: v for k, v in imgs.items() if k != "input_img"}
                    if imgs
                    else None
                ),
                inputImage=imgs.get("input_img"),
            )
            if app_common.normalize_output_formats(input.outputFormats):
                entry["exports"] = app_common.build_pipeline_exports(
                    input.outputFormats,
                    item,
                    log_id=log_id,
                    file_storage=self.context["file_storage"],
                    return_urls=self.context["return_urls"],
                    url_expires_in=self.context["url_expires_in"],
                )
            layout_parsing_results.append(entry)

        return schemas.pp_structurev3.InferResult(
            layoutParsingResults=layout_parsing_results,
            dataInfo=data_info,
        )
