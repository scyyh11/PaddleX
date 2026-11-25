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

from paddlex.inference.serving.app_shared.document_export import (
    InvalidMarkdownImageError,
)
from paddlex.inference.serving.app_shared.ocr_input import create_ocr_context
from paddlex.inference.serving.app_shared.paddleocr_vl import (
    build_restructure_pages_result,
)
from paddlex_hps_server import BaseTritonPythonModel, protocol, schemas


class TritonPythonModel(BaseTritonPythonModel):
    @property
    def pipeline_creation_kwargs(self):
        return {"initial_predictor": False}

    def initialize(self, args):
        super().initialize(args)
        self.context = create_ocr_context(self.app_config)

    def get_input_model_type(self):
        return schemas.paddleocr_vl.RestructurePagesRequest

    def get_result_model_type(self):
        return schemas.paddleocr_vl.RestructurePagesResult

    def run(self, input, log_id):
        try:
            return build_restructure_pages_result(
                self.pipeline,
                input,
                log_id=log_id,
                file_storage=self.context["file_storage"],
                return_urls=self.context["return_urls"],
                url_expires_in=self.context["url_expires_in"],
            )
        except InvalidMarkdownImageError as exc:
            return protocol.create_aistudio_output_without_result(
                422, str(exc), log_id=log_id
            )
