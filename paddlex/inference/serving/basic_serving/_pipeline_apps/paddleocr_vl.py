# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any

from .....utils.deps import function_requires_deps, is_dep_available
from ...app_shared.paddleocr_vl import (
    build_infer_result,
    build_restructure_pages_result,
    get_inference_kwargs,
)
from ...infra import utils as serving_utils
from ...infra.config import AppConfig
from ...infra.models import AIStudioResultResponse
from ...schemas.paddleocr_vl import (
    INFER_ENDPOINT,
    RESTRUCTURE_PAGES_ENDPOINT,
    InferRequest,
    InferResult,
    RestructurePagesRequest,
    RestructurePagesResult,
)
from .._app import create_app, primary_operation
from ._common import ocr as ocr_common

if is_dep_available("fastapi"):
    from fastapi import FastAPI


@function_requires_deps("fastapi")
def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> "FastAPI":
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @primary_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(
        request: InferRequest,
    ) -> AIStudioResultResponse[InferResult]:
        pipeline = ctx.pipeline

        log_id = request.logId if request.logId else serving_utils.generate_log_id()
        visualize_enabled = (
            request.visualize if request.visualize is not None else ctx.config.visualize
        )
        images, data_info = await ocr_common.get_images(request, ctx)

        preds = await pipeline.infer(images, **get_inference_kwargs(request))
        result = await serving_utils.call_async(
            build_infer_result,
            pipeline.pipeline,
            request,
            images,
            data_info,
            preds,
            log_id=log_id,
            visualize=visualize_enabled,
            file_storage=ctx.extra["file_storage"],
            return_urls=ctx.extra["return_urls"],
            url_expires_in=ctx.extra["url_expires_in"],
            max_output_img_size=ctx.extra["max_output_img_size"],
        )
        return AIStudioResultResponse[InferResult](logId=log_id, result=result)

    @primary_operation(
        app,
        RESTRUCTURE_PAGES_ENDPOINT,
        "restructurePages",
    )
    async def _restructure_pages(
        request: RestructurePagesRequest,
    ) -> AIStudioResultResponse[RestructurePagesResult]:
        log_id = request.logId if request.logId else serving_utils.generate_log_id()
        result = await serving_utils.call_async(
            build_restructure_pages_result,
            ctx.pipeline.pipeline,
            request,
            log_id=log_id,
            file_storage=ctx.extra["file_storage"],
            return_urls=ctx.extra["return_urls"],
            url_expires_in=ctx.extra["url_expires_in"],
        )
        return AIStudioResultResponse[RestructurePagesResult](
            logId=log_id, result=result
        )

    return app
