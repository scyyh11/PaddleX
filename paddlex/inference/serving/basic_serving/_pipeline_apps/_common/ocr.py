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

from typing import List, Tuple, Union

import numpy as np

from ......utils import logging
from ......utils.deps import is_dep_available
from ....app_shared.ocr_input import (
    OCRInputError,
    create_ocr_context,
    decode_ocr_images,
    get_file_type,
)
from ....infra import utils as serving_utils
from ....infra.models import ImageInfo, PDFInfo, TIFFInfo
from ....schemas.shared.ocr import BaseInferRequest
from ..._app import AppContext

if is_dep_available("fastapi"):
    from fastapi import HTTPException


def update_app_context(app_context: AppContext) -> None:
    app_context.extra.update(create_ocr_context(app_context.config))


async def get_images(
    request: BaseInferRequest, app_context: AppContext
) -> Tuple[List[np.ndarray], Union[ImageInfo, PDFInfo, TIFFInfo]]:
    try:
        file_type = get_file_type(request)
        file_bytes = await serving_utils.get_raw_bytes_async(
            request.file,
            app_context.aiohttp_session,
        )
        images, data_info = await serving_utils.call_async(
            decode_ocr_images,
            file_bytes,
            file_type,
            app_context.extra["max_num_input_imgs"],
        )
    except OCRInputError as exc:
        logging.error("Failed to read input file: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as e:
        logging.error("Failed to read input file: %s", e)
        raise HTTPException(status_code=422, detail="Invalid input file") from e
    return images, data_info
