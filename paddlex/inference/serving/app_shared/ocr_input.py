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

import warnings
from typing import Final, Tuple

from typing_extensions import Literal

from ..infra import utils as serving_utils
from ..infra.config import AppConfig
from ..infra.storage import SupportsGetURL, create_storage
from ..schemas.shared.ocr import BaseInferRequest


class OCRInputError(ValueError):
    """The request file could not be accepted or decoded."""


DEFAULT_MAX_NUM_INPUT_IMGS: Final[int] = 10
DEFAULT_MAX_OUTPUT_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)
DEFAULT_URL_EXPIRES_IN: Final[int] = -1


def create_ocr_context(config: AppConfig) -> dict:
    extra_cfg = config.extra or {}
    context = {}
    context["file_storage"] = None
    if "file_storage" in extra_cfg:
        context["file_storage"] = create_storage(extra_cfg["file_storage"])
    return_urls = config.return_urls
    if "return_img_urls" in extra_cfg:
        warnings.warn(
            "`Serving.extra.return_img_urls` is deprecated; use the top-level "
            "`Serving.return_urls` field instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if return_urls is None:
            return_urls = bool(extra_cfg["return_img_urls"])
    if return_urls is None:
        return_urls = False
    context["return_urls"] = return_urls
    if context["return_urls"]:
        file_storage = context["file_storage"]
        if not file_storage:
            raise ValueError(
                "The file storage must be properly configured when URLs need to be returned."
            )
        if not isinstance(file_storage, SupportsGetURL):
            raise TypeError(
                f"`{type(file_storage).__name__}` does not support getting URLs."
            )
    context["url_expires_in"] = extra_cfg.get("url_expires_in", DEFAULT_URL_EXPIRES_IN)
    context["max_num_input_imgs"] = extra_cfg.get(
        "max_num_input_imgs", DEFAULT_MAX_NUM_INPUT_IMGS
    )
    context["max_output_img_size"] = extra_cfg.get(
        "max_output_img_size", DEFAULT_MAX_OUTPUT_IMG_SIZE
    )
    return context


def get_file_type(request: BaseInferRequest) -> Literal["PDF", "IMAGE"]:
    if request.fileType is not None:
        return "PDF" if request.fileType == 0 else "IMAGE"
    if not serving_utils.is_url(request.file):
        raise OCRInputError("File type cannot be determined")
    file_type = serving_utils.infer_file_type(request.file)
    if file_type not in ("PDF", "IMAGE"):
        raise OCRInputError("Unsupported file type")
    return file_type


def decode_ocr_images(file_bytes, file_type, max_num_input_imgs):
    try:
        return serving_utils.file_to_images(
            file_bytes, file_type, max_num_imgs=max_num_input_imgs
        )
    except serving_utils.ImageTooLargeError as exc:
        raise OCRInputError(str(exc)) from exc
    except Exception as exc:
        raise OCRInputError("Invalid input file") from exc
