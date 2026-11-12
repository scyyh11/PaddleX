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

from typing import Optional

from paddlex.inference.serving.app_shared.image_output import (
    postprocess_image,
    postprocess_images,
    prune_result,
)

from .storage import Storage

__all__ = [
    "postprocess_image",
    "postprocess_images",
    "prune_result",
    "normalize_output_formats",
    "build_pipeline_exports",
    "refill_paddleocr_vl_images_from_markdown",
]


def normalize_output_formats(output_formats):
    """Which extra formats the client requested (e.g. docx). Delegates to serving schemas."""
    from paddlex.inference.serving.schemas.shared.export import (
        normalize_output_formats as _normalize_output_formats,
    )

    return _normalize_output_formats(output_formats)


def build_pipeline_exports(
    output_formats,
    result_obj,
    *,
    log_id: str,
    file_storage: Optional[Storage] = None,
    return_urls: bool = False,
    url_expires_in: int = -1,
):
    """Build optional `exports` payload for validated `outputFormats`."""
    from paddlex.inference.serving.app_shared.document_export import (
        build_pipeline_exports as _build_pipeline_exports,
    )

    return _build_pipeline_exports(
        output_formats,
        result_obj,
        log_id=log_id,
        file_storage=file_storage,
        return_urls=return_urls,
        url_expires_in=url_expires_in,
    )


def refill_paddleocr_vl_images_from_markdown(result, markdown_images):
    """Restore PIL images from markdown image map before DOCX export (PaddleOCR-VL restructure)."""
    from paddlex.inference.serving.app_shared.document_export import (
        refill_paddleocr_vl_images_from_markdown as _refill,
    )

    return _refill(result, markdown_images)
