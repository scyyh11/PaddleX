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

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL.Image import Image

from ....utils.deps import function_requires_deps, is_dep_available
from ..infra import utils as serving_utils
from ..infra.storage import Storage
from .output_payload import store_output_bytes

if is_dep_available("opencv-contrib-python"):
    import cv2


def prune_result(result: dict) -> dict:
    KEYS_TO_REMOVE = ["input_path", "page_index"]

    def _process_obj(obj):
        if isinstance(obj, dict):
            return {
                k: _process_obj(v) for k, v in obj.items() if k not in KEYS_TO_REMOVE
            }
        elif isinstance(obj, list):
            return [_process_obj(item) for item in obj]
        else:
            return obj

    return _process_obj(result)


@function_requires_deps("opencv-contrib-python")
def postprocess_image(
    image: np.ndarray,
    log_id: str,
    filename: str,
    *,
    file_storage: Optional[Storage] = None,
    return_url: bool = False,
    url_expires_in: int = -1,
    max_img_size: Optional[Tuple[int, int]] = None,
) -> str:
    ext = os.path.splitext(filename)[1]
    h, w = image.shape[:2]
    if max_img_size is not None:
        max_height, max_width = max_img_size
        if h > max_height or w > max_width:
            scale = min(max_height / h, max_width / w)
            image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    return store_output_bytes(
        serving_utils.image_array_to_bytes(image, ext=ext),
        log_id,
        filename,
        file_storage=file_storage,
        return_url=return_url,
        url_expires_in=url_expires_in,
    )


def postprocess_images(
    images: Dict[str, Union[Image, np.ndarray]],
    log_id: str,
    filename_template: str = "{key}.jpg",
    file_storage: Optional[Storage] = None,
    return_urls: bool = False,
    url_expires_in: int = -1,
    max_img_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, str]:
    output_images: Dict[str, str] = {}
    for key, img in images.items():
        output_images[key] = postprocess_image(
            (
                cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
                if isinstance(img, Image)
                else img
            ),
            log_id=log_id,
            filename=filename_template.format(key=key),
            file_storage=file_storage,
            return_url=return_urls,
            url_expires_in=url_expires_in,
            max_img_size=max_img_size,
        )
    return output_images
