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

from typing import Optional

from ..infra.storage import Storage, SupportsGetURL
from ..infra.utils import base64_encode


def store_output_bytes(
    data: bytes,
    log_id: str,
    filename: str,
    *,
    file_storage: Optional[Storage] = None,
    return_url: bool = False,
    url_expires_in: int = -1,
) -> str:
    if return_url:
        if not file_storage:
            raise ValueError(
                "`file_storage` must not be None when URLs need to be returned."
            )
        if not isinstance(file_storage, SupportsGetURL):
            raise TypeError("The provided storage does not support getting URLs.")

    key = f"{log_id}/{filename}"
    if file_storage is not None:
        file_storage.set(key, data)
        if return_url:
            assert isinstance(file_storage, SupportsGetURL)
            return file_storage.get_url(key, expires_in=url_expires_in)
    return base64_encode(data)
