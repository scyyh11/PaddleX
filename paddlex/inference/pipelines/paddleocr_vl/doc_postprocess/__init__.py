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

"""Document-level post-processing for multi-page PaddleOCR-VL results."""

from .config import DocPostprocessConfig
from .models import ENV_VAR, ModelBundleError, find_models_dir
from .pipeline import DocPostprocessor
from .result import DocPostprocessResult

__all__ = [
    "DocPostprocessConfig",
    "DocPostprocessor",
    "DocPostprocessResult",
    "ModelBundleError",
    "ENV_VAR",
    "find_models_dir",
]
