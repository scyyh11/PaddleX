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


def _freeze(value):
    if isinstance(value, dict):
        return frozenset((_freeze(k), _freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return type(value), tuple(_freeze(item) for item in value)
    return type(value), value


def group_request_indices(inference_kwargs):
    """Group requests with the same effective arguments in first-seen order."""
    groups = {}
    for index, kwargs in enumerate(inference_kwargs):
        groups.setdefault(_freeze(kwargs), []).append(index)
    return list(groups.values())
