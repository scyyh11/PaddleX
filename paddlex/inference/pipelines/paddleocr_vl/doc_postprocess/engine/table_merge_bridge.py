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

"""Bridge between the engine HTML cache and PaddleX table merging."""

from ....layout_parsing.merge_table import detect_table_headers
from .fast_html import append_rows
from .runtime_context import html_cache, readonly_soup


def readonly_soup_factory(html):
    """Parse table HTML through the request-local readonly soup cache."""
    return readonly_soup(html, "html.parser")


def merge_to_html(previous, current, skip):
    """Append non-header rows of ``current`` onto ``previous`` and cache the result."""
    result = append_rows(previous, current, skip)
    text = str(result)
    cache = html_cache.get(None)
    if cache is not None:
        cache[(text, "html.parser")] = result
    return text


def cached_table_merger(previous, current):
    """Drop-in ``table_merger`` callback for ``merge_tables_across_pages``."""
    skip, _ = detect_table_headers(previous, current)
    return merge_to_html(previous, current, skip)
