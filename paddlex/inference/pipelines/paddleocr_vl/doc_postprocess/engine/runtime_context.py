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

"""Request-local immutable analyses and instance-scoped model loading."""

from contextvars import ContextVar
from functools import wraps

from ..config import DocPostprocessConfig
from .fast_html import parse

raw_pages = ContextVar("raw_pages")
html_cache = ContextVar("html_cache")
backend = ContextVar("backend", default="cpu")
input_pages = ContextVar("input_pages", default=None)
analysis_cache = ContextVar("analysis_cache")
config = ContextVar("doc_postprocess_config", default=DocPostprocessConfig())
model_store = ContextVar("doc_postprocess_model_store", default=None)
default_fold = ContextVar("doc_postprocess_default_fold", default=0)
table_features_v2 = ContextVar("table_features_v2", default=False)


def get_config():
    return config.get()


def document_pages():
    return raw_pages.get()


def readonly_soup(text, parser):
    cache = html_cache.get(None)
    key = (text, parser)
    if cache is None:
        return parse(text)
    if key not in cache:
        cache[key] = parse(text)
    return cache[key]


def load_model(path):
    store = model_store.get()
    if store is None:
        raise RuntimeError("Document post-processing model store is not initialized.")
    return store.load(path)


def memoize(function):
    @wraps(function)
    def wrapped(*args):
        cache = analysis_cache.get(None)
        if cache is None:
            return function(*args)
        key = (function.__module__, function.__name__, args)
        if key not in cache:
            cache[key] = function(*args)
        return cache[key]

    return wrapped
