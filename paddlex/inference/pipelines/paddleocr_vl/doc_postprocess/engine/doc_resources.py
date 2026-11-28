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

"""Resolve model family directories and fold routes from the active ModelStore."""

import pathlib

from . import runtime_context

FAMILIES = frozenset(
    {
        "headings",
        "parents",
        "spans",
        "graph",
        "tables",
        "paragraph-runtime",
        "graphic",
        "paragraph-pairs",
        "text",
        "structure",
        "joint-spans",
    }
)


def _model_exists(directory, stem):
    d = pathlib.Path(directory)
    return (
        (d / f"{stem}.safetensors").is_file() and (d / f"{stem}.json").is_file()
    ) or (d / f"{stem}.joblib").is_file()


def path(family):
    """Directory of a model family inside the active ``ModelStore``."""
    store = runtime_context.model_store.get()
    if store is None:
        raise RuntimeError(
            f"Cannot resolve model family {family!r}: no active ModelStore in context."
        )
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family: {family!r}")
    return pathlib.Path(store.root) / family


def heading_route():
    """Fold selection across all four heading model families."""
    final_stems = [
        (path("headings"), "fold-final"),
        (path("parents"), "parent-fold-final"),
        (path("spans"), "span-fold-final"),
        (path("graph"), "graph-fold-final"),
    ]
    if all(_model_exists(d, stem) for d, stem in final_stems):
        return {"fold": "final", "fit": "final_model"}
    return {
        "fold": runtime_context.default_fold.get(),
        "fit": "default_fold",
    }


def fold(root):
    """Return ``'final'`` when present in ``root``, otherwise the default fold."""
    return (
        "final"
        if _model_exists(root, "fold-final")
        else runtime_context.default_fold.get()
    )
