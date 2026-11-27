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

DIRECTORIES = {
    "/models": "headings",
    "/models-parent": "parents",
    "/models-span": "spans",
    "/models-graph": "graph",
    "/table-models": "tables",
    "/runtime-paragraph-models": "paragraph-runtime",
    "/graphic-models": "graphic",
    "/paragraph-models": "paragraph-pairs",
    "/models-text": "text",
    "/models-structure": "structure",
    "/models-joint-span": "joint-spans",
}


def _model_exists(directory, stem):
    d = pathlib.Path(directory)
    return (
        (d / f"{stem}.safetensors").is_file() and (d / f"{stem}.json").is_file()
    ) or (d / f"{stem}.joblib").is_file()


def path(mount):
    """Directory of a model family inside the active ``ModelStore``."""
    mount = str(mount)
    store = runtime_context.model_store.get()
    if store is None or mount not in DIRECTORIES:
        return pathlib.Path(mount)
    return pathlib.Path(store.root) / DIRECTORIES[mount]


def heading_route():
    """Route of a document without an out-of-fold record."""
    final_stems = [
        (path("/models"), "fold-final"),
        (path("/models-parent"), "parent-fold-final"),
        (path("/models-span"), "span-fold-final"),
        (path("/models-graph"), "graph-fold-final"),
    ]
    if all(_model_exists(d, stem) for d, stem in final_stems):
        return {"fold": "final", "fit": "final_model"}
    return {
        "fold": runtime_context.default_fold.get(),
        "fit": "default_fold_for_new_document",
    }


def fold(folds, doc, root):
    """Manifest fold of a development document; ``final`` or default fold otherwise."""
    if doc in folds:
        return folds[doc]
    return (
        "final"
        if _model_exists(root, "fold-final")
        else runtime_context.default_fold.get()
    )
