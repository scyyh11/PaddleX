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

"""Document-level result of the multi-page post-processing."""

import json
import pathlib

import numpy as np
from PIL import Image


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class DocPostprocessResult(dict):
    """Markdown and structured pages of one multi-page document.

    Keys: ``doc_name``, ``input_path``, ``page_count``, ``markdown_text``,
    ``pages``, ``images``, ``events``, and ``timings``.
    """

    @property
    def markdown(self):
        """Markdown in the PaddleX result format (``markdown_texts``, ``markdown_images``)."""
        images = {
            path: Image.fromarray(np.ascontiguousarray(crop[:, :, ::-1]))
            for path, crop in self.get("images", {}).items()
        }
        return {
            "markdown_texts": self["markdown_text"],
            "markdown_images": images,
            "page_index": None,
            "input_path": self.get("input_path"),
        }

    @property
    def json(self):
        return {
            "res": {
                "doc_name": self["doc_name"],
                "input_path": self.get("input_path"),
                "page_count": self["page_count"],
                "pages": self["pages"],
                "timings": self["timings"],
            }
        }

    def _target(self, save_path, suffix):
        path = pathlib.Path(save_path)
        if path.suffix.lower() == suffix:
            return path
        return path / f"{self['doc_name']}{suffix}"

    def save_to_markdown(self, save_path):
        """Write ``<doc_name>.md`` (or ``save_path`` itself when it ends with .md) and its images."""
        import cv2

        target = self._target(save_path, ".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self["markdown_text"], encoding="utf-8")
        for relative, crop in self.get("images", {}).items():
            path = target.parent / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), crop)
        return target

    def save_to_json(self, save_path):
        """Write the structured pages and timings to ``<doc_name>.json``."""
        target = self._target(save_path, ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.json, ensure_ascii=False, indent=1, default=_jsonable),
            encoding="utf-8",
        )
        return target

    def save_all(self, save_path):
        self.save_to_json(save_path)
        self.save_to_markdown(save_path)

    def print(self):
        timings = ", ".join(f"{k}={v:.1f}" for k, v in self["timings"].items())
        print(
            f"{self['doc_name']}: {self['page_count']} pages, "
            f"{len(self['markdown_text'])} Markdown characters ({timings} ms)"
        )
