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

"""Assemble independent 1.6 corpora, with either existing or subsequently captured target features."""
import argparse
from collections import Counter
import json
from pathlib import Path

from common import digest, load_rows, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--before-capture", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Select a fresh output")
    staging = args.output.with_name(args.output.name + ".staging")
    folder = staging / "features"
    folder.mkdir(parents=True, exist_ok=False)
    splits = (
        ["manifest.jsonl", "dev_test.jsonl"]
        if args.before_capture
        else ["manifest.jsonl"]
    )
    ids, images, docs, sources, inputs = set(), set(), {}, set(), []
    merged = {name: [] for name in splits}
    for directory in args.input:
        selection_path = directory / "selection.json"
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text())
        else:
            provenance = json.loads((directory / "provenance.json").read_text())
            selection = {**provenance, "sources": [provenance["source"]]}
        for name in splits:
            path = directory / name
            if name == "dev_test.jsonl" and not path.is_file():
                continue
            if (
                name == "manifest.jsonl"
                and digest(path) != selection["manifest_sha256"]
                or name == "dev_test.jsonl"
                and "dev_test_sha256" in selection
                and digest(path) != selection["dev_test_sha256"]
            ):
                raise ValueError(f"Modified selection {path}")
            selected = load_rows(path)
            inputs.append(dict(path=str(path.resolve()), sha256=digest(path)))
            for r in selected:
                sid, image_hash, split = r["id"], r["image_sha256"], r["split"]
                allowed = (
                    ("test",) if name == "dev_test.jsonl" else ("train", "validation")
                )
                if r["source"] not in selection["sources"] or split not in allowed:
                    raise ValueError(f"Unexpected source or split: {sid}")
                if (
                    sid in ids
                    or image_hash in images
                    or docs.setdefault(r["document_id"], split) != split
                ):
                    raise ValueError(
                        f"Repeated sample, image hash, or overlapping document: {sid}"
                    )
                if not args.before_capture:
                    f = directory / "features" / (sid + ".pt")
                    if not f.is_file():
                        raise ValueError(f"Missing teacher states: {f}")
                    (folder / f.name).symlink_to(f.resolve())
                sources.add(r["source"])
                ids.add(sid)
                images.add(image_hash)
                merged[name].append(r)
    for name, rows in merged.items():
        (staging / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    selection = dict(
        sources=sorted(sources),
        manifest_sha256=digest(staging / "manifest.jsonl"),
        inputs=inputs,
        counts=dict(
            Counter(
                f"{r['split']}/{r['task']}" for rows in merged.values() for r in rows
            )
        ),
        document_splits_exclusive=True,
        duplicate_image_hashes=0,
    )
    if args.before_capture:
        selection["dev_test_sha256"] = digest(staging / "dev_test.jsonl")
    write_json(staging / "selection.json", selection)
    staging.rename(args.output)


if __name__ == "__main__":
    main()
