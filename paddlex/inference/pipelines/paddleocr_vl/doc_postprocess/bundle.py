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

"""Assemble, validate, or export a portable document post-processing model bundle.

    python -m paddlex.inference.pipelines.paddleocr_vl.doc_postprocess.bundle check [DIR]
    python -m paddlex.inference.pipelines.paddleocr_vl.doc_postprocess.bundle export-portable SRC DST
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

from .models import FAMILIES, find_models_dir, problems, read_bundle
from .portable_trees import save_portable_package


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _place(source, target, link):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if link:
        os.symlink(os.path.abspath(source), target)
    else:
        shutil.copyfile(source, target)


def build(out, sources, folds="default", default_fold=0, link=False):
    out = pathlib.Path(out)
    files = {}
    for family, (stem_pattern, required) in FAMILIES.items():
        pattern = f"{stem_pattern}.joblib"
        source = pathlib.Path(sources[family])
        names = list(required)
        if folds == "all":
            prefix, suffix = pattern.split("{}")
            names += sorted(p.name for p in source.glob(f"{prefix}*{suffix}"))
        else:
            for fold in ("final", default_fold):
                if (source / pattern.format(fold)).is_file():
                    names.append(pattern.format(fold))
        for name in dict.fromkeys(names):
            _place(source / name, out / family / name, link)
            files[f"{family}/{name}"] = _sha256(source / name)
    settings = {"default_fold": default_fold, "folds": folds, "files": files}
    (out / "bundle.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    return out


def export_portable(src, dst, default_fold=None):
    """Convert a ``.joblib`` bundle directory into a pickle-free ``.safetensors`` bundle."""
    import joblib

    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    src_settings = read_bundle(src)
    if default_fold is None:
        default_fold = src_settings.get("default_fold", 0)

    files = {}
    for family, (stem_pattern, required) in FAMILIES.items():
        src_dir = src / family
        dst_dir = dst / family
        dst_dir.mkdir(parents=True, exist_ok=True)

        for req in required:
            src_req = src_dir / req
            if src_req.is_file():
                if req == "manifest.json":
                    manifest = json.loads(src_req.read_text(encoding="utf-8"))
                    manifest["folds"] = {}
                    (dst_dir / req).write_text(
                        json.dumps(manifest, indent=1), encoding="utf-8"
                    )
                else:
                    shutil.copyfile(src_req, dst_dir / req)
                files[f"{family}/{req}"] = _sha256(dst_dir / req)

        for fold in ("final", default_fold):
            stem = stem_pattern.format(fold)
            jb = src_dir / f"{stem}.joblib"
            if jb.is_file():
                package = joblib.load(jb)
                sf_path, js_path = save_portable_package(package, dst_dir / stem)
                files[f"{family}/{sf_path.name}"] = _sha256(sf_path)
                files[f"{family}/{js_path.name}"] = _sha256(js_path)

    settings = {
        "default_fold": default_fold,
        "folds": "default",
        "format": "safetensors",
        "files": files,
    }
    (dst / "bundle.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    return dst


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m paddlex.inference.pipelines.paddleocr_vl.doc_postprocess.bundle"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("build", help="assemble a bundle from model directories")
    make.add_argument("out")
    for family in FAMILIES:
        make.add_argument(f"--{family}", required=True, metavar="DIR")
    make.add_argument("--folds", choices=["default", "all"], default="default")
    make.add_argument("--default-fold", type=int, default=0)
    make.add_argument("--link", action="store_true", help="symlink instead of copying")

    verify = commands.add_parser("check", help="check that a bundle is complete")
    verify.add_argument("dir", nargs="?")

    export = commands.add_parser(
        "export-portable",
        help="export a .joblib bundle into a pickle-free .safetensors bundle",
    )
    export.add_argument("src")
    export.add_argument("dst")
    export.add_argument("--default-fold", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "build":
        sources = {
            family: getattr(args, family.replace("-", "_")) for family in FAMILIES
        }
        out = build(args.out, sources, args.folds, args.default_fold, args.link)
        missing = problems(out)
    elif args.command == "export-portable":
        out = export_portable(args.src, args.dst, default_fold=args.default_fold)
        missing = problems(out)
    else:
        out = find_models_dir(args.dir)
        missing = problems(out)

    for message in missing:
        print("missing:", message, file=sys.stderr)
    print(f"{out}: {'incomplete' if missing else 'ok'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
