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

"""CPU-only bitwise parity and paired complete image-processor timing."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("Threads and repeats must be positive")
    os.environ.update(
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
        OMP_WAIT_POLICY="PASSIVE",
    )
    import numpy as np
    from PIL import Image
    import PIL
    import torch
    import transformers
    from transformers import AutoProcessor
    from vllm019_image_lut import make_exact_pil_preprocess

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    ).image_processor
    original = processor.preprocess
    candidate = make_exact_pil_preprocess(processor)
    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    cases = []
    for row in rows:
        with Image.open(row["image"]) as image:
            cases.append((row["task"], image.convert("RGB")))
    rng = np.random.default_rng(17)
    sizes = [(28, 28), (27, 29), (56, 84), (337, 337), (1024, 1536), (2048, 80)]
    synthetic = [
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        for w, h in sizes
    ]
    codes = np.arange(28 * 28 * 3, dtype=np.uint16).reshape(28, 28, 3).astype(np.uint8)
    synthetic.append(Image.fromarray(codes))
    checks = []

    def array(value):
        return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    def compare(label, image, options):
        outcomes = []
        notices = []
        for operation in (original, candidate):
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                try:
                    result = operation(images=image, **options)
                    outcomes.append((result, None))
                except Exception as exc:
                    outcomes.append((None, type(exc).__name__))
            notices.append([(w.category.__name__, str(w.message)) for w in emitted])
        a, b = outcomes
        same = a[1] == b[1] and notices[0] == notices[1]
        if a[1] is None and b[1] is None:
            same &= a[0].keys() == b[0].keys()
            for name in a[0]:
                left, right = array(a[0][name]), array(b[0][name])
                same &= left.dtype == right.dtype and left.shape == right.shape
                same &= left.tobytes() == right.tobytes()
        checks.append(
            dict(
                case=label,
                bitwise_equal=bool(same),
                native_exception=a[1],
                warnings_equal=notices[0] == notices[1],
            )
        )
        if not same:
            raise AssertionError(f"Exact preprocessing parity failed: {label}")

    for index, (_, image) in enumerate(cases):
        compare(
            f"manifest_{index}",
            image,
            {"return_tensors": "pt", "input_data_format": "channels_last"},
        )
    for index, image in enumerate(synthetic):
        compare(f"synthetic_{index}", image, {"return_tensors": "pt"})
    edge_sizes = [
        (1, 1),
        (3, 3),
        (1, 3),
        (3, 1),
        (28, 1),
        (28, 3),
        (1, 28),
        (3, 28),
        (56, 3),
        (3, 56),
    ]
    for index, (width, height) in enumerate(edge_sizes):
        edge = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        for variant, extra in enumerate(
            ({}, {"input_data_format": "channels_last"}, {"do_resize": False})
        ):
            compare(
                f"small_pil_{index}_{variant}", edge, {"return_tensors": "pt", **extra}
            )
    image = synthetic[2]
    options = [
        {},
        {"return_tensors": "np"},
        {"return_tensors": "pt"},
        {"do_resize": False},
        {"do_rescale": False},
        {"do_normalize": False},
        {"do_rescale": False, "do_normalize": False},
        {"data_format": "channels_last"},
        {"do_convert_rgb": False},
        {"image_mean": [0.1, 0.2, 0.3], "image_std": [0.7, 0.6, 0.5]},
        {"image_mean": 0.25, "image_std": 0.75},
        {"rescale_factor": 1 / 127.5},
        {"resample": Image.Resampling.NEAREST},
        {"resample": Image.Resampling.BILINEAR},
        {"resample": Image.Resampling.LANCZOS},
        {"image_mean": [0, 0], "image_std": [1, 1]},
        {"data_format": None},
        {"input_data_format": "channels_first"},
        {"unknown_pixel_option": 123},
        {"rescale_factor": 0.0, "do_normalize": False},
        {"rescale_factor": -0.0, "do_normalize": False},
        {"rescale_factor": 0.0, "image_mean": 0.0, "image_std": [1.0, -1.0, 1.0]},
        {"rescale_factor": -0.0, "image_mean": [0.0, -0.0, 0.0], "image_std": 1.0},
    ]
    for index, option in enumerate(options):
        compare(f"options_{index}", image, option)
    fallback_inputs = [
        image.convert("L"),
        image.convert("RGBA"),
        np.asarray(image),
        np.asarray(image).astype(np.float32) / 255,
        [image, image],
        [image],
        [[image]],
        [[[image]]],
    ]
    for index, value in enumerate(fallback_inputs):
        compare(f"input_contract_{index}", value, {"return_tensors": "pt"})
    compare("video_fallback", None, {"videos": [image], "return_tensors": "pt"})
    retained = candidate(image, return_tensors="pt")
    retained_bytes = array(retained["pixel_values"]).tobytes()
    candidate(synthetic[0], return_tensors="pt")
    retained_equal = retained_bytes == array(retained["pixel_values"]).tobytes()
    assert retained_equal

    timings = defaultdict(lambda: defaultdict(list))
    # Both functions warmed before randomized pairs; image decode/imports excluded.
    for _, image in cases:
        original(image, return_tensors="pt", input_data_format="channels_last")
        candidate(image, return_tensors="pt", input_data_format="channels_last")
    shuffle = random.Random(19)
    for _ in range(args.repeats):
        order = list(range(len(cases)))
        shuffle.shuffle(order)
        for index in order:
            _, image = cases[index]
            methods = [("native", original), ("lut", candidate)]
            shuffle.shuffle(methods)
            for name, operation in methods:
                started = time.perf_counter()
                result = operation(
                    image, return_tensors="pt", input_data_format="channels_last"
                )
                timings[index][name].append((time.perf_counter() - started) * 1000)
                del result
    groups = {}
    for task in ("all", *sorted({task for task, _ in cases})):
        selected = [
            i for i, (kind, _) in enumerate(cases) if task == "all" or task == kind
        ]
        totals = {
            name: sum(statistics.median(timings[i][name]) for i in selected)
            for name in ("native", "lut")
        }
        groups[task] = dict(
            images=len(selected),
            sum_image_medians_ms=totals,
            component_speedup=totals["native"] / totals["lut"],
            saved_ms=totals["native"] - totals["lut"],
        )
    output = dict(
        passed=all(c["bitwise_equal"] for c in checks),
        checks=checks,
        retained_outputs_unchanged=retained_equal,
        counters=candidate.profile_counts,
        groups=groups,
        repeats=args.repeats,
        threads=args.threads,
        versions=dict(
            torch=torch.__version__,
            transformers=transformers.__version__,
            numpy=np.__version__,
            pillow=PIL.__version__,
        ),
        cuda_initialized=torch.cuda.is_initialized(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        processor_source_sha256=hashlib.sha256(
            (args.model / "image_processing_paddleocr_vl.py").read_bytes()
        ).hexdigest(),
        scope="CPU complete image_processor.preprocess including resize, normalization, patch packing and PyTorch tensor conversion. Preloaded PIL images; excludes disk decode, tokenizer/renderer, H2D, vision and decode. Timings are randomized per-image pairs; sum per-image medians. No image-result caching.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": output["passed"],
                "checks": len(checks),
                "cuda_initialized": output["cuda_initialized"],
                "groups": groups,
                "output": str(args.output.resolve()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
