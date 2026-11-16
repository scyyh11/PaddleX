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

"""Generate a pinned 1.6 teacher corpus and collect vLLM hidden states.

Start one "generate" and one "features" process. Use "audit" after both finish.
Each phase has private model, compiler and processor caches.
"""
import argparse
from collections import Counter, defaultdict
import importlib
import json
import os
from pathlib import Path
import time

from common import ROOT as BASE, load_rows as read_rows, digest as sha256

DATA = BASE / "data/v16_clean"

MODEL = BASE / "models/PaddleOCR-VL-1.6"
MAX_MODEL_LEN = 4096
MAX_GENERATION = 512
BATCH = 48
LAYERS = (4, 10, 16)
SOURCES = (
    ("manifest.jsonl", "teacher.jsonl", "features"),
    ("dev_test.jsonl", "dev_test_teacher.jsonl", "dev_test_features"),
)


def environment(phase):
    if (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        or "," in os.environ["CUDA_VISIBLE_DEVICES"]
    ):
        raise RuntimeError("Select exactly one CUDA device with CUDA_VISIBLE_DEVICES")
    module_dir = str(Path(__file__).resolve().parent)
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join([module_dir] + [p for p in paths if p])
    cache = DATA / "runtime" / phase
    for key, name in (
        ("HF_HOME", "hf"),
        ("VLLM_CACHE_ROOT", "vllm"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
        ("XDG_CACHE_HOME", "xdg"),
        ("TORCH_EXTENSIONS_DIR", "extensions"),
    ):
        os.environ[key] = str(cache / name)
    os.environ.update(
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_USE_V2_MODEL_RUNNER="1",
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="3",
        TOKENIZERS_PARALLELISM="false",
    )


def target_metadata():
    meta = dict(
        model=str(MODEL),
        target_revision="c5630abae1d940eafe0697512a0325494b02ab42",
        target_sha256=sha256(MODEL / "model.safetensors"),
        tokenizer_sha256=sha256(MODEL / "tokenizer.json"),
        template_sha256=sha256(MODEL / "chat_template.jinja"),
        processor_config_sha256=sha256(MODEL / "preprocessor_config.json"),
    )
    assert (
        meta["target_sha256"]
        == "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db"
    )
    assert (
        meta["tokenizer_sha256"]
        == "c8a215a59183d0d0781adc33bacd3ce6162716f7fd568fb30234a74d69803a7d"
    )
    assert (
        meta["template_sha256"]
        == "2f27812dab7f333e471884e0c803d807f11953d5453140dfb1aaba234f872bc8"
    )
    assert (
        meta["processor_config_sha256"]
        == "111872ab1e8bb7fd040ac5087bfced7ab8f011f02139b088cba294964c3b1d0e"
    )
    return meta


def setup(phase):
    from transformers import AutoProcessor
    from importlib.metadata import version

    if version("vllm") != "0.32.0":
        raise RuntimeError("Feature capture is validated only on vLLM 0.32.0")
    from vllm import LLM

    processor = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True)
    ip = processor.image_processor
    assert (ip.min_pixels, ip.max_pixels) == (112896, 1003520)
    resize = importlib.import_module(type(ip).__module__).smart_resize
    tokenized = {}
    for task, prompt in [
        ("ocr", "OCR:"),
        ("formula", "Formula Recognition:"),
        ("table", "Table Recognition:"),
    ]:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        template = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert template.endswith(prompt + "\nAssistant:\n")
        tokens = processor.tokenizer.encode(template, add_special_tokens=False)
        assert tokens.count(100295) == 1
        tokenized[task] = (prompt, tokens)

    features = phase == "features"
    options = (
        dict(
            worker_extension_cls="feature_worker.FeatureCaptureExtension",
            enforce_eager=True,
        )
        if features
        else dict(max_cudagraph_capture_size=BATCH)
    )
    engine = LLM(
        model=str(MODEL),
        tokenizer=str(MODEL),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1 if features else BATCH,
        max_num_batched_tokens=4096 if features else 8192,
        kv_cache_memory_bytes=2 * 1024**3,
        gpu_memory_utilization=0.20,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 1},
        seed=0,
        **options,
    )
    if features:
        print(
            engine.collective_rpc("install_feature_capture", args=(LAYERS,)), flush=True
        )

    def make_request(row, generated=()):
        from PIL import Image

        prompt, raw = tokenized[row["task"]]
        assert prompt == row["prompt"]
        with Image.open(row["image"]) as source:
            rgb = source.convert("RGB")
        height, width = resize(
            rgb.height,
            rgb.width,
            factor=28,
            min_pixels=ip.min_pixels,
            max_pixels=ip.max_pixels,
        )
        patches = height * width // (28 * 28)
        i = raw.index(100295)
        expected = raw[:i] + [100295] * patches + raw[i + 1 :] + list(generated)
        return (
            {
                "prompt_token_ids": raw + list(generated),
                "multi_modal_data": {"image": rgb},
            },
            expected,
        )

    return engine, make_request


def generate(engine, prepare, meta):
    from vllm import SamplingParams

    params = SamplingParams(temperature=0, seed=0, max_tokens=MAX_GENERATION)
    for manifest, teacher, _ in SOURCES:
        path = DATA / teacher
        rows = read_rows(DATA / manifest)
        existing = read_rows(path) if path.exists() else []
        assert all(r["target_sha256"] == meta["target_sha256"] for r in existing)
        done = {r["id"] for r in existing}
        assert len(done) == len(existing) and done <= {r["id"] for r in rows}
        pending = [r for r in rows if r["id"] not in done]
        with path.open("a", buffering=1) as stream:
            for i in range(0, len(pending), BATCH):
                batch = pending[i : i + BATCH]
                requests, expected = zip(*(prepare(r) for r in batch))
                outputs = engine.generate(list(requests), params, use_tqdm=False)
                for row, expanded, result in zip(batch, expected, outputs, strict=True):
                    assert list(result.prompt_token_ids) == expanded, row["id"]
                    out = result.outputs[0]
                    record = {k: row[k] for k in ("id", "split", "document_id", "task")}
                    record.update(
                        **meta,
                        prompt_token_ids=expanded,
                        token_ids=list(out.token_ids),
                        text=out.text,
                        finish_reason=out.finish_reason,
                        temperature=0,
                        max_tokens=MAX_GENERATION,
                        processor_pixel_limits=[112896, 1003520],
                    )
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(teacher, "wrote", i + len(batch), "of", len(pending), flush=True)
        path.with_suffix(".done").write_text(str(len(rows)) + "\n")


def capture(engine, prepare, meta):
    from vllm import SamplingParams

    params = SamplingParams(temperature=0, seed=0, max_tokens=1)
    for manifest, teacher, directory in SOURCES:
        path = DATA / teacher
        features = DATA / directory
        features.mkdir(exist_ok=True)
        rows = {r["id"]: r for r in read_rows(DATA / manifest)}
        while not path.exists():
            time.sleep(1)
        captured = set()
        with path.open() as stream:
            while len(captured) < len(rows):
                pos = stream.tell()
                raw = stream.readline()
                if not raw.endswith("\n"):
                    if path.with_suffix(".done").exists():
                        raise ValueError(
                            f"Incomplete {teacher}, {len(rows)-len(captured)} rows missing"
                        )
                    stream.seek(pos)
                    time.sleep(0.5)
                    continue
                record = json.loads(raw)
                row = rows[record["id"]]
                assert all(
                    record[k] == row[k] for k in ("id", "split", "task", "document_id")
                )
                assert all(record[k] == v for k, v in meta.items())
                assert row["id"] not in captured
                captured.add(row["id"])
                file = features / (row["id"] + ".pt")
                if file.exists():
                    continue
                generated = record["token_ids"]
                request, expected = prepare(row, generated)
                assert expected == record["prompt_token_ids"] + generated
                assert len(expected) < MAX_MODEL_LEN
                engine.collective_rpc("feature_capture_begin")
                output = engine.generate([request], params, use_tqdm=False)[0]
                assert list(output.prompt_token_ids) == expected
                metadata = {k: row[k] for k in ("id", "document_id", "split", "task")}
                metadata.update(
                    **meta,
                    generated_ids=generated,
                    token_ids=expected,
                    prompt_len=len(record["prompt_token_ids"]),
                    teacher_finish_reason=record["finish_reason"],
                    source_category=row.get("source_category"),
                    processor_pixel_limits=[112896, 1003520],
                )
                saved = engine.collective_rpc(
                    "feature_capture_save", args=(str(file), metadata)
                )[0]
                assert saved["all_positions_and_tokens_verified"]
                if len(captured) % 24 == 0 or len(captured) == len(rows):
                    print(directory, len(captured), "/", len(rows), flush=True)


def audit(meta):
    import torch
    import vllm
    import transformers
    from common import load_frozen_weights

    torch.set_num_threads(3)
    counts, generated_counts, reason_counts = Counter(), Counter(), defaultdict(Counter)
    token_range = [2**63 - 1, 0]
    minima = {}
    # Project a fixed sample of saved final states through the version-matched
    # frozen head. Batched decode and full-sequence replay may differ at ties.
    samples, labels = [], []
    for name, teacher, directory in SOURCES:
        rows = {r["id"]: r for r in read_rows(DATA / name)}
        records = {r["id"]: r for r in read_rows(DATA / teacher)}
        root = DATA / directory
        assert rows.keys() == records.keys() == {p.stem for p in root.glob("*.pt")}
        for sid, row in rows.items():
            record = records[sid]
            x = torch.load(root / (sid + ".pt"), weights_only=True, map_location="cpu")
            assert all(x[k] == record[k] == v for k, v in meta.items())
            assert all(x[k] == row[k] for k in ("id", "document_id", "split", "task"))
            gen, prompt = record["token_ids"], record["prompt_token_ids"]
            assert x["prompt_len"] == len(prompt) and x["generated_ids"] == gen and gen
            assert x["token_ids"].tolist() == prompt + gen
            n = len(x["token_ids"])
            assert x["positions"].tolist() == list(range(n))
            assert x["aux_layer_ids"] == list(LAYERS)
            assert x["features"].shape == (n, 1024) and x["aux_features"].shape == (
                n,
                3,
                1024,
            )
            assert all(
                t.dtype == torch.bfloat16 and t.isfinite().all()
                for t in (x["features"], x["aux_features"])
            )
            key = row["split"] + "/" + row["task"]
            if row["split"] != "test" and counts[key] < 16:
                for position in sorted(
                    {len(prompt) - 1, len(prompt) - 1 + len(gen) // 2, n - 2}
                ):
                    samples.append(x["features"][position].clone())
                    labels.append(int(x["token_ids"][position + 1]))
            counts[key] += 1
            generated_counts[key] += len(gen)
            minima[key] = min(minima.get(key, len(gen)), len(gen))
            reason_counts[key][record["finish_reason"]] += 1
            token_range = [min(token_range[0], min(gen)), max(token_range[1], max(gen))]
    _, head = load_frozen_weights(MODEL, "cpu")
    predictions = []
    for vectors in torch.stack(samples).split(32):
        predictions.extend(
            torch.nn.functional.linear(vectors, head).argmax(-1).tolist()
        )
    matches = sum(a == b for a, b in zip(predictions, labels, strict=True))
    assert matches / len(labels) > 0.95, "Poor frozen 1.6 head/token alignment"
    summary = {
        **meta,
        "rows": counts,
        "generated_tokens": generated_counts,
        "min_generated": minima,
        "versions": {
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
        },
        "sampling": {
            "temperature": 0,
            "seed": 0,
            "max_new_tokens": MAX_GENERATION,
            "dtype": "bfloat16",
            "max_model_len": MAX_MODEL_LEN,
            "batch_size": BATCH,
            "prefix_caching": False,
        },
        "selection": json.loads((DATA / "selection.json").read_text()),
        "teacher_sha256": {
            teacher: sha256(DATA / teacher) for _, teacher, _ in SOURCES
        },
        "head_audit_next_token_top1": {
            "matches": matches,
            "sampled_positions": len(labels),
            "method": "CPU BF16 projection of genuine final states at prompt end, generation midpoint and generation end, up to 16 crops/train or validation task.",
        },
        "finish_reasons": reason_counts,
        "generated_id_range": token_range,
        "aux_layers": list(LAYERS),
        "validation": "Per-row prompt expansion, token identity, position coverage, feature dimensions, bfloat16, finiteness and checkpoint hash verified.",
    }
    (DATA / "capture_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["generate", "features", "audit"])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="Prepared manifests and private output/cache directory for this run",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=MODEL,
        help="Pinned local PaddleOCR-VL-1.6 checkpoint",
    )
    parser.add_argument(
        "--include-heldout",
        action="store_true",
        help="Also capture dev_test.jsonl separately; never used for training",
    )
    parser.add_argument(
        "--max-generation",
        type=int,
        default=MAX_GENERATION,
        help="Maximum generated response length, including longer tables",
    )
    args = parser.parse_args()
    if not 1 <= args.max_generation < MAX_MODEL_LEN - 2048:
        parser.error(
            "Choose at most 2047 generated tokens to preserve space for the image prompt"
        )
    MODEL = args.target.resolve()
    SOURCES = SOURCES if args.include_heldout else SOURCES[:1]
    MAX_GENERATION = args.max_generation
    phase, DATA = args.phase, args.data_dir.resolve()
    environment(phase)
    meta = target_metadata()
    if phase == "audit":
        audit(meta)
    else:
        engine, prepare = setup(phase)
        if phase == "generate":
            generate(engine, prepare, meta)
        else:
            capture(engine, prepare, meta)
