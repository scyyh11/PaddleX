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

"""Train a one-layer EAGLE3 from scratch on independent OCR/formula/table documents."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F

from common import target_output_scale
from common import (
    ROOT,
    TARGET_REVISION,
    TARGET_SHA256,
    Eagle3,
    digest,
    evaluate,
    fixed_anchors,
    load_corpus,
    load_frozen_weights,
    load_rows,
    rollouts,
    write_json,
)


CLEAN_DATA = ROOT / "data/v16_clean"
SOURCES = {
    "ocr": {
        "https://huggingface.co/datasets/wikimedia/wikipedia",
        "https://docs.python.org/3.12/",
    },
    "formula": {"procedurally generated expressions rendered with Matplotlib MathText"},
    "table": {"https://huggingface.co/datasets/docling-project/FinTabNet_OTSL"},
}


def audit_rows(rows):
    ids, image_hashes, documents = set(), {}, {}
    for row in rows:
        sid, task, split = row["id"], row["task"], row["split"]
        if (
            sid in ids
            or task not in SOURCES
            or row["source"] not in SOURCES[task]
            or split not in ("train", "validation", "test")
        ):
            raise ValueError(
                f"Duplicate ID, invalid split or non-allowlisted source: {sid}"
            )
        ids.add(sid)
        if not row["document_id"]:
            raise ValueError(f"Missing source document: {sid}")
        if documents.setdefault(row["document_id"], split) != split:
            raise ValueError(f"Cross-split document overlap: {sid}")
        actual_hash = digest(row["image"])
        if row["image_sha256"] != actual_hash:
            raise ValueError(f"Image bytes differ from frozen manifest: {sid}")
        if image_hashes.setdefault(actual_hash, split) != split:
            raise ValueError(f"Cross-split image hash overlap: {sid}")
    return dict(
        rows_by_source=dict(Counter(r["source"] for r in rows)),
        source_documents_by_split=dict(Counter(documents.values())),
        distinct_image_hashes_by_split=dict(Counter(image_hashes.values())),
    )


def private_ids(train, target, full_vocab_size):
    counts = Counter()
    for x in train:
        counts.update(x["tokens"][x["prompt"] :].tolist())
    base_vocab = json.loads((target / "tokenizer.json").read_text())["model"]["vocab"]
    if min(counts) < 0 or max(counts) >= full_vocab_size or len(counts) > 40000:
        raise ValueError(
            "Training vocabulary exceeds model or native compact-kernel limits"
        )
    size = max(32768, len(counts))
    chosen = set(counts)
    # Preserve broad character coverage before filling by official BPE ID order.
    # Neither held-out responses nor another draft's mapping are read.
    ordered = sorted(base_vocab.items(), key=lambda kv: (len(kv[0]) != 1, kv[1]))
    for _, token_id in ordered:
        if len(chosen) >= size:
            break
        chosen.add(token_id)
    absolute = torch.tensor(sorted(chosen), dtype=torch.long)
    return absolute, dict(
        train_response_tokens=sum(counts.values()),
        train_observed_ids=len(counts),
        tokenizer_only_ids=len(chosen) - len(counts),
        inherited_draft_ids=0,
        private_vocab_size=len(chosen),
        tokenizer_basis="Base official 1.6 BPE token IDs, single-codepoint tokens first, then ascending ID; skip IDs already observed in training",
    )


def train_step(model, by_task, rng, embed, head, absolute, inverse, optimizer, args):
    tasks = sorted(by_task)
    items = [rng.choice(by_task[rng.choice(tasks)]) for _ in range(args.batch)]
    picks = []
    for x in items:
        positions = range(x["prompt"] - 1, len(x["tokens"]) - 2)
        picks.append(rng.sample(positions, min(args.anchors_per_crop, len(positions))))
    # The frozen target's final state at t+1 gives its distribution for t+2.
    # These states enter loss computation only; recurrent inputs are computed by the draft.
    na = max(map(len, picks))
    teacher = torch.zeros(
        4, len(items), na, head.shape[1], dtype=torch.bfloat16, device=embed.device
    )
    for b, (item, indices) in enumerate(zip(items, picks)):
        idx = torch.tensor(indices) + 1
        for step in range(4):
            teacher[step, b, : len(indices)] = item["final"][
                (idx + step).clamp(max=len(item["final"]) - 1)
            ]
    weighted_loss, weight_sum, count = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for step, (logits, gold, valid) in enumerate(
            rollouts(model, items, picks, embed, head, absolute, True)
        ):
            draft_log_prob = F.log_softmax(logits[valid].float(), dim=-1)
            hard = F.nll_loss(draft_log_prob, inverse[gold[valid]], reduction="sum")
            with torch.no_grad():
                target_prob = F.softmax(
                    F.linear(teacher[step][valid], head).float(), dim=-1
                )
            soft = -(target_prob * draft_log_prob).sum()
            loss = 0.65 * hard + 0.35 * soft
            n = valid.sum()
            weighted_loss = weighted_loss + loss * 0.8**step
            weight_sum = weight_sum + n * 0.8**step
            count = count + n
        loss = weighted_loss / weight_sum
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return float(loss.detach()), int(count), float(grad)


def score(model, items, anchors, embed, head, absolute):
    result = evaluate(model, items, anchors, embed, head, absolute)
    # Equal task weights for every clean model-selection decision.
    result.pop("weighted_prefix")
    result["macro_prefix"] = sum(
        result[t]["accepted_draft_prefix"] for t in SOURCES
    ) / len(SOURCES)
    return result


def export(model, staging, config, head, absolute, metadata):
    staging.mkdir(exist_ok=True)
    weights = {
        k: v.detach().to("cpu", torch.bfloat16).contiguous()
        for k, v in model.state_dict().items()
    }
    weights["lm_head.weight"] = head.cpu().contiguous()
    weights["d2t"] = absolute.cpu() - torch.arange(len(absolute))
    temporary = staging / "model.safetensors.tmp"
    save_file(weights, temporary)
    temporary.chmod(0o644)
    temporary.replace(staging / "model.safetensors")
    write_json(staging / "config.json", config)
    write_json(staging / "training_metadata.json", metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=CLEAN_DATA)
    p.add_argument("--target", type=Path, default=ROOT / "models/PaddleOCR-VL-1.6")
    p.add_argument(
        "--run", type=Path, default=ROOT / "results/v16_clean/eagle3_training"
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "models/PaddleOCR-VL-1.6-EAGLE3-K4-clean"
    )
    p.add_argument("--steps", type=int, default=7200)
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--anchors-per-task", type=int, default=384)
    p.add_argument("--anchors-per-crop", type=int, default=48)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3.5e-4)
    p.add_argument("--seed", type=int, default=20261107)
    args = p.parse_args()
    if args.output.exists() or args.run.exists():
        p.error("Select fresh run and output directories")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error("Select exactly one CUDA device with CUDA_VISIBLE_DEVICES")
    target_sha = digest(args.target / "model.safetensors")
    if target_sha != TARGET_SHA256:
        raise ValueError("Target does not match pinned PaddleOCR-VL-1.6")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    trainval_path = args.data / "manifest.jsonl"
    selection = json.loads((args.data / "selection.json").read_text())
    if selection["manifest_sha256"] != digest(trainval_path):
        raise ValueError("Clean manifest differs from frozen source selection")
    trainval = load_rows(trainval_path)
    if {r["split"] for r in trainval} != {"train", "validation"}:
        raise ValueError(
            "Training manifest must include only training and selection documents"
        )
    trainer_audit = audit_rows(trainval)
    args.run.mkdir(parents=True)
    staging = args.run / "selected_checkpoint"
    train_rows = [x for x in trainval if x["split"] == "train"]
    val_rows = [x for x in trainval if x["split"] == "validation"]
    train = load_corpus(train_rows, args.data / "features", target_sha, "cpu")
    val = load_corpus(val_rows, args.data / "features", target_sha, "cuda")
    for item in train:
        f = torch.load(
            args.data / "features" / (item["id"] + ".pt"),
            weights_only=True,
            map_location="cpu",
        )
        item["final"] = f["features"].view(-1, 1024).to(torch.bfloat16)
        if len(item["final"]) != len(item["tokens"]):
            raise ValueError(f"Teacher length mismatch: {item['id']}")
    anchors = fixed_anchors(val, args.anchors_per_task, args.seed)
    write_json(args.run / "validation_anchors.json", anchors)
    by_task = {t: [x for x in train if x["task"] == t] for t in SOURCES}
    embed, full_head = load_frozen_weights(args.target, "cuda")
    ids, vocab = private_ids(train, args.target, len(full_head))
    absolute = ids.cuda()
    head = full_head[absolute].contiguous()
    inverse = torch.full((len(full_head),), -100, device="cuda", dtype=torch.long)
    inverse[absolute] = torch.arange(len(absolute), device="cuda")
    scale = target_output_scale(args.target)
    model = Eagle3(layers=1, output_scale=scale).cuda()
    # Write initialization once and obtain the model's native architectural config independent of any other checkpoint.
    initial_dir = args.run / "initial_transformer"
    model.export(initial_dir)
    arch = json.loads((initial_dir / "config.json").read_text())
    arch.update(
        draft_vocab_size=len(absolute),
        has_own_lm_head=True,
        has_own_embed_tokens=False,
        ocr_compact_vocab_source="Clean training responses and official PaddleOCR-VL-1.6 BPE vocabulary",
    )
    base = dict(
        target=str(args.target),
        target_sha256=target_sha,
        target_revision=TARGET_REVISION,
        seed=args.seed,
        warmstart=None,
        initialization="Random N(0,0.02) linear weights; fc initialized to average three target features; RMSNorm output gain equals mean frozen official 1.6 final gain.",
        initial_output_scale=scale,
        initial_transformer_sha256=digest(initial_dir / "model.safetensors"),
        manifest=str(trainval_path),
        manifest_sha256=digest(trainval_path),
        source_audit=trainer_audit,
        selection_source_sha256=digest(args.data / "selection.json"),
        teacher_sha256=digest(args.data / "teacher.jsonl"),
        train_samples=len(train),
        validation_samples=len(val),
        excluded_short_train=len(train_rows) - len(train),
        excluded_short_validation=len(val_rows) - len(val),
        train_tasks={t: len(v) for t, v in by_task.items()},
        private_vocab=vocab,
        model_parameters=sum(p.numel() for p in model.parameters()),
        prior_draft_data_weights_or_vocab_read=False,
        omnidocbench_used=False,
        postfreeze_holdout_used=False,
        target_tokenizer_sha256=digest(args.target / "tokenizer.json"),
        target_chat_template_sha256=digest(args.target / "chat_template.jinja"),
        target_processor_sha256=digest(args.target / "preprocessor_config.json"),
        training="Uniform random task, then crop and up to 48 generation anchors; 4-step teacher-token rollout with draft-predicted hidden states, per-step weight 0.8^step, cross-entropy mixture of 65% hard 1.6 token and 35% frozen 1.6 distribution on compact head.",
        selection="Maximum mean of OCR/formula/table greedy accepted K4 prefixes on clean validation documents. Checkpoint rounded to BF16 for each evaluation.",
        validation_anchors_sha256=digest(args.run / "validation_anchors.json"),
        source_sha256={
            f: digest(Path(__file__).parent / f)
            for f in ("train_clean.py", "common.py", "eagle_model.py")
        },
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    write_json(args.run / "run_config.json", base)
    initial_score = score(model, val, anchors, embed, head, absolute)
    write_json(args.run / "initial_validation.json", initial_score)
    print(json.dumps(dict(initial_validation=initial_score)), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = -1.0
    labels = 0
    started = time.monotonic()
    with (args.run / "metrics.jsonl").open("w", buffering=1) as fh:
        for step in range(1, args.steps + 1):
            ramp = min(1.0, step / 160)
            opt.param_groups[0]["lr"] = (
                args.lr
                * ramp
                * (0.12 + 0.88 * (1 + math.cos(math.pi * step / args.steps)) / 2)
            )
            loss, n, grad = train_step(
                model, by_task, rng, embed, head, absolute, inverse, opt, args
            )
            labels += n
            if step % args.eval_every == 0 or step == args.steps:
                scores = score(model, val, anchors, embed, head, absolute)
                record = dict(
                    step=step,
                    validation=scores,
                    labels=labels,
                    loss=loss,
                    grad=grad,
                    seconds=time.monotonic() - started,
                )
                fh.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if scores["macro_prefix"] > best:
                    best = scores["macro_prefix"]
                    export(model, staging, arch, head, absolute, {**base, **record})
    selected = json.loads((staging / "training_metadata.json").read_text())
    loaded = load_file(staging / "model.safetensors")
    saved_ids = torch.arange(len(absolute)) + loaded.pop("d2t")
    if not torch.equal(saved_ids, ids) or not torch.equal(
        loaded.pop("lm_head.weight"), full_head[absolute].cpu()
    ):
        raise RuntimeError("Serialized head or d2t mismatched frozen official target")
    model.load_state_dict(loaded, strict=True)
    reproduced = score(model, val, anchors, embed, head, absolute)
    if reproduced != selected["validation"]:
        raise RuntimeError(
            "Validation of reloaded BF16 weights did not reproduce selected result"
        )
    checksum = digest(staging / "model.safetensors")
    write_json(
        staging / "compact_vocab.json",
        dict(
            **vocab,
            target_sha256=target_sha,
            output_model_sha256=checksum,
            full_vocab_size=len(full_head),
            absolute_target_ids=saved_ids.tolist(),
            vocabulary_source_split="train plus data-independent official target tokenizer",
            mapping_semantics="absolute_target_id = private_index + d2t[private_index]",
            compact_head_rows_equal_frozen_target=True,
            embedding_remains_target_shared=True,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(args.output)
    summary = dict(
        initial_validation=initial_score,
        selected_step=selected["step"],
        selected_validation=reproduced,
        selected_labels=selected["labels"],
        total_labels=labels,
        seconds=time.monotonic() - started,
        weights_sha256=checksum,
        reloaded_validation_reproduced=True,
    )
    write_json(args.run / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
