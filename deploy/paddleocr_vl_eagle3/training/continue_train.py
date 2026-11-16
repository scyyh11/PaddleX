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

"""Resume a clean EAGLE3 from a frozen 1.6 checkpoint with recurrent hidden and logit distillation."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import time

from safetensors.torch import load_file
import torch
from torch.nn import functional as F

from common import (
    ROOT,
    TARGET_SHA256,
    TARGET_REVISION,
    Eagle3,
    batch_inputs,
    digest,
    fixed_anchors,
    load_corpus,
    load_frozen_weights,
    load_rows,
    write_json,
)
from train_clean import SOURCES, export, score


def load_split(rows, directory, device, with_final):
    # load_corpus verifies IDs, feature shape, pinned revision and exact generated token IDs.
    items = load_corpus(rows, directory, TARGET_SHA256, device)
    if with_final:
        for x in items:
            f = torch.load(
                directory / (x["id"] + ".pt"), weights_only=True, map_location="cpu"
            )
            x["final"] = (
                f["features"].view(-1, 1024).to(device=device, dtype=torch.bfloat16)
            )
            if len(x["final"]) != len(x["tokens"]):
                raise ValueError(f"Teacher feature count mismatch: {x['id']}")
    return items


def load_splits(directories):
    clean_directory = directories[0].resolve()
    image_split, doc_split, ids = {}, {}, set()
    groups, validations, provenance = [], [], []
    allowed = set().union(*SOURCES.values())
    for directory in directories:
        filename = directory / "manifest.jsonl"
        rows = load_rows(filename)
        audit = json.loads((directory / "selection.json").read_text())
        if digest(filename) != audit["manifest_sha256"]:
            raise ValueError(f"Manifest does not match frozen selection: {filename}")
        if directory.resolve() == clean_directory:
            permitted = allowed
        else:
            permitted = set(audit["sources"])
            if not permitted or "omnidocbench" in str(permitted).lower():
                raise ValueError("Invalid external sources")
        for r in rows:
            split = r["split"]
            if split not in {"validation", "train"} or r["source"] not in permitted:
                raise ValueError(f"Disallowed split or source: {r['id']}")
            if r["id"] in ids or r["image_sha256"] in image_split:
                raise ValueError(f"Duplicated sample or image: {r['id']}")
            ids.add(r["id"])
            image_split[r["image_sha256"]] = split
            if doc_split.setdefault(r["document_id"], split) != split:
                raise ValueError(f"Document spans both splits: {r['document_id']}")
        train = load_split(
            [r for r in rows if r["split"] == "train"],
            directory / "features",
            "cpu",
            True,
        )
        val = load_split(
            [r for r in rows if r["split"] == "validation"],
            directory / "features",
            "cuda",
            False,
        )
        groups.append(train)
        validations.append(val)
        provenance.append(
            dict(
                directory=str(directory),
                manifest_sha256=digest(filename),
                sources=dict(Counter(r["source"] for r in rows)),
                train=len(train),
                validation=len(val),
            )
        )
    return groups, validations, provenance


class PrivateHeadResidual(torch.nn.Module):
    """Token-specific low-rank residual, folded into the private head for vLLM."""

    def __init__(self, rows, width, rank):
        super().__init__()
        self.down = torch.nn.Linear(width, rank, bias=False)
        self.up = torch.nn.Linear(rank, rows, bias=False)
        torch.nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(width))
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.up(self.down(hidden))

    @torch.no_grad()
    def fold(self, head):
        return (
            (head.float() + self.up.weight @ self.down.weight)
            .to(torch.bfloat16)
            .contiguous()
        )


def run_step(model, buckets, rng, embed, head, inverse, args, adapter):
    # Give each task explicit sampling probability. If external data is available,
    # sample primarily from its source-matched documents while replaying the old corpus.
    tasks = rng.choices(
        ["ocr", "formula", "table"], weights=[0.45, 0.28, 0.27], k=args.batch
    )
    items = []
    for t in tasks:
        corpus = 1 if len(buckets) > 1 and rng.random() < args.external_weight else 0
        items.append(rng.choice(buckets[corpus][t]))
    indices = []
    for x in items:
        population = range(x["prompt"] - 1, len(x["tokens"]) - 2)
        indices.append(rng.sample(population, min(args.anchors, len(population))))
    aux, tokens, origins, active, lengths = batch_inputs(items, indices, embed.device)
    width = aux.shape[1]
    b = torch.arange(len(items), device="cuda")[:, None]
    gold_final = torch.zeros(
        4, len(items), origins.shape[1], 1024, dtype=torch.bfloat16, device="cuda"
    )
    for i, x in enumerate(items):
        base = torch.tensor(indices[i])
        for step in range(4):
            gold_final[step, i, : len(base)] = x["final"][
                (base + 1 + step).clamp(max=len(x["final"]) - 1)
            ]
    loss_sum, denom, labels = 0.0, 0.0, 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, pre, kv = model(aux, tokens[:, 1 : width + 1], embed)
        h, pre = h[b, origins], pre[b, origins]
        for step in range(4):
            pos = origins + 2 + step
            valid = active & (pos < lengths)
            gold = tokens[b, pos.clamp(max=tokens.shape[1] - 1)]
            h_valid = h[valid]
            teacher_h = gold_final[step][valid]
            draft_logits = F.linear(h_valid, head)
            if adapter is not None:
                draft_logits = draft_logits + adapter(h_valid)
            logits = draft_logits.float()
            with torch.no_grad():
                probs = (F.linear(teacher_h, head).float() / args.temperature).softmax(
                    -1
                )
            soft = (
                -(probs * (logits / args.temperature).log_softmax(-1)).sum()
                * args.temperature**2
            )
            hard = F.cross_entropy(
                logits, inverse[gold[valid]], reduction="sum", ignore_index=-100
            )
            align = (
                1.0 - F.cosine_similarity(h_valid.float(), teacher_h.float(), dim=-1)
            ).sum()
            n = valid.sum()
            w = 0.82**step
            loss_sum += w * (
                args.hard_weight * hard
                + (1.0 - args.hard_weight) * soft
                + args.align_weight * align
            )
            denom += w * n
            labels += n
            if step < 3:
                h, pre, kv = model(
                    pre,
                    gold,
                    embed,
                    positions=origins + step + 1,
                    caches=kv,
                    already_combined=True,
                    anchor_positions=origins,
                    base_length=width,
                )
    loss = loss_sum / denom
    loss.backward()
    return loss.detach().item(), int(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, action="append", required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", type=Path, default=ROOT / "models/PaddleOCR-VL-1.6")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--anchors", type=int, default=48)
    p.add_argument("--validation-anchors", type=int, default=512)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--hard-weight", type=float, default=0.65)
    p.add_argument("--align-weight", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1.3e-4)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--external-weight", type=float, default=0.78)
    p.add_argument("--vocab-size", type=int, default=40000)
    p.add_argument("--head-rank", type=int, default=0)
    p.add_argument("--head-lr", type=float, default=1.0e-4)
    args = p.parse_args()
    if args.run.exists() or args.output.exists() or len(args.data) not in (1, 2):
        p.error(
            "Use fresh paths and pass the independent clean corpus first, plus at most one external corpus"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error("Select exactly one CUDA device with CUDA_VISIBLE_DEVICES")
    if digest(args.target / "model.safetensors") != TARGET_SHA256:
        p.error("Unexpected target checksum")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True)
    prior_meta = json.loads((args.prior / "training_metadata.json").read_text())
    if (
        prior_meta.get("omnidocbench_used") is not False
        or prior_meta["target_sha256"] != TARGET_SHA256
    ):
        raise ValueError("Prior draft has not documented clean 1.6 provenance")
    groups, validations, source = load_splits(args.data)
    embeds, full_head = load_frozen_weights(args.target, "cuda")
    arch = json.loads((args.prior / "config.json").read_text())
    state = load_file(args.prior / "model.safetensors")
    offsets = state.pop("d2t")
    prior_ids = offsets + torch.arange(len(offsets))
    prior_head = state.pop("lm_head.weight")
    if not torch.equal(prior_head, full_head[prior_ids].cpu()):
        raise ValueError("Prior draft private head differs from target")
    buckets = []
    counts = Counter()
    for items in groups:
        by_task = defaultdict(list)
        for x in items:
            by_task[x["task"]].append(x)
            counts.update(x["tokens"][x["prompt"] :].tolist())
        buckets.append(by_task)
    ids = [i for i, n in counts.most_common(args.vocab_size)]
    chosen = set(ids)
    for i in prior_ids.tolist():
        if len(chosen) >= args.vocab_size:
            break
        chosen.add(i)
    coverages = []
    prior_set = set(prior_ids.tolist())
    for val in validations:
        counter = defaultdict(Counter)
        for x in val:
            counter[x["task"]].update(x["tokens"][x["prompt"] :].tolist())
        coverages.append(
            {
                t: dict(
                    tokens=sum(c.values()),
                    absent_prior=sum(n for i, n in c.items() if i not in prior_set),
                    absent_new=sum(n for i, n in c.items() if i not in chosen),
                )
                for t, c in counter.items()
            }
        )
    absolute = torch.tensor(sorted(chosen), device="cuda")
    head = full_head[absolute].contiguous()
    inverse = torch.full((len(full_head),), -100, device="cuda", dtype=torch.long)
    inverse[absolute] = torch.arange(len(absolute), device="cuda")
    model = Eagle3(arch["num_hidden_layers"]).cuda()
    model.load_state_dict(state, strict=True)
    adapter = (
        PrivateHeadResidual(len(absolute), head.shape[1], args.head_rank).cuda()
        if args.head_rank
        else None
    )

    def draft_head():
        return adapter.fold(head) if adapter is not None else head

    anchors = [
        fixed_anchors(v, args.validation_anchors, args.seed) for v in validations
    ]
    write_json(args.run / "validation_anchors.json", anchors)
    arch.update(
        draft_vocab_size=len(absolute),
        ocr_compact_vocab_source="1.6 independent continuation training tokens plus clean prior fallback",
    )
    base = dict(
        target=str(args.target),
        target_sha256=TARGET_SHA256,
        target_revision=TARGET_REVISION,
        warmstart=str(args.prior),
        warmstart_sha256=digest(args.prior / "model.safetensors"),
        sources=source,
        training_response_tokens=sum(counts.values()),
        train_compact_coverage=sum(n for i, n in counts.items() if i in chosen)
        / sum(counts.values()),
        eligible_train_observed_ids=len(counts),
        private_vocab_size=len(chosen),
        validation_coverage=coverages,
        training="Four-step recurrent distillation, teacher BF16 normalized hidden cosine and temperature-scaled compact-head soft/hard CE",
        learned_head_residual_rank=args.head_rank,
        selection="Highest fixed document-balanced clean validation K4 greedy macro prefix; when available the external corpus receives 80% selection weight, base clean 20%",
        omnidocbench_used=False,
        postfreeze_holdout_used=False,
        anchors_sha256=digest(args.run / "validation_anchors.json"),
        script_sha256=digest(Path(__file__)),
        args={
            k: (
                str(v)
                if isinstance(v, Path)
                else [str(x) for x in v] if k == "data" else v
            )
            for k, v in vars(args).items()
        },
    )
    write_json(args.run / "run_config.json", base)

    def validate():
        effective_head = draft_head()
        scores = [
            score(model, v, a, embeds, effective_head, absolute)
            for v, a in zip(validations, anchors)
        ]
        criterion = 0.2 * scores[0]["macro_prefix"] + 0.8 * scores[-1]["macro_prefix"]
        return dict(corpora=scores, criterion=criterion)

    initial = validate()
    write_json(args.run / "initial_validation.json", initial)
    print(json.dumps(dict(initial=initial, sources=source)), flush=True)
    best = initial["criterion"]
    best_step = 0
    staging = args.run / "selected_checkpoint"
    export(
        model, staging, arch, head, absolute, {**base, "step": 0, "validation": initial}
    )
    trainable = list(model.parameters())
    optimizer_groups = [dict(params=trainable, lr=args.lr, base_lr=args.lr)]
    if adapter is not None:
        trainable = trainable + list(adapter.parameters())
        optimizer_groups.append(
            dict(
                params=list(adapter.parameters()), lr=args.head_lr, base_lr=args.head_lr
            )
        )
    opt = torch.optim.AdamW(optimizer_groups, lr=args.lr, weight_decay=0.01)
    start = time.monotonic()
    labels = 0
    with (args.run / "metrics.jsonl").open("w", buffering=1) as f:
        for step in range(1, args.steps + 1):
            opt.zero_grad(set_to_none=True)
            loss, n = run_step(
                model, buckets, rng, embeds, head, inverse, args, adapter
            )
            grad = torch.nn.utils.clip_grad_norm_(
                trainable, 1.0, error_if_nonfinite=True
            )
            for group in opt.param_groups:
                group["lr"] = (
                    group["base_lr"]
                    * min(1.0, step / 120)
                    * (0.12 + 0.88 * (1 + math.cos(math.pi * step / args.steps)) / 2)
                )
            opt.step()
            labels += n
            if step % args.eval_every == 0 or step == args.steps:
                val = validate()
                record = dict(
                    step=step,
                    validation=val,
                    labels=labels,
                    loss=loss,
                    grad=float(grad),
                    seconds=time.monotonic() - start,
                )
                f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if val["criterion"] > best:
                    best_step, best = step, val["criterion"]
                    export(
                        model, staging, arch, draft_head(), absolute, {**base, **record}
                    )
    saved = load_file(staging / "model.safetensors")
    saved_head = saved.pop("lm_head.weight")
    if not torch.equal(
        saved.pop("d2t") + torch.arange(len(absolute)), absolute.cpu()
    ) or (adapter is None and not torch.equal(saved_head, head.cpu())):
        raise ValueError("Exported vocabulary does not match frozen target")
    head_is_frozen = torch.equal(saved_head, head.cpu())
    head = saved_head.cuda()
    adapter = None
    model.load_state_dict(saved, strict=True)
    final = validate()
    if (
        final
        != json.loads((staging / "training_metadata.json").read_text())["validation"]
    ):
        raise ValueError("Serialized checkpoint failed validation")
    checksum = digest(staging / "model.safetensors")
    write_json(
        staging / "compact_vocab.json",
        dict(
            full_vocab_size=len(full_head),
            absolute_target_ids=absolute.cpu().tolist(),
            vocabulary_source_split="clean training plus clean prior",
            compact_head_rows_equal_frozen_target=head_is_frozen,
            output_model_sha256=checksum,
            embedding_remains_target_shared=True,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(args.output)
    write_json(
        args.run / "summary.json",
        dict(
            best_step=best_step,
            before=initial,
            selected=final,
            labels=labels,
            seconds=time.monotonic() - start,
            weights_sha256=checksum,
            output=str(args.output),
        ),
    )
    print("Published", args.output, best_step, best, flush=True)


if __name__ == "__main__":
    main()
