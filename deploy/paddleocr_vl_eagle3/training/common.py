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

"""Shared VL-1.6 checkpoint, corpus and recurrent-training helpers."""
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from eagle_model import Eagle3 as Eagle3  # noqa: F401 - public training API

ROOT = Path(__file__).resolve().parents[1]
TASK_WEIGHTS = {"ocr": 0.45, "table": 0.35, "formula": 0.20}
LAYERS = [4, 10, 16]
K = 4
TARGET_SHA256 = "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db"
TARGET_REVISION = "c5630abae1d940eafe0697512a0325494b02ab42"


def load_frozen_weights(model_dir, device="cpu", dtype=torch.bfloat16):
    """Read only target embedding/head without loading the visual encoder."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    wanted = {"embed": "model.embed_tokens.weight", "head": "lm_head.weight"}
    found = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for label, key in wanted.items():
                if key in f.keys():
                    found[label] = f.get_tensor(key).to(device=device, dtype=dtype)
    if len(found) != len(wanted):
        raise ValueError(f"Missing target tensors: {set(wanted) - set(found)}")
    return found["embed"].detach(), found["head"].detach()


def target_output_scale(model_dir):
    """Estimate the frozen final feature scale from the target RMSNorm gain."""
    from safetensors import safe_open

    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            if "model.norm.weight" in f.keys():
                return f.get_tensor("model.norm.weight").float().mean().item()
    raise ValueError("Could not find target model.norm.weight for output calibration")


def load_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def load_corpus(rows, directory, target_hash, device):
    items = []
    for row in rows:
        path = directory / (row["id"] + ".pt")
        item = torch.load(path, map_location="cpu", weights_only=True)
        if (
            item["target_sha256"] != target_hash
            or item["target_revision"] != TARGET_REVISION
        ):
            raise ValueError(f"Unexpected teacher checksum in {path}")
        if any(item[key] != row[key] for key in ("id", "task", "split", "document_id")):
            raise ValueError(f"Feature metadata disagrees with manifest: {path}")
        tokens = item["token_ids"].view(-1).long()
        prompt = int(item["prompt_len"])
        if not 1 <= prompt <= len(tokens):
            raise ValueError(f"Invalid prompt length in {path}")
        if not torch.equal(
            tokens[prompt:], torch.as_tensor(item["generated_ids"]).reshape(-1).long()
        ):
            raise ValueError(
                f"Generated response differs from feature tokens in {path}"
            )
        aux = item["aux_features"]
        if list(item["aux_layer_ids"]) != LAYERS or tuple(aux.shape) != (
            len(tokens),
            3,
            1024,
        ):
            raise ValueError(f"Unexpected auxiliary layer or shape in {path}")
        if not torch.equal(
            item["positions"].view(-1).long(), torch.arange(len(tokens))
        ):
            raise ValueError(f"Unexpected token positions in {path}")
        if len(tokens) - prompt >= 2:
            items.append(
                dict(
                    id=row["id"],
                    task=row["task"],
                    prompt=prompt,
                    doc=row["document_id"],
                    aux=aux.to(device, torch.bfloat16),
                    tokens=tokens.to(device),
                )
            )
    if {x["task"] for x in items} != set(TASK_WEIGHTS):
        raise ValueError(
            "Need OCR, table and formula responses with usable labels in both splits"
        )
    return items


def fixed_anchors(items, count, seed):
    """Round-robin documents in each task, sampling distinct positions per crop."""
    out = {}
    for task in TASK_WEIGHTS:
        task_items = sorted(
            (x for x in items if x["task"] == task), key=lambda x: (x["doc"], x["id"])
        )
        rng = random.Random(f"{seed}/{task}")
        positions = {}
        for x in task_items:
            choices = list(range(x["prompt"] - 1, len(x["tokens"]) - 2))
            rng.shuffle(choices)
            positions[x["id"]] = choices
        plans = defaultdict(list)
        for i in range(max(map(len, positions.values()))):
            for x in task_items:
                if i < len(positions[x["id"]]):
                    plans[x["doc"]].append((x["id"], positions[x["id"]][i]))
        doc_ids = sorted(plans)
        rng.shuffle(doc_ids)
        sampled = [
            (plans[d][i])
            for i in range(max(map(len, plans.values())))
            for d in doc_ids
            if i < len(plans[d])
        ]
        for sid, a in sampled[:count]:
            out.setdefault(sid, []).append(a)
    return out


def batch_inputs(items, selected, device):
    lengths = [len(x["tokens"]) for x in items]
    width = max(lengths) - 2
    nanchors = max(map(len, selected))
    b = len(items)
    aux = torch.zeros(b, width, 3, 1024, device=device, dtype=torch.bfloat16)
    token_ids = torch.zeros(b, width + 2, device=device, dtype=torch.long)
    origins = torch.zeros(b, nanchors, device=device, dtype=torch.long)
    active = torch.zeros_like(origins, dtype=torch.bool)
    for i, (item, picks, length) in enumerate(zip(items, selected, lengths)):
        aux[i, : length - 2] = item["aux"][: length - 2]
        token_ids[i, :length] = item["tokens"]
        origins[i, : len(picks)] = torch.tensor(picks, device=device)
        active[i, : len(picks)] = True
    return (
        aux,
        token_ids,
        origins,
        active,
        torch.tensor(lengths, device=device)[:, None],
    )


def rollouts(model, items, selected, embed, head, absolute, teacher_forcing):
    """Each anchor sees the causal prefix and its own recurrent KVs only."""
    aux, tokens, origins, active, lengths = batch_inputs(items, selected, embed.device)
    width = aux.shape[1]
    batch_ix = torch.arange(len(items), device=embed.device)[:, None]
    h, pre, caches = model(aux, tokens[:, 1 : width + 1], embed)
    h, pre = h[batch_ix, origins], pre[batch_ix, origins]
    for step in range(K):
        idx = origins + 2 + step
        valid = active & (idx < lengths)
        gold = tokens[batch_ix, idx.clamp(max=tokens.shape[1] - 1)]
        logits = F.linear(h, head)
        yield logits, gold, valid
        if step + 1 < K:
            known = gold if teacher_forcing else absolute[logits.argmax(-1)]
            h, pre, caches = model(
                pre,
                known,
                embed,
                positions=origins + step + 1,
                caches=caches,
                already_combined=True,
                anchor_positions=origins,
                base_length=width,
            )


@torch.inference_mode()
def evaluate(model, items, anchors, embed, head, absolute):
    """BF16 serialized-weight greedy K=4 rollouts, including partial endpoints."""
    model = copy.deepcopy(model).to(torch.bfloat16).eval()
    histogram = defaultdict(lambda: Counter())
    for item in items:
        selected = anchors.get(item["id"], [])
        for start in range(0, len(selected), 64):
            picks = [selected[start : start + 64]]
            prefix_hits = []
            for logits, gold, valid in rollouts(
                model, [item], picks, embed, head, absolute, False
            ):
                prefix_hits.append(absolute[logits.argmax(-1)].eq(gold) & valid)
            lengths = (
                torch.stack(prefix_hits, dim=-1)
                .cumprod(dim=-1)
                .sum(-1)
                .reshape(-1)
                .tolist()
            )
            for task in (item["task"], "all"):
                histogram[task].update(lengths)
    result = {}
    for task, bins in histogram.items():
        n = sum(bins.values())
        result[task] = dict(
            anchors=n,
            first_accuracy=1 - bins[0] / n,
            accepted_draft_prefix=sum(k * v for k, v in bins.items()) / n,
            prefix_histogram=[bins[k] for k in range(K + 1)],
        )
    result["weighted_prefix"] = sum(
        TASK_WEIGHTS[t] * result[t]["accepted_draft_prefix"] for t in TASK_WEIGHTS
    )
    return result
