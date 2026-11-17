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

"""Explicit GPU differential checks against the unmodified vLLM 0.19 sampler."""

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import torch
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vllm019_greedy_sampler import eligible, try_greedy_rejection_sample


def make_sampling_metadata(batch):
    empty = torch.empty(batch, device="cuda")
    processor = MinTokensLogitsProcessor(None, torch.device("cuda"), False)
    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=empty,
        presence_penalties=empty,
        repetition_penalties=empty,
        output_token_ids=[[] for _ in range(batch)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors([processor]),
    )


def make_metadata(lengths, logits, reject_at):
    target_rows, bonus_rows, drafts, cumulative_drafts, cumulative_samples = (
        [],
        [],
        [],
        [],
        [],
    )
    offset = 0
    for length in lengths:
        target_rows.extend(range(offset, offset + length))
        bonus_rows.append(offset + length)
        ids = logits[offset : offset + length].float().argmax(-1).cpu().tolist()
        if reject_at < length:
            ids[reject_at] = (ids[reject_at] + 1) % logits.shape[-1]
        drafts.extend(ids)
        offset += length + 1
        cumulative_drafts.append(len(drafts))
        cumulative_samples.append(offset)

    def tensor(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    return SpecDecodeMetadata(
        draft_token_ids=tensor(drafts),
        num_draft_tokens=lengths,
        cu_num_draft_tokens=tensor(cumulative_drafts),
        cu_num_sampled_tokens=tensor(cumulative_samples),
        target_logits_indices=tensor(target_rows),
        bonus_logits_indices=tensor(bonus_rows),
        logits_indices=tensor(list(range(offset))),
    )


def event_time_us(function, iterations):
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


@torch.inference_mode()
def main(args):
    torch.manual_seed(17)
    original = RejectionSampler(Sampler())
    checked = 0
    for dtype in (torch.bfloat16, torch.float32):
        for lengths in ([0], [1], [2], [3], [4], [4, 0, 2, 1]):
            sm = make_sampling_metadata(len(lengths))
            rows = sum(lengths) + len(lengths)
            for special in ("random", "ties", "nan", "positive_inf", "negative_inf"):
                logits = torch.randn(rows, 103424, device="cuda", dtype=dtype)
                if special == "ties":
                    logits[:, 3] = logits[:, 9] = 100
                elif special == "nan":
                    logits[:, 7] = logits[:, 11] = float("nan")
                elif special == "positive_inf":
                    logits[:, 2] = logits[:, 8] = float("inf")
                elif special == "negative_inf":
                    logits.fill_(-float("inf"))
                for reject_at in range(max(lengths) + 1):
                    metadata = make_metadata(lengths, logits, reject_at)
                    expected = original(metadata, None, logits, sm)
                    actual = try_greedy_rejection_sample(metadata, logits, sm)
                    assert actual is not None and actual.logprobs_tensors is None
                    assert torch.equal(
                        actual.sampled_token_ids, expected.sampled_token_ids
                    ), (str(dtype), lengths, special, reject_at)
                    checked += 1

    logits = torch.randn(5, 103424, device="cuda", dtype=torch.bfloat16)
    metadata = make_metadata([4], logits, 4)
    sm = make_sampling_metadata(1)
    assert eligible(metadata, logits, sm)
    for change in (
        {"all_greedy": False},
        {"all_random": True},
        {"max_num_logprobs": 0},
        {"no_penalties": False},
        {"bad_words_token_ids": {0: [[0]]}},
        {"allowed_token_ids_mask": torch.zeros_like(logits[:1], dtype=torch.bool)},
    ):
        assert (
            try_greedy_rejection_sample(metadata, logits, replace(sm, **change)) is None
        )
    processor = sm.logitsprocs.non_argmax_invariant[0]
    processor.min_toks[0] = (10, [], {0})
    assert not eligible(metadata, logits, sm)
    processor.min_toks.clear()

    timings = {}
    if args.benchmark:
        timings = {
            "original_eager_us": event_time_us(
                lambda: original(metadata, None, logits, sm), args.iterations
            ),
            "candidate_eager_us": event_time_us(
                lambda: try_greedy_rejection_sample(metadata, logits, sm),
                args.iterations,
            ),
        }
    result = {"sampler_cases": checked, "all_exact": True, "timings": timings}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    main(args)
