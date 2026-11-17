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

"""Opt-in, post-warmup stage events for the unchanged vLLM 0.19 worker.

Intervals measure stream elapsed time including CPU submission gaps. Nested
stages must not be added to their parent. Only reset/collect synchronize.
"""
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
import time

import torch
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.v1.utils import compute_iteration_details
from hps_vllm019_int8_worker import Vllm019Int8Worker


class ProfileVllm019Worker(Vllm019Int8Worker):
    def install_stage_profile(self, pool_size=32768):
        if hasattr(self, "_ocr_profile"):
            return {"already_installed": True}
        runner = self.model_runner
        if runner.__class__.__module__ != "vllm.v1.worker.gpu_model_runner":
            raise RuntimeError("Expected vLLM 0.19 GPUModelRunner")
        state = self._ocr_profile = dict(
            enabled=False,
            trace=False,
            phase="unknown",
            cursor=0,
            rows=[],
            pool=[
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(pool_size)
            ],
        )
        for start, end in state["pool"]:
            start.record()
            end.record()
        torch.cuda.synchronize()

        def wrap(obj, name, label):
            original = getattr(obj, name)

            @wraps(original)
            def wrapped(*args, **kwargs):
                if not state["enabled"] and not state["trace"]:
                    return original(*args, **kwargs)
                if name == "execute_model":
                    scheduler_output = args[0] if args else kwargs["scheduler_output"]
                    # Use scheduler state, not runner.requests: the latter still
                    # describes the previous step until _update_states runs.
                    details = compute_iteration_details(scheduler_output)
                    state["phase"] = (
                        "prefill"
                        if details.num_ctx_requests
                        else "decode" if details.num_generation_requests else "idle"
                    )
                stage = label + "/" + state["phase"]
                if not state["enabled"]:
                    with torch.profiler.record_function("OCR_STAGE/" + stage):
                        return original(*args, **kwargs)
                if state["cursor"] >= len(state["pool"]):
                    raise RuntimeError(
                        "Stage event pool exhausted; increase --event-pool-size"
                    )
                start, end = state["pool"][state["cursor"]]
                state["cursor"] += 1
                marker = (
                    torch.profiler.record_function("OCR_STAGE/" + stage)
                    if state["trace"]
                    else nullcontext()
                )
                with marker:
                    start.record()
                    before = time.perf_counter()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        end.record()
                        state["rows"].append(
                            (stage, start, end, (time.perf_counter() - before) * 1000)
                        )

            setattr(obj, name, wrapped)

        for name, stage in (
            ("execute_model", "execute_total"),
            ("sample_tokens", "sample_total"),
            ("_execute_mm_encoder", "encoder_total"),
            ("_model_forward", "target_forward"),
            ("_sample", "target_sample"),
            ("propose_draft_token_ids", "draft_total"),
        ):
            wrap(runner, name, stage)
        wrap(runner.drafter, "propose", "draft_propose")
        wrap(runner.model, "compute_logits", "target_logits")
        target = runner.model
        while isinstance(target, CUDAGraphWrapper):
            target = target.unwrap()
        wrap(target, "encode_image", "vision")
        wrap(target.mlp_AR, "forward", "projector")
        return {
            "runner": runner.__class__.__module__,
            "event_pool_size": pool_size,
            "synchronization": "reset/collect only",
            "nested": {
                "execute_total": ["encoder_total", "target_forward", "target_logits"],
                "sample_total": ["target_sample", "draft_total"],
                "encoder_total": ["vision", "projector"],
                "draft_total": ["draft_propose"],
            },
        }

    def stage_profile_reset(self, enabled=True, trace=False):
        state = self._ocr_profile
        torch.cuda.synchronize()
        state.update(enabled=enabled, trace=trace, phase="unknown", cursor=0, rows=[])
        return {"enabled": enabled}

    def stage_profile_collect(self):
        state = self._ocr_profile
        torch.cuda.synchronize()
        state["enabled"] = False
        state["trace"] = False
        summary = defaultdict(
            lambda: dict(calls=0, gpu_stream_ms=0.0, cpu_submit_ms=0.0)
        )
        for stage, start, end, cpu in state["rows"]:
            summary[stage]["calls"] += 1
            summary[stage]["gpu_stream_ms"] += start.elapsed_time(end)
            summary[stage]["cpu_submit_ms"] += cpu
        return dict(summary)
