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

"""Opt-in static speculative indices for native vLLM 0.19 B1/K4/Q5 decode.

The five index tensors are constant for this exact CPU-described batch shape.
Draft token IDs retain independent storage and current GPU values, including
when async scheduling supplies those values from a previous GPU proposal.
Importing this module installs no hooks and does not initialize CUDA.
"""

from collections import Counter
from functools import wraps
import importlib.metadata

import numpy as np
import torch

COUNTS = Counter()


def eligible(num_spec_tokens, num_draft_tokens, cu_num_scheduled_tokens, input_ids):
    """Admission reads CPU arrays and tensor metadata only; no GPU scalar sync."""
    return (
        num_spec_tokens == 4
        and isinstance(num_draft_tokens, np.ndarray)
        and isinstance(cu_num_scheduled_tokens, np.ndarray)
        and num_draft_tokens.shape == (1,)
        and cu_num_scheduled_tokens.shape == (1,)
        and num_draft_tokens[0] == 4
        and cu_num_scheduled_tokens[0] == 5
        and input_ids.is_cuda
        and input_ids.ndim == 1
        and input_ids.shape[0] >= 5
        and input_ids.dtype == torch.int32
        and input_ids.stride(0) == 1
    )


def build_metadata(input_ids, cached_indices=None, metadata_cls=None):
    """Build B1/K4/Q5 metadata and return `(metadata, reusable_indices)`.

    The caller must have checked `eligible`. CPU input is also accepted here
    so the integer mapping and ownership contract can be checked without CUDA.
    Consumers must treat the cached index tensors as read-only, as the native
    runner, sampler, and proposer do. The draft token tensor is newly cloned.
    """
    if metadata_cls is None:
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

        metadata_cls = SpecDecodeMetadata
    if cached_indices is None:
        cached_indices = (
            torch.tensor([4], device=input_ids.device, dtype=torch.int32),
            torch.tensor([5], device=input_ids.device, dtype=torch.int32),
            torch.arange(5, device=input_ids.device, dtype=torch.int32),
            torch.arange(4, device=input_ids.device, dtype=torch.int32),
            torch.tensor([4], device=input_ids.device, dtype=torch.int32),
        )
    cu_drafts, cu_sampled, logits_indices, target_indices, bonus_indices = (
        cached_indices
    )
    result = metadata_cls(
        draft_token_ids=input_ids[1:5].clone(),
        num_draft_tokens=[4],
        cu_num_draft_tokens=cu_drafts,
        cu_num_sampled_tokens=cu_sampled,
        target_logits_indices=target_indices,
        bonus_logits_indices=bonus_indices,
        logits_indices=logits_indices,
    )
    return result, cached_indices


def _make_wrapper(original):
    @wraps(original)
    def calculate(self, num_draft_tokens, cu_num_scheduled_tokens):
        ids = self.input_ids.gpu
        if not eligible(
            self.num_spec_tokens, num_draft_tokens, cu_num_scheduled_tokens, ids
        ):
            COUNTS["fallback_calls"] += 1
            return original(self, num_draft_tokens, cu_num_scheduled_tokens)
        cache = getattr(self, "_paddleocr_vllm019_spec_indices", None)
        if cache is not None and cache[0].device != ids.device:
            cache = None
        result, cache = build_metadata(ids, cache)
        self._paddleocr_vllm019_spec_indices = cache
        COUNTS["fast_calls"] += 1
        return result

    calculate._paddleocr_vllm019_spec_metadata = True
    return calculate


def install():
    """Explicitly patch the old V1 runner, retaining its general fallback."""
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("Static speculative metadata requires vLLM 0.19.0")
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner._calc_spec_decode_metadata
    if not getattr(original, "_paddleocr_vllm019_spec_metadata", False):
        GPUModelRunner._calc_spec_decode_metadata = _make_wrapper(original)
