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

"""Explicit graph argmax for the existing strict greedy sampler.

Only vllm019_greedy_sampler.try_greedy_rejection_sample is replaced. The outer
native sampler type/hook guards and original eligible() remain in control.
The original integer selector immediately consumes borrowed graph argmax
storage and writes a newly allocated, independently owned SamplerOutput.
torch.argmax is never patched; the existing guarded sampler installer is reused.

Scope: the existing PaddleOCR-VL 1.6 B1/K4 strict-greedy profile, contiguous
CUDA BF16/FP32 [Q,103424], Q1..7, pinned vLLM/PyTorch/Triton on A100 80GB.
BF16 uses the original exact FP32 conversion. Unsupported calls use the
original try function, including all its own eligibility/fallback behavior.

One strongly owned cache holds at most seven graphs. It is bound to the first
device, stream and Python thread; a changed owner falls back rather than racing
shared inputs/output. The caller must submit source production/consumption on
that stream. Graph replay is supported only by eager sampler calls: every call
during CUDA capture uses the original sampler, even with a prepared plan.
prepare() may build plans outside capture; otherwise first-use construction
incurs one-time warmup cost before eager replay.
Keep the module/cache alive until every graph using its buffers is retired.
"""

from collections import Counter
from functools import wraps
import importlib
import importlib.metadata
import threading

import torch
import triton

from vllm019_graph_argmax import GraphArgmaxCache
from vllm019_stable_argmax import VOCAB

COUNTS = Counter()
_INSTALLATION = None
_CACHE = None
_OWNER = None
_DEVICE_SUPPORT = {}
_CACHE_LOCK = threading.Lock()


def _runtime_supported(device):
    if device not in _DEVICE_SUPPORT:
        props = torch.cuda.get_device_properties(device)
        _DEVICE_SUPPORT[device] = (
            str(torch.__version__) == "2.10.0+cu128"
            and torch.version.cuda == "12.8"
            and triton.__version__ == "3.6.0"
            and importlib.metadata.version("vllm") == "0.19.0"
            and (props.major, props.minor, props.multi_processor_count) == (8, 0, 108)
            and "A100" in props.name
            and props.total_memory >= 79 * 1024**3
        )
    return _DEVICE_SUPPORT[device]


def _profile_reason(spec, logits):
    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_cuda
        or logits.dtype not in (torch.bfloat16, torch.float32)
        or logits.ndim != 2
        or logits.shape[1] != VOCAB
        or not 1 <= logits.shape[0] <= 7
        or not logits.is_contiguous()
    ):
        return "logits_profile"
    if len(spec.num_draft_tokens) != 1 or not 0 <= spec.num_draft_tokens[0] <= 4:
        return "batch_profile"
    return None


def _owner(device):
    stream = torch.cuda.current_stream(device)
    return device, stream.cuda_stream, threading.get_ident()


def _get_cache(device, queries):
    """Return (cache, reason); neither initialize nor replay inside capture."""
    # First-use preparation can release the GIL. Serialize owner admission so
    # another Python thread cannot acquire the same persistent cache meanwhile.
    with _CACHE_LOCK:
        return _get_cache_locked(device, queries)


def _get_cache_locked(device, queries):
    global _CACHE, _OWNER
    if not _runtime_supported(device):
        return None, "runtime_profile"
    owner = _owner(device)
    if _OWNER is not None and owner != _OWNER:
        return None, "stream_or_thread"
    with torch.cuda.device(device):
        # This runtime rejects nested graph replay even for an existing plan.
        if torch.cuda.is_current_stream_capturing():
            return None, "capture"
        if _CACHE is None:
            _CACHE = GraphArgmaxCache(device)
            _OWNER = owner
        if queries not in _CACHE.plans:
            _CACHE.prepare(queries)
            COUNTS["prepared_plans"] += 1
    return _CACHE, None


def _make_try(original, sampler_module, output_type):
    @wraps(original)
    def sample(metadata, logits, sampling_metadata):
        spec = metadata
        if not sampler_module.eligible(spec, logits, sampling_metadata):
            reason = "ineligible"
        else:
            reason = _profile_reason(spec, logits)
        cache = None
        if reason is None:
            cache, reason = _get_cache(logits.device, logits.shape[0])
        if reason is not None:
            COUNTS["fallback_calls"] += 1
            COUNTS[f"fallback_{reason}"] += 1
            return original(spec, logits, sampling_metadata)

        columns = spec.max_spec_len + 1
        # Allocate the owned result before borrowing shared argmax storage.
        output = torch.empty(
            (len(spec.num_draft_tokens), columns),
            device=logits.device,
            dtype=torch.int32,
        )
        fp32_logits = logits.to(torch.float32)
        argmax = cache(fp32_logits)
        # No subsequent cache replay may intervene before this same-stream read.
        sampler_module._select_greedy_tokens[(len(spec.num_draft_tokens),)](
            argmax,
            spec.draft_token_ids,
            spec.cu_num_draft_tokens,
            spec.target_logits_indices,
            spec.bonus_logits_indices,
            output,
            OUTPUT_COLUMNS=columns,
            BLOCK=triton.next_power_of_2(columns),
            num_warps=4,
        )
        COUNTS["graph_argmax_calls"] += 1
        return output_type(sampled_token_ids=output, logprobs_tensors=None)

    sample._paddleocr_vllm019_sampler_graph = True
    return sample


def prepare(queries=tuple(range(1, 8)), device="cuda"):
    """Prepare outside capture for eager calls on the same stream/thread."""
    if _INSTALLATION is None:
        raise RuntimeError("Call vllm019_sampler_graph.install() before prepare()")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Graph sampler preparation requires CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    for q in queries:
        if type(q) is not int or not 1 <= q <= 7:
            raise ValueError("Prepared query counts must be integers in 1..7")
        _, reason = _get_cache(device, q)
        if reason is not None:
            raise RuntimeError(f"Cannot prepare graph sampler: {reason}")
    return metadata()


def install():
    """Install explicitly; preserve the existing native sampler wrapper guards."""
    global _INSTALLATION
    if _INSTALLATION is not None:
        module, original, wrapped = _INSTALLATION
        if module.try_greedy_rejection_sample is not wrapped:
            raise RuntimeError("The sampler hook was replaced after installation")
        return metadata()
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("Graph sampler requires vLLM 0.19.0")
    module = importlib.import_module("vllm019_greedy_sampler")
    output_type = importlib.import_module("vllm.v1.outputs").SamplerOutput
    original = module.try_greedy_rejection_sample
    if getattr(original, "_paddleocr_vllm019_sampler_graph", False):
        raise RuntimeError("Graph sampler hook is installed by another module instance")
    # This installs or retains the adapter's original _uses_native_sampler guard.
    module.install()
    wrapped = _make_try(original, module, output_type)
    module.try_greedy_rejection_sample = wrapped
    _INSTALLATION = (module, original, wrapped)
    return metadata()


def metadata():
    return {
        "installed": _INSTALLATION is not None,
        "scope": "existing PaddleOCR-VL 1.6 B1/K4 strict greedy; BF16/FP32 Q1..7 x 103424",
        "runtime": "vLLM 0.19.0 / torch 2.10.0+cu128 / Triton 3.6.0 / A100 80GB SM80/108SM",
        "argmax": "GraphArgmaxCache with exact BF16-to-FP32 conversion",
        "selector": "original integer _select_greedy_tokens",
        "output_ownership": "fresh independent int32 SamplerOutput; borrowed argmax consumed immediately",
        "stream_policy": "cache bound to first CUDA device, stream and Python thread; others use original function",
        "capture_policy": "original sampler during every CUDA capture; graph argmax supports eager replay only",
        "counter_scope": "Python calls, excluding captured graph replays",
        "counts": dict(COUNTS),
        "cache": None if _CACHE is None else _CACHE.metadata(),
    }
