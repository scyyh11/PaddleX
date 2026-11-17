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

"""Pinned CUDA 12 native-Q1 target projection recipes for image HTTP decoding.

Explicitly install before constructing the target. The actual target decoder
and ParallelLMHead objects are marked; no prefix assumption selects the head.
Only their Q2..5 calls use locked Lt plans. Q1 and larger image prefills retain
native F.linear. Extremely short text-only prefills or final prefill chunks of
2..5 tokens are outside this image HTTP profile; no stale Python phase flag is
used to classify them.

Recipes below come from CUDA 12.8.4.1 native Q1 logs, not CUDA 13 assumptions.
Current-weight GPU checks passed 405/405 eager and changed-input graph cases.
End-to-end regression is still required for this integrated adapter. Unsupported
runtime/configuration, missing plans, or changed storage raise instead of
silently returning the numerically different native Q2..5 GEMM.
"""

from collections import Counter
from functools import wraps
import importlib.metadata

import torch

from vllm019_locked_lt import LockedLt

SHAPES = {
    "qkv": (2560, 1024),
    "o": (1024, 2048),
    "gateup": (6144, 1024),
    "down": (1024, 3072),
    "head": (103424, 1024),
}
# cublasLt.h: 64x64 tile=15, 128x64=18, 64x5 stages=17, 32x6=12,
# REDUCTION_SCHEME_INPLACE=1. Source: vllm019-linear-parity.cublaslt.log.
RECIPES = {
    "qkv": dict(
        algo_id=31,
        tile=15,
        stages=17,
        split_k=1,
        reduction=0,
        swizzle=0,
        row_batch=False,
        sm_count_target=108,
    ),
    "o": dict(algo_id=13, custom_option=75, row_batch=True, sm_count_target=108),
    "gateup": dict(
        algo_id=6,
        tile=15,
        stages=17,
        split_k=1,
        reduction=0,
        swizzle=0,
        row_batch=False,
        sm_count_target=108,
    ),
    "down": dict(algo_id=13, custom_option=75, row_batch=True, sm_count_target=108),
    "head": dict(
        algo_id=6,
        tile=18,
        stages=12,
        split_k=2,
        reduction=1,
        swizzle=1,
        row_batch=False,
        sm_count_target=108,
    ),
}
PLANS = {}
COUNTS = Counter()
_INSTALLED = False


def _check_runtime(device=None):
    if (
        importlib.metadata.version("vllm") != "0.19.0"
        or importlib.metadata.version("nvidia-cublas-cu12") != "12.8.4.1"
        or torch.__version__.split("+")[0] != "2.10.0"
        or torch.version.cuda != "12.8"
    ):
        raise RuntimeError(
            "Target Lt requires vLLM0.19.0/torch2.10.0/CUDA12.8/cuBLAS12.8.4.1"
        )
    from vllm import envs

    if (
        envs.VLLM_BATCH_INVARIANT
        or str(torch.backends.cuda.preferred_blas_library()) != "_BlasBackend.Cublas"
        or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is not True
        or getattr(
            torch.backends.cuda.matmul,
            "allow_bf16_reduced_precision_reduction_split_k",
            None,
        )
        is not True
    ):
        raise RuntimeError(
            "Target Lt native-Q1 backend/precision flags differ from the measured configuration"
        )
    if device is not None:
        props = torch.cuda.get_device_properties(device)
        if (
            props.major != 8
            or props.minor != 0
            or props.multi_processor_count != 108
            or "A100" not in props.name
            or props.total_memory < 79 * 1024**3
        ):
            raise RuntimeError("Target Lt recipes require A100 80GB SM80 with 108 SMs")


def _mark_target(target, config, linear_type, embedding_type, head_type):
    parallel = config.parallel_config
    if (
        config.model_config.dtype != torch.bfloat16
        or config.quant_config is not None
        or any(
            getattr(parallel, name) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
            )
        )
    ):
        raise RuntimeError("Target Lt requires unquantized BF16 with TP/PP/DP/DCP=1")
    layers = target.language_model.model.layers
    if len(layers) != 18:
        raise RuntimeError("Target Lt requires the 18-layer PaddleOCR-VL-1.6 decoder")
    selected = []
    for decoder in layers:
        selected.extend(
            (
                ("qkv", decoder.self_attn.qkv_proj),
                ("o", decoder.self_attn.o_proj),
                ("gateup", decoder.mlp.gate_up_proj),
                ("down", decoder.mlp.down_proj),
            )
        )
    head = target.language_model.lm_head
    if type(head) is not head_type:
        raise RuntimeError("Target Lt requires the actual target ParallelLMHead")
    selected.append(("head", head))
    if len({id(layer) for _, layer in selected}) != 73:
        raise RuntimeError("Unexpected shared target projection modules")
    for kind, layer in selected:
        method_type = embedding_type if kind == "head" else linear_type
        if (
            tuple(layer.weight.shape) != SHAPES[kind]
            or layer.weight.dtype != torch.bfloat16
            or type(layer.quant_method) is not method_type
            or getattr(layer, "bias", None) is not None
        ):
            raise RuntimeError(f"Unexpected target {kind} projection contract")
    # Mark only after the complete target contract has passed.
    for kind, layer in selected:
        layer._vllm019_target_linear_kind = kind
    target._vllm019_target_linear_layers = selected
    target._vllm019_target_linear_prepared = False


def _prepare_target(target):
    if target._vllm019_target_linear_prepared:
        raise RuntimeError(
            "Target Lt does not support reloading captured target weights"
        )
    selected = target._vllm019_target_linear_layers
    device = selected[0][1].weight.device
    _check_runtime(device)
    pending = {}
    try:
        for kind, layer in selected:
            weight = layer.weight
            if (
                not weight.is_cuda
                or weight.device != device
                or weight.dtype != torch.bfloat16
                or tuple(weight.shape) != SHAPES[kind]
                or not weight.is_contiguous()
            ):
                raise RuntimeError(f"Target {kind} weight changed during loading")
            key = (weight.data_ptr(), kind)
            if key in PLANS or key in pending:
                raise RuntimeError("Target Lt weight storage was unexpectedly reused")
            plan = LockedLt(
                weight, RECIPES[kind], workspace_bytes=4096 if kind == "head" else 0
            )
            pending[key] = plan
            if plan.library_version != 120804:
                raise RuntimeError(
                    "Target Lt loaded a different cuBLASLt library version"
                )
            for queries in (2, 3, 4, 5):
                plan.prepare(queries)
        PLANS.update(pending)
        target._vllm019_target_linear_prepared = True
        COUNTS["prepared_targets"] += 1
    except Exception:
        for plan in pending.values():
            try:
                plan.close()
            except Exception:
                pass
        raise


def _execute(x, weight, kind):
    rows = x.numel() // x.shape[-1]
    if rows < 2 or rows > 5:
        COUNTS[f"{kind}_native_q{rows}"] += 1
        return torch.nn.functional.linear(x, weight)
    if kind not in SHAPES or tuple(weight.shape) != SHAPES[kind]:
        raise RuntimeError("Target Lt projection identity/shape changed")
    plan = PLANS.get((weight.data_ptr(), kind))
    if (
        plan is None
        or rows not in plan.plans
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or x.device != weight.device
        or x.shape[-1] != weight.shape[-1]
    ):
        raise RuntimeError("Target Q2..5 Lt call lacks a matching prepared BF16 plan")
    # Native F.linear accepts noncontiguous leading dimensions; materialize
    # their unchanged BF16 values before applying the checked matrix layout.
    flat = x.reshape(rows, x.shape[-1]).contiguous()
    result = plan(flat)
    COUNTS[f"{kind}_locked_q{rows}"] += 1
    return result.reshape(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("vllm019_target_linear::linear", mutates_args=())
def target_linear(x: torch.Tensor, weight: torch.Tensor, kind: str) -> torch.Tensor:
    return _execute(x, weight, kind)


@target_linear.register_fake
def _target_linear_fake(x, weight, kind):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _make_apply_wrapper(original):
    @wraps(original)
    def apply(self, layer, x, bias=None):
        kind = getattr(layer, "_vllm019_target_linear_kind", None)
        if kind is None:
            return original(self, layer, x, bias)
        if (
            bias is not None
            or kind not in SHAPES
            or tuple(layer.weight.shape) != SHAPES[kind]
        ):
            raise RuntimeError(
                "Marked target projection contract changed; native Q2..5 fallback is forbidden"
            )
        return target_linear(x, layer.weight, kind)

    return apply


def _make_target_init(original, linear_type, embedding_type, head_type):
    @wraps(original)
    def init(self, *, vllm_config, prefix=""):
        original(self, vllm_config=vllm_config, prefix=prefix)
        _mark_target(self, vllm_config, linear_type, embedding_type, head_type)

    return init


def _make_load_weights(original):
    @wraps(original)
    def load_weights(self, weights):
        if getattr(self, "_vllm019_target_linear_prepared", False):
            raise RuntimeError(
                "Reloading target weights after Lt preparation is unsupported"
            )
        result = original(self, weights)
        _prepare_target(self)
        return result

    return load_weights


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _check_runtime()
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        ParallelLMHead,
    )
    from vllm.model_executor.models.paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration as Target,
    )

    Target.__init__ = _make_target_init(
        Target.__init__,
        UnquantizedLinearMethod,
        UnquantizedEmbeddingMethod,
        ParallelLMHead,
    )
    Target.load_weights = _make_load_weights(Target.load_weights)
    UnquantizedLinearMethod.apply = _make_apply_wrapper(UnquantizedLinearMethod.apply)
    UnquantizedEmbeddingMethod.apply = _make_apply_wrapper(
        UnquantizedEmbeddingMethod.apply
    )
    _INSTALLED = True


def metadata():
    return {
        "installed": _INSTALLED,
        "scope": "Marked target decoder + actual target LM head; Q2..5 image HTTP verification",
        "unsupported_scope": "Extremely short text-only prefills/final prefill chunks of 2..5 tokens",
        "target_arithmetic": "CUDA12 native-Q1 recipes; BF16 weights/output, native reduction schemes",
        "native_queries": "Q1 and >5-row image prefills",
        "environment": "vLLM0.19.0/torch2.10.0/cuBLAS12.8.4.1/A10080GB-SM80-108SM",
        "recipes": {kind: dict(recipe) for kind, recipe in RECIPES.items()},
        "persistent_plans": len(PLANS),
        "prepared_query_shapes": sum(len(plan.plans) for plan in PLANS.values()),
        "workspace_bytes": sum(plan.workspace.numel() for plan in PLANS.values()),
        "counts": dict(COUNTS),
        "counter_scope": "Python eager/warmup/capture dispatch, not CUDA graph replay",
    }
