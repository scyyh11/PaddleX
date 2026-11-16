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

"""Opt-in native Siglip max_seqlen: one CUDA scalar read per encoder forward.

The original SiglipEncoder.forward and original attention custom op still run.
Only the max_seqlen argument is replaced with a fresh CPU int32 scalar carrying
the value of the actual GPU scalar. The remaining 26 native .item() calls read
CPU memory, preserving the native FA argument, wrapper, backend and kernels.

There is no shape-derived length or cross-forward cache. A ContextVar gives each
native encoder invocation its own cache; nested invocations and exceptions reset
it correctly. Cache hits require identical strong-owned scalar and cu_seqlens
Tensor objects. Unsupported calls keep the original path. Import installs no
hooks. Native method identities and dispatch bindings are checked before reuse;
this relies on the pinned native forward not mutating metadata between layers.
Object identity alone does not detect in-place Tensor value changes.
This does not make the whole vision forward CUDA-graph-capturable: the
first scalar read and native pids.max/arange metadata sync still remain.
"""

from contextvars import ContextVar
from functools import wraps
import importlib.metadata
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

_ACTIVE = ContextVar("paddle_v019_vision_scalar_scope", default=None)
_INSTALLATION = None
COUNTS = {
    "encoder_scopes": 0,
    "encoder_fallbacks": 0,
    "gpu_scalar_reads": 0,
    "scalar_cache_hits": 0,
    "native_attention_calls": 0,
    "attention_fallbacks": 0,
    "metadata_identity_fallbacks": 0,
}


class _ScalarScope:
    def __init__(self):
        self.source = None
        self.cu_seqlens = None
        self.cpu_scalar = None

    def resolve(self, source, cu_seqlens):
        if self.source is None:
            # No CPU shape assumption: read the scalar produced by this very
            # forward's native (cu_seqlens[1:] - cu_seqlens[:-1]).max().
            value = source.item()
            self.cpu_scalar = torch.tensor(value, dtype=torch.int32, device="cpu")
            self.source = source
            self.cu_seqlens = cu_seqlens
            COUNTS["gpu_scalar_reads"] += 1
        elif source is not self.source or cu_seqlens is not self.cu_seqlens:
            COUNTS["metadata_identity_fallbacks"] += 1
            return None
        else:
            COUNTS["scalar_cache_hits"] += 1
        return self.cpu_scalar


def _native_module(module):
    return (
        not module._forward_pre_hooks
        and not module._forward_hooks
        and getattr(module, "_compiled_call_impl", None) is None
        and "forward" not in vars(module)
    )


def _bound_native(instance, name, function):
    method = getattr(instance, name, None)
    return (
        getattr(method, "__self__", None) is instance
        and getattr(method, "__func__", None) is function
    )


def _native_python_function(function, module, qualname):
    """Reject ordinary preexisting monkey patches, not authenticate source bytes."""
    return (
        inspect.isfunction(function)
        and not hasattr(function, "__wrapped__")
        and function.__module__ == module.__name__
        and function.__qualname__ == qualname
        and function.__code__.co_name == qualname.rsplit(".", 1)[-1]
        and Path(function.__code__.co_filename).resolve()
        == Path(module.__file__).resolve()
    )


def _owned_hooks_intact(state):
    return (
        state.encoder_type.forward is state.wrapped_forward
        and state.attention_module.vit_flash_attn_wrapper is state.wrapped_attention
    )


def _native_methods_intact(state):
    return (
        state.layer_type.forward is state.layer_forward
        and state.attention_type.forward is state.self_attention_forward
        and state.mm_type.forward is state.mm_forward
        and state.mm_type.forward_cuda is state.mm_forward_cuda
        and state.mm_type._forward_fa is state.mm_forward_fa
        and state.wrappers_module.vit_flash_attn_wrapper is state.original_attention
    )


def _encoder_eligible(encoder, state):
    from torch.nn.modules import module as nn_module

    if (
        not _owned_hooks_intact(state)
        or not _native_methods_intact(state)
        or type(encoder) is not state.encoder_type
        or not _bound_native(encoder, "forward", state.wrapped_forward)
        or encoder.attn_backend != state.backend
        or len(encoder.layers) != 27
        or encoder.config.hidden_size != 1152
        or encoder.config.num_attention_heads != 16
        or not _native_module(encoder)
        or nn_module._global_forward_pre_hooks
        or nn_module._global_forward_hooks
        or torch.compiler.is_compiling()
    ):
        return False
    return all(
        type(layer) is state.layer_type
        and type(layer.self_attn) is state.attention_type
        and type(layer.self_attn.attn) is state.mm_type
        and _bound_native(layer, "forward", state.layer_forward)
        and _bound_native(layer.self_attn, "forward", state.self_attention_forward)
        and _bound_native(layer.self_attn.attn, "forward", state.mm_forward)
        and _bound_native(layer.self_attn.attn, "forward_cuda", state.mm_forward_cuda)
        and _bound_native(layer.self_attn.attn, "_forward_fa", state.mm_forward_fa)
        and _bound_native(
            layer.self_attn.attn, "_forward_method", state.mm_forward_cuda
        )
        and layer.self_attn.tp_size == 1
        and layer.self_attn.attn.attn_backend == state.backend
        and layer.self_attn.attn._fa_version == 2
        and all(
            _native_module(module)
            for module in (layer, layer.self_attn, layer.self_attn.attn)
        )
        for layer in encoder.layers
    )


def _attention_eligible(
    q, k, v, batch_size, is_rocm_aiter, fa_version, cu_seqlens, max_seqlen
):
    if batch_size != 1 or is_rocm_aiter or fa_version != 2:
        return False
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v, cu_seqlens, max_seqlen)):
        return False
    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.ndim == 4
        and q.shape[0] == 1
        and q.shape[1] > 0
        and tuple(q.shape[2:]) == (16, 72)
        and all(
            t.device == q.device and t.dtype == q.dtype and t.shape == q.shape
            for t in (k, v)
        )
        and cu_seqlens.device == q.device
        and cu_seqlens.dtype == torch.int32
        and tuple(cu_seqlens.shape) == (2,)
        and cu_seqlens.is_contiguous()
        and max_seqlen.device == q.device
        and max_seqlen.dtype == torch.int32
        and max_seqlen.ndim == 0
    )


def _wrap_encoder(original, eligible):
    @wraps(original)
    def forward(self, *args, **kwargs):
        if not eligible(self):
            COUNTS["encoder_fallbacks"] += 1
            # Mask a surrounding scope even for unsupported nested encoders.
            token = _ACTIVE.set(None)
        else:
            COUNTS["encoder_scopes"] += 1
            token = _ACTIVE.set(_ScalarScope())
        try:
            return original(self, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    forward._paddle_v019_vision_scalar_once = True
    return forward


def _wrap_attention(original):
    @wraps(original)
    def attention(
        q,
        k,
        v,
        batch_size,
        is_rocm_aiter,
        fa_version,
        scale=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        scope = _ACTIVE.get()
        replacement = None
        if scope is not None and _attention_eligible(
            q, k, v, batch_size, is_rocm_aiter, fa_version, cu_seqlens, max_seqlen
        ):
            replacement = scope.resolve(max_seqlen, cu_seqlens)
        if replacement is None:
            COUNTS["attention_fallbacks"] += 1
        else:
            COUNTS["native_attention_calls"] += 1
            max_seqlen = replacement
        # Preserve the original vllm::flash_attn_maxseqlen_wrapper custom op,
        # including its own .item(), einops views, kwargs and FA implementation.
        return original(
            q=q,
            k=k,
            v=v,
            batch_size=batch_size,
            is_rocm_aiter=is_rocm_aiter,
            fa_version=fa_version,
            scale=scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    attention._paddle_v019_vision_scalar_once = True
    return attention


def install():
    """Explicit worker-side installation; call before the first measured image."""
    global _INSTALLATION
    if _INSTALLATION is not None:
        if not (
            _owned_hooks_intact(_INSTALLATION) and _native_methods_intact(_INSTALLATION)
        ):
            raise RuntimeError(
                "Vision scalar hooks or native methods were replaced after installation"
            )
        return metadata()
    if importlib.metadata.version("vllm") != "0.19.0":
        raise RuntimeError("vision scalar experiment requires vLLM 0.19.0")
    from vllm.model_executor.models import paddleocr_vl as model_module
    from vllm.model_executor.layers.attention import (
        mm_encoder_attention as attention_module,
    )
    from vllm.model_executor import custom_op
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.ops import vit_attn_wrappers

    state = SimpleNamespace(
        encoder_type=model_module.SiglipEncoder,
        layer_type=model_module.SiglipEncoderLayer,
        attention_type=model_module.SiglipAttention,
        mm_type=attention_module.MMEncoderAttention,
        attention_module=attention_module,
        wrappers_module=vit_attn_wrappers,
        backend=AttentionBackendEnum.FLASH_ATTN,
        original_forward=model_module.SiglipEncoder.forward,
        original_attention=attention_module.vit_flash_attn_wrapper,
        layer_forward=model_module.SiglipEncoderLayer.forward,
        self_attention_forward=model_module.SiglipAttention.forward,
        mm_forward=attention_module.MMEncoderAttention.forward,
        mm_forward_cuda=attention_module.MMEncoderAttention.forward_cuda,
        mm_forward_fa=attention_module.MMEncoderAttention._forward_fa,
    )
    if state.original_attention is not vit_attn_wrappers.vit_flash_attn_wrapper:
        raise RuntimeError("MMEncoderAttention already has a nonnative FA wrapper")
    expected_functions = (
        (state.original_forward, model_module, "SiglipEncoder.forward"),
        (state.layer_forward, model_module, "SiglipEncoderLayer.forward"),
        (state.self_attention_forward, model_module, "SiglipAttention.forward"),
        (state.mm_forward, custom_op, "CustomOp.forward"),
        (state.mm_forward_cuda, attention_module, "MMEncoderAttention.forward_cuda"),
        (state.mm_forward_fa, attention_module, "MMEncoderAttention._forward_fa"),
        (state.original_attention, vit_attn_wrappers, "vit_flash_attn_wrapper"),
    )
    if not all(_native_python_function(*entry) for entry in expected_functions):
        raise RuntimeError(
            "Vision scalar reuse requires the unmodified native Python methods"
        )
    state.wrapped_forward = _wrap_encoder(
        state.original_forward, lambda encoder: _encoder_eligible(encoder, state)
    )
    state.wrapped_attention = _wrap_attention(state.original_attention)
    state.encoder_type.forward = state.wrapped_forward
    state.attention_module.vit_flash_attn_wrapper = state.wrapped_attention
    _INSTALLATION = state
    return metadata()


def uninstall():
    """Restore only our own hooks; no GPU synchronization or memory changes."""
    global _INSTALLATION
    if _INSTALLATION is None:
        return
    state = _INSTALLATION
    if not _owned_hooks_intact(state):
        raise RuntimeError(
            "Refusing to overwrite hooks installed after this experiment"
        )
    state.encoder_type.forward = state.original_forward
    state.attention_module.vit_flash_attn_wrapper = state.original_attention
    _INSTALLATION = None


def metadata():
    owned = _INSTALLATION is not None and _owned_hooks_intact(_INSTALLATION)
    native = _INSTALLATION is not None and _native_methods_intact(_INSTALLATION)
    return {
        "installed": _INSTALLATION is not None,
        "enabled": owned and native,
        "hook_identity_ok": owned,
        "native_method_identity_ok": native,
        "scope": "native PaddleOCR-VL Siglip 27 layers, TP1, BF16 H16 D72, FA2, B1",
        "max_seqlen_source": "actual CUDA scalar; one item per encoder forward",
        "cache_lifetime": "fresh per-forward ContextVar; no cross-forward cache",
        "attention_path": "original custom op with CPU int32 scalar",
        "gpu_workspace_bytes": 0,
        "cuda_graph_supported": False,
        **COUNTS,
    }
