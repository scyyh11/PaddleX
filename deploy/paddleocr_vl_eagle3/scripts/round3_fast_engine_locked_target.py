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

"""Experimental exact-AR arithmetic for small-query target verification.

Target BF16 weights/head remain unchanged. The default AR1 Lt algorithms are
fixed for QKV, gate/up and head. O/down use the original GEMV algorithm with
strided row batches whose individual matrices retain M=1 dimensions.
Only target 2..9-token calls are replaced;
single-token calls and the image prefills (>9 tokens) keep default kernels.
"""
from collections import defaultdict
import torch
from round3_locked_lt import LockedLt
import round3_attention_rows  # noqa: F401; installs attention and auxiliary-state adapters
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

RECIPES = {
    0: dict(algo_id=31, tile=15, stages=17, split_k=1, reduction=0, swizzle=0),
    1: dict(
        algo_id=13,
        tile=None,
        stages=None,
        split_k=None,
        reduction=None,
        swizzle=None,
        custom_option=75,
        row_batch=True,
    ),
    2: dict(algo_id=6, tile=15, stages=17, split_k=1, reduction=0, swizzle=0),
    3: dict(
        algo_id=13,
        tile=None,
        stages=None,
        split_k=None,
        reduction=None,
        swizzle=None,
        custom_option=75,
        row_batch=True,
    ),
    4: dict(algo_id=6, tile=18, stages=12, split_k=2, reduction=1, swizzle=1),
}
for _kind, _recipe in RECIPES.items():
    _recipe["workspace_bytes"] = 4096 if _kind == 4 else 0
NAMES = {0: "qkv", 1: "o", 2: "gateup", 3: "down", 4: "head"}
SHAPES = {
    (2560, 1024): 0,
    (1024, 2048): 1,
    (6144, 1024): 2,
    (1024, 3072): 3,
    (103424, 1024): 4,
}
PLANS = {}
COUNTS = defaultdict(int)
PREFIXES = set()
DTYPES = set()


@torch.library.custom_op("round3_locked_target::linear", mutates_args=())
def target_linear(x: torch.Tensor, w: torch.Tensor, kind: int) -> torch.Tensor:
    n = x.numel() // x.shape[-1]
    DTYPES.add((str(x.dtype), str(w.dtype)))
    assert x.dtype == w.dtype == torch.bfloat16
    if n == 1 or n > 9:
        COUNTS[f"{NAMES[kind]}_original_n{n}"] += 1
        return torch.nn.functional.linear(x, w)
    flat = x.reshape(n, x.shape[-1]).contiguous()
    COUNTS[f"{NAMES[kind]}_locked_n{n}"] += 1
    key = (w.data_ptr(), kind)
    if key not in PLANS:
        PLANS[key] = LockedLt(w, **RECIPES[kind])
    y = PLANS[key](flat)
    return y.reshape(*x.shape[:-1], w.shape[0])


@target_linear.register_fake
def target_linear_fake(x, w, kind):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def install():
    if getattr(UnquantizedLinearMethod, "_round3_locked_target", False):
        return
    original = UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        prefix = getattr(layer, "prefix", "")
        kind = SHAPES.get(tuple(layer.weight.shape))
        if (
            bias is None
            and kind is not None
            and prefix.startswith("language_model.model.layers.")
        ):
            PREFIXES.add(prefix)
            return target_linear(x, layer.weight, kind)
        return original(self, layer, x, bias)

    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._round3_locked_target = True
    original_head = UnquantizedEmbeddingMethod.apply

    def head(self, layer, x, bias=None):
        prefix = getattr(layer, "prefix", "")
        if (
            bias is None
            and prefix == "language_model.lm_head"
            and tuple(layer.weight.shape) == (103424, 1024)
        ):
            PREFIXES.add(prefix)
            return target_linear(x, layer.weight, 4)
        return original_head(self, layer, x, bias)

    UnquantizedEmbeddingMethod.apply = head


install()
