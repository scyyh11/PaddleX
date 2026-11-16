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

"""Speculative decode graphs managed by vLLM; vision stays on its native path."""
from round3_fast_engine_locked_target import install
import round3_attention_rows as rows
import torch
import triton

install()

_wide_row_dispatches = 0


def _install_wide_decode_attention():
    from vllm.v1.attention.backends.flash_attn import FA4DenseAttentionKernel

    original = FA4DenseAttentionKernel.kernel

    def kernel(*args, **kw):
        # Captured decode advertises the *engine* context bound, even when a
        # request's actual KV length is short. Preserve that full capacity.
        bound = kw.get("max_seqlen_k", 0)
        if args or not 4096 < bound <= 12288:
            return original(*args, **kw)
        q, k = kw.get("q"), kw.get("k")
        cu, tables, lengths = (
            kw.get("cu_seqlens_q"),
            kw.get("block_table"),
            kw.get("seqused_k"),
        )
        if (
            q is None
            or k is None
            or cu is None
            or tables is None
            or lengths is None
            or kw.get("fa_version") != 2
            or kw.get("causal") is not True
            or q.shape[0] != 5
            or kw.get("max_seqlen_q") != 5
            or q.shape[1:] != (16, 128)
            or k.shape[-2:] != (2, 128)
            or q.dtype != torch.bfloat16
            or cu.numel() != 2
            or lengths.numel() != 1
            or tables.shape[0] != 1
            or tables.stride(-1) != 1
            or kw.get("alibi_slopes") is not None
            or kw.get("softcap", 0) != 0
            or kw.get("return_softmax_lse", False)
            or kw.get("window_size") not in (None, (-1, -1), [-1, -1])
        ):
            return original(*args, **kw)
        n, cols = q.shape[0], tables.shape[1]
        key = (n, cols, q.device.index)
        if key not in rows._BUFFERS:
            opts = {"device": q.device, "dtype": torch.int32}
            rows._BUFFERS[key] = (
                torch.arange(n + 1, **opts),
                torch.zeros(n + 1, **opts),
                torch.empty(n, **opts),
                torch.empty((n, cols), **opts),
            )
        rowcu, dummy, rowlens, rowbt = rows._BUFFERS[key]
        rows._metadata[(n,)](
            cu,
            lengths,
            tables,
            rowlens,
            rowbt,
            1,
            cols,
            tables.stride(0),
            1,
            triton.next_power_of_2(cols),
            num_warps=4,
        )
        global _wide_row_dispatches
        _wide_row_dispatches += 1
        return torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            k,
            kw["v"],
            kw.get("out"),
            rowcu,
            dummy,
            rowlens,
            None,
            rowbt,
            None,
            1,
            bound,
            0.0,
            kw["softmax_scale"],
            False,
            True,
            -1,
            -1,
            0.0,
            False,
            32,
            None,
        )[0]

    FA4DenseAttentionKernel.kernel = staticmethod(kernel)


_install_wide_decode_attention()


class DecodeGraphEagleWorker:
    def hps_eagle_metadata(self):
        runner = self.model_runner
        proposer = runner.speculator
        config = proposer.vllm_config
        data = {
            "method": config.speculative_config.method,
            "speculator": type(proposer).__name__,
            "draft": type(proposer.model).__name__,
            "target": type(runner.model).__name__,
            "checkpoint": proposer.draft_model_config.model,
            "k": config.speculative_config.num_speculative_tokens,
            "aux_layers": list(runner.model._ocr_native_aux_layers),
            "max_model_len": config.model_config.max_model_len,
            "draft_position_embeddings": proposer.model.config.max_position_embeddings,
            "async_scheduling": config.scheduler_config.async_scheduling,
            "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            "capture_sizes": config.compilation_config.cudagraph_capture_sizes,
            "draft_head_execution": "bf16",
            "wide_target_row_dispatches_at_startup": _wide_row_dispatches,
        }
        if (
            data["method"] != "eagle3"
            or data["speculator"] != "EagleSpeculator"
            or data["target"] != "PaddleOCRVLForConditionalGeneration"
            or data["draft"] != "Eagle3LlamaForCausalLM"
            or data["k"] != 4
            or data["aux_layers"] != [4, 10, 16]
            or data["async_scheduling"]
            or data["cudagraph_mode"] != "FULL_DECODE_ONLY"
            or data["draft_position_embeddings"] < data["max_model_len"]
        ):
            raise RuntimeError(f"Unexpected native decode graph worker: {data}")
        return data
