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

"""EAGLE3 worker contract for vLLM 0.19.0's V1 EagleProposer.

The target keeps BF16 weights, with verified CUDA12 native-Q1 linear plans for
small verification blocks and FA2 AR1 query endpoints/split counts. The optional
finite-QKV packed kernel is admitted separately. This worker requires the pinned
A100 80GB/108-SM profile; CUDA13 recipes and FA4 hooks are not used.
Regressions compare against the same vLLM 0.19 autoregressive baseline.
"""
import os

from vllm019_model import install
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm019_ar1_attention import COUNTS, install as install_ar1_attention
from vllm019_target_linear import (
    install as install_target_linear,
    metadata as target_linear_metadata,
)
from vllm019_packed_attention import (
    LIBRARY_ENV,
    install as install_packed_attention,
    metadata as packed_attention_metadata,
)

install()
install_ar1_attention()
install_target_linear()
if os.environ.get(LIBRARY_ENV):
    install_packed_attention()


class Vllm019EagleWorker:
    def hps_eagle_metadata(self):
        runner = self.model_runner
        proposer = runner.drafter
        config = proposer.vllm_config
        draft = proposer.model
        target = runner.model
        while isinstance(target, CUDAGraphWrapper):
            target = target.unwrap()
        data = {
            "runtime_profile": "vllm-0.19.0-ar1-attention-target",
            "method": config.speculative_config.method,
            "speculator": type(proposer).__name__,
            "target": type(target).__name__,
            "draft": type(draft).__name__,
            "checkpoint": proposer.draft_model_config.model,
            "k": config.speculative_config.num_speculative_tokens,
            "aux_layers": list(target._ocr_native_aux_layers),
            "draft_head_dtype": str(draft.lm_head.weight.dtype),
            "draft_head_execution": "bf16",
            "max_model_len": config.model_config.max_model_len,
            "draft_position_embeddings": draft.config.max_position_embeddings,
            "async_scheduling": config.scheduler_config.async_scheduling,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            "draft_cudagraph_mode": proposer.cudagraph_dispatcher.cudagraph_mode.name,
            "capture_sizes": config.compilation_config.cudagraph_capture_sizes,
            "target_projection": "CUDA12 native-Q1 BF16 recipes for marked target Q2..5",
            "target_linear": target_linear_metadata(),
            "target_attention": "native FA2 AR1 row batch with per-query fallback",
            "attention_metadata": "fused exact integer causal endpoints and page tables",
            "attention_row_batch_device": "A100 80GB, SM80, 108 SMs",
            "ar1_attention_startup": dict(COUNTS),
            "attention_counter_scope": "Python initialization/capture calls, not graph replay counts",
            "cuda13_locked_kernels": False,
        }
        packed = packed_attention_metadata()
        data["packed_attention"] = packed
        if packed["enabled"]:
            data["target_attention"] = (
                "opt-in finite-QKV packed FA2 with original exact AR1 fallback; "
                "see packed_attention counts for startup/capture usage"
            )
        expected = {
            "method": "eagle3",
            "speculator": "EagleProposer",
            "target": "PaddleOCRVLForConditionalGeneration",
            "draft": "Eagle3LlamaForCausalLM",
            "draft_head_dtype": "torch.bfloat16",
            "k": 4,
            "aux_layers": [4, 10, 16],
        }
        if (
            not all(data[key] == value for key, value in expected.items())
            or data["cudagraph_mode"]
            not in ("NONE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
            or data["draft_cudagraph_mode"]
            != (
                "PIECEWISE"
                if data["cudagraph_mode"] == "FULL_AND_PIECEWISE"
                else "NONE"
            )
            or data["draft_position_embeddings"] < data["max_model_len"]
            or (data["async_scheduling"] and data["max_num_seqs"] != 1)
        ):
            raise RuntimeError(f"Unexpected vLLM 0.19.0 EAGLE3 worker: {data}")
        return data
