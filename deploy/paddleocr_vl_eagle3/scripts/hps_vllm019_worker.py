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

The target keeps native BF16 linear layers and native FA2 AR1 query rows for
small verification blocks. A100 80GB/108-SM devices batch these rows with the
original AR1 split count; other devices use per-query calls. CUDA 13 cuBLASLt
recipes and FA4 hooks are not used. Regressions compare against a 0.19 baseline.
"""
from vllm019_model import install
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm019_ar1_attention import COUNTS, install as install_ar1_attention

install()
install_ar1_attention()


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
            "target_projection": "native vLLM 0.19.0 BF16",
            "target_attention": "native FA2 AR1 row batch with per-query fallback",
            "attention_row_batch_device": "A100 80GB, SM80, 108 SMs",
            "ar1_attention_startup": dict(COUNTS),
            "attention_counter_scope": "Python initialization/capture calls, not graph replay counts",
            "cuda13_locked_kernels": False,
        }
        expected = {
            "method": "eagle3",
            "speculator": "EagleProposer",
            "target": "PaddleOCRVLForConditionalGeneration",
            "draft": "Eagle3LlamaForCausalLM",
            "draft_head_dtype": "torch.bfloat16",
            "k": 4,
            "aux_layers": [4, 10, 16],
            "async_scheduling": False,
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
        ):
            raise RuntimeError(f"Unexpected vLLM 0.19.0 EAGLE3 worker: {data}")
        return data
