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

"""Enable Paddle's auxiliary decoder states and AR1 target verification for HPS."""
from round3_fast_engine_locked_target import install

# Install before vLLM traces the target. The module also installs per-row FA2
# for target contexts up to 4096. Longer contexts use native vLLM attention.
install()


class NativeEagleWorker:
    def hps_eagle_metadata(self):
        runner = self.model_runner
        speculator = runner.speculator
        draft = speculator.model
        config = speculator.vllm_config
        metadata = {
            "method": config.speculative_config.method,
            "speculator": type(speculator).__name__,
            "target": type(runner.model).__name__,
            "draft": type(draft).__name__,
            "checkpoint": speculator.draft_model_config.model,
            "draft_head_dtype": str(draft.lm_head.weight.dtype),
            "k": config.speculative_config.num_speculative_tokens,
            "aux_layers": list(runner.model._ocr_native_aux_layers),
            "eager": config.model_config.enforce_eager,
            "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
            "async_scheduling": config.scheduler_config.async_scheduling,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "max_model_len": config.model_config.max_model_len,
            "draft_position_embeddings": draft.config.max_position_embeddings,
            "target_projection": "AR1-ordered cuBLASLt for small verification blocks",
            "target_attention": "per-row FA2 to 4096 tokens, native FA2 beyond 4096",
        }
        expected = {
            "method": "eagle3",
            "speculator": "EagleSpeculator",
            "target": "PaddleOCRVLForConditionalGeneration",
            "draft": "Eagle3LlamaForCausalLM",
            "draft_head_dtype": "torch.bfloat16",
            "k": 4,
            "aux_layers": [4, 10, 16],
            "eager": False,
            "cudagraph_mode": "NONE",
            "async_scheduling": False,
            "max_num_seqs": 8,
        }
        if (
            not all(metadata[key] == value for key, value in expected.items())
            or metadata["draft_position_embeddings"] < metadata["max_model_len"]
        ):
            raise RuntimeError(f"Unexpected HPS recovery worker: {metadata}")
        return metadata
