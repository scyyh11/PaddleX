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

"""Private INT8 draft top-1 on vLLM 0.19.0's native greedy proposer."""
from hps_vllm019_worker import Vllm019EagleWorker
from round3_eagle_fused_int8_top1 import install, metadata
from round3_eagle_fused_int8_head import PLANS

install()


class Vllm019Int8Worker(Vllm019EagleWorker):
    def hps_eagle_metadata(self):
        data = super().hps_eagle_metadata()
        proposer = self.model_runner.drafter
        model = proposer.model
        weight = model.lm_head.weight
        plan = PLANS.get(weight.data_ptr())
        if (
            not proposer.use_local_argmax_reduction
            or model.draft_id_to_target_id is None
            or plan is None
            or plan["private_vocab_size"] != model.config.draft_vocab_size
            or plan["target_vocab_size"] != model.config.vocab_size
        ):
            raise RuntimeError("vLLM 0.19.0 private INT8 draft head was not prepared")
        # In 0.19 dummy_run captures draft.forward only. The top-1 call happens
        # in the proposer outside that graph, so zero startup dispatches is valid.
        data.update(
            draft_head_execution="int8 top-1",
            draft_head_prepared=True,
            draft_top1_in_forward_graph=False,
            draft_head_startup=metadata(),
        )
        return data
