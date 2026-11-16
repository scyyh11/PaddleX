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
from vllm019_draft_graph_buckets import (
    install as install_draft_loop,
    metadata as draft_loop_metadata,
)
from vllm019_greedy_sampler import (
    install as install_greedy_sampler,
    COUNTS as SAMPLER_COUNTS,
)
from vllm019_spec_metadata import (
    install as install_spec_metadata,
    COUNTS as SPEC_METADATA_COUNTS,
)
from vllm019_draft_staging import (
    install as install_draft_staging,
    metadata as staging_metadata,
)
from vllm019_sampler_graph import (
    install as install_sampler_graph,
    metadata as sampler_graph_metadata,
)
from vllm019_vision_scalar import (
    install as install_vision_scalar,
    metadata as vision_scalar_metadata,
)

install()
install_draft_loop()
install_greedy_sampler()
install_spec_metadata()
install_draft_staging()
install_sampler_graph()
install_vision_scalar()


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
        # Native dummy_run captures forward only. Our complete proposal graphs
        # capture top-1 too, lazily on real decode, so startup counts can be zero.
        data.update(
            runtime_profile="vllm-0.19.0-eagle3-bucketed",
            draft_head_execution="int8 top-1",
            draft_head_prepared=True,
            draft_top1_in_forward_graph=False,
            draft_top1_in_proposal_graph=True,
            draft_head_startup=metadata(),
            draft_loop=draft_loop_metadata(proposer),
            draft_execution="complete Q5 proposal graphs with context buckets and guarded fallbacks",
            target_sampler="FP32 stable argmax graph and strict greedy integer selection with upstream fallback",
            spec_metadata="fixed B1 K4 Q5 GPU indices with upstream fallback",
            target_sampler_calls=dict(SAMPLER_COUNTS),
            spec_metadata_calls=dict(SPEC_METADATA_COUNTS),
            draft_staging=staging_metadata(),
            target_argmax=sampler_graph_metadata(),
            vision_scalar=vision_scalar_metadata(),
        )
        return data
