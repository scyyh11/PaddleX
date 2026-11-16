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

"""Use the existing INT8 top-1 implementation only on the EAGLE draft head."""
from hps_decode_graph_eagle_worker import DecodeGraphEagleWorker
from round3_eagle_fused_int8_top1 import install, metadata

install()


class DecodeGraphInt8Worker(DecodeGraphEagleWorker):
    def hps_eagle_metadata(self):
        data = super().hps_eagle_metadata()
        head = metadata()
        if (
            not self.model_runner.speculator.use_local_argmax_reduction
            or not head["top1_dispatches"]
        ):
            raise RuntimeError(
                "INT8 draft top-1 was not used during decode graph capture"
            )
        data["draft_head_execution"] = "int8 top-1"
        return data
