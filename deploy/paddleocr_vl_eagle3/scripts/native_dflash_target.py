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

"""PaddleOCR-VL auxiliary-state adapter used by the EAGLE3 worker.

The historical filename is retained to make the exported runtime traceable to
the experiments. This module only installs the target EAGLE3 contract; DFlash
instrumentation and the old final-state-only experiment are not exported.
"""


def install():
    from vllm.model_executor.models.paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration as Target,
    )

    if getattr(Target, "_ocr_native_dflash_adapter", False):
        return
    original_forward = Target.forward

    def set_aux(self, layers):
        layers = tuple(layers)
        if layers != (4, 10, 16):
            raise ValueError("This checkpoint requires auxiliary layer IDs (4, 10, 16)")
        self.language_model.model._set_aux_hidden_state_layers(layers)
        self._ocr_native_aux_layers = layers
        self._ocr_native_return_aux = True

    def default_aux(self):
        return (4, 10, 16)

    def forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        if not getattr(self, "_ocr_native_return_aux", False):
            return result
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or len(result[1]) != len(self._ocr_native_aux_layers)
        ):
            raise RuntimeError("PaddleOCR-VL did not return all three EAGLE3 states")
        return result

    Target.supports_eagle3 = True
    Target.has_own_lm_head = False
    Target.has_own_embed_tokens = False
    Target.set_aux_hidden_state_layers = set_aux
    Target.get_eagle3_default_aux_hidden_state_layers = default_aux
    Target.forward = forward
    Target._ocr_native_dflash_adapter = True


install()
