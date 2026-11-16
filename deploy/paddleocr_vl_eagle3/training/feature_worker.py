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

"""Collect real vLLM teacher-forced states; eager, TP1 and one request only."""
from pathlib import Path
import torch


class FeatureCaptureExtension:
    def install_feature_capture(self, layers=(4, 10, 16)):
        r = self.model_runner
        if not r.vllm_config.model_config.enforce_eager:
            raise ValueError("Feature collection requires eager execution")
        inner = r.model.language_model.model
        inner._set_aux_hidden_state_layers(tuple(layers))
        state = {"active": False, "chunks": [], "batch": None, "layers": list(layers)}
        self._ocr_features = state
        original_batch = r.step_timing.record_batch

        def record_batch(batch, full_cudagraph):
            if state["active"]:
                if batch.num_reqs != 1:
                    raise ValueError(
                        "Only a single teacher-forced request is supported"
                    )
                n = batch.num_tokens
                state["batch"] = (
                    batch.positions[:n].detach().cpu().clone(),
                    batch.input_ids[:n].detach().cpu().clone(),
                )
            return original_batch(batch, full_cudagraph)

        r.step_timing.record_batch = record_batch
        original_forward = r.model.forward

        def forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            hidden, aux = result
            if state["active"]:
                positions, tokens = state["batch"]
                n = len(positions)
                if len(aux) != len(layers):
                    raise RuntimeError("Missing real intermediate hidden states")
                state["chunks"].append(
                    (
                        positions,
                        tokens,
                        hidden[:n].detach().cpu().clone(),
                        torch.stack(aux, 1)[:n].detach().cpu().clone(),
                    )
                )
            return hidden

        r.model.forward = forward
        return {
            "runner": type(r).__module__,
            "layers": list(layers),
            "semantics": "residual stream after zero-based decoder layers 3/9/15; final normalized output separately",
        }

    def feature_capture_begin(self):
        self._ocr_features.update(active=True, chunks=[], batch=None)
        return True

    def feature_capture_save(self, path, metadata):
        s = self._ocr_features
        s["active"] = False
        length = len(metadata["token_ids"])
        final = torch.empty(length, 1024, dtype=torch.bfloat16)
        aux = torch.empty(length, len(s["layers"]), 1024, dtype=torch.bfloat16)
        seen = torch.zeros(length, dtype=torch.bool)
        expected = torch.tensor(metadata["token_ids"], dtype=torch.long)
        for positions, tokens, h, a in s["chunks"]:
            valid = (positions >= 0) & (positions < length)
            positions, tokens, h, a = (
                positions[valid].long(),
                tokens[valid],
                h[valid],
                a[valid],
            )
            if not torch.equal(expected[positions], tokens.long()):
                raise RuntimeError(
                    "Actual vLLM teacher-forced tokens differ from requested sequence"
                )
            if seen[positions].any():
                raise RuntimeError("Duplicate teacher-forced feature positions")
            final[positions], aux[positions], seen[positions] = h, a, True
        if not seen.all():
            raise RuntimeError(
                f"Missing {int((~seen).sum())} teacher feature positions"
            )
        result = {
            **metadata,
            "token_ids": expected,
            "features": final,
            "aux_features": aux,
            "positions": torch.arange(length),
            "aux_layer_ids": s["layers"],
            "feature_backend": "vllm 0.32 eager teacher forcing, same remote image processor as deployed target",
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(result, tmp)
        tmp.replace(path)
        s["chunks"] = []
        return {
            "path": str(path),
            "tokens": length,
            "final_shape": list(final.shape),
            "aux_shape": list(aux.shape),
            "all_positions_and_tokens_verified": True,
        }
