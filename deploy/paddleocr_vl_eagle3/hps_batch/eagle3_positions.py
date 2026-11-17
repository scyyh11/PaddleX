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

"""Keep this trained draft's absolute token positions separate from target M-RoPE.

The dense VL-1.6 draft was trained with ordinary positions 0, 1, ... across
both image and text tokens. Native vLLM 0.19 instead copies the target's first
M-RoPE axis into a text-only draft. An image makes that axis differ from the
absolute token ordinal. Besides mismatching training, its later native EAGLE
steps use the same position to address draft KV slots.

Reconstruct the draft's positions from the native per-request attention
metadata. Target M-RoPE, token shifts, rejection handling, attention, sampling,
and subsequent native EAGLE steps remain unchanged. No CPU/GPU synchronization
or device-specific kernel is introduced.
"""

import functools


def write_token_ordinal_positions(output, query_start_loc, seq_lens, token_arange):
    """Fill one flattened batch using each request's absolute query interval.

    ``seq_lens`` includes the current target query and any rejected-token
    padding. The native proposer independently selects the last accepted
    position and subtracts rejected lengths before recurrent drafting.
    """
    import torch

    num_tokens = output.shape[0]
    flat_indices = token_arange[:num_tokens]
    # right=True assigns a query boundary to the next request. All boundaries
    # and lengths stay on device, including asynchronous scheduler corrections.
    request_indices = torch.bucketize(flat_indices, query_start_loc[1:], right=True)
    positions = (
        flat_indices + seq_lens[request_indices] - query_start_loc[1:][request_indices]
    )
    output.copy_(positions)


def uses_trained_paddle_position_contract(proposer):
    target = proposer.vllm_config.model_config
    return (
        proposer.method == "eagle3"
        and getattr(target.hf_config, "model_type", None) == "paddleocr_vl"
        and target.uses_mrope
        and not proposer.uses_mrope
    )


def install_proposer_hook(proposer_cls):
    original = proposer_cls.set_inputs_first_pass
    if getattr(original, "_paddlex_hps_eagle3_positions", False):
        return

    @functools.wraps(original)
    def first_pass(
        self,
        target_token_ids,
        next_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        cad,
        num_rejected_tokens_gpu,
    ):
        scoped = uses_trained_paddle_position_contract(self)
        if scoped and self.needs_extra_input_slots:
            raise ValueError(
                "PaddleOCR draft token positions require sequential EAGLE3"
            )
        result = original(
            self,
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
        )
        if scoped:
            num_tokens, _, native_metadata = result
            write_token_ordinal_positions(
                self.positions[:num_tokens],
                native_metadata.query_start_loc,
                native_metadata.seq_lens,
                self.arange,
            )
        return result

    first_pass._paddlex_hps_eagle3_positions = True
    proposer_cls.set_inputs_first_pass = first_pass


def install():
    from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

    install_proposer_hook(SpecDecodeBaseProposer)
