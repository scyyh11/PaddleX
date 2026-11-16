# Third-party notices

The optional packed attention build under `kernels/fa2_sm80/` uses pinned
FlashAttention and CUTLASS sources and PyTorch headers. Its
[NOTICE](kernels/fa2_sm80/NOTICE.md) and [license directory](kernels/fa2_sm80/LICENSES)
retain upstream attribution and redistribution terms. The build copies these
notices alongside the shared library; they must accompany binary distribution.

The draft graph adapters (`scripts/vllm019_draft_loop.py`,
`vllm019_draft_full_graph.py`, and `vllm019_draft_graph_buckets.py`) adapt the
input preparation and autoregressive recurrence of vLLM's
[`vllm/v1/spec_decode/eagle.py`](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/spec_decode/eagle.py).
The speculative metadata and greedy selection adapters preserve the contracts
of that release's GPU model runner and rejection sampler.
Copyright contributors to the vLLM project; licensed under Apache-2.0.
The adaptations add guarded CUDA graph capture, integer metadata fusion and
fixed-shape index reuse. The Apache-2.0 license is included in this repository's
root `LICENSE` file.

`scripts/vllm019_ar1_attention.py::ar1_num_splits` adapts the split heuristic from
FlashAttention commit `29210221863736a08f71a866459e368ad1ac4a95`,
[`csrc/flash_attn/flash_api.cpp`](https://github.com/vllm-project/flash-attention/blob/29210221863736a08f71a866459e368ad1ac4a95/csrc/flash_attn/flash_api.cpp).
The original license is retained below.

```text
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
