# Third-party notices

The build orchestration and Python adapter are copyright2026 PaddlePaddle
Authors and licensed under Apache-2.0; see LICENSE. Original and adapted CUDA
source retains the applicable upstream BSD-3-Clause notices below.

The attention algorithm, generated patched header and integer row mask derive
from [FlashAttention](https://github.com/vllm-project/flash-attention/tree/29210221863736a08f71a866459e368ad1ac4a95),
commit29210221863736a08f71a866459e368ad1ac4a95. Original individual source headers
include copyright2023/2024 Tri Dao. Its project license states copyright2022
the respective contributors; its complete LICENSE and AUTHORS are reproduced
in LICENSES/flash-attention.LICENSE and LICENSES/flash-attention.AUTHORS.

[CUTLASS](https://github.com/NVIDIA/cutlass/tree/62750a2b75c802660e4894434dc55e839f322277),
commit62750a2b75c802660e4894434dc55e839f322277, supplies CUTE/CUTLASS kernel types.
Copyright2017–2025 NVIDIA CORPORATION & AFFILIATES. Its complete BSD-3-Clause
license is reproduced in LICENSES/cutlass.LICENSE. Downloaded headers preserve
their original file notices.

PyTorch2.10.0 headers supply the original Philox type declarations. The shared
library does not link the Torch/c10 runtime. The installed distribution's full
license and notice are reproduced in LICENSES/pytorch.LICENSE and
LICENSES/pytorch.NOTICE to retain the applicable notices when distributing the
built artifact. PyTorch and the CUDA toolkit/runtime are installed separately.

Binary redistributions must accompany the library with this NOTICE.md and the
LICENSES directory copied into the build output. No upstream project or
contributor endorses this experimental adaptation.

The generated source changes only query packing/masking and skips loads of
trailing empty output partials. Its complete LSE reduction and all nonempty
split arithmetic remain the pinned upstream implementation. This description
is provenance, not a promise of numerical equivalence outside the documented
finite-input profile.
