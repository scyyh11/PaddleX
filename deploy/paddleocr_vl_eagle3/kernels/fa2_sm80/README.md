# PaddleOCR-VL: experimental FA2 packed queries for vLLM 0.19

This optional build bundle targets the existing PaddleOCR-VL-1.6 EAGLE3 service
on one A10080GB. It builds a separate CUDA library and Python adapter; it does
not modify the installed vLLM package or replace attention automatically.

The rebuilt library has passed component bitwise tests and finite crop/page
regressions through the service entrypoint. It remains explicitly opt-in with
the numerical boundary documented below. No all-document accuracy guarantee
or stable 2× service speedup is claimed by this bundle. See the service's
[validation report](../../benchmarks/vllm019_speed_optimization.json).

## Build

Use the existing pinned runtime: Linux x86-64, Python3.12, vLLM0.19.0,
PyTorch2.10.0+cu128, and the CUDA12.8 development toolkit with **nvcc12.8.93**.
The CUDA runtime libraries and PyTorch CUDA headers must be installed. This
script requires no Python build dependencies beyond the standard library and
the already installed PyTorch distribution. It does not execute GPU code.

```bash
python build.py --output-dir build
```

If nvcc is not on PATH, supply `--nvcc /path/to/cuda-12.8/bin/nvcc`.
The script downloads the two official pinned archives, checks their SHA256,
extracts only headers/notices, generates the exact validated patched header,
and builds SM80 with the original FA2 numerical compiler flags. It refuses
unrecognized source, toolchain or extra NVCC environment flags.

For an air-gapped build, first obtain these exact archives and place them in
a cache directory, then use `--offline --cache-dir /path/to/cache`:

| Dependency | Commit | Archive SHA256 |
| --- | --- | --- |
| FlashAttention | `29210221863736a08f71a866459e368ad1ac4a95` | `fe98fd6aa07e30203ee0bb9619cb53f30703f71c22bbfa91909aa3b43b7018f1` |
| CUTLASS | `62750a2b75c802660e4894434dc55e839f322277` | `78816d6c6d97793b5b59ef2a702174cb85b78dfcefc8fe2489964de2e42f17d2` |

The filenames are `<dependency>-<commit>.tar.gz`. Official download URLs are
recorded in `build.py`. `--prepare-only` verifies/extracts sources and produces
the patched header without compiling. Use a fresh output directory for every
build; do not replace a library referenced by active CUDA graphs.

Outputs include `libpaddleocr_vl_fa2_sm80.so`, `build_manifest.json`, the patched
header, build log and required license notices. Keep the library, manifest and
license materials together when deploying. CUDA runtime libraries are provided
by the pinned inference environment, not redistributed by this bundle.

## Adapter interface

```python
from packed_fa2 import PackedFA2

# Construct once, before graph capture; no per-request plan construction.
plan = PackedFA2(
    queries=5,
    max_seqlen_k=12288,
    block_table_columns=block_table.shape[1],
    device=q.device,
    library_path="build/libpaddleocr_vl_fa2_sm80.so",
)
out = plan(q, k, v, cu_seqlens_q, seqused_k, block_table, out=output_buffer)
```

The constructor checks artifact/source identity, pinned runtime, device and
profile, then allocates its workspaces. Calls read current GPU causal metadata
without a host synchronization. Every tensor is recorded on the current CUDA
stream so the caching allocator tracks in-flight use. Warmup, capture and
replay may use different streams with explicit caller-provided ordering.
Concurrent reuse of one plan is unsupported.

Keep the plan, its CDLL and all graph-bound external inputs/outputs alive until
all graphs referencing them have been retired. `record_stream` protects
in-flight allocator use; it does not make an input safe to destroy before a
future graph replay. The default output is persistent and overwritten by the
next call, so copy it when retaining results. Integration into the service
should keep plans in worker-owned state and use the existing output buffers.

Unsupported shapes/runtime are rejected. The integration layer must select
native attention before calling this adapter for unsupported profiles. This
bundle supplies no monkey patch and no global environment override.

## Numerical scope

- A10080GB, SM80,108 SMs; BF16 Q/K/V;16 query heads,2 KV heads, head dimension128.
- Single request, Q2–Q7, page size16, contiguous head dimension, aligned rows.
- Fixed context bound4096 or12288 and native scale `1/sqrt(128)`.
- GPU endpoints must not exceed the declared bound; page IDs must address the
  supplied KV cache. The caller/engine owns those metadata invariants.
- Finite Q/K/V only. There is no per-call GPU `isfinite` scan. Model-weight and
  representative activation finite checks belong to deployment validation.

The native AR1 attention evaluates each query as8 GQA rows in an M64 tile.
This candidate packs Q5/Q7 into40/56 rows of the same tile. It preserves the
native SM80 traits, QK/softmax/PV arithmetic and split count. The complete LSE
reduction tree remains unchanged. Only trailing completely empty Oaccum loads
are omitted; one explicit native FTZ zero FMA preserves the original tail's
signed-zero behavior.

**Known boundary:** a future query's V containing NaN/Inf can contaminate an
earlier packed query through `0×V`; individual native Q1 instead clears V
past that query's endpoint. This is a known counterexample. Neither the finite
input tests nor mathematical attention equivalence prove all-BF16-pattern or
all-document output equivalence.

## Validation record and deployment gate

The rebuilt public bundle passed 3656/3656 raw-bit differential checks:
1956 core checks, 224 zero/tiny/signed-zero comparisons, and 1476 additional
stride/nonzero-query-start checks. These cover Q2–Q7, changed-input CUDA graph
replay, random page mappings, input scales 0.1/1/5/10, zero length, padding,
128-key boundaries and long context. The reference independently constructs
per-query metadata and calls the original native FA2 operator. Rebuilding is
not itself GPU validation; rerun the public gate for the actual artifact:

```bash
# From deploy/paddleocr_vl_eagle3:
CUDA_VISIBLE_DEVICES=0 python tests/gpu_vllm019_packed_attention.py \
  --library kernels/fa2_sm80/build/libpaddleocr_vl_fa2_sm80.so \
  --output results/packed-fa2-parity.json
```

Before enabling a rebuilt library, rerun those component gates, then compare
crop token IDs/stop reasons, independent pages, long contexts and complete
DocLayoutV3/postprocessing results against the same baseline. Warm both engines
and include first-use graph capture costs separately. Report VLM-only and
whole-pipeline latency separately.

The service enables this library only with
`--vllm019-packed-attention-library PATH`. Its guarded adapter prepares the Q5
plan before graph capture, reports separate packed/fallback counts and retains
the original AR1 path for unsupported inputs. Draft and pipeline configuration
remain controlled by the service configuration.

See [NOTICE.md](NOTICE.md) and [LICENSES](LICENSES) for original attribution and
redistribution conditions. Download caches, extracted build directories and
compiled binaries are excluded from the source bundle.
