# VL-1.6 full-page client and benchmark platform integration

The exported client runs the official PaddleX `PaddleOCR-VL-1.6` pipeline with
`PP-DocLayoutV3`, calls the local EAGLE3 VLM for each detected crop, and returns
Markdown plus JSON containing source-image coordinates and reading order.
The default registration name is `PaddleOCR-VL-1.6-EAGLE3-DocLayoutV3`.
This export includes the current full-page client; earlier v1 crop-client
experiments are not part of the deployment.

## Environment and model paths

Run commands from the `deploy/paddleocr_vl_eagle3` directory. Install the layout
client in a separate Python 3.12 environment from vLLM:

```bash
python3.12 -m venv .venv-layout
.venv-layout/bin/python -m pip install -r requirements-layout.txt
```

Use a CUDA-compatible PaddlePaddle GPU wheel for the host. The pinned versions
record the tested environment; package installation alone does not validate a
new CUDA/driver combination.

Provide these local model directories before startup:

```text
models/PaddleOCR-VL-1.6/
models/PaddleOCR-VL-1.6-EAGLE3-K4-dense/
models/hps_layout/PP-DocLayoutV3_infer/
```

The target and draft paths are checked against the VLM readiness response.
For an existing model installation, set `OCR_EAGLE_TARGET` and
`OCR_EAGLE_DRAFT` to the same paths in both the backend and client processes.
`OCR_LAYOUT_MODEL_DIR` overrides the layout model directory. Paths are resolved
before comparison. Model weights and datasets are not included in Git.

## Check one local page

Start the VLM backend using the repository's runtime instructions, then run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-layout/bin/python scripts/benchmark_eagle3_client.py \
  --pipeline page --vlm-version 1.6 --device gpu:0 \
  --vlm-url http://127.0.0.1:18780/v1 \
  --self-test-image /path/to/document.png \
  --output results/document.json
```

The JSON contains `md` and `json` fields, exactly as submitted to the benchmark.
The local check requires neither a platform URL nor a platform token and does
not register a client. Point `CUDA_VISIBLE_DEVICES` at the physical GPU used for
layout; `--device gpu:0` is the index within that visible set. The benchmark
measurements used layout and VLM on the same A100.

## Register with a platform

Set the platform origin explicitly. There is no built-in platform address:

```bash
export OCR_BENCH_SERVER_URL='https://benchmark.example.org'
export OCR_BENCH_TOKEN_FILE='/path/to/private/platform-token'
CUDA_VISIBLE_DEVICES=0 .venv-layout/bin/python scripts/benchmark_eagle3_client.py \
  --device gpu:0 \
  --vlm-url http://127.0.0.1:18780/v1 \
  --timing-log results/page_timings.jsonl
```

The token file should contain only the token and be readable only by its owner.
`OCR_BENCH_API_TOKEN` is an alternative; when both are set, that variable takes
precedence. Neither the token nor platform addresses belong in source control.
Use `--server-url` to override `OCR_BENCH_SERVER_URL`.

This is a client for the original benchmark template's protocol, not an
arbitrary benchmark service. A compatible server must provide:

- WebSocket `/api/client/ws`: client sends `register` with `token`, `model_name`
  and `server_url`, and receives `registered`.
- `infer_task` messages with `experiment_id` and a `files` object mapping a page
  identifier to an HTTP(S) image URL; relative URLs use the platform origin.
- The explicit task types `e2e`, `pip`, `e2e_experiment`, or `pip_experiment`.
  For messages without a concrete type, the client reads same-origin
  `/api/experiments/{id}` and, where needed, `/api/datasets/{id}` using
  `X-API-Token` to resolve the task. Generic `client`/`custom` experiments can
  use the configured page task when compatible.
- `infer_result`, `log`, and `task_done` client messages, plus `ping`/`pong`
  heartbeats. A failed page reports an `error` object without an invented OCR
  result. Image downloads do not carry the platform API token.

Registering does not create or launch a platform experiment. Select the
registered client in the platform and start the experiment there.

## Preserved behavior and limitations

Each page and each crop are processed serially. The official V3 layout preset
uses a confidence threshold of 0.3; `temperature=0`, `min_pixels=112896` and
`max_pixels=1003520` match the evaluated page pipeline. Chart recognition,
seal recognition, Markdown assembly and other defaults come from the pinned
PaddleX 1.6 pipeline preset.

A crop starts with a 1024-token output limit. If it returns `length`, the client
retries the identical crop and prompt once, raising the limit to the backend's
advertised maximum, capped at 8192. If any crop still does not finish with
`stop`, the entire page is reported failed. This includes pathological repeated
generation: raising the limit does not guarantee a complete output. The export
preserves this behavior; it does not silently submit a truncated page.

A valid page with no recognized blocks can return empty Markdown successfully.
Some platforms classify every empty Markdown file as failed, even for pages
without text. Distinguish that platform rule from a client inference exception.

The client checks target/draft identity and execution profile. It retries a
page across a compatible backend replacement and suspends registration while
the backend is unavailable. VLM URLs must use the local loopback HTTP service;
remote/container-separated VLM endpoints require an explicit adaptation.

Timing rows contain `download_s`, `inference_s`, and their sum `total_s`.
`inference_s` includes layout, VLM requests, and page postprocessing, but not
platform scoring or result transmission. A failed download has `inference_s`
set to `null`.

## CPU protocol regression checks

```bash
.venv-layout/bin/python -m unittest discover -s tests -p 'test_benchmark_page*.py' -v
```

These use mocked HTTP/WebSocket responses and do not load models, allocate a
GPU, register a platform client, or establish real network connections.
