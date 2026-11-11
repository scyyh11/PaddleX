#!/usr/bin/env bash
# Usage: deploy_gpu_stack.sh GPU RUNTIME_DIR TRITON_HTTP_PORT GATEWAY_PORT VLM_PORT
# Per-GPU HPS stack for GPUs other than 0: Triton container paddleocr_hps_gpu<GPU>
# (HTTP/GRPC/metrics on TRITON_HTTP_PORT, +1, +2) and gateway container paddleocr_hps_api_gpu<GPU>.
# A running Triton container is stopped and kept as paddleocr_hps_gpu<GPU>_<previous runtime dir>.
# RUNTIME_DIR comes from prepare_hps_variant.py.
set -euo pipefail
source "$(dirname "$(realpath "$0")")/env.sh"
GPU=$1
ROOT=$(realpath "$2")
HTTP=$3
GRPC=$((HTTP+1))
METRICS=$((HTTP+2))
GATEWAY=$4
VLM=$5
NAME=paddleocr_hps_gpu${GPU}
API=paddleocr_hps_api_gpu${GPU}
[ "$GPU" != 0 ] || { echo "GPU 0 uses deploy_hps_variant.sh" >&2; exit 1; }
grep -q "server_url: http://${L20_VLM_HOST}:${VLM}/v1" "$ROOT/runtime/server/pipeline_config.yaml"
DEV_SHELL_MOUNT=(); ENTRY=(--entrypoint /usr/bin/python3); CMD=(/usr/local/bin/dev-shell /bin/bash server.sh)
if [ -n "$L20_DEV_SHELL" ]; then DEV_SHELL_MOUNT=(-v "$L20_DEV_SHELL":/usr/local/bin/dev-shell:ro); else ENTRY=(--entrypoint /bin/bash); CMD=(server.sh); fi
cd "$ROOT"
if docker inspect "$NAME" >/dev/null 2>&1; then
  previous=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" | sed -n 's#^PYTHONPATH=.*/##p')
  docker stop "$NAME"
  docker rename "$NAME" "${NAME}_${previous}"
fi
docker run -d --name "$NAME" --user 0 --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$GPU" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e PADDLEX_HPS_DEVICE_TYPE=gpu \
  -e PADDLE_PDX_CACHE_HOME="$L20_CACHE_HOME/paddlex" \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True -e PADDLE_PDX_MODEL_SOURCE=bos \
  -e PYTHONUNBUFFERED=1 -e PYTHONPATH="$ROOT" \
  -v "$L20_WORK_ROOT":"$L20_WORK_ROOT" "${DEV_SHELL_MOUNT[@]}" \
  -p 127.0.0.1:${HTTP}:8000 -p 127.0.0.1:${GRPC}:8001 -p 127.0.0.1:${METRICS}:8002 \
  --shm-size=4g "${ENTRY[@]}" -w "$ROOT/runtime/server" \
  "$L20_TRITON_IMAGE" "${CMD[@]}"
if ! docker inspect -f '{{.State.Running}}' "$API" 2>/dev/null | grep -q true; then
  docker rm "$API" >/dev/null 2>&1 || true
  API_ENTRY=(); [ -n "$L20_DEV_SHELL" ] && API_ENTRY=(--entrypoint /usr/local/bin/dev-shell)
  docker run -d --name "$API" --user 0 --network host \
    -e HPS_VLM_URL="http://${L20_VLM_HOST}:${VLM}" -e HPS_TRITON_URL="127.0.0.1:${GRPC}" \
    -e HPS_MAX_CONCURRENT_INFERENCE_REQUESTS=16 -e HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS=64 \
    -e HPS_INFERENCE_TIMEOUT=600 -e PYTHONUNBUFFERED=1 -e PYTHONPATH="$L20_PADDLEX_ROOT" \
    -v "$L20_WORK_ROOT":"$L20_WORK_ROOT" "${DEV_SHELL_MOUNT[@]}" \
    -w "$L20_GATEWAY_ROOT/official/gateway" "${API_ENTRY[@]}" \
    "$L20_GATEWAY_IMAGE" \
    "$L20_GATEWAY_ROOT/gateway_venv/bin/python" -m uvicorn --host 0.0.0.0 --port "$GATEWAY" --workers 4 app:app
fi
for i in $(seq 1 60); do
  if curl -fsS -o /dev/null -m 3 "http://127.0.0.1:${HTTP}/v2/health/ready"; then echo "triton gpu${GPU} ready after $((i*5))s"; break; fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${HTTP}/v2/models/layout-parsing/config" > effective_triton_config.json
for i in $(seq 1 24); do
  if curl -fsS -m 5 "http://127.0.0.1:${GATEWAY}/health/ready"; then echo; exit 0; fi
  sleep 5
done
echo "gateway ${GATEWAY} not ready" >&2
exit 1
