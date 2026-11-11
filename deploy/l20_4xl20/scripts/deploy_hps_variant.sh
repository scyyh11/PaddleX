#!/usr/bin/env bash
# Usage: deploy_hps_variant.sh RUNTIME_DIR RENAME_SUFFIX
# GPU 0: stops the running Triton container, keeps it as paddleocr_hps_test_pipeline_<RENAME_SUFFIX>,
# and starts the same image with RUNTIME_DIR/runtime/server. The existing gateway
# (paddleocr_hps_test_api, port 8288) is left running and reconnects by itself.
set -euo pipefail
source "$(dirname "$(realpath "$0")")/env.sh"
ROOT=$(realpath "$1")
SUFFIX=$2
DEV_SHELL_MOUNT=(); ENTRY=(--entrypoint /usr/bin/python3); CMD=(/usr/local/bin/dev-shell /bin/bash server.sh)
if [ -n "$L20_DEV_SHELL" ]; then DEV_SHELL_MOUNT=(-v "$L20_DEV_SHELL":/usr/local/bin/dev-shell:ro); else ENTRY=(--entrypoint /bin/bash); CMD=(server.sh); fi
cd "$ROOT"
test -f runtime/server/model_repo/layout-parsing/config.pbtxt
curl -fsS http://127.0.0.1:8280/v2/models/layout-parsing/config > previous_config_before.json
docker stop paddleocr_hps_test_pipeline
docker rename paddleocr_hps_test_pipeline "paddleocr_hps_test_pipeline_${SUFFIX}"
docker run -d --name paddleocr_hps_test_pipeline --user 0 --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e PADDLEX_HPS_DEVICE_TYPE=gpu \
  -e PADDLE_PDX_CACHE_HOME="$L20_CACHE_HOME/paddlex" \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True -e PADDLE_PDX_MODEL_SOURCE=bos \
  -e PYTHONUNBUFFERED=1 -e PYTHONPATH="$ROOT" \
  -v "$L20_WORK_ROOT":"$L20_WORK_ROOT" "${DEV_SHELL_MOUNT[@]}" \
  -p 127.0.0.1:8280:8000 -p 127.0.0.1:8281:8001 -p 127.0.0.1:8282:8002 \
  --shm-size=4g "${ENTRY[@]}" -w "$ROOT/runtime/server" \
  "$L20_TRITON_IMAGE" "${CMD[@]}"
for i in $(seq 1 60); do
  if curl -fsS -o /dev/null -m 3 http://127.0.0.1:8280/v2/health/ready; then echo "triton ready after $((i*5))s"; break; fi
  sleep 5
done
curl -fsS http://127.0.0.1:8280/v2/models/layout-parsing/config > effective_triton_config.json
curl -fsS http://127.0.0.1:8288/health/ready
