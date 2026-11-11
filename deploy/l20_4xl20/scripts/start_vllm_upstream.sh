#!/usr/bin/env bash
# Usage: start_vllm_upstream.sh GPU PORT VENV_DIR [TAG [NAME=VALUE ...]]
# Serves PaddleOCR-VL-1.6 with upstream vLLM (`vllm serve`), using the same flags,
# chat template and served model name as PaddleX genai_server.
# Extra NAME=VALUE pairs are exported for the server, e.g. L20_PADDLEOCR_VL_BATCHED=1.
# GPU_MEM_UTIL (default 0.5, same as PaddleX) overrides --gpu-memory-utilization;
# API_SERVERS (default 4, same as PaddleX) overrides --api-server-count;
# VLLM_EXTRA_ARGS is appended to the vllm serve command line.
set -euo pipefail
source "$(dirname "$(realpath "$0")")/env.sh"
GPU=$1
PORT=$2
VENV=$(realpath "$3")
TAG=${4:-stock}
EXTRA_ENV="${*:5}"
VERSION=$(basename "$(dirname "$VENV")")
LOGDIR=${L20_LOG_ROOT}/${VERSION}/serve_gpu${GPU}_${PORT}_${TAG}
if curl -fsS -o /dev/null -m 3 "http://${L20_VLM_HOST}:${PORT}/health"; then
  echo "port ${PORT} already serving; refusing to start a second server" >&2
  exit 1
fi
mkdir -p "$LOGDIR"
docker exec -d "$L20_VLM_CONTAINER" ${L20_DEV_SHELL:+dev-shell} bash -c "cd ${LOGDIR} && \
  export CUDA_VISIBLE_DEVICES=${GPU} HF_HOME=${L20_CACHE_HOME}/huggingface TMPDIR=${L20_CACHE_HOME}/tmp \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 ${EXTRA_ENV} && unset PYTHONPATH && \
  echo \$\$ > ${LOGDIR}/service.pid && \
  exec ${VENV}/bin/vllm serve ${L20_MODEL_DIR} --served-model-name PaddleOCR-VL-1.6-0.9B --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.5} --max-model-len 16384 --max-num-batched-tokens 131072 --api-server-count ${API_SERVERS:-4} \
    --chat-template ${L20_CHAT_TEMPLATE} --host 0.0.0.0 --port ${PORT} ${VLLM_EXTRA_ARGS:-} > ${LOGDIR}/server.log 2>&1"
for i in $(seq 1 120); do
  if curl -fsS -o /dev/null -m 3 "http://${L20_VLM_HOST}:${PORT}/health"; then echo "vllm ${VERSION} ${TAG} gpu${GPU}:${PORT} ready after $((i*5))s"; exit 0; fi
  sleep 5
done
echo "vllm ${VERSION} ${TAG} gpu${GPU}:${PORT} not ready; see ${LOGDIR}/server.log" >&2
exit 1
