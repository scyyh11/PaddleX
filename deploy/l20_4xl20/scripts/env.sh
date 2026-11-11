# Machine-specific defaults for the deploy/l20_4xl20 scripts.
# Every value can be overridden by exporting it before running a script.
# Defaults match the 4x L20 development machine this setup was measured on.

: "${L20_WORK_ROOT:=/ssd1/huangyizhan}"
: "${L20_PADDLEX_ROOT:=$L20_WORK_ROOT/PaddleX}"
: "${L20_CACHE_HOME:=$L20_WORK_ROOT/cache}"
: "${L20_MODEL_DIR:=$L20_CACHE_HOME/paddlex/official_models/PaddleOCR-VL-1.6}"
: "${L20_CHAT_TEMPLATE:=$L20_PADDLEX_ROOT/paddlex/inference/genai/chat_templates/PaddleOCR-VL-1.6-0.9B.jinja}"
# Directory for per-server vLLM logs (one subdirectory per version/GPU/port/tag).
: "${L20_LOG_ROOT:=$L20_WORK_ROOT/benchmarks/vllm_versions}"

# vLLM runs inside this container; L20_VLM_HOST is its address as seen from the host.
: "${L20_VLM_CONTAINER:=paddleocr_dev}"
: "${L20_VLM_HOST:=192.168.5.3}"

# PaddleX HPS images (pipeline = Triton server, gateway = FastAPI app).
: "${L20_TRITON_IMAGE:=sha256:19a2235b6ce918fb49de77c82800b13551bc56f27b280326f39522edb5c7ba7f}"
: "${L20_GATEWAY_IMAGE:=44e5147cab1d}"
# Directory holding the official HPS gateway (official/gateway/app.py) and its venv (gateway_venv).
: "${L20_GATEWAY_ROOT:=$L20_WORK_ROOT/benchmarks/locust_vl_20261105/hps_default}"

# Wrapper that returns ENOSYS for clone3/faccessat2 under this host's old seccomp filter.
# Set to an empty string on hosts that do not need it.
: "${L20_DEV_SHELL:=$L20_WORK_ROOT/.devcontainer/dev-shell}"
