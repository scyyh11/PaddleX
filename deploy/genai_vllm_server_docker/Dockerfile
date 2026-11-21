ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.19.0-x86_64
FROM ${VLLM_BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 wget \
    && rm -rf /var/lib/apt/lists/* \
    && if ! command -v python >/dev/null; then ln -s "$(command -v python3)" /usr/local/bin/python; fi

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG PADDLEX_VERSION
ARG PADDLEX_WHEEL
ARG VLLM_VERSION=0.19.0
# 0.33.0's base image already provides its CUDA/Torch/vLLM stack. Do not
# re-resolve that stack and replace its GPU libraries with PyPI wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=paddlex-wheel,target=/paddlex-build \
    if [ -n "${PADDLEX_WHEEL}" ] && [ -z "${PADDLEX_VERSION}" ]; then \
        python -m pip install "/paddlex-build/${PADDLEX_WHEEL}"; \
    elif [ -n "${PADDLEX_VERSION}" ] && [ -z "${PADDLEX_WHEEL}" ]; then \
        python -m pip install "paddlex${PADDLEX_VERSION}"; \
    else \
        echo "Set exactly one of PADDLEX_VERSION or PADDLEX_WHEEL." >&2; \
        exit 2; \
    fi \
    && case "${VLLM_VERSION}" in \
        0.19.0) \
            printf '%s\n' 'transformers==4.57.6' > /tmp/vllm-constraints.txt \
            && PIP_CONSTRAINT=/tmp/vllm-constraints.txt paddlex --install genai-vllm-server \
            && rm /tmp/vllm-constraints.txt ;; \
        0.33.0) \
            python -c 'from importlib.metadata import version; from packaging.version import Version; assert all(Version(version(k)).public == v for k, v in (("vllm", "0.33.0"), ("torch", "2.14.0"), ("transformers", "5.20.2"))), "Use the pinned vLLM 0.33.0 base image"; from paddlex.utils.deps import require_genai_engine_plugin; require_genai_engine_plugin("vllm-server")' ;; \
        *) echo "Unsupported vLLM version: ${VLLM_VERSION}" >&2; exit 2 ;; \
    esac \
    && python -c 'import sys; from importlib.metadata import version; sys.exit(version("vllm").split("+")[0] != sys.argv[1])' "${VLLM_VERSION}"

EXPOSE 8080

ENTRYPOINT []
CMD ["paddlex_genai_server", "--model_name", "PaddleOCR-VL-0.9B", "--host", "0.0.0.0", "--port", "8080", "--backend", "vllm"]
