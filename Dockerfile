FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV TZ=Asia/Shanghai

WORKDIR /app/opening_strength_fit

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# `core` builds a minimal local image; `cluster` preserves every production adapter.
ARG DEPENDENCY_PROFILE=cluster
COPY pyproject.toml requirements.lock ./
RUN python -c 'import os, tomllib; project = tomllib.load(open("pyproject.toml", "rb"))["project"]; profile = os.environ["DEPENDENCY_PROFILE"]; optional = project.get("optional-dependencies", {}); assert profile == "core" or profile in optional, f"unknown dependency profile: {profile}"; print("\n".join([*project.get("dependencies", []), *optional.get(profile, [])]))' \
        > /tmp/runtime-requirements.txt \
    && python -m pip install \
        -c requirements.lock \
        -r /tmp/runtime-requirements.txt \
    && rm /tmp/runtime-requirements.txt

ARG INSTALL_TORCH_CUDA=0
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_PACKAGE=""
RUN if [ "${INSTALL_TORCH_CUDA}" = "1" ]; then \
        test -n "${TORCH_PACKAGE}"; \
        python -m pip install --index-url "${TORCH_CUDA_INDEX_URL}" "${TORCH_PACKAGE}"; \
    fi

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-deps .

ARG CACHE_BUST=dev
RUN test -n "$CACHE_BUST"

ARG SOURCE_REVISION
RUN test -n "$SOURCE_REVISION"
ENV OPENING_STRENGTH_SOURCE_REVISION=${SOURCE_REVISION}
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}

COPY experiments/runs ./experiments/runs
COPY experiments/canonical ./experiments/canonical
COPY experiments/config_templates ./experiments/config_templates
COPY experiments/scripts ./experiments/scripts
COPY experiments/incubator.toml ./experiments/incubator.toml

CMD ["osf", "--help"]
