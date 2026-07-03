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

COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN python -m pip install -c requirements.lock .

ARG INSTALL_TORCH_CUDA=0
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_PACKAGE=torch
RUN if [ "${INSTALL_TORCH_CUDA}" = "1" ]; then \
        python -m pip install --index-url "${TORCH_CUDA_INDEX_URL}" "${TORCH_PACKAGE}"; \
    fi

ARG CACHE_BUST=dev
RUN test -n "$CACHE_BUST"

COPY . .

CMD ["osf-train", "--help"]
