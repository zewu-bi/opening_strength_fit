FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV TZ=Asia/Shanghai
ENV PYTHONPATH=/app/opening_strength_fit/src
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app/opening_strength_fit

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-dev \
        libgomp1 \
        ocl-icd-opencl-dev \
        opencl-headers \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN grep -v '^lightgbm' requirements.txt > /tmp/requirements-no-lightgbm.txt \
    && python -m pip install -r /tmp/requirements-no-lightgbm.txt \
    && python -m pip install --no-binary=lightgbm --no-deps \
        --config-settings=cmake.define.USE_GPU=ON \
        lightgbm

ARG CACHE_BUST=dev
RUN test -n "$CACHE_BUST"

COPY . .

CMD ["python", "scripts/run_experiment.py", "--help"]
