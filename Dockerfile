FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV TZ=Asia/Shanghai
ENV PYTHONPATH=/app/opening_strength_fit/src

WORKDIR /app/opening_strength_fit

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install .

ARG CACHE_BUST=dev
RUN test -n "$CACHE_BUST"

COPY . .

CMD ["python", "scripts/run_experiment.py", "--help"]
