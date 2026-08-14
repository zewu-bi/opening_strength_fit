#!/usr/bin/env bash
set -euo pipefail

HFCLI_BIN="${HFCLI_BIN:-${HOME}/.local/bin/hfcli}"
NAMESPACE="${NAMESPACE:-bizewu}"
POLL_SECONDS="${POLL_SECONDS:-60}"
ACTIVE_GPU_JOBS=()

kctl() {
  "${HFCLI_BIN}" kubectl -n "${NAMESPACE}" "$@"
}

suspend_active_gpu_jobs() {
  local job
  for job in "${ACTIVE_GPU_JOBS[@]}"; do
    kctl patch job "${job}" --type merge \
      -p '{"spec":{"suspend":true}}' >/dev/null 2>&1 || true
  done
}
trap suspend_active_gpu_jobs EXIT INT TERM

wait_job() {
  local job="$1"
  local expected="$2"
  local last_report=0
  while true; do
    local succeeded failed suspended now
    succeeded="$(kctl get job "${job}" -o jsonpath='{.status.succeeded}')"
    failed="$(kctl get job "${job}" -o jsonpath='{range .status.conditions[?(@.type=="Failed")]}{.status}{end}')"
    suspended="$(kctl get job "${job}" -o jsonpath='{.spec.suspend}')"
    if [[ "${succeeded:-0}" -ge "${expected}" ]]; then
      printf '%s COMPLETE %s %s/%s\n' "$(date -u +%FT%TZ)" "${job}" "${succeeded}" "${expected}"
      return 0
    fi
    if [[ "${failed}" == "True" ]]; then
      printf '%s FAILED %s\n' "$(date -u +%FT%TZ)" "${job}" >&2
      kctl logs -l job-name="${job}" --all-containers --tail=160 >&2 || true
      return 1
    fi
    if [[ "${suspended}" == "true" ]]; then
      printf '%s START %s\n' "$(date -u +%FT%TZ)" "${job}"
      kctl patch job "${job}" --type merge -p '{"spec":{"suspend":false}}' >/dev/null
    fi
    now="$(date +%s)"
    if (( now - last_report >= 300 )); then
      printf '%s WAIT %s %s/%s\n' "$(date -u +%FT%TZ)" "${job}" "${succeeded:-0}" "${expected}"
      last_report="${now}"
    fi
    sleep "${POLL_SECONDS}"
  done
}

GPU_JOBS=(
  os-nn-ds350-base-w1101-h1m-v1
  os-nn-ds350-ord-w1101-h1m-v1
  os-nn-ds350-base-w1101-h3m-v1
  os-nn-ds350-ord-w1101-h3m-v1
)
ACTIVE_GPU_JOBS=("${GPU_JOBS[@]}")
PIDS=()
for job in "${ACTIVE_GPU_JOBS[@]}"; do
  wait_job "${job}" 8 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

ACTIVE_GPU_JOBS=()
wait_job os-ds350-four-window-limit-tables-v1 1
trap - EXIT INT TERM
printf '%s PIPELINE_COMPLETE\n' "$(date -u +%FT%TZ)"
