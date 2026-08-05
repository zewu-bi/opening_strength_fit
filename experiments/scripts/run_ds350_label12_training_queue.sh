#!/usr/bin/env bash
set -euo pipefail

# One label is one 8-shard Indexed Job. Keep only one label active so the full
# 12-case matrix reserves at most eight GPUs at a time.
JOBS=(
  os-nn-ds350-w0931-h1m-v2
  os-nn-ds350-w0931-h3m-v2
  os-nn-ds350-w0931-h10m-v2
  os-nn-ds350-w0931-h1h-v2
  os-nn-ds350-w0931-hclose-v2
  os-nn-ds350-w1001-h1m-v2
  os-nn-ds350-w1001-h3m-v2
  os-nn-ds350-w1001-h10m-v2
  os-nn-ds350-w1001-h1h-v2
  os-nn-ds350-w1001-hclose-v2
  os-nn-ds350-w1401-h1m-v2
  os-nn-ds350-w1401-h3m-v2
)

POLL_SECONDS="${POLL_SECONDS:-60}"
CURRENT_JOB=""

suspend_current_job() {
  if [[ -n "${CURRENT_JOB}" ]]; then
    hfcli kubectl patch job "${CURRENT_JOB}" --type merge -p '{"spec":{"suspend":true}}' \
      >/dev/null 2>&1 || true
  fi
}
trap suspend_current_job INT TERM

for JOB in "${JOBS[@]}"; do
  CURRENT_JOB="${JOB}"
  hfcli kubectl get job "${JOB}" >/dev/null

  while true; do
    SUCCEEDED="$(hfcli kubectl get job "${JOB}" -o jsonpath='{.status.succeeded}')"
    FAILED="$(hfcli kubectl get job "${JOB}" -o jsonpath='{range .status.conditions[?(@.type=="Failed")]}{.status}{end}')"
    SUSPENDED="$(hfcli kubectl get job "${JOB}" -o jsonpath='{.spec.suspend}')"

    if [[ "${SUCCEEDED:-0}" -ge 8 ]]; then
      echo "completed ${JOB} (8/8)"
      break
    fi
    if [[ "${FAILED}" == "True" ]]; then
      echo "job failed: ${JOB}" >&2
      exit 1
    fi
    if [[ "${SUSPENDED}" == "true" ]]; then
      echo "starting ${JOB}"
      hfcli kubectl patch job "${JOB}" --type merge -p '{"spec":{"suspend":false}}' >/dev/null
    fi
    echo "waiting ${JOB}: ${SUCCEEDED:-0}/8"
    sleep "${POLL_SECONDS}"
  done
done

CURRENT_JOB=""
echo "all 12 ds350 label jobs completed"
