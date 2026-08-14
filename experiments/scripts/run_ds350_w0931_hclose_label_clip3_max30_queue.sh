#!/usr/bin/env bash
set -euo pipefail

JOBS=(os-nn-ds350-clip3-w0931-hclose-v1)

POLL_SECONDS="${POLL_SECONDS:-60}"
CURRENT_JOB=""
HFCLI_BIN="${HFCLI_BIN:-$(command -v hfcli || true)}"
if [[ -z "${HFCLI_BIN}" && -x "${HOME}/.local/bin/hfcli" ]]; then
  HFCLI_BIN="${HOME}/.local/bin/hfcli"
fi
if [[ -z "${HFCLI_BIN}" ]]; then
  echo "hfcli was not found; set HFCLI_BIN to its absolute path" >&2
  exit 127
fi

kctl() {
  "${HFCLI_BIN}" kubectl "$@"
}

suspend_current_job() {
  if [[ -n "${CURRENT_JOB}" ]]; then
    kctl patch job "${CURRENT_JOB}" --type merge -p '{"spec":{"suspend":true}}' \
      >/dev/null 2>&1 || true
  fi
}
trap suspend_current_job INT TERM

for JOB in "${JOBS[@]}"; do
  CURRENT_JOB="${JOB}"
  kctl get job "${JOB}" >/dev/null

  while true; do
    SUCCEEDED="$(kctl get job "${JOB}" -o jsonpath='{.status.succeeded}')"
    FAILED="$(kctl get job "${JOB}" -o jsonpath='{range .status.conditions[?(@.type=="Failed")]}{.status}{end}')"
    SUSPENDED="$(kctl get job "${JOB}" -o jsonpath='{.spec.suspend}')"

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
      kctl patch job "${JOB}" --type merge -p '{"spec":{"suspend":false}}' >/dev/null
    fi
    echo "waiting ${JOB}: ${SUCCEEDED:-0}/8"
    sleep "${POLL_SECONDS}"
  done
done

CURRENT_JOB=""
echo "ds350 w0931 hclose label-clip3 max-30 job observed complete"
