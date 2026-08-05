#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="bizewu"

wait_for_job() {
  local job_name="$1"
  local expected="$2"
  while true; do
    local succeeded failed active failed_condition
    succeeded="$(hfcli kubectl -n "${NAMESPACE}" get job "${job_name}" -o jsonpath='{.status.succeeded}')"
    failed="$(hfcli kubectl -n "${NAMESPACE}" get job "${job_name}" -o jsonpath='{.status.failed}')"
    active="$(hfcli kubectl -n "${NAMESPACE}" get job "${job_name}" -o jsonpath='{.status.active}')"
    failed_condition="$(hfcli kubectl -n "${NAMESPACE}" get job "${job_name}" -o jsonpath='{range .status.conditions[?(@.type=="Failed")]}{.status}{end}')"
    succeeded="${succeeded:-0}"
    failed="${failed:-0}"
    active="${active:-0}"
    echo "pipeline job=${job_name} active=${active} succeeded=${succeeded}/${expected} failed=${failed}"
    if [ "${failed_condition}" = "True" ]; then
      echo "pipeline stopping because ${job_name} reached Failed condition" >&2
      return 1
    fi
    if [ "${succeeded}" -ge "${expected}" ]; then
      return 0
    fi
    sleep 30
  done
}

wait_for_job "os-long-label-raw-w0931-1010-v1" 7

hfcli kubectl -n "${NAMESPACE}" apply \
  -f experiments/jobs/opening_0931_0940_labels_10m_1h_close_job.yaml \
  -f experiments/jobs/opening_1001_1010_labels_10m_1h_close_job.yaml

wait_for_job "os-long-labels-w0931-0940" 7
wait_for_job "os-long-labels-w1001-1010" 7

hfcli kubectl -n "${NAMESPACE}" apply \
  -f experiments/jobs/opening_0931_0940_labels_10m_1h_close_split_job.yaml \
  -f experiments/jobs/opening_1001_1010_labels_10m_1h_close_split_job.yaml

wait_for_job "os-long-labelsplit-w0931-0940" 7
wait_for_job "os-long-labelsplit-w1001-1010" 7

echo "long-horizon label pipeline complete"
