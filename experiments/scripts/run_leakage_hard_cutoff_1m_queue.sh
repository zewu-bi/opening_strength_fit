#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

wait_for_job() {
  local job_name="$1"
  local expected="$2"
  while true; do
    local succeeded failed active
    succeeded="$(hfcli kubectl get job "${job_name}" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
    failed="$(hfcli kubectl get job "${job_name}" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
    active="$(hfcli kubectl get job "${job_name}" -o jsonpath='{.status.active}' 2>/dev/null || true)"
    succeeded="${succeeded:-0}"
    failed="${failed:-0}"
    active="${active:-0}"
    printf 'queue job=%s succeeded=%s/%s active=%s failed=%s time=%s\n' \
      "${job_name}" "${succeeded}" "${expected}" "${active}" "${failed}" "$(date -u +%FT%TZ)"
    if (( succeeded >= expected )); then
      return 0
    fi
    local failed_condition
    failed_condition="$(hfcli kubectl get job "${job_name}" -o jsonpath='{range .status.conditions[?(@.type=="Failed")]}{.status}{end}' 2>/dev/null || true)"
    if [[ "${failed_condition}" == "True" ]]; then
      printf 'queue stopping: job %s reached Failed condition\n' "${job_name}" >&2
      return 1
    fi
    sleep 60
  done
}

cd "${ROOT}"
wait_for_job os-leak-cutoff-clock-base-v2 350

if hfcli kubectl get job os-leak-cutoff-clock-reduce-v2 >/dev/null 2>&1; then
  printf 'queue reusing existing job os-leak-cutoff-clock-reduce-v2\n'
else
  hfcli kubectl apply -f experiments/jobs/support/leakage_audit_2026_08/leakage_hard_cutoff_clock_reduce_v2_job.yaml
fi
wait_for_job os-leak-cutoff-clock-reduce-v2 35

if hfcli kubectl get job os-leak-cutoff-1m-nn-v1 >/dev/null 2>&1; then
  printf 'queue reusing existing job os-leak-cutoff-1m-nn-v1\n'
else
  hfcli kubectl apply -f experiments/jobs/support/leakage_audit_2026_08/leakage_hard_cutoff_1m_nn_train_v1_job.yaml
fi
wait_for_job os-leak-cutoff-1m-nn-v1 4

experiments/scripts/apply_leakage_audit_code_configmap.sh
if hfcli kubectl get job os-leak-cutoff-1m-summary-v1 >/dev/null 2>&1; then
  printf 'queue reusing existing job os-leak-cutoff-1m-summary-v1\n'
else
  hfcli kubectl apply -f experiments/jobs/support/leakage_audit_2026_08/leakage_hard_cutoff_1m_summary_v1_job.yaml
fi
wait_for_job os-leak-cutoff-1m-summary-v1 15
printf 'queue complete: hard-cutoff 1m summaries are ready\n'
