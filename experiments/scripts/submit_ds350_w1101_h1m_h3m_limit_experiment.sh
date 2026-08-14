#!/usr/bin/env bash
set -euo pipefail

HFCLI_BIN="${HFCLI_BIN:-${HOME}/.local/bin/hfcli}"
NAMESPACE="${NAMESPACE:-bizewu}"

kctl() {
  "${HFCLI_BIN}" kubectl -n "${NAMESPACE}" "$@"
}

HFCLI_BIN="${HFCLI_BIN}" experiments/scripts/apply_ds350_1m_experiment_code_configmap.sh

kctl create configmap os-ds350-four-window-limit-tables-code-v1 \
  --from-file=build_ds350_clip_tables.py=experiments/scripts/build_ds350_clip_tables.py \
  --from-file=build_ds350_four_window_limit_tables.py=experiments/scripts/build_ds350_four_window_limit_tables.py \
  --dry-run=client -o yaml | kctl apply -f -

kctl apply -f experiments/jobs/support/opening_1101_1110_limit_dependence_v1/nn_ds350_w1101_h1m_h3m_limit_train_max30_v1_jobs.yaml
kctl apply -f experiments/jobs/support/opening_1101_1110_limit_dependence_v1/ds350_four_window_limit_tables_job.yaml

printf '%s\n' "submitted suspended 11:01 h1m/h3m training and analysis jobs"
