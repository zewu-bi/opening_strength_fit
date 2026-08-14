#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

hfcli kubectl create configmap os-leakage-audit-code-v1 \
  --namespace bizewu \
  --from-file=sampling.py="${ROOT}/src/opening_strength_fit/sampling.py" \
  --from-file=training_dataset_features.py="${ROOT}/src/opening_strength_fit/training_dataset_features.py" \
  --from-file=training_dataset_build.py="${ROOT}/src/opening_strength_fit/commands/training_dataset_build.py" \
  --from-file=training.py="${ROOT}/src/opening_strength_fit/training.py" \
  --from-file=training_args.py="${ROOT}/src/opening_strength_fit/training_args.py" \
  --from-file=summarize_leakage_kill_run.py="${ROOT}/experiments/scripts/summarize_leakage_kill_run.py" \
  --from-file=leakage_kill_lgbm_36m_v1.toml="${ROOT}/experiments/config_templates/leakage_kill_lgbm_36m_v1.toml" \
  --dry-run=client \
  -o yaml \
  | hfcli kubectl apply -f -
