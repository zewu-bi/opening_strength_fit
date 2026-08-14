#!/usr/bin/env bash
set -euo pipefail

HFCLI_BIN="${HFCLI_BIN:-$(command -v hfcli || true)}"
if [[ -z "${HFCLI_BIN}" && -x "${HOME}/.local/bin/hfcli" ]]; then
  HFCLI_BIN="${HOME}/.local/bin/hfcli"
fi
if [[ -z "${HFCLI_BIN}" ]]; then
  echo "hfcli was not found; set HFCLI_BIN to its absolute path" >&2
  exit 127
fi

CONFIGMAP_ARGS=(
  -n bizewu
  create configmap os-ds350-1m-experiment-code-v1
  --from-file=horizon_label_split.py=src/opening_strength_fit/commands/horizon_label_split.py
  --from-file=long_horizon_label_split.py=src/opening_strength_fit/commands/long_horizon_label_split.py
  --from-file=training_modeling.py=src/opening_strength_fit/training_modeling.py
)

"${HFCLI_BIN}" kubectl "${CONFIGMAP_ARGS[@]}" \
  --dry-run=client -o yaml | "${HFCLI_BIN}" kubectl apply -f -
