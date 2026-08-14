#!/usr/bin/env bash
set -euo pipefail

HFCLI="${HFCLI:-$HOME/.local/bin/hfcli}"
NAMESPACE="${NAMESPACE:-bizewu}"

wait_job() {
  local job="$1"
  local last_report=0
  while true; do
    local complete failed active succeeded now
    complete="$(${HFCLI} kubectl get job "${job}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')"
    failed="$(${HFCLI} kubectl get job "${job}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}')"
    active="$(${HFCLI} kubectl get job "${job}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.active}')"
    succeeded="$(${HFCLI} kubectl get job "${job}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.succeeded}')"
    if [[ "${complete}" == "True" ]]; then
      printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "${job}"
      return 0
    fi
    if [[ "${failed}" == "True" ]]; then
      printf '%s FAILED %s\n' "$(date -u +%FT%TZ)" "${job}" >&2
      local pod
      for pod in $(${HFCLI} kubectl get pods -n "${NAMESPACE}" \
        -l job-name="${job}" -o name); do
        ${HFCLI} kubectl logs -n "${NAMESPACE}" "${pod}" --tail=160 >&2 || true
      done
      return 1
    fi
    now=$(date +%s)
    if (( now - last_report >= 300 )); then
      printf '%s WAIT %s active=%s succeeded=%s\n' \
        "$(date -u +%FT%TZ)" "${job}" "${active:-0}" "${succeeded:-0}"
      last_report=${now}
    fi
    sleep 30
  done
}

job_exists() {
  ${HFCLI} kubectl get job "$1" -n "${NAMESPACE}" >/dev/null 2>&1
}

ensure_split_job() {
  local year="$1"
  local job="os-ds-labelsplit-w1101-y${year}-v2"
  if job_exists "${job}"; then
    printf '%s\n' "${job}"
    return
  fi
  ${HFCLI} kubectl apply -f - >/dev/null <<YAML
apiVersion: batch/v1
kind: Job
metadata: {name: ${job}, namespace: ${NAMESPACE}, labels: {app: ${job}}}
spec:
  activeDeadlineSeconds: 86400
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    metadata: {labels: {app: ${job}}}
    spec:
      restartPolicy: Never
      imagePullSecrets: [{name: highfort}]
      volumes:
        - {name: output, persistentVolumeClaim: {claimName: bizewu-private-data}}
        - {name: config, configMap: {name: os-ds350-w1101-config-v1}}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - {key: kubernetes.io/hostname, operator: NotIn, values: [node7, node8]}
      containers:
        - name: splitter
          image: registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260804-horizon-labels-v3
          workingDir: /app/opening_strength_fit
          command: [/bin/bash, -lc]
          args: ['exec osf-split-horizon-labels --config /mnt/window-config/opening_1101_1110_labels_horizon_split.toml --year ${year} --overwrite']
          volumeMounts:
            - {name: output, mountPath: /mnt/output}
            - {name: config, mountPath: /mnt/window-config, readOnly: true}
          resources:
            requests: {cpu: '4', memory: 8Gi}
            limits: {cpu: '12', memory: 32Gi}
YAML
  printf '%s\n' "${job}"
}

ensure_close_job() {
  local year="$1"
  local job="os-ds-labelclose-w1101-y${year}-v1"
  if job_exists "${job}"; then
    printf '%s\n' "${job}"
    return
  fi
  ${HFCLI} kubectl apply -f - >/dev/null <<YAML
apiVersion: batch/v1
kind: Job
metadata: {name: ${job}, namespace: ${NAMESPACE}, labels: {app: ${job}}}
spec:
  activeDeadlineSeconds: 259200
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    metadata: {labels: {app: ${job}}}
    spec:
      restartPolicy: Never
      imagePullSecrets: [{name: highfort}]
      volumes:
        - {name: output, persistentVolumeClaim: {claimName: bizewu-private-data}}
        - {name: config, configMap: {name: os-ds350-w1101-config-v1}}
        - {name: code, configMap: {name: os-ds350-w1101-close-code-v1}}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - {key: kubernetes.io/hostname, operator: NotIn, values: [node7, node8]}
      containers:
        - name: builder
          image: registry.corp.highfortfunds.com/bizewu/opening-strength-fit@sha256:f04ef3ab7372000162e1b92bf86e96f5f6e0e36f15c163d75c9a3c1cc554ab09
          workingDir: /app/opening_strength_fit
          command: [/bin/bash, -lc]
          args: ['exec python /mnt/window-code/close_label_build.py --config /mnt/window-config/opening_1101_1110_labels_close.toml --year ${year} --overwrite']
          volumeMounts:
            - {name: output, mountPath: /mnt/output}
            - {name: config, mountPath: /mnt/window-config, readOnly: true}
            - {name: code, mountPath: /mnt/window-code, readOnly: true}
          resources:
            requests: {cpu: '4', memory: 16Gi}
            limits: {cpu: '12', memory: 64Gi}
YAML
  printf '%s\n' "${job}"
}

ensure_validation_job() {
  local year="$1"
  local job="os-ds350-validate-w1101-y${year}-v1"
  if job_exists "${job}"; then
    printf '%s\n' "${job}"
    return
  fi
  ${HFCLI} kubectl apply -f - >/dev/null <<YAML
apiVersion: batch/v1
kind: Job
metadata: {name: ${job}, namespace: ${NAMESPACE}, labels: {app: ${job}}}
spec:
  activeDeadlineSeconds: 86400
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
  template:
    metadata: {labels: {app: ${job}}}
    spec:
      restartPolicy: Never
      imagePullSecrets: [{name: highfort}]
      volumes:
        - {name: output, persistentVolumeClaim: {claimName: bizewu-private-data}}
        - {name: code, configMap: {name: os-ds350-w1101-validation-code-v1}}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - {key: kubernetes.io/hostname, operator: NotIn, values: [node7, node8]}
      containers:
        - name: validator
          image: registry.corp.highfortfunds.com/bizewu/opening-strength-fit@sha256:8eb66ef731709af980fa7e047ccb59748c0ed8f4ed36aa3460ee9b1f4a9fe0b6
          command: [/bin/bash, -lc]
          args:
            - >-
              exec python /mnt/window-code/validate_ds350_training_pair.py
              --feature-root /mnt/output/opening_strength_fit/datasets/opening_1101_1110_features_350
              --label-root /mnt/output/opening_strength_fit/datasets/opening_1101_1110_labels_h1m_v2
              --label-root /mnt/output/opening_strength_fit/datasets/opening_1101_1110_labels_hclose_v1
              --year ${year} --expected-feature-count 350
              --output /mnt/output/opening_strength_fit/datasets/opening_1101_1110_validation/year=${year}/report.json
          volumeMounts:
            - {name: output, mountPath: /mnt/output}
            - {name: code, mountPath: /mnt/window-code, readOnly: true}
          resources:
            requests: {cpu: '4', memory: 32Gi}
            limits: {cpu: '12', memory: 96Gi}
YAML
  printf '%s\n' "${job}"
}

finish_year() {
  local year="$1"
  local feature_job="os-ds-feat350-w1101-y${year}-v1"
  local label_job="os-ds-label5-w1101-y${year}-v1"
  local split_job close_job validation_job
  wait_job "${label_job}"
  split_job=$(ensure_split_job "${year}")
  wait_job "${split_job}"
  close_job=$(ensure_close_job "${year}")
  wait_job "${feature_job}"
  wait_job "${close_job}"
  validation_job=$(ensure_validation_job "${year}")
  wait_job "${validation_job}"
  printf '%s YEAR_COMPLETE %s\n' "$(date -u +%FT%TZ)" "${year}"
}

pids=()
for year in 2019 2020 2021 2022 2023 2024 2025; do
  finish_year "${year}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  printf '%s PIPELINE_FAILED\n' "$(date -u +%FT%TZ)" >&2
  exit 1
fi
printf '%s PIPELINE_COMPLETE\n' "$(date -u +%FT%TZ)"
