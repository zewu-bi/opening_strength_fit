import argparse
import os
import textwrap
from datetime import date
from pathlib import Path

import pandas as pd

from opening_strength_fit.commands.k8s_analysis_rendering import render_pool_internal_analysis_job
from opening_strength_fit.commands.k8s_rendering_common import (
    avoid_nodes_affinity_yaml as _avoid_nodes_affinity_yaml,
)
from opening_strength_fit.commands.k8s_rendering_common import (
    env_from_secrets_yaml as _env_from_secrets_yaml,
)
from opening_strength_fit.commands.k8s_rendering_common import k8s_job_name as _k8s_job_name
from opening_strength_fit.commands.k8s_rendering_common import (
    node_selector_yaml as _node_selector_yaml,
)
from opening_strength_fit.commands.k8s_rendering_common import (
    training_config_map_mount_yaml as _training_config_map_mount_yaml,
)
from opening_strength_fit.commands.k8s_rendering_common import (
    training_config_map_volume_yaml as _training_config_map_volume_yaml,
)
from opening_strength_fit.commands.k8s_rendering_common import (
    wait_for_specific_paths_yaml as _wait_for_specific_paths_yaml,
)
from opening_strength_fit.config import config_value as get
from opening_strength_fit.config import load_toml, run_id, slug
from opening_strength_fit.k8s import KUBERNETES_NAME_LIMIT
from opening_strength_fit.pvc_layout import (
    output_layout,
    rolling_shard_dir_name,
    run_output_dir,
    yearly_shard_dir_name,
)

DEFAULT_IMAGE_ENV = "OPENING_STRENGTH_IMAGE"
_RUN_KIND_COMMANDS = {
    "alpha_conditioned_rolling_validation": "osf-run-alpha-conditioned-rolling-validation",
    "ask_level_attribution": "osf-ask-level-attribution",
    "cache_transform": "osf-build-target-label-cache",
    "capacity_acceptance": "osf-analyze-capacity-acceptance",
    "capacity_audit": "osf-audit-capacity",
    "clickhouse_labeled_cache": "osf-build-labeled-cache",
    "execution_context": "osf-extract-execution-context",
    "exposure_audit": "osf-audit-exposure",
    "exposure_input": "osf-build-exposure-input",
    "feature_audit": "osf-audit-feature-dependence",
    "feature_hygiene": "osf-audit-feature-hygiene",
    "gap_risk_attribution": "osf-run-gap-risk-attribution",
    "labeled_cache": "osf-build-labeled-cache",
    "learned_risk_layer": "osf-run-learned-risk-layer",
    "next_close_label_cache": "osf-build-next-close-labels",
    "score_risk_sweep": "osf-run-score-risk-sweep",
    "strategy_acceptance": "osf-audit-strategy-acceptance",
    "target_cache": "osf-build-target-label-cache",
}


def load_config(path: Path) -> dict:
    return load_toml(path)


def resolve_render_image(image: str, *, allow_mutable: bool = False) -> str:
    resolved = (image or os.environ.get(DEFAULT_IMAGE_ENV, "")).strip()
    if not resolved:
        raise SystemExit(
            f"missing container image: pass --image or set {DEFAULT_IMAGE_ENV} "
            "to an immutable tag or digest"
        )
    image_ref = resolved.rsplit("/", 1)[-1]
    if image_ref.endswith(":latest") and not allow_mutable:
        raise SystemExit("refusing mutable image tag ':latest'; pass an immutable tag or digest")
    return resolved


def training_command(config: dict) -> str:
    run_kind = str(get(config, "run", "kind", "experiment")).strip().lower()
    command = _RUN_KIND_COMMANDS.get(run_kind)
    if command:
        return command
    if run_kind not in {"experiment", "exploration"}:
        raise SystemExit(f"Unsupported run.kind for k8s rendering: {run_kind}")
    model_name = str(get(config, "model", "name", "ridge")).strip().lower()
    if model_name in {
        "ridge",
        "gbm",
        "hist_gbm",
        "hist_gradient_boosting",
        "lightgbm",
        "lgbm",
        "torch_mlp",
        "mlp",
        "nn",
        "ensemble",
        "clock_segment_lightgbm",
        "clock_segment_lgbm",
        "segmented_lightgbm",
    }:
        return "osf-train"
    raise SystemExit(f"Unsupported model.name for k8s rendering: {model_name}")


def _rolling_completion_files(config: dict, command: str) -> list[str]:
    run_kind = str(get(config, "run", "kind", "")).strip().lower()
    if run_kind == "capacity_acceptance":
        return ["capacity_acceptance_trace.json", "capacity_acceptance_daily_summary.csv"]
    if run_kind == "capacity_audit":
        return ["capacity_audit_trace.json", "capacity_audit_summary.csv"]
    if run_kind == "strategy_acceptance":
        return ["_SUCCESS", "strategy_acceptance_summary.csv"]
    if run_kind == "ask_level_attribution":
        return ["ask_level_attribution_trace.json", "_SUCCESS"]
    if run_kind == "execution_context":
        return ["execution_context_trace.json", "_SUCCESS"]
    if run_kind == "exposure_input":
        return ["exposure_input_trace.json", "exposure_input.parquet"]
    if run_kind == "exposure_audit":
        return ["exposure_audit_trace.json", "exposure_audit_summary.csv"]
    if run_kind == "alpha_conditioned_rolling_validation":
        return ["rolling_summary.csv"]
    if run_kind == "feature_audit":
        return ["feature_audit_trace.json", "feature_audit_metrics.csv"]
    if run_kind == "feature_hygiene":
        return ["feature_hygiene_trace.json", "feature_hygiene.csv"]
    if command == "osf-train":
        return ["_SUCCESS", "metrics_by_year.csv", "predictions.parquet"]
    return ["metrics_by_year.csv"]


def _shell_file_check(files: list[str]) -> str:
    return " && ".join(f'[ -f "${{OUT}}/{file}" ]' for file in files)


def _gpu_count(resources: dict) -> str:
    gpu_limit = str(resources.get("gpu_limit", resources.get("nvidia_gpu", "0")) or "0")
    if gpu_limit in {"", "0", "0.0", "none", "None"}:
        return ""
    return gpu_limit


def _gpu_resource_line(resources: dict, indent: int = 14) -> str:
    gpu_count = _gpu_count(resources)
    if not gpu_count:
        return ""
    return f'{" " * indent}nvidia.com/gpu: "{gpu_count}"\n'


def _gpu_tolerations_yaml(resources: dict, indent: int = 14) -> str:
    if not _gpu_count(resources):
        return ""
    return textwrap.indent(
        textwrap.dedent(
            """\
            tolerations:
              - key: has_gpu
                operator: Equal
                value: "true"
                effect: NoSchedule
            """
        ),
        " " * indent,
    )


def _gpu_opencl_bootstrap_yaml(resources: dict, indent: int) -> str:
    if not _gpu_count(resources):
        return ""
    return textwrap.indent(
        textwrap.dedent(
            """\
            mkdir -p /etc/OpenCL/vendors
            echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd
            """
        ),
        " " * indent,
    )


def _wait_for_paths_script(config: dict) -> str:
    paths = get(config, "k8s", "wait_for_paths", []) or []
    if isinstance(paths, str):
        paths = paths.replace(",", " ").split()
    paths = [str(path).strip() for path in paths if str(path).strip()]
    if not paths:
        return ""

    timeout_seconds = int(get(config, "k8s", "wait_for_path_timeout_seconds", 21600))
    interval_seconds = int(get(config, "k8s", "wait_for_path_interval_seconds", 60))
    return _wait_for_specific_paths_yaml(
        paths,
        timeout_seconds=timeout_seconds,
        interval_seconds=interval_seconds,
        indent=0,
    )


def _wait_for_paths_yaml(config: dict, indent: int) -> str:
    script = _wait_for_paths_script(config)
    if not script:
        return ""
    return textwrap.indent(script, " " * indent)


def _scheduler_yaml(config: dict, resources: dict, indent: int = 14) -> str:
    return (
        _node_selector_yaml(config, indent=indent)
        + _avoid_nodes_affinity_yaml(config, indent=indent)
        + _gpu_tolerations_yaml(
            resources,
            indent=indent,
        )
    )


def _training_command_yaml(
    *,
    command: str,
    config_path: Path,
    output_dir: str,
    resources: dict,
    wait_for_paths: str = "",
    indent: int = 18,
) -> str:
    if _gpu_count(resources) or wait_for_paths.strip():
        preamble_lines: list[str] = []
        if _gpu_count(resources):
            preamble_lines.extend(
                [
                    "mkdir -p /etc/OpenCL/vendors",
                    "echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd",
                ]
            )
        if wait_for_paths.strip():
            preamble_lines.extend(wait_for_paths.rstrip().splitlines())
        lines = [
            "command:",
            "  - /bin/bash",
            "  - -lc",
            "  - |",
            "    set -euo pipefail",
        ]
        lines.extend(f"    {line}" for line in preamble_lines)
        lines.extend(
            [
                f"    exec {command} \\",
                f"      --config {config_path.as_posix()} \\",
                f"      --output-dir {output_dir}",
            ]
        )
        return textwrap.indent("\n".join(lines) + "\n", " " * indent)
    return textwrap.indent(
        textwrap.dedent(
            f"""\
            command:
              - {command}
              - --config
              - {config_path.as_posix()}
              - --output-dir
              - {output_dir}
            """
        ),
        " " * indent,
    )


def _year_from_config(config: dict, key: str) -> int:
    value = get(config, "window", key, None)
    if not value:
        raise SystemExit(
            f"--sharded requires [window].{key}; set explicit test_start_date/test_end_date"
        )
    return date.fromisoformat(str(value)).year


def _month_range_from_config(config: dict) -> list[str]:
    start = get(config, "window", "test_start_month", None)
    end = get(config, "window", "test_end_month", None)
    if not start or not end:
        raise SystemExit(
            "--sharded monthly requires [window].test_start_month and [window].test_end_month"
        )
    return [str(month) for month in pd.period_range(str(start), str(end), freq="M")]


def _month_windows_from_config(config: dict) -> list[tuple[str, str]]:
    months = _month_range_from_config(config)
    test_months = int(get(config, "window", "test_months", 1) or 1)
    stride_months = int(get(config, "window", "test_stride_months", test_months) or test_months)
    if test_months < 1:
        raise SystemExit("[window].test_months must be >= 1")
    if stride_months < 1:
        raise SystemExit("[window].test_stride_months must be >= 1")

    first = pd.Period(months[0], freq="M")
    last = pd.Period(months[-1], freq="M")
    windows: list[tuple[str, str]] = []
    test_start = first
    while test_start <= last:
        test_end = test_start + test_months - 1
        if test_end > last:
            break
        windows.append((str(test_start), str(test_end)))
        test_start += stride_months
    if not windows:
        raise SystemExit(
            "sharded rolling monthly produced no test windows; check "
            "[window].test_months/test_stride_months/test_start_month/test_end_month"
        )
    return windows


def _shard_parallelism(config: dict, resources: dict) -> int:
    raw = get(config, "k8s", "shard_parallelism", resources.get("shard_parallelism", 1))
    return max(1, int(raw or 1))


def _shard_job_mode(config: dict) -> str:
    mode = str(get(config, "k8s", "shard_job_mode", "indexed") or "indexed")
    mode = mode.strip().lower().replace("-", "_")
    if mode not in {"indexed", "separate"}:
        raise SystemExit("k8s.shard_job_mode must be 'indexed' or 'separate'")
    return mode


def _window_mode(config: dict) -> str:
    return str(get(config, "window", "mode", "chronological"))


def _k8s_env_from(config: dict, indent: int = 18) -> str:
    secret_names = []
    clickhouse_secret = str(get(config, "k8s", "clickhouse_secret", "") or "")
    if clickhouse_secret:
        secret_names.append(clickhouse_secret)
    env_secrets = get(config, "k8s", "env_secrets", []) or []
    if isinstance(env_secrets, str):
        secret_names.extend(env_secrets.replace(",", " ").split())
    else:
        secret_names.extend(str(item) for item in env_secrets)
    ceph_secret = str(get(config, "k8s", "ceph_secret", "") or "")
    if ceph_secret:
        secret_names.append(ceph_secret)
    return _env_from_secrets_yaml(secret_names, indent=indent)


def render_training_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    job_name = get(config, "k8s", "job_name", f"opening-strength-{slug(run_id_value)}")
    namespace = get(config, "k8s", "namespace", "bizewu")
    pull_secret = get(config, "k8s", "image_pull_secret", "highfort")
    pvc = get(config, "k8s", "pvc", "bizewu-private-data")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    output_dir = run_output_dir(config, run_id_value, mount_path=str(mount_path))
    resources = config.get("k8s", {}).get("resources", {})
    cpu_request = resources.get("cpu_request", "4")
    cpu_limit = resources.get("cpu_limit", "8")
    memory_request = resources.get("memory_request", "16Gi")
    memory_limit = resources.get("memory_limit", "32Gi")
    gpu_resource_line = _gpu_resource_line(resources, indent=22)
    scheduler_yaml = _scheduler_yaml(config, resources, indent=14)
    command = training_command(config)
    env_from = _k8s_env_from(config, indent=18)
    command_yaml = _training_command_yaml(
        command=command,
        config_path=config_path,
        output_dir=output_dir,
        resources=resources,
        wait_for_paths=_wait_for_paths_script(config),
        indent=18,
    )

    return textwrap.dedent(
        f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
        spec:
          backoffLimit: 0
          ttlSecondsAfterFinished: 86400
          template:
            spec:
              restartPolicy: Never
              imagePullSecrets:
                - name: {pull_secret}
              volumes:
                - name: opening-strength-output
                  persistentVolumeClaim:
                    claimName: {pvc}
{scheduler_yaml.rstrip()}
              containers:
                - name: opening-strength-fit
                  image: {image}
                  imagePullPolicy: Always
{env_from}                  workingDir: /app/opening_strength_fit
{command_yaml.rstrip()}
                  volumeMounts:
                    - name: opening-strength-output
                      mountPath: {mount_path}
                  resources:
                    requests:
                      cpu: "{cpu_request}"
                      memory: {memory_request}
{gpu_resource_line.rstrip()}
                    limits:
                      cpu: "{cpu_limit}"
                      memory: {memory_limit}
{gpu_resource_line.rstrip()}
        """
    )


def render_sharded_training_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    explicit_job_name = str(get(config, "k8s", "job_name", "") or "").strip()
    job_name = explicit_job_name or _k8s_job_name("opening-strength", run_id_value, "sharded")
    namespace = get(config, "k8s", "namespace", "bizewu")
    pull_secret = get(config, "k8s", "image_pull_secret", "highfort")
    pvc = get(config, "k8s", "pvc", "bizewu-private-data")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    layout = output_layout(config)
    output_dir = run_output_dir(config, run_id_value, mount_path=str(mount_path))
    resources = config.get("k8s", {}).get("resources", {})
    cpu_request = resources.get("cpu_request", "4")
    cpu_limit = resources.get("cpu_limit", "8")
    memory_request = resources.get("memory_request", "16Gi")
    memory_limit = resources.get("memory_limit", "32Gi")
    gpu_resource_line = _gpu_resource_line(resources, indent=26)
    scheduler_yaml = _scheduler_yaml(config, resources, indent=18)
    command = training_command(config)
    env_from = _k8s_env_from(config, indent=18)
    opencl_bootstrap = _gpu_opencl_bootstrap_yaml(resources, indent=26)
    wait_for_paths = _wait_for_paths_yaml(config, indent=26)
    config_map_volume = _training_config_map_volume_yaml(config, indent=20)
    config_map_mount = _training_config_map_mount_yaml(config, indent=24)
    if _window_mode(config) == "rolling_monthly":
        env_from = _k8s_env_from(config, indent=22)
        month_windows = _month_windows_from_config(config)
        test_starts = " ".join(start for start, _ in month_windows)
        test_ends = " ".join(end for _, end in month_windows)
        shard_job_mode = _shard_job_mode(config)
        index_suffix_chars = (2 if shard_job_mode == "separate" else 1) + len(
            str(len(month_windows) - 1)
        )
        if not explicit_job_name:
            job_name = _k8s_job_name(
                "opening-strength",
                run_id_value,
                "sharded",
                max_length=KUBERNETES_NAME_LIMIT - index_suffix_chars,
            )
        shard_parallelism = _shard_parallelism(config, resources)
        train_months = int(get(config, "window", "train_months", 12))
        test_months = int(get(config, "window", "test_months", 1) or 1)
        test_stride_months = int(
            get(config, "window", "test_stride_months", test_months) or test_months
        )
        completion_files = _rolling_completion_files(config, command)
        completion_check = _shell_file_check(completion_files)
        completion_label = ", ".join(completion_files)
        rolling_dir_expression = rolling_shard_dir_name("${TEST_START}", "${TEST_END}", layout)
        per_index_backoff = get(config, "k8s", "backoff_limit_per_index", None)
        backoff_field = "backoffLimitPerIndex" if per_index_backoff is not None else "backoffLimit"
        backoff_value = int(per_index_backoff) if per_index_backoff is not None else 0
        if shard_job_mode == "separate":
            config_map_volume = _training_config_map_volume_yaml(config, indent=32)
            scheduler_yaml = _scheduler_yaml(config, resources, indent=30)
            env_from = _k8s_env_from(config, indent=34)
            opencl_bootstrap = _gpu_opencl_bootstrap_yaml(resources, indent=38)
            wait_for_paths = _wait_for_paths_yaml(config, indent=38)
            config_map_mount = _training_config_map_mount_yaml(config, indent=36)
            gpu_resource_line = _gpu_resource_line(resources, indent=38)
            manifests: list[str] = []
            for index, (test_start, test_end) in enumerate(month_windows):
                suffix = f"-s{index}"
                if len(job_name) + len(suffix) > KUBERNETES_NAME_LIMIT:
                    raise SystemExit(
                        f"separate shard job name exceeds {KUBERNETES_NAME_LIMIT} characters: "
                        f"{job_name}{suffix}"
                    )
                shard_job_name = f"{job_name}{suffix}"
                rolling_dir = rolling_shard_dir_name(test_start, test_end, layout)
                manifests.append(
                    textwrap.dedent(
                        f"""\
                        apiVersion: batch/v1
                        kind: Job
                        metadata:
                          name: {shard_job_name}
                          namespace: {namespace}
                        spec:
                          backoffLimit: 0
                          ttlSecondsAfterFinished: 86400
                          template:
                            spec:
                              restartPolicy: Never
                              imagePullSecrets:
                                - name: {pull_secret}
                              volumes:
                                - name: opening-strength-output
                                  persistentVolumeClaim:
                                    claimName: {pvc}
{config_map_volume.rstrip()}
{scheduler_yaml.rstrip()}
                              containers:
                                - name: opening-strength-fit
                                  image: {image}
                                  imagePullPolicy: Always
{env_from}                                  workingDir: /app/opening_strength_fit
                                  command:
                                    - /bin/bash
                                    - -lc
                                    - |
                                      set -euo pipefail
{opencl_bootstrap.rstrip()}
{wait_for_paths.rstrip()}
                                      ROOT={output_dir}
                                      mkdir -p "${{ROOT}}"
                                      TEST_START={test_start}
                                      TEST_END={test_end}
                                      OUT="${{ROOT}}/{rolling_dir}"
                                      if {completion_check}; then
                                        echo "test window ${{TEST_START}}..${{TEST_END}}: required outputs already exist ({completion_label}), skipping ${{OUT}}"
                                        exit 0
                                      fi

                                      echo
                                      echo "running {run_id_value} shard test=${{TEST_START}}..${{TEST_END}} index={index}"
                                      echo "output_dir=${{OUT}}"

                                      {command} \\
                                        --config {config_path.as_posix()} \\
                                        --rolling-monthly \\
                                        --train-months {train_months} \\
                                        --test-months {test_months} \\
                                        --test-stride-months {test_stride_months} \\
                                        --test-start-month "${{TEST_START}}" \\
                                        --test-end-month "${{TEST_END}}" \\
                                        --output-dir "${{OUT}}"
                                  volumeMounts:
                                    - name: opening-strength-output
                                      mountPath: {mount_path}
{config_map_mount.rstrip()}
                                  resources:
                                    requests:
                                      cpu: "{cpu_request}"
                                      memory: {memory_request}
{gpu_resource_line.rstrip()}
                                    limits:
                                      cpu: "{cpu_limit}"
                                      memory: {memory_limit}
{gpu_resource_line.rstrip()}
                        """
                    ).rstrip()
                )
            return "\n---\n".join(manifests) + "\n"
        return textwrap.dedent(
            f"""\
            apiVersion: batch/v1
            kind: Job
            metadata:
              name: {job_name}
              namespace: {namespace}
            spec:
              {backoff_field}: {backoff_value}
              completionMode: Indexed
              completions: {len(month_windows)}
              parallelism: {shard_parallelism}
              ttlSecondsAfterFinished: 86400
              template:
                spec:
                  restartPolicy: Never
                  imagePullSecrets:
                    - name: {pull_secret}
                  volumes:
                    - name: opening-strength-output
                      persistentVolumeClaim:
                        claimName: {pvc}
{config_map_volume.rstrip()}
{scheduler_yaml.rstrip()}
                  containers:
                    - name: opening-strength-fit
                      image: {image}
                      imagePullPolicy: Always
{env_from}                      workingDir: /app/opening_strength_fit
                      env:
                        - name: JOB_COMPLETION_INDEX
                          valueFrom:
                            fieldRef:
                              fieldPath: "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                      command:
                        - /bin/bash
                        - -lc
                        - |
                          set -euo pipefail
{opencl_bootstrap.rstrip()}
{wait_for_paths.rstrip()}
                          ROOT={output_dir}
                          mkdir -p "${{ROOT}}"
                          TEST_STARTS=({test_starts})
                          TEST_ENDS=({test_ends})
                          INDEX="${{JOB_COMPLETION_INDEX:-}}"
                          if [ -z "${{INDEX}}" ]; then
                            echo "missing JOB_COMPLETION_INDEX for indexed shard job" >&2
                            exit 1
                          fi
                          if [ "${{INDEX}}" -lt 0 ] || [ "${{INDEX}}" -ge "${{#TEST_STARTS[@]}}" ]; then
                            echo "JOB_COMPLETION_INDEX out of range: ${{INDEX}}" >&2
                            exit 1
                          fi

                          TEST_START="${{TEST_STARTS[${{INDEX}}]}}"
                          TEST_END="${{TEST_ENDS[${{INDEX}}]}}"
                          OUT="${{ROOT}}/{rolling_dir_expression}"
                          if {completion_check}; then
                            echo "test window ${{TEST_START}}..${{TEST_END}}: required outputs already exist ({completion_label}), skipping ${{OUT}}"
                            exit 0
                          fi

                          echo
                          echo "running {run_id_value} shard test=${{TEST_START}}..${{TEST_END}} index=${{INDEX}}"
                          echo "output_dir=${{OUT}}"

                          {command} \\
                            --config {config_path.as_posix()} \\
                            --rolling-monthly \\
                            --train-months {train_months} \\
                            --test-months {test_months} \\
                            --test-stride-months {test_stride_months} \\
                            --test-start-month "${{TEST_START}}" \\
                            --test-end-month "${{TEST_END}}" \\
                            --output-dir "${{OUT}}"
                      volumeMounts:
                        - name: opening-strength-output
                          mountPath: {mount_path}
{config_map_mount.rstrip()}
                      resources:
                        requests:
                          cpu: "{cpu_request}"
                          memory: {memory_request}
{gpu_resource_line.rstrip()}
                        limits:
                          cpu: "{cpu_limit}"
                          memory: {memory_limit}
{gpu_resource_line.rstrip()}
            """
        )
    test_start_year = _year_from_config(config, "test_start_date")
    test_end_year = _year_from_config(config, "test_end_date")
    years = " ".join(str(year) for year in range(test_start_year, test_end_year + 1))
    yearly_dir_expression = yearly_shard_dir_name("${YEAR}", layout)

    return textwrap.dedent(
        f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
        spec:
          backoffLimit: 0
          ttlSecondsAfterFinished: 86400
          template:
            spec:
              restartPolicy: Never
              imagePullSecrets:
                - name: {pull_secret}
              volumes:
                - name: opening-strength-output
                  persistentVolumeClaim:
                    claimName: {pvc}
{_scheduler_yaml(config, resources, indent=14).rstrip()}
              containers:
                - name: opening-strength-fit
                  image: {image}
                  imagePullPolicy: Always
{env_from}                  workingDir: /app/opening_strength_fit
                  command:
                    - /bin/bash
                    - -lc
                    - |
                      set -euo pipefail
{_gpu_opencl_bootstrap_yaml(resources, indent=22).rstrip()}
                      ROOT={output_dir}
                      mkdir -p "${{ROOT}}"

                      for YEAR in {years}; do
                        OUT="${{ROOT}}/{yearly_dir_expression}"
                        if [ -f "${{OUT}}/metrics_by_year.csv" ]; then
                          echo "year ${{YEAR}}: metrics already exist, skipping ${{OUT}}"
                          continue
                        fi

                        echo
                        echo "running {run_id_value} shard year=${{YEAR}}"
                        echo "output_dir=${{OUT}}"

                        {command} \\
                          --config {config_path.as_posix()} \\
                          --test-start-date "${{YEAR}}-01-01" \\
                          --test-end-date "${{YEAR}}-12-31" \\
                          --output-dir "${{OUT}}"
                      done
                  volumeMounts:
                    - name: opening-strength-output
                      mountPath: {mount_path}
                  resources:
                    requests:
                      cpu: "{cpu_request}"
                      memory: {memory_request}
{_gpu_resource_line(resources, indent=22).rstrip()}
                    limits:
                      cpu: "{cpu_limit}"
                      memory: {memory_limit}
{_gpu_resource_line(resources, indent=22).rstrip()}
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="TOML run config.")
    parser.add_argument(
        "--image",
        default="",
        help=f"Container image tag to run. Defaults to ${DEFAULT_IMAGE_ENV}.",
    )
    parser.add_argument(
        "--allow-mutable-image",
        action="store_true",
        help="Allow rendering a mutable image tag such as :latest.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/jobs",
        help="Directory for rendered Job manifests.",
    )
    parser.add_argument(
        "--sharded",
        action="store_true",
        help="Render a sequential per-year or per-month training job.",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Render a cluster-side pool-internal analysis job for the configured run.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    image = resolve_render_image(args.image, allow_mutable=args.allow_mutable_image)
    run_id_value = run_id(config, config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analysis:
        analysis_path = output_dir / f"{run_id_value}_pool_internal_analysis_job.yaml"
        analysis_path.write_text(
            render_pool_internal_analysis_job(config_path, config, image).rstrip() + "\n",
            encoding="utf-8",
        )
        print("rendered_k8s_jobs:")
        print(f"  analysis: {analysis_path}")
        return

    suffix = "_sharded" if args.sharded else ""
    training_path = output_dir / f"{run_id_value}{suffix}_job.yaml"
    if args.sharded:
        training_path.write_text(
            render_sharded_training_job(config_path, config, image).rstrip() + "\n",
            encoding="utf-8",
        )
    else:
        training_path.write_text(
            render_training_job(config_path, config, image).rstrip() + "\n",
            encoding="utf-8",
        )

    print("rendered_k8s_jobs:")
    print(f"  training: {training_path}")


if __name__ == "__main__":
    main()
