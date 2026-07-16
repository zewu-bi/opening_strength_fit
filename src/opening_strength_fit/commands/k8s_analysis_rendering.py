from __future__ import annotations

import shlex
import textwrap
from datetime import date
from pathlib import Path

import pandas as pd

from opening_strength_fit.commands.artifact_sync_metrics import DEFAULT_NEXT_CLOSE_LABEL_PVC_DIR
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
    wait_for_specific_paths_yaml as _wait_for_specific_paths_yaml,
)
from opening_strength_fit.config import config_value as get
from opening_strength_fit.config import run_id
from opening_strength_fit.pvc_layout import (
    output_layout,
    rolling_shard_dir_name,
    run_output_dir,
    yearly_shard_dir_name,
)


def _analysis_config(config: dict) -> dict:
    analysis = config.get("analysis", {})
    if not isinstance(analysis, dict):
        return {}
    pool_internal = analysis.get("pool_internal", {})
    return pool_internal if isinstance(pool_internal, dict) else {}


def _analysis_get(config: dict, key: str, default):
    return _analysis_config(config).get(key, default)


def _analysis_list(config: dict, key: str, default: list[str] | tuple[str, ...]) -> list[str]:
    value = _analysis_get(config, key, default)
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = [str(item) for item in value]
    return [part.strip() for part in parts if part and part.strip()]


def _analysis_bool(config: dict, key: str, default: bool) -> bool:
    value = _analysis_get(config, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _analysis_resources(config: dict) -> dict:
    resources = _analysis_config(config).get("resources", {})
    return resources if isinstance(resources, dict) else {}


def _analysis_env_from(config: dict, indent: int = 22) -> str:
    secret_names = []
    clickhouse_secret = str(
        _analysis_get(config, "clickhouse_secret", get(config, "k8s", "clickhouse_secret", ""))
        or ""
    )
    if clickhouse_secret:
        secret_names.append(clickhouse_secret)
    secret_names.extend(_analysis_list(config, "env_secrets", []))
    ceph_secret = str(_analysis_get(config, "ceph_secret", "") or "")
    if ceph_secret:
        secret_names.append(ceph_secret)
    return _env_from_secrets_yaml(secret_names, indent=indent)


def _analysis_scheduler_yaml(config: dict, indent: int = 14) -> str:
    analysis = _analysis_config(config)
    scheduler_config = {"k8s": {}}
    if isinstance(analysis.get("node_selector", {}), dict):
        scheduler_config["k8s"]["node_selector"] = analysis.get("node_selector", {})
    if "avoid_nodes" in analysis:
        scheduler_config["k8s"]["avoid_nodes"] = analysis.get("avoid_nodes", [])
    else:
        scheduler_config["k8s"]["avoid_nodes"] = get(config, "k8s", "avoid_nodes", [])
    return _node_selector_yaml(scheduler_config, indent=indent) + _avoid_nodes_affinity_yaml(
        scheduler_config, indent=indent
    )


def _window_mode(config: dict) -> str:
    return str(get(config, "window", "mode", "chronological"))


def _month_range_from_config(config: dict) -> list[str]:
    start = get(config, "window", "test_start_month", None)
    end = get(config, "window", "test_end_month", None)
    if not start or not end:
        raise SystemExit(
            "--analysis monthly requires [window].test_start_month and [window].test_end_month"
        )
    return [str(month) for month in pd.period_range(str(start), str(end), freq="M")]


def _month_windows_from_config(config: dict) -> list[tuple[str, str]]:
    months = _month_range_from_config(config)
    test_months = int(get(config, "window", "test_months", 1) or 1)
    stride_months = int(get(config, "window", "test_stride_months", test_months) or test_months)
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
    return windows


def _year_from_config(config: dict, key: str) -> int:
    value = get(config, "window", key, None)
    if not value:
        raise SystemExit(f"--analysis requires [window].{key} for yearly waits")
    return date.fromisoformat(str(value)).year


def _shell_command_yaml(args: list[str], indent: int) -> str:
    quoted = [shlex.quote(str(arg)) for arg in args]
    first, *rest = quoted
    lines = [f"{' ' * indent}{first} \\"]
    for index, item in enumerate(rest):
        suffix = " \\" if index < len(rest) - 1 else ""
        lines.append(f"{' ' * (indent + 2)}{item}{suffix}")
    return "\n".join(lines)


def _analysis_prediction_inputs(config: dict, pvc_dir: str) -> list[str]:
    predictions = _analysis_list(config, "predictions", [])
    return predictions or [pvc_dir]


def _analysis_wait_paths(config: dict, pvc_dir: str) -> list[str]:
    explicit = _analysis_list(config, "wait_for_paths", [])
    if explicit:
        return explicit
    completion_file = str(_analysis_get(config, "wait_for_completion_file", "metrics_by_year.csv"))
    completion_file = completion_file.strip()
    if not completion_file:
        completion_file = "predictions.parquet"
    if _window_mode(config) == "rolling_monthly":
        layout = output_layout(config)
        return [
            f"{pvc_dir.rstrip('/')}/"
            f"{rolling_shard_dir_name(start_month, end_month, layout)}/{completion_file}"
            for start_month, end_month in _month_windows_from_config(config)
        ]
    try:
        start_year = _year_from_config(config, "test_start_date")
        end_year = _year_from_config(config, "test_end_date")
    except SystemExit:
        return [f"{pvc_dir.rstrip('/')}/{completion_file}"]
    layout = output_layout(config)
    return [
        f"{pvc_dir.rstrip('/')}/{yearly_shard_dir_name(year, layout)}/{completion_file}"
        for year in range(start_year, end_year + 1)
    ]


def _analysis_command_args(
    config: dict,
    *,
    config_path: Path,
    run_id_value: str,
    pvc_dir: str,
    analysis_dir: str,
    report_dir: str,
) -> list[str]:
    variant = str(_analysis_get(config, "variant", run_id_value) or run_id_value)
    plot_prefix = str(_analysis_get(config, "plot_prefix", variant) or variant)
    plot_variant_label = str(_analysis_get(config, "plot_variant_label", variant) or variant)
    top_n = int(_analysis_get(config, "top_n", get(config, "evaluation", "top_n", 100)) or 100)
    pool_lag = int(_analysis_get(config, "pool_date_lag_sessions", 0) or 0)
    next_close_label_input = str(
        _analysis_get(config, "next_close_label_input", DEFAULT_NEXT_CLOSE_LABEL_PVC_DIR)
        or DEFAULT_NEXT_CLOSE_LABEL_PVC_DIR
    )
    args = ["osf-analyze-pool-internal-top100"]
    for prediction_input in _analysis_prediction_inputs(config, pvc_dir):
        args.extend(["--predictions", prediction_input])
    args.extend(
        [
            "--next-close-label-input",
            next_close_label_input,
            "--run-id",
            run_id_value,
            "--variant",
            variant,
            "--output-dir",
            analysis_dir,
            "--report-dir",
            report_dir,
            "--plot-prefix",
            plot_prefix,
            "--plot-variant-label",
            plot_variant_label,
            "--plot-period",
            str(_analysis_get(config, "plot_period", "month") or "month"),
            "--top-n",
            str(top_n),
            "--pool-date-lag-sessions",
            str(pool_lag),
        ]
    )
    for pool in _analysis_list(config, "pools", ["universe", "S", "M", "L"]):
        args.extend(["--pool", pool])
    if _analysis_bool(config, "weekly_enabled", False):
        args.extend(
            [
                "--weekly-report-dir",
                str(_analysis_get(config, "weekly_report_dir", f"{analysis_dir}/weekly")),
                "--weekly-output-prefix",
                str(_analysis_get(config, "weekly_output_prefix", plot_prefix) or plot_prefix),
                "--weekly-rolling-weeks",
                str(int(_analysis_get(config, "weekly_rolling_weeks", 4) or 4)),
            ]
        )
    return args


def render_pool_internal_analysis_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    namespace = get(config, "k8s", "namespace", "bizewu")
    pull_secret = get(config, "k8s", "image_pull_secret", "highfort")
    pvc = get(config, "k8s", "pvc", "bizewu-private-data")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    pvc_dir = run_output_dir(config, run_id_value, mount_path=str(mount_path))
    analysis_dir = str(
        _analysis_get(config, "output_dir", f"{pvc_dir.rstrip('/')}/analysis/pool_internal_top100")
    )
    report_dir = str(_analysis_get(config, "report_dir", f"{analysis_dir.rstrip('/')}/reports"))
    job_name = str(
        _analysis_get(config, "job_name", _k8s_job_name("os-analyze", run_id_value, "pool")) or ""
    )
    resources = _analysis_resources(config)
    wait_for_paths = _wait_for_specific_paths_yaml(
        _analysis_wait_paths(config, pvc_dir),
        timeout_seconds=int(_analysis_get(config, "wait_for_path_timeout_seconds", 86400) or 86400),
        interval_seconds=int(_analysis_get(config, "wait_for_path_interval_seconds", 120) or 120),
        indent=22,
    )
    command_yaml = _shell_command_yaml(
        _analysis_command_args(
            config,
            config_path=config_path,
            run_id_value=run_id_value,
            pvc_dir=pvc_dir,
            analysis_dir=analysis_dir,
            report_dir=report_dir,
        ),
        indent=22,
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
{_analysis_scheduler_yaml(config, indent=14).rstrip()}
              containers:
                - name: opening-strength-fit
                  image: {image}
                  imagePullPolicy: Always
{_analysis_env_from(config, indent=18)}                  workingDir: /app/opening_strength_fit
                  command:
                    - /bin/bash
                    - -lc
                    - |
                      set -euo pipefail
{wait_for_paths.rstrip()}
                      rm -rf {shlex.quote(analysis_dir)}
                      mkdir -p {shlex.quote(analysis_dir)}
                      mkdir -p {shlex.quote(report_dir)}
{command_yaml}
                  volumeMounts:
                    - name: opening-strength-output
                      mountPath: {mount_path}
                  resources:
                    requests:
                      cpu: "{resources.get("cpu_request", "4")}"
                      memory: {resources.get("memory_request", "128Gi")}
                    limits:
                      cpu: "{resources.get("cpu_limit", "8")}"
                      memory: {resources.get("memory_limit", "256Gi")}
        """
    )
