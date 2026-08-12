from __future__ import annotations

import shlex
import textwrap
from pathlib import Path

from opening_strength_fit import k8s_rendering_support as _render_support
from opening_strength_fit.commands.artifact_sync_metrics import DEFAULT_NEXT_CLOSE_LABEL_PVC_DIR
from opening_strength_fit.config import coerce_bool, coerce_str_list, run_id
from opening_strength_fit.config import config_value as get
from opening_strength_fit.pvc_layout import (
    output_layout,
    rolling_shard_dir_name,
    run_output_dir,
    yearly_shard_dir_name,
)

_training_support = _render_support


def _analysis_config(config: dict) -> dict:
    analysis = config.get("analysis", {})
    pool_internal = analysis.get("pool_internal", {}) if isinstance(analysis, dict) else {}
    return pool_internal if isinstance(pool_internal, dict) else {}


def _analysis_get(config: dict, key: str, default):
    return _analysis_config(config).get(key, default)


def _analysis_list(config: dict, key: str, default: list[str] | tuple[str, ...]) -> list[str]:
    return coerce_str_list(_analysis_get(config, key, default))


def _analysis_bool(config: dict, key: str, default: bool) -> bool:
    return coerce_bool(_analysis_get(config, key, default))


def _analysis_resources(config: dict) -> dict:
    resources = _analysis_config(config).get("resources", {})
    return resources if isinstance(resources, dict) else {}


def _analysis_env_from(config: dict, indent: int = 22) -> str:
    clickhouse_secret = get(config, "k8s", "clickhouse_secret", "")
    secret_names = [
        str(_analysis_get(config, "clickhouse_secret", clickhouse_secret) or ""),
        *_analysis_list(config, "env_secrets", []),
        str(_analysis_get(config, "ceph_secret", "") or ""),
    ]
    return _render_support.env_from_secrets_yaml(secret_names, indent=indent)


def _analysis_scheduler_yaml(config: dict, indent: int = 14) -> str:
    analysis = _analysis_config(config)
    selector = analysis.get("node_selector", {})
    scheduler_config = {
        "k8s": {
            "node_selector": selector if isinstance(selector, dict) else {},
            "avoid_nodes": analysis.get("avoid_nodes", get(config, "k8s", "avoid_nodes", [])),
        }
    }
    toleration_resources = (
        {"gpu_limit": 1} if _analysis_bool(config, "tolerate_gpu_nodes", False) else {}
    )
    return (
        _render_support.node_selector_yaml(scheduler_config, indent=indent)
        + _training_support.gpu_tolerations_yaml(toleration_resources, indent=indent)
        + _render_support.avoid_nodes_affinity_yaml(scheduler_config, indent=indent)
    )


def _shell_command_yaml(args: list[str], indent: int) -> str:
    quoted = [shlex.quote(str(arg)) for arg in args]
    first, *rest = quoted
    lines = [f"{' ' * indent}{first} \\"]
    for index, item in enumerate(rest):
        suffix = " \\" if index < len(rest) - 1 else ""
        lines.append(f"{' ' * (indent + 2)}{item}{suffix}")
    return "\n".join(lines)


def _analysis_wait_paths(config: dict, pvc_dir: str) -> list[str]:
    explicit = _analysis_list(config, "wait_for_paths", [])
    if explicit:
        return explicit
    completion_file = (
        str(_analysis_get(config, "wait_for_completion_file", "metrics_by_year.csv")).strip()
        or "predictions.parquet"
    )
    if _training_support.window_mode(config) == "rolling_monthly":
        layout = output_layout(config)
        return [
            f"{pvc_dir.rstrip('/')}/"
            f"{rolling_shard_dir_name(start_month, end_month, layout)}/{completion_file}"
            for start_month, end_month in _training_support.month_windows_from_config(config)
        ]
    try:
        start_year = _training_support.year_from_config(config, "test_start_date")
        end_year = _training_support.year_from_config(config, "test_end_date")
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
    for prediction_input in _analysis_list(config, "predictions", []) or [pvc_dir]:
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
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    pvc_dir = run_output_dir(config, run_id_value, mount_path=str(mount_path))
    analysis_dir = str(
        _analysis_get(config, "output_dir", f"{pvc_dir.rstrip('/')}/analysis/pool_internal_top100")
    )
    report_dir = str(_analysis_get(config, "report_dir", f"{analysis_dir.rstrip('/')}/reports"))
    default_job_name = _render_support.k8s_job_name("os-analyze", run_id_value, "pool")
    job_name = str(_analysis_get(config, "job_name", default_job_name) or "")
    resources = _analysis_resources(config)
    wait_for_paths = _render_support.wait_for_specific_paths_yaml(
        _analysis_wait_paths(config, pvc_dir),
        timeout_seconds=int(_analysis_get(config, "wait_for_path_timeout_seconds", 86400) or 86400),
        interval_seconds=int(_analysis_get(config, "wait_for_path_interval_seconds", 120) or 120),
        indent=0,
    )
    command_yaml = _shell_command_yaml(
        _analysis_command_args(
            config,
            run_id_value=run_id_value,
            pvc_dir=pvc_dir,
            analysis_dir=analysis_dir,
            report_dir=report_dir,
        ),
        indent=0,
    )
    resources_yaml = _render_support.container_resources_yaml(
        resources,
        indent=10,
        defaults=("4", "128Gi", "8", "256Gi"),
    )
    script = textwrap.indent(
        f"""\
set -euo pipefail
{wait_for_paths.rstrip()}
rm -rf {shlex.quote(analysis_dir)}
mkdir -p {shlex.quote(analysis_dir)}
mkdir -p {shlex.quote(report_dir)}
{command_yaml}
""",
        " " * 14,
    )
    return _render_support.job_manifest_yaml(
        _render_support.configured_job_pod_header_yaml(config, job_name),
        _analysis_scheduler_yaml(config, indent=6),
        image,
        script,
        mount_path,
        resources_yaml,
        env_from=_analysis_env_from(config, indent=10),
        env_before_workdir=True,
    )
