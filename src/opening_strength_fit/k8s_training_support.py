from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pandas as pd

from opening_strength_fit.config import config_value as get
from opening_strength_fit.k8s_rendering_support import (
    avoid_nodes_affinity_yaml,
    env_from_secrets_yaml,
    node_selector_yaml,
    wait_for_specific_paths_yaml,
)


def rolling_completion_files(config: dict, command: str) -> list[str]:
    run_kind = str(get(config, "run", "kind", "")).strip().lower()
    outputs = {
        "capacity_acceptance": [
            "capacity_acceptance_trace.json",
            "capacity_acceptance_daily_summary.csv",
        ],
        "capacity_audit": ["capacity_audit_trace.json", "capacity_audit_summary.csv"],
        "strategy_acceptance": ["_SUCCESS", "strategy_acceptance_summary.csv"],
        "ask_level_attribution": ["ask_level_attribution_trace.json", "_SUCCESS"],
        "execution_context": ["execution_context_trace.json", "_SUCCESS"],
        "exposure_input": ["exposure_input_trace.json", "exposure_input.parquet"],
        "exposure_audit": ["exposure_audit_trace.json", "exposure_audit_summary.csv"],
        "alpha_conditioned_rolling_validation": ["rolling_summary.csv"],
        "feature_audit": ["feature_audit_trace.json", "feature_audit_metrics.csv"],
        "feature_hygiene": ["feature_hygiene_trace.json", "feature_hygiene.csv"],
    }
    if run_kind in outputs:
        return outputs[run_kind]
    if command == "osf-train":
        return ["_SUCCESS", "metrics_by_year.csv", "predictions.parquet"]
    return ["metrics_by_year.csv"]


def shell_file_check(files: list[str]) -> str:
    return " && ".join(f'[ -f "${{OUT}}/{file}" ]' for file in files)


def gpu_count(resources: dict) -> str:
    gpu_limit = str(resources.get("gpu_limit", resources.get("nvidia_gpu", "0")) or "0")
    if gpu_limit in {"", "0", "0.0", "none", "None"}:
        return ""
    return gpu_limit


def gpu_resource_line(resources: dict, indent: int = 14) -> str:
    count = gpu_count(resources)
    if not count:
        return ""
    return f'{" " * indent}nvidia.com/gpu: "{count}"\n'


def gpu_tolerations_yaml(resources: dict, indent: int = 14) -> str:
    if not gpu_count(resources):
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


def gpu_opencl_bootstrap_yaml(resources: dict, indent: int) -> str:
    if not gpu_count(resources):
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


def wait_for_paths_script(config: dict) -> str:
    paths = get(config, "k8s", "wait_for_paths", []) or []
    if isinstance(paths, str):
        paths = paths.replace(",", " ").split()
    paths = [str(path).strip() for path in paths if str(path).strip()]
    if not paths:
        return ""
    return wait_for_specific_paths_yaml(
        paths,
        timeout_seconds=int(get(config, "k8s", "wait_for_path_timeout_seconds", 21600)),
        interval_seconds=int(get(config, "k8s", "wait_for_path_interval_seconds", 60)),
        indent=0,
    )


def wait_for_paths_yaml(config: dict, indent: int) -> str:
    script = wait_for_paths_script(config)
    if not script:
        return ""
    return textwrap.indent(script, " " * indent)


def scheduler_yaml(config: dict, resources: dict, indent: int = 14) -> str:
    return (
        node_selector_yaml(config, indent=indent)
        + avoid_nodes_affinity_yaml(config, indent=indent)
        + gpu_tolerations_yaml(resources, indent=indent)
    )


def training_command_yaml(
    *,
    command: str,
    config_path: Path,
    output_dir: str,
    resources: dict,
    wait_for_paths: str = "",
    indent: int = 18,
) -> str:
    if gpu_count(resources) or wait_for_paths.strip():
        preamble_lines: list[str] = []
        if gpu_count(resources):
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


def year_from_config(config: dict, key: str) -> int:
    value = get(config, "window", key, None)
    if not value:
        raise SystemExit(
            f"--sharded requires [window].{key}; set explicit test_start_date/test_end_date"
        )
    return date.fromisoformat(str(value)).year


def month_range_from_config(config: dict) -> list[str]:
    start = get(config, "window", "test_start_month", None)
    end = get(config, "window", "test_end_month", None)
    if not start or not end:
        raise SystemExit(
            "--sharded monthly requires [window].test_start_month and [window].test_end_month"
        )
    return [str(month) for month in pd.period_range(str(start), str(end), freq="M")]


def month_windows_from_config(config: dict) -> list[tuple[str, str]]:
    months = month_range_from_config(config)
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


def shard_parallelism(config: dict, resources: dict) -> int:
    raw = get(config, "k8s", "shard_parallelism", resources.get("shard_parallelism", 1))
    return max(1, int(raw or 1))


def shard_job_mode(config: dict) -> str:
    mode = str(get(config, "k8s", "shard_job_mode", "indexed") or "indexed")
    mode = mode.strip().lower().replace("-", "_")
    if mode not in {"indexed", "separate"}:
        raise SystemExit("k8s.shard_job_mode must be 'indexed' or 'separate'")
    return mode


def window_mode(config: dict) -> str:
    return str(get(config, "window", "mode", "chronological"))


def k8s_env_from(config: dict, indent: int = 18) -> str:
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
    return env_from_secrets_yaml(secret_names, indent=indent)
