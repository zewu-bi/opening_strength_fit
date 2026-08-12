from __future__ import annotations

import hashlib
import re
import textwrap
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from opening_strength_fit.config import coerce_str_list, slug
from opening_strength_fit.config import config_value as get
from opening_strength_fit.k8s import KUBERNETES_NAME_LIMIT


def _mapping_yaml_lines(name: str, values: Mapping[str, object] | None, indent: int) -> list[str]:
    if not values:
        return []
    prefix = " " * indent
    return [f"{prefix}{name}:", *(f"{prefix}  {key}: {value}" for key, value in values.items())]


def configured_job_pod_header_yaml(
    config: dict,
    job_name: object,
    *,
    annotations: Mapping[str, object] | None = None,
    labels: Mapping[str, object] | None = None,
    spec_lines: Sequence[str] = ("backoffLimit: 0",),
    pod_labels: Mapping[str, object] | None = None,
    volume_lines: Sequence[str] = (),
    ttl_seconds_after_finished: int = 86400,
    indent: int = 0,
) -> str:
    k8s = config.get("k8s", {})
    pod_metadata = (
        ["    metadata:", *_mapping_yaml_lines("labels", pod_labels, 6)] if pod_labels else []
    )
    lines = [
        "apiVersion: batch/v1",
        "kind: Job",
        "metadata:",
        f"  name: {job_name}",
        f"  namespace: {k8s.get('namespace', 'bizewu')}",
        *_mapping_yaml_lines("annotations", annotations, 2),
        *_mapping_yaml_lines("labels", labels, 2),
        "spec:",
        *(f"  {line}" for line in spec_lines),
        f"  ttlSecondsAfterFinished: {ttl_seconds_after_finished}",
        "  template:",
        *pod_metadata,
        "    spec:",
        "      restartPolicy: Never",
        "      imagePullSecrets:",
        f"        - name: {k8s.get('image_pull_secret', 'highfort')}",
        "      volumes:",
        "        - name: opening-strength-output",
        "          persistentVolumeClaim:",
        f"            claimName: {k8s.get('pvc', 'bizewu-private-data')}",
        *volume_lines,
    ]
    return textwrap.indent("\n".join(lines) + "\n", " " * indent)


def node_selector_yaml(config: dict, indent: int) -> str:
    node_selector = config.get("k8s", {}).get("node_selector", {})
    if not node_selector:
        return ""
    values = {key: f'"{value}"' for key, value in sorted(node_selector.items())}
    return "\n".join(_mapping_yaml_lines("nodeSelector", values, indent)) + "\n"


def avoid_nodes_affinity_yaml(config: dict, indent: int) -> str:
    avoid_nodes = config.get("k8s", {}).get("avoid_nodes", [])
    if isinstance(avoid_nodes, str):
        avoid_nodes = avoid_nodes.replace(",", " ").split()
    nodes = [str(node).strip() for node in avoid_nodes if str(node).strip()]
    required_label_values = config.get("k8s", {}).get(
        "required_node_label_values",
        {},
    )
    if not isinstance(required_label_values, dict):
        raise SystemExit("k8s.required_node_label_values must be a table")
    required = []
    for key, raw_values in sorted(required_label_values.items()):
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if cleaned:
            required.append((str(key).strip(), cleaned))
    if not nodes and not required:
        return ""

    expressions = [("kubernetes.io/hostname", "NotIn", nodes)] if nodes else []
    expressions.extend((key, "In", values) for key, values in required)
    lines = [
        f"{' ' * indent}affinity:",
        f"{' ' * (indent + 2)}nodeAffinity:",
        f"{' ' * (indent + 4)}requiredDuringSchedulingIgnoredDuringExecution:",
        f"{' ' * (indent + 6)}nodeSelectorTerms:",
        f"{' ' * (indent + 8)}- matchExpressions:",
    ]
    for key, operator, values in expressions:
        lines.extend(
            [
                f"{' ' * (indent + 10)}- key: {key}",
                f"{' ' * (indent + 12)}operator: {operator}",
                f"{' ' * (indent + 12)}values:",
                *(f"{' ' * (indent + 14)}- {value}" for value in values),
            ]
        )
    return "\n".join(lines) + "\n"


def wait_for_specific_paths_yaml(
    paths: list[str],
    *,
    timeout_seconds: int,
    interval_seconds: int,
    indent: int,
) -> str:
    quoted_paths = " ".join(f'"{path.replace(chr(34), chr(92) + chr(34))}"' for path in paths)
    return textwrap.indent(
        f"""\
WAIT_PATH_TIMEOUT_SECONDS={timeout_seconds}
WAIT_PATH_INTERVAL_SECONDS={interval_seconds}
WAIT_PATH_STARTED=${{SECONDS}}
WAIT_PATHS=({quoted_paths})
for WAIT_PATH in "${{WAIT_PATHS[@]}}"; do
  until [ -f "${{WAIT_PATH}}" ]; do
    if [ $((SECONDS - WAIT_PATH_STARTED)) -ge "${{WAIT_PATH_TIMEOUT_SECONDS}}" ]; then
      echo "timed out waiting for dependency file: ${{WAIT_PATH}}" >&2
      exit 1
    fi
    echo "waiting for dependency file: ${{WAIT_PATH}}"
    sleep "${{WAIT_PATH_INTERVAL_SECONDS}}"
  done
  echo "dependency file is ready: ${{WAIT_PATH}}"
done
""",
        " " * indent,
    )


def env_from_secrets_yaml(secret_names: list[str], indent: int) -> str:
    unique_names = list(dict.fromkeys(item for name in secret_names if (item := str(name).strip())))
    if not unique_names:
        return ""
    lines = [
        "envFrom:",
        *(line for name in unique_names for line in ("  - secretRef:", f"      name: {name}")),
    ]
    return textwrap.indent("\n".join(lines) + "\n", " " * indent)


def output_volume_mount_yaml(mount_path: object, indent: int) -> str:
    return textwrap.indent(
        f"- name: opening-strength-output\n  mountPath: {mount_path}\n", " " * indent
    )


def job_manifest_yaml(
    header: str,
    scheduler: str,
    image: str,
    script: str,
    mount_path: object,
    resources: str,
    *,
    name: str = "opening-strength-fit",
    env_from: str = "",
    env_before_workdir: bool = False,
    indexed: bool = False,
    config_volume: str = "",
    extra_mounts: str = "",
    command_yaml: str = "",
) -> str:
    before_workdir, after_workdir = (env_from, "") if env_before_workdir else ("", env_from)
    indexed_env = (
        ""
        if not indexed
        else """\
          env:
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: "metadata.annotations['batch.kubernetes.io/job-completion-index']"
"""
    )
    command_yaml = (
        command_yaml
        or f"""\
          command:
            - /bin/bash
            - -lc
            - |
{script.rstrip()}
"""
    )
    return f"""\
{header}{config_volume}{scheduler}      containers:
        - name: {name}
          image: {image}
          imagePullPolicy: Always
{before_workdir}          workingDir: /app/opening_strength_fit
{after_workdir}{indexed_env}{command_yaml.rstrip()}
          volumeMounts:
{output_volume_mount_yaml(mount_path, 12).rstrip()}
{extra_mounts}{resources.rstrip()}
"""


def container_resources_yaml(
    resources: Mapping[str, object],
    *,
    indent: int,
    defaults: tuple[str, str, str, str] = ("4", "16Gi", "8", "32Gi"),
    gpu_count: str = "",
    blank_without_gpu: bool = False,
) -> str:
    cpu_request, memory_request, cpu_limit, memory_limit = defaults
    prefix = " " * indent
    child = " " * (indent + 4)
    gpu = [f'{child}nvidia.com/gpu: "{gpu_count}"'] if gpu_count else []
    lines = [
        f"{prefix}resources:",
        f"{prefix}  requests:",
        f'{child}cpu: "{resources.get("cpu_request", cpu_request)}"',
        f"{child}memory: {resources.get('memory_request', memory_request)}",
        *gpu,
        *(("",) if blank_without_gpu and not gpu else ()),
        f"{prefix}  limits:",
        f'{child}cpu: "{resources.get("cpu_limit", cpu_limit)}"',
        f"{child}memory: {resources.get('memory_limit', memory_limit)}",
        *gpu,
    ]
    return "\n".join(lines) + "\n"


def rolling_completion_files(config: dict, command: str) -> list[str]:
    if configured := coerce_str_list(get(config, "k8s", "completion_files", [])):
        return configured
    outputs = {
        "capacity_acceptance": "capacity_acceptance_trace.json capacity_acceptance_daily_summary.csv",
        "capacity_audit": "capacity_audit_trace.json capacity_audit_summary.csv",
        "strategy_acceptance": "_SUCCESS strategy_acceptance_summary.csv",
        "ask_level_attribution": "ask_level_attribution_trace.json _SUCCESS",
        "execution_context": "execution_context_trace.json _SUCCESS",
        "exposure_input": "exposure_input_trace.json exposure_input.parquet",
        "exposure_audit": "exposure_audit_trace.json exposure_audit_summary.csv",
        "alpha_conditioned_rolling_validation": "rolling_summary.csv",
        "feature_audit": "feature_audit_trace.json feature_audit_metrics.csv",
        "feature_hygiene": "feature_hygiene_trace.json feature_hygiene.csv",
    }
    default = (
        "_SUCCESS metrics_by_year.csv predictions.parquet"
        if command == "osf-train"
        else "metrics_by_year.csv"
    )
    return outputs.get(str(get(config, "run", "kind", "")).strip().lower(), default).split()


def shell_file_check(files: list[str]) -> str:
    return " && ".join(f'[ -f "${{OUT}}/{file}" ]' for file in files)


def gpu_count(resources: dict) -> str:
    value = str(resources.get("gpu_limit", resources.get("nvidia_gpu", "0")) or "0")
    return "" if value in {"", "0", "0.0", "none", "None"} else value


def training_resources_yaml(resources: dict, indent: int) -> str:
    return container_resources_yaml(
        resources, indent=indent, gpu_count=gpu_count(resources), blank_without_gpu=True
    )


def gpu_tolerations_yaml(resources: dict, indent: int = 14) -> str:
    if not gpu_count(resources) and not resources.get("tolerate_gpu_nodes", False):
        return ""
    return textwrap.indent(
        'tolerations:\n  - key: has_gpu\n    operator: Equal\n    value: "true"\n'
        "    effect: NoSchedule\n",
        " " * indent,
    )


def gpu_opencl_bootstrap_yaml(resources: dict, indent: int) -> str:
    if not gpu_count(resources):
        return ""
    return textwrap.indent(
        "mkdir -p /etc/OpenCL/vendors\n"
        "echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd\n",
        " " * indent,
    )


def wait_for_paths_yaml(config: dict, indent: int) -> str:
    paths = coerce_str_list(get(config, "k8s", "wait_for_paths", []))
    return (
        wait_for_specific_paths_yaml(
            paths,
            timeout_seconds=int(get(config, "k8s", "wait_for_path_timeout_seconds", 21600)),
            interval_seconds=int(get(config, "k8s", "wait_for_path_interval_seconds", 60)),
            indent=indent,
        )
        if paths
        else ""
    )


def scheduler_yaml(config: dict, resources: dict, indent: int = 14) -> str:
    return (
        node_selector_yaml(config, indent)
        + avoid_nodes_affinity_yaml(config, indent)
        + gpu_tolerations_yaml(resources, indent)
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
        preamble = (gpu_opencl_bootstrap_yaml(resources, 0) + wait_for_paths).rstrip().splitlines()
        lines = [
            "command:",
            "  - /bin/bash",
            "  - -lc",
            "  - |",
            "    set -euo pipefail",
            *(f"    {line}" for line in preamble),
            f"    exec {command} \\",
            f"      --config {config_path.as_posix()} \\",
            f"      --output-dir {output_dir}",
        ]
        return textwrap.indent("\n".join(lines) + "\n", " " * indent)
    return textwrap.indent(
        f"command:\n  - {command}\n  - --config\n  - {config_path.as_posix()}\n"
        f"  - --output-dir\n  - {output_dir}\n",
        " " * indent,
    )


def year_from_config(config: dict, key: str) -> int:
    if not (value := get(config, "window", key, None)):
        raise SystemExit(
            f"--sharded requires [window].{key}; set explicit test_start_date/test_end_date"
        )
    return date.fromisoformat(str(value)).year


def month_windows_from_config(config: dict) -> list[tuple[str, str]]:
    start, end = (
        get(config, "window", key, None) for key in ("test_start_month", "test_end_month")
    )
    if not start or not end:
        raise SystemExit(
            "--sharded monthly requires [window].test_start_month and [window].test_end_month"
        )
    months = pd.period_range(str(start), str(end), freq="M")
    test_months = int(get(config, "window", "test_months", 1) or 1)
    stride = int(get(config, "window", "test_stride_months", test_months) or test_months)
    if test_months < 1 or stride < 1:
        key = "test_months" if test_months < 1 else "test_stride_months"
        raise SystemExit(f"[window].{key} must be >= 1")
    windows = [
        (str(months[index]), str(months[index + test_months - 1]))
        for index in range(0, len(months) - test_months + 1, stride)
    ]
    if not windows:
        raise SystemExit(
            "sharded rolling monthly produced no test windows; check "
            "[window].test_months/test_stride_months/test_start_month/test_end_month"
        )
    return windows


def shard_parallelism(config: dict, resources: dict) -> int:
    value = get(config, "k8s", "shard_parallelism", resources.get("shard_parallelism", 1))
    return max(1, int(value or 1))


def shard_job_mode(config: dict) -> str:
    mode = str(get(config, "k8s", "shard_job_mode", "indexed") or "indexed")
    mode = mode.strip().lower().replace("-", "_")
    if mode not in {"indexed", "separate"}:
        raise SystemExit("k8s.shard_job_mode must be 'indexed' or 'separate'")
    return mode


def window_mode(config: dict) -> str:
    return str(get(config, "window", "mode", "chronological"))


def k8s_env_from(config: dict, indent: int = 18) -> str:
    return env_from_secrets_yaml(
        [
            str(get(config, "k8s", "clickhouse_secret", "") or ""),
            *coerce_str_list(get(config, "k8s", "env_secrets", [])),
            str(get(config, "k8s", "ceph_secret", "") or ""),
        ],
        indent,
    )


def _training_config_map(config: dict) -> tuple[str, str, str, str] | None:
    k8s = config.get("k8s", {})
    name = str(k8s.get("config_map_name", "") or "").strip()
    mount_path = str(k8s.get("config_map_mount_path", "") or "").strip()
    sub_path = str(k8s.get("config_map_sub_path", "") or "").strip()
    volume_name = str(k8s.get("config_map_volume_name", "run-config") or "").strip()
    if not name and not mount_path and not sub_path:
        return None
    if not name or not mount_path or not volume_name:
        raise SystemExit(
            "k8s config-map mounting requires config_map_name, config_map_mount_path, "
            "and a non-empty config_map_volume_name"
        )
    return name, volume_name, mount_path, sub_path


def training_config_map_volume_yaml(config: dict, indent: int) -> str:
    if (spec := _training_config_map(config)) is None:
        return ""
    name, volume_name, *_ = spec
    return textwrap.indent(
        f"- name: {volume_name}\n  configMap:\n    name: {name}\n",
        " " * indent,
    )


def training_config_map_mount_yaml(config: dict, indent: int) -> str:
    if (spec := _training_config_map(config)) is None:
        return ""
    _, volume_name, mount_path, sub_path = spec
    sub_path_yaml = f"  subPath: {sub_path}\n" if sub_path else ""
    return textwrap.indent(
        f"- name: {volume_name}\n  mountPath: {mount_path}\n{sub_path_yaml}", " " * indent
    )


def compact_run_slug(run_id_value: str, *, max_length: int) -> str:
    run_slug = slug(run_id_value)
    if len(run_slug) <= max_length:
        return run_slug

    tokens = [token for token in run_slug.split("-") if token]
    model = tokens[0] if tokens else ""
    delay = next((token for token in tokens if re.fullmatch(r"delay\d+", token)), "")
    horizon = next((token for token in tokens if re.fullmatch(r"\d+[mhd]", token)), "")
    weight = next((token for token in tokens if re.fullmatch(r"w\d+", token)), "")
    version = next((token for token in reversed(tokens) if re.fullmatch(r"v\d+", token)), "")

    candidates: list[list[str]] = []
    if "mixed" in tokens and weight:
        candidates.extend(
            (
                [model, delay, horizon, "mixed", weight, version],
                [model, delay, "mixed", weight],
                ["mixed", weight, version],
            )
        )
    if "rolling" in tokens:
        candidates.extend(([model, delay, horizon, "roll", version], [model, delay, "roll"]))
    candidates.append(
        [
            "roll" if token == "rolling" else token
            for token in tokens
            if (
                token == model
                or token in {"mixed", "top100"}
                or re.fullmatch(r"delay\d+", token)
                or re.fullmatch(r"\d+[mhd]", token)
                or re.fullmatch(r"w\d+", token)
                or re.fullmatch(r"v\d+", token)
                or token == "rolling"
            )
        ]
    )

    for parts in candidates:
        compact = "-".join(part for part in parts if part)
        if compact and len(compact) <= max_length:
            return compact

    return run_slug[:max_length].rstrip("-")


def k8s_job_name(
    prefix: str,
    run_id_value: str,
    suffix: str = "",
    *,
    max_length: int = KUBERNETES_NAME_LIMIT,
) -> str:
    candidate = "-".join(part.strip("-") for part in (prefix, slug(run_id_value), suffix) if part)
    if len(candidate) <= max_length:
        return candidate

    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
    tail = f"-{suffix}-{digest}" if suffix else f"-{digest}"
    head = f"{prefix.strip('-')}-"
    keep = max_length - len(head) - len(tail)
    if (keep := max_length - len(head) - len(tail)) < 1:
        keep = max_length - len(tail) - 1
        head = ""
    return f"{head}{compact_run_slug(run_id_value, max_length=keep)}{tail}"
