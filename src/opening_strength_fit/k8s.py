from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from opening_strength_fit.config import config_value, load_toml, run_id, slug
from opening_strength_fit.pvc_layout import output_layout, run_output_dir

DEFAULT_IMAGE = "registry.corp.highfortfunds.com/bizewu/opening-strength-fit:latest"
KUBERNETES_NAME_LIMIT = 63


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    pvc_dir: str
    namespace: str
    pvc: str
    mount_path: str
    pull_secret: str
    image: str
    test_start_year: int
    test_end_year: int
    test_start_month: str = ""
    test_end_month: str = ""
    test_months: int = 1
    test_stride_months: int = 1
    kind: str = "experiment"
    local_dir: str = ""
    pool_internal_analysis_enabled: bool = False
    pool_internal_analysis_dir: str = ""
    pool_internal_record_prefix: str = ""
    pool_internal_archive_profile: str = ""
    output_layout: str = "legacy"

    @property
    def job_name(self) -> str:
        return slug(f"opening-strength-{self.run_id}")


def _year_from_date(value: object, default: int) -> int:
    if not value:
        return default
    return date.fromisoformat(str(value)).year


def _year_from_month(value: object, default: int) -> int:
    if not value:
        return default
    return int(str(value).split("-", 1)[0])


def _nested_mapping(config: dict, *keys: str) -> dict:
    values = config
    for key in keys:
        value = values.get(key, {}) if isinstance(values, dict) else {}
        if not isinstance(value, dict):
            return {}
        values = value
    return values


def _nested_bool(config: dict, keys: tuple[str, ...], default: bool) -> bool:
    values = _nested_mapping(config, *keys[:-1])
    value = values.get(keys[-1], default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _nested_str(config: dict, keys: tuple[str, ...], default: str) -> str:
    values = _nested_mapping(config, *keys[:-1])
    value = values.get(keys[-1], default)
    return default if value is None else str(value)


def load_run_spec(path: str | Path) -> RunSpec:
    config_path = Path(path)
    config = load_toml(config_path)
    run_id_value = run_id(config, config_path)
    mount_path = str(config_value(config, "k8s", "mount_path", "/mnt/output"))
    pvc_dir = run_output_dir(config, run_id_value, mount_path=mount_path)
    default_pool_internal_dir = f"{pvc_dir.rstrip('/')}/analysis/pool_internal_top100"
    start_year = _year_from_date(
        config_value(config, "window", "test_start_date", None),
        0,
    )
    end_year = _year_from_date(
        config_value(config, "window", "test_end_date", None),
        start_year,
    )
    test_start_month = str(config_value(config, "window", "test_start_month", "") or "")
    test_end_month = str(config_value(config, "window", "test_end_month", "") or "")
    if start_year <= 0 and test_start_month:
        start_year = _year_from_month(test_start_month, 0)
    if end_year <= 0 and test_end_month:
        end_year = _year_from_month(test_end_month, start_year)
    return RunSpec(
        run_id=run_id_value,
        pvc_dir=pvc_dir,
        namespace=str(config_value(config, "k8s", "namespace", "bizewu")),
        pvc=str(config_value(config, "k8s", "pvc", "bizewu-private-data")),
        mount_path=mount_path,
        pull_secret=str(config_value(config, "k8s", "image_pull_secret", "highfort")),
        image=str(config_value(config, "k8s", "helper_image", DEFAULT_IMAGE)),
        test_start_year=start_year,
        test_end_year=end_year,
        test_start_month=test_start_month,
        test_end_month=test_end_month,
        test_months=int(config_value(config, "window", "test_months", 1) or 1),
        test_stride_months=int(
            config_value(
                config,
                "window",
                "test_stride_months",
                config_value(config, "window", "test_months", 1),
            )
            or 1
        ),
        kind=str(config_value(config, "run", "kind", "experiment")),
        local_dir=str(config_value(config, "output", "local_dir", "") or ""),
        pool_internal_analysis_enabled=_nested_bool(
            config,
            ("analysis", "pool_internal", "enabled"),
            False,
        ),
        pool_internal_analysis_dir=_nested_str(
            config,
            ("analysis", "pool_internal", "output_dir"),
            default_pool_internal_dir,
        ),
        pool_internal_record_prefix=_nested_str(
            config,
            ("analysis", "pool_internal", "record_prefix"),
            run_id_value,
        ),
        pool_internal_archive_profile=_nested_str(
            config,
            ("analysis", "pool_internal", "archive", "profile"),
            "",
        ),
        output_layout=output_layout(config),
    )


def run_command(
    command: list[str],
    *,
    capture_output: bool = False,
    stdout=None,
    check: bool = True,
) -> str:
    result = subprocess.run(
        command,
        check=check,
        text=capture_output,
        capture_output=capture_output,
        stdout=stdout,
    )
    return result.stdout if capture_output else ""


def command_succeeds(command: list[str]) -> bool:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.returncode == 0


def temporary_pod_name(pod_prefix: str, run_id_value: str) -> str:
    prefix = slug(pod_prefix)
    run_slug = slug(run_id_value)
    candidate = f"{prefix}-{run_slug}".strip("-")
    if len(candidate) <= KUBERNETES_NAME_LIMIT:
        return candidate

    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
    keep = KUBERNETES_NAME_LIMIT - len(prefix) - len(digest) - 2
    if keep < 1:
        prefix_keep = KUBERNETES_NAME_LIMIT - len(digest) - 1
        return f"{prefix[:prefix_keep].rstrip('-')}-{digest}"
    return f"{prefix}-{run_slug[:keep].rstrip('-')}-{digest}"


def ensure_temp_pod(
    hfcli: str,
    spec: RunSpec,
    timeout: str,
    pod_prefix: str,
    *,
    dry_run: bool = False,
) -> str:
    pod_name = temporary_pod_name(pod_prefix, spec.run_id)
    overrides = {
        "apiVersion": "v1",
        "spec": {
            "imagePullSecrets": [{"name": spec.pull_secret}],
            "containers": [
                {
                    "name": "opening-strength-helper",
                    "image": spec.image,
                    "command": ["/bin/sh", "-c", "sleep 3600"],
                    "volumeMounts": [
                        {
                            "name": "opening-strength-output",
                            "mountPath": spec.mount_path,
                        }
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "opening-strength-output",
                    "persistentVolumeClaim": {"claimName": spec.pvc},
                }
            ],
        },
    }
    delete_command = [
        hfcli,
        "kubectl",
        "delete",
        "pod",
        pod_name,
        "-n",
        spec.namespace,
        "--ignore-not-found",
    ]
    create_command = [
        hfcli,
        "kubectl",
        "run",
        pod_name,
        "-n",
        spec.namespace,
        "--restart=Never",
        f"--image={spec.image}",
        f"--overrides={json.dumps(overrides)}",
        "--command",
        "--",
        "/bin/sh",
        "-c",
        "sleep 3600",
    ]
    wait_command = [
        hfcli,
        "kubectl",
        "wait",
        "--for=condition=Ready",
        f"pod/{pod_name}",
        "-n",
        spec.namespace,
        f"--timeout={timeout}",
    ]
    if dry_run:
        print("temporary_pull_pod:")
        print(f"  name: {pod_name}")
        print(f"  delete: {' '.join(delete_command)}")
        print(f"  create: {' '.join(create_command)}")
        print(f"  wait: {' '.join(wait_command)}")
        return pod_name

    run_command(delete_command)
    run_command(create_command)
    run_command(wait_command)
    return pod_name


def delete_temp_pod(hfcli: str, namespace: str, pod_name: str) -> None:
    run_command(
        [
            hfcli,
            "kubectl",
            "delete",
            "pod",
            pod_name,
            "-n",
            namespace,
            "--ignore-not-found",
        ]
    )
