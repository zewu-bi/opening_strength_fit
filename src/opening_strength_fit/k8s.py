from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from opening_strength_fit.config import coerce_bool, load_toml, run_id, slug
from opening_strength_fit.config import config_value as get
from opening_strength_fit.pvc_layout import output_layout, run_output_dir

DEFAULT_IMAGE = "registry.corp.highfortfunds.com/bizewu/opening-strength-fit:latest"
KUBERNETES_NAME_LIMIT = 63
RENDER_MODES = {
    "training": ("training", "_job.yaml"),
    "sharded": ("sharded_training", "_sharded_job.yaml"),
    "indexed": ("indexed_builder", "_job.yaml"),
    "top1000": ("pool_internal_analysis", "_job.yaml"),
    "analysis": ("pool_internal_analysis", "_pool_internal_analysis_job.yaml"),
    "matrix": ("sharded_training", "_matrix_job.yaml"),
}


@dataclass(frozen=True)
class RenderedJobSpec:
    mode: str
    kind: str
    sha256: str
    suffix: str


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


def rendered_job_specs(config: dict) -> tuple[RenderedJobSpec, ...]:
    specs = []
    sections = (
        ((_nested_mapping(config, "k8s")), {"indexed", "matrix", "top1000", "training", "sharded"}),
        ((_nested_mapping(config, "k8s", "sharded")), {"sharded"}),
        ((_nested_mapping(config, "analysis", "pool_internal")), {"analysis"}),
    )
    for section, allowed_modes in sections:
        mode = str(section.get("render_mode", "") or "").strip()
        digest = str(section.get("render_sha256", "") or "").strip()
        if not mode and not digest:
            continue
        if mode not in allowed_modes:
            raise ValueError(f"render_mode must be one of: {', '.join(sorted(allowed_modes))}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("render_sha256 must be a lowercase SHA-256 digest")
        if mode == "sharded" and any(item.mode == "sharded" for item in specs):
            raise ValueError("sharded render_mode must be declared only once")
        kind, suffix = RENDER_MODES[mode]
        specs.append(RenderedJobSpec(mode, kind, digest, suffix))
    return tuple(specs)


def rendered_job_image(config: dict, mode: str) -> str:
    k8s_image = str(_nested_mapping(config, "k8s").get("helper_image", "") or "").strip()
    mode_image = str(_nested_mapping(config, "k8s", mode).get("helper_image", "") or "").strip()
    if mode == "analysis" and not mode_image:
        mode_image = str(
            _nested_mapping(config, "analysis", "pool_internal").get("helper_image", "") or ""
        ).strip()
    return mode_image or k8s_image


def render_config_for_mode(config: dict, mode: str) -> dict:
    overrides = _nested_mapping(config, "k8s", mode)
    if not overrides:
        return config
    rendered = deepcopy(config)
    rendered["k8s"].update(overrides)
    return rendered


def _year_from_date(value: object, default: int) -> int:
    return date.fromisoformat(str(value)).year if value else default


def _year_from_month(value: object, default: int) -> int:
    return int(str(value).split("-", 1)[0]) if value else default


def _nested_mapping(config: dict, *keys: str) -> dict:
    values = config
    for key in keys:
        value = values.get(key, {}) if isinstance(values, dict) else {}
        if not isinstance(value, dict):
            return {}
        values = value
    return values


def load_run_spec(path: str | Path) -> RunSpec:
    config_path = Path(path)
    config = load_toml(config_path)
    run_id_value = run_id(config, config_path)
    mount_path = str(get(config, "k8s", "mount_path", "/mnt/output"))
    pvc_dir = run_output_dir(config, run_id_value, mount_path=mount_path)
    default_pool_internal_dir = f"{pvc_dir.rstrip('/')}/analysis/pool_internal_top100"
    start_year = _year_from_date(get(config, "window", "test_start_date", None), 0)
    end_year = _year_from_date(get(config, "window", "test_end_date", None), start_year)
    test_start_month = str(get(config, "window", "test_start_month", "") or "")
    test_end_month = str(get(config, "window", "test_end_month", "") or "")
    if start_year <= 0 and test_start_month:
        start_year = _year_from_month(test_start_month, 0)
    if end_year <= 0 and test_end_month:
        end_year = _year_from_month(test_end_month, start_year)
    analysis = _nested_mapping(config, "analysis", "pool_internal")

    def analysis_str(key: str, default: str) -> str:
        value = analysis.get(key, default)
        return default if value is None else str(value)

    return RunSpec(
        run_id=run_id_value,
        pvc_dir=pvc_dir,
        namespace=str(get(config, "k8s", "namespace", "bizewu")),
        pvc=str(get(config, "k8s", "pvc", "bizewu-private-data")),
        mount_path=mount_path,
        pull_secret=str(get(config, "k8s", "image_pull_secret", "highfort")),
        image=str(get(config, "k8s", "helper_image", DEFAULT_IMAGE)),
        test_start_year=start_year,
        test_end_year=end_year,
        test_start_month=test_start_month,
        test_end_month=test_end_month,
        test_months=int(get(config, "window", "test_months", 1) or 1),
        test_stride_months=int(
            get(config, "window", "test_stride_months", get(config, "window", "test_months", 1))
            or 1
        ),
        kind=str(get(config, "run", "kind", "experiment")),
        local_dir=str(get(config, "output", "local_dir", "") or ""),
        pool_internal_analysis_enabled=coerce_bool(analysis.get("enabled", False)),
        pool_internal_analysis_dir=analysis_str("output_dir", default_pool_internal_dir),
        pool_internal_record_prefix=analysis_str("record_prefix", run_id_value),
        pool_internal_archive_profile=str(
            _nested_mapping(analysis, "archive").get("profile", "") or ""
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
    kubectl = [hfcli, "kubectl"]
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
        *kubectl,
        "delete",
        "pod",
        pod_name,
        "-n",
        spec.namespace,
        "--ignore-not-found",
    ]
    create_command = [
        *kubectl,
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
        *kubectl,
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
