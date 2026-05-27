from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import subprocess

from opening_strength_fit.config import config_value, load_toml, run_id, slug


DEFAULT_IMAGE = "registry.corp.highfortfunds.com/bizewu/opening-strength-fit:latest"


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


def load_run_spec(path: str | Path) -> RunSpec:
    config_path = Path(path)
    config = load_toml(config_path)
    run_id_value = run_id(config, config_path)
    mount_path = str(config_value(config, "k8s", "mount_path", "/mnt/output"))
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
        pvc_dir=str(
            config_value(
                config,
                "output",
                "k8s_dir",
                f"{mount_path}/opening_strength_fit/{run_id_value}",
            )
        ),
        namespace=str(config_value(config, "k8s", "namespace", "bizewu")),
        pvc=str(config_value(config, "k8s", "pvc", "bizewu-private-data")),
        mount_path=mount_path,
        pull_secret=str(config_value(config, "k8s", "image_pull_secret", "highfort")),
        image=DEFAULT_IMAGE,
        test_start_year=start_year,
        test_end_year=end_year,
        test_start_month=test_start_month,
        test_end_month=test_end_month,
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


def ensure_temp_pod(
    hfcli: str,
    spec: RunSpec,
    timeout: str,
    pod_prefix: str,
    *,
    dry_run: bool = False,
) -> str:
    pod_name = f"{pod_prefix}-{slug(spec.run_id)}"
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
