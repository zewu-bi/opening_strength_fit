from __future__ import annotations

from pathlib import Path

from opening_strength_fit.k8s import RunSpec, command_succeeds, run_command


def remote_file_exists(
    hfcli: str,
    namespace: str,
    pod_name: str,
    remote_path: str,
) -> bool:
    return command_succeeds(
        [
            hfcli,
            "kubectl",
            "exec",
            "-n",
            namespace,
            pod_name,
            "--",
            "/bin/sh",
            "-lc",
            f"test -f '{remote_path}'",
        ]
    )


def fetch_binary_file(
    hfcli: str,
    namespace: str,
    pod_name: str,
    remote_path: str,
    local_path: Path,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        hfcli,
        "kubectl",
        "exec",
        "-n",
        namespace,
        pod_name,
        "--",
        "cat",
        remote_path,
    ]
    with local_path.open("wb") as file:
        run_command(command, stdout=file)


def fetch_remote_file_if_exists(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    remote_path: str,
    local_path: Path,
) -> bool:
    if not remote_file_exists(hfcli, spec.namespace, pod_name, remote_path):
        return False
    print(f"fetching {remote_path} -> {local_path}")
    fetch_binary_file(hfcli, spec.namespace, pod_name, remote_path, local_path)
    return True
