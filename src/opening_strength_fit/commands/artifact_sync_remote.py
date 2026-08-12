from __future__ import annotations

import subprocess
import tarfile
from functools import partial
from pathlib import Path

from opening_strength_fit.k8s import RunSpec, command_succeeds, run_command


def _kubectl_exec(hfcli: str, namespace: str, pod_name: str, *command: str) -> list[str]:
    return [hfcli, "kubectl", "exec", "-n", namespace, pod_name, "--", *command]


def _remote_path_exists(
    hfcli: str,
    namespace: str,
    pod_name: str,
    remote_path: str,
    kind: str,
) -> bool:
    return command_succeeds(
        _kubectl_exec(hfcli, namespace, pod_name, "/bin/sh", "-lc", f"test -{kind} '{remote_path}'")
    )


remote_file_exists = partial(_remote_path_exists, kind="f")
remote_dir_exists = partial(_remote_path_exists, kind="d")


def fetch_binary_file(
    hfcli: str,
    namespace: str,
    pod_name: str,
    remote_path: str,
    local_path: Path,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = _kubectl_exec(hfcli, namespace, pod_name, "cat", remote_path)
    with local_path.open("wb") as file:
        run_command(command, stdout=file)


def fetch_remote_directory_if_exists(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    remote_dir: str,
    local_dir: Path,
) -> bool:
    if not remote_dir_exists(hfcli, spec.namespace, pod_name, remote_dir):
        return False
    print(f"fetching {remote_dir}/ -> {local_dir}/")
    local_dir.mkdir(parents=True, exist_ok=True)
    command = _kubectl_exec(
        hfcli, spec.namespace, pod_name, "tar", "-C", remote_dir, "-cf", "-", "."
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    if process.stdout is None:
        raise SystemExit(f"failed to open tar stream for {remote_dir}")
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(local_dir)
    finally:
        process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return True


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
