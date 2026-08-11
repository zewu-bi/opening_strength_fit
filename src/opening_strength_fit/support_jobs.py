"""Render compact support-job specifications as Kubernetes JSON."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import tomllib
from pathlib import Path
from typing import Any

_MATRIX_PLACEHOLDER = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")


def load_support_jobs(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    defaults = document.get("defaults", {})
    jobs = document.get("jobs", [])
    if not isinstance(defaults, dict) or not isinstance(jobs, list):
        raise ValueError("support-job spec requires [defaults] and [[jobs]] tables")

    by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        job_id = str(job.get("id", "")).strip()
        if not job_id:
            raise ValueError("every support job requires a non-empty id")
        if job_id in by_id:
            raise ValueError(f"duplicate support-job id: {job_id}")
        by_id[job_id] = job
    return defaults, by_id


def _shell_argument(value: object, variables: set[str]) -> str:
    text = str(value)
    referenced = set(_MATRIX_PLACEHOLDER.findall(text))
    unknown = referenced - variables
    if unknown:
        raise ValueError(f"unknown matrix variable(s): {', '.join(sorted(unknown))}")
    if not referenced:
        return shlex.quote(text)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    expanded = _MATRIX_PLACEHOLDER.sub(lambda match: f"${{{match.group(1)}}}", escaped)
    return f'"{expanded}"'


def _indexed_command(job: dict[str, Any]) -> tuple[list[str], list[str]]:
    matrix = job.get("matrix", [])
    variables = {str(item["name"]) for item in matrix}
    lines = ["set -euo pipefail"]
    for item in matrix:
        name = str(item["name"])
        values = " ".join(shlex.quote(str(value)) for value in item["values"])
        lines.extend(
            [
                f"{name}_VALUES=({values})",
                f'{name}="${{{name}_VALUES[${{JOB_COMPLETION_INDEX:?missing JOB_COMPLETION_INDEX}}]}}"',
            ]
        )
    command = " ".join(
        ["python", shlex.quote(str(job["script"]))]
        + [_shell_argument(value, variables) for value in job.get("arguments", [])]
    )
    lines.append(f"exec {command}")
    return ["/bin/bash", "-lc"], ["\n".join(lines) + "\n"]


def render_support_job(defaults: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    name = str(job["name"])
    code_volume, config_map, mount_path, *sub_path = job["code"]
    request_cpu, request_memory, limit_cpu, limit_memory = job["resources"]

    volumes: list[dict[str, Any]] = []
    mounts: list[dict[str, Any]] = []
    if job.get("output", True):
        volumes.append(
            {
                "name": "output",
                "persistentVolumeClaim": {"claimName": defaults["output_pvc"]},
            }
        )
        mounts.append({"name": "output", "mountPath": defaults["output_mount"]})
    volumes.append({"name": code_volume, "configMap": {"name": config_map}})
    code_mount = {"name": code_volume, "mountPath": mount_path}
    if sub_path:
        code_mount["subPath"] = sub_path[0]
    mounts.append(code_mount)

    container: dict[str, Any] = {
        "name": job.get("container", "analysis"),
        "image": job.get("image", defaults[job.get("image_key", "image")]),
        "imagePullPolicy": defaults["image_pull_policy"],
    }
    if working_dir := job.get("working_dir"):
        container["workingDir"] = working_dir
    secrets = job.get("secrets", [job["secret"]] if job.get("secret") else [])
    if secrets:
        container["envFrom"] = [{"secretRef": {"name": secret}} for secret in secrets]
    if "completions" in job:
        container["env"] = [
            {
                "name": "JOB_COMPLETION_INDEX",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                    }
                },
            }
        ]
    command = (
        _indexed_command(job)
        if job.get("matrix")
        else (
            ["python", job["script"]],
            None,
        )
    )
    container.update(
        {
            "command": command[0],
            "volumeMounts": mounts,
            "resources": {
                "requests": {"cpu": request_cpu, "memory": request_memory},
                "limits": {"cpu": limit_cpu, "memory": limit_memory},
            },
        }
    )
    if command[1] is not None:
        container["args"] = command[1]
    elif arguments := job.get("arguments"):
        container["command"].extend(str(value) for value in arguments)

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "imagePullSecrets": [{"name": defaults["image_pull_secret"]}],
    }
    if job.get("gpu"):
        pod_spec.update(
            {
                "nodeSelector": {"has_gpu": "true"},
                "tolerations": [
                    {
                        "key": "has_gpu",
                        "operator": "Equal",
                        "value": "true",
                        "effect": "NoSchedule",
                    }
                ],
            }
        )
    if job.get("avoid_default_nodes"):
        pod_spec["affinity"] = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "kubernetes.io/hostname",
                                    "operator": "NotIn",
                                    "values": defaults["avoid_nodes"],
                                }
                            ]
                        }
                    ]
                }
            }
        }
    pod_spec.update({"volumes": volumes, "containers": [container]})

    metadata: dict[str, Any] = {"name": name, "namespace": defaults["namespace"]}
    if job.get("metadata_label"):
        metadata["labels"] = {"app": name}
    spec: dict[str, Any] = {
        "activeDeadlineSeconds": job["deadline"],
        "backoffLimit": job.get("backoff", 0),
        "ttlSecondsAfterFinished": defaults["ttl_seconds"],
        "template": {
            "metadata": {"labels": {"app": name}},
            "spec": pod_spec,
        },
    }
    if "completions" in job:
        spec.update(
            {
                "completionMode": "Indexed",
                "completions": job["completions"],
                "parallelism": job.get("parallelism", job["completions"]),
            }
        )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": metadata,
        "spec": spec,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--job", action="append", dest="job_ids")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    defaults, jobs = load_support_jobs(args.spec)
    requested = args.job_ids or list(jobs)
    unknown = [job_id for job_id in requested if job_id not in jobs]
    if unknown:
        parser.error(f"unknown job id(s): {', '.join(unknown)}")
    rendered = [render_support_job(defaults, jobs[job_id]) for job_id in requested]
    document = (
        rendered[0]
        if len(rendered) == 1
        else {"apiVersion": "v1", "kind": "List", "items": rendered}
    )
    text = json.dumps(document, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
