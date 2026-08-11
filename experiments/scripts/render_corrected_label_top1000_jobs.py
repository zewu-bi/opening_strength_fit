from __future__ import annotations

import json
from pathlib import Path

WINDOWS = ("0931_0940", "1001_1010", "1401_1410")
HORIZONS = ("short1m", "short3m")
RUN_TEMPLATE = "nn_v6_w{window}_{horizon}_corrected_nextclose_36m_grouped_gated_v2_mse"
IMAGE = (
    "registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260731-corrected-label-matrix"
)


def _job(*, run_id: str, window: str, horizon: str, mode: str, node: str) -> dict:
    short_window = window.replace("_", "-")
    short_horizon = horizon.removeprefix("short")
    job_name = f"os-top1000-corrected-{short_window}-{short_horizon}-{mode}"
    output_root = (
        f"/mnt/output/opening_strength_fit/nn/{run_id}/analysis/top1000_corrected_acceptance"
    )
    label_root = f"/mnt/output/opening_strength_fit/cache/opening_2019_2025_next_close_decision_clock_state_clock6_{window}"
    compat_root = f"/tmp/{window}-{horizon}-next-label-compat"
    source_pattern = f"opening_${{YEAR}}_next_close_decision_clock_state_clock6_{window}.parquet"
    command = [
        "set -euo pipefail",
        f"mkdir -p {compat_root} {output_root}",
        "for YEAR in 2019 2020 2021 2022 2023 2024 2025; do",
        f"  ln -sf {label_root}/{source_pattern} {compat_root}/opening_${{YEAR}}_next_close_labels_v1.parquet",
        "done",
    ]
    args = [
        "python /opt/analysis/run_top1000_rank_bucket_diagnostics.py",
        "--prediction-root /mnt/output/opening_strength_fit/nn",
        f"--next-label-root {compat_root}",
        "--pool-path lml.bzw@ssd/data/pool_L.parquet",
        f"--output-dir {output_root if mode == 'rank' else output_root + '/hist'}",
        f"--variant corrected_{window}_{horizon}",
        f"--run-id {run_id}",
    ]
    if mode == "hist":
        args += ["--top1000-bucket-return-histogram-only", "--histogram-bin-width-bps 100"]
    command.append(" ".join(args))
    command.append(f"touch {output_root}/_{mode.upper()}_SUCCESS")
    memory_request, memory_limit = ("256Gi", "384Gi") if mode == "rank" else ("192Gi", "320Gi")
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": "bizewu"},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "imagePullSecrets": [{"name": "highfort"}],
                    "nodeSelector": {"kubernetes.io/hostname": node},
                    "tolerations": [
                        {
                            "key": "has_gpu",
                            "operator": "Equal",
                            "value": "true",
                            "effect": "NoSchedule",
                        }
                    ],
                    "volumes": [
                        {
                            "name": "opening-strength-output",
                            "persistentVolumeClaim": {"claimName": "bizewu-private-data"},
                        },
                        {
                            "name": "analysis-script",
                            "configMap": {"name": "os-top1000-rank-bucket-script-v3"},
                        },
                    ],
                    "containers": [
                        {
                            "name": "opening-strength-fit",
                            "image": IMAGE,
                            "imagePullPolicy": "Always",
                            "envFrom": [
                                {"secretRef": {"name": "opening-strength-clickhouse"}},
                                {"secretRef": {"name": "xy-fit-ceph-credentials"}},
                            ],
                            "workingDir": "/app/opening_strength_fit",
                            "command": ["/bin/bash", "-lc", "\n".join(command)],
                            "volumeMounts": [
                                {"name": "opening-strength-output", "mountPath": "/mnt/output"},
                                {
                                    "name": "analysis-script",
                                    "mountPath": "/opt/analysis",
                                    "readOnly": True,
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "8", "memory": memory_request},
                                "limits": {"cpu": "16", "memory": memory_limit},
                            },
                        }
                    ],
                }
            },
        },
    }


def main() -> None:
    output_dir = Path("experiments/jobs/support/corrected_label_3x2_grid_2022_2025_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_nodes = ("node13", "node16")
    cases = (
        item
        for candidate_window in WINDOWS
        for item in ((candidate_window, "short1m"), (candidate_window, "short3m"))
    )
    for index, (window, horizon) in enumerate(cases):
        run_id = RUN_TEMPLATE.format(window=window, horizon=horizon)
        for mode, node in (("rank", rank_nodes[index % 2]), ("hist", "node14")):
            path = output_dir / f"{run_id}_top1000_{mode}_job.yaml"
            path.write_text(
                json.dumps(
                    _job(
                        run_id=run_id,
                        window=window,
                        horizon=horizon,
                        mode=mode,
                        node=node,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(path)


if __name__ == "__main__":
    main()
