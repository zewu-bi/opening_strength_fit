import argparse
from datetime import date
import textwrap
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.config import config_value as get
from opening_strength_fit.config import load_toml, run_id, slug


DEFAULT_IMAGE = "registry.corp.highfortfunds.com/bizewu/opening-strength-fit:latest"


def load_config(path: Path) -> dict:
    return load_toml(path)


def training_script(config: dict) -> str:
    run_kind = str(get(config, "run", "kind", "experiment")).strip().lower()
    if run_kind == "cache":
        return "scripts/materialize_labeled_caches.py"
    model_name = str(get(config, "model", "name", "ridge")).strip().lower()
    if model_name in {
        "ridge",
        "gbm",
        "hist_gbm",
        "hist_gradient_boosting",
        "lightgbm",
        "lgbm",
    }:
        return "scripts/run_experiment.py"
    raise SystemExit(f"Unsupported model.name for k8s rendering: {model_name}")


def is_cache_run(config: dict) -> bool:
    return str(get(config, "run", "kind", "experiment")).strip().lower() == "cache"


def _gpu_count(resources: dict) -> str:
    gpu_limit = str(resources.get("gpu_limit", resources.get("nvidia_gpu", "0")) or "0")
    if gpu_limit in {"", "0", "0.0", "none", "None"}:
        return ""
    return gpu_limit


def _gpu_resource_line(resources: dict, indent: int = 14) -> str:
    gpu_count = _gpu_count(resources)
    if not gpu_count:
        return ""
    return f"{' ' * indent}nvidia.com/gpu: \"{gpu_count}\"\n"


def _node_selector_yaml(config: dict, indent: int = 14) -> str:
    node_selector = config.get("k8s", {}).get("node_selector", {})
    if not node_selector:
        return ""
    lines = [f"{' ' * indent}nodeSelector:"]
    for key, value in sorted(node_selector.items()):
        lines.append(f"{' ' * (indent + 2)}{key}: \"{value}\"")
    return "\n".join(lines) + "\n"


def _gpu_tolerations_yaml(resources: dict, indent: int = 14) -> str:
    if not _gpu_count(resources):
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


def _gpu_opencl_bootstrap_yaml(resources: dict, indent: int) -> str:
    if not _gpu_count(resources):
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


def _scheduler_yaml(config: dict, resources: dict, indent: int = 14) -> str:
    return _node_selector_yaml(config, indent=indent) + _gpu_tolerations_yaml(
        resources,
        indent=indent,
    )


def _training_command_yaml(
    *,
    script: str,
    config_path: Path,
    output_dir: str,
    resources: dict,
    indent: int = 18,
) -> str:
    if _gpu_count(resources):
        return textwrap.indent(
            textwrap.dedent(
                f"""\
                command:
                  - /bin/bash
                  - -lc
                  - |
                    set -euo pipefail
                    mkdir -p /etc/OpenCL/vendors
                    echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd
                    exec python {script} \\
                      --config {config_path.as_posix()} \\
                      --output-dir {output_dir}
                """
            ),
            " " * indent,
        )
    return textwrap.indent(
        textwrap.dedent(
            f"""\
            command:
              - python
              - {script}
              - --config
              - {config_path.as_posix()}
              - --output-dir
              - {output_dir}
            """
        ),
        " " * indent,
    )


def _year_from_config(config: dict, key: str) -> int:
    value = get(config, "window", key, None)
    if not value:
        raise SystemExit(
            f"--sharded requires [window].{key}; set explicit test_start_date/test_end_date"
        )
    return date.fromisoformat(str(value)).year


def _month_range_from_config(config: dict) -> list[str]:
    start = get(config, "window", "test_start_month", None)
    end = get(config, "window", "test_end_month", None)
    if not start or not end:
        raise SystemExit(
            "--sharded monthly requires [window].test_start_month and "
            "[window].test_end_month"
        )
    return [str(month) for month in pd.period_range(str(start), str(end), freq="M")]


def _window_mode(config: dict) -> str:
    return str(get(config, "window", "mode", "chronological"))


def _clickhouse_env_from(config: dict, indent: int = 18) -> str:
    secret = str(get(config, "k8s", "clickhouse_secret", "") or "")
    if not secret:
        return ""
    return textwrap.indent(
        textwrap.dedent(
            f"""\
            envFrom:
              - secretRef:
                  name: {secret}
            """
        ),
        " " * indent,
    )


def render_training_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    job_name = get(config, "k8s", "job_name", f"opening-strength-{slug(run_id_value)}")
    namespace = get(config, "k8s", "namespace", "bizewu")
    pull_secret = get(config, "k8s", "image_pull_secret", "highfort")
    pvc = get(config, "k8s", "pvc", "bizewu-private-data")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    output_dir = get(
        config,
        "output",
        "k8s_dir",
        f"{mount_path}/opening_strength_fit/{run_id_value}",
    )
    resources = config.get("k8s", {}).get("resources", {})
    cpu_request = resources.get("cpu_request", "4")
    cpu_limit = resources.get("cpu_limit", "8")
    memory_request = resources.get("memory_request", "16Gi")
    memory_limit = resources.get("memory_limit", "32Gi")
    gpu_resource_line = _gpu_resource_line(resources, indent=22)
    scheduler_yaml = _scheduler_yaml(config, resources, indent=14)
    script = training_script(config)
    env_from = _clickhouse_env_from(config, indent=18)
    command_yaml = _training_command_yaml(
        script=script,
        config_path=config_path,
        output_dir=output_dir,
        resources=resources,
        indent=18,
    )

    return textwrap.dedent(
        f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
        spec:
          backoffLimit: 0
          ttlSecondsAfterFinished: 86400
          template:
            spec:
              restartPolicy: Never
              imagePullSecrets:
                - name: {pull_secret}
              volumes:
                - name: opening-strength-output
                  persistentVolumeClaim:
                    claimName: {pvc}
{scheduler_yaml.rstrip()}
              containers:
                - name: opening-strength-fit
                  image: {image}
                  imagePullPolicy: Always
{env_from}                  workingDir: /app/opening_strength_fit
{command_yaml.rstrip()}
                  volumeMounts:
                    - name: opening-strength-output
                      mountPath: {mount_path}
                  resources:
                    requests:
                      cpu: "{cpu_request}"
                      memory: {memory_request}
{gpu_resource_line.rstrip()}
                    limits:
                      cpu: "{cpu_limit}"
                      memory: {memory_limit}
{gpu_resource_line.rstrip()}
        """
    )


def render_materialize_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    job_name = get(config, "k8s", "job_name", f"opening-strength-{slug(run_id_value)}")
    namespace = get(config, "k8s", "namespace", "bizewu")
    pull_secret = get(config, "k8s", "image_pull_secret", "highfort")
    pvc = get(config, "k8s", "pvc", "bizewu-private-data")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    resources = config.get("k8s", {}).get("resources", {})
    cpu_request = resources.get("cpu_request", "4")
    cpu_limit = resources.get("cpu_limit", "8")
    memory_request = resources.get("memory_request", "16Gi")
    memory_limit = resources.get("memory_limit", "32Gi")
    env_from = _clickhouse_env_from(config, indent=18)

    return textwrap.dedent(
        f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
        spec:
          backoffLimit: 0
          ttlSecondsAfterFinished: 86400
          template:
            spec:
              restartPolicy: Never
              imagePullSecrets:
                - name: {pull_secret}
              volumes:
                - name: opening-strength-output
                  persistentVolumeClaim:
                    claimName: {pvc}
              containers:
                - name: materialize-labeled-cache
                  image: {image}
                  imagePullPolicy: Always
{env_from}                  workingDir: /app/opening_strength_fit
                  command:
                    - python
                    - scripts/materialize_labeled_caches.py
                    - --config
                    - {config_path.as_posix()}
                  volumeMounts:
                    - name: opening-strength-output
                      mountPath: {mount_path}
                  resources:
                    requests:
                      cpu: "{cpu_request}"
                      memory: {memory_request}
                    limits:
                      cpu: "{cpu_limit}"
                      memory: {memory_limit}
        """
    )


def render_reader_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    job_name = f"opening-strength-read-{slug(run_id_value)}"
    namespace = get(config, "k8s", "namespace", "bizewu")
    pull_secret = get(config, "k8s", "image_pull_secret", "highfort")
    pvc = get(config, "k8s", "pvc", "bizewu-private-data")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    output_dir = get(
        config,
        "output",
        "k8s_dir",
        f"{mount_path}/opening_strength_fit/{run_id_value}",
    )

    return textwrap.dedent(
        f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
        spec:
          backoffLimit: 0
          ttlSecondsAfterFinished: 3600
          template:
            spec:
              restartPolicy: Never
              imagePullSecrets:
                - name: {pull_secret}
              volumes:
                - name: opening-strength-output
                  persistentVolumeClaim:
                    claimName: {pvc}
              containers:
                - name: reader
                  image: {image}
                  imagePullPolicy: Always
                  workingDir: /app/opening_strength_fit
                  command:
                    - python
                    - scripts/summarize_opening_results.py
                    - --input-dir
                    - {output_dir}
                  volumeMounts:
                    - name: opening-strength-output
                      mountPath: {mount_path}
        """
    )


def render_sharded_training_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    job_name = f"opening-strength-{slug(run_id_value)}-sharded"
    namespace = get(config, "k8s", "namespace", "bizewu")
    pull_secret = get(config, "k8s", "image_pull_secret", "highfort")
    pvc = get(config, "k8s", "pvc", "bizewu-private-data")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    output_dir = get(
        config,
        "output",
        "k8s_dir",
        f"{mount_path}/opening_strength_fit/{run_id_value}",
    )
    resources = config.get("k8s", {}).get("resources", {})
    cpu_request = resources.get("cpu_request", "4")
    cpu_limit = resources.get("cpu_limit", "8")
    memory_request = resources.get("memory_request", "16Gi")
    memory_limit = resources.get("memory_limit", "32Gi")
    gpu_resource_line = _gpu_resource_line(resources, indent=26)
    scheduler_yaml = _scheduler_yaml(config, resources, indent=18)
    script = training_script(config)
    env_from = _clickhouse_env_from(config, indent=18)
    opencl_bootstrap = _gpu_opencl_bootstrap_yaml(resources, indent=26)
    if _window_mode(config) == "rolling_monthly":
        env_from = _clickhouse_env_from(config, indent=22)
        months = " ".join(_month_range_from_config(config))
        train_months = int(get(config, "window", "train_months", 12))
        return textwrap.dedent(
            f"""\
            apiVersion: batch/v1
            kind: Job
            metadata:
              name: {job_name}
              namespace: {namespace}
            spec:
              backoffLimit: 0
              ttlSecondsAfterFinished: 86400
              template:
                spec:
                  restartPolicy: Never
                  imagePullSecrets:
                    - name: {pull_secret}
                  volumes:
                    - name: opening-strength-output
                      persistentVolumeClaim:
                        claimName: {pvc}
{scheduler_yaml.rstrip()}
                  containers:
                    - name: opening-strength-fit
                      image: {image}
                      imagePullPolicy: Always
{env_from}                      workingDir: /app/opening_strength_fit
                      command:
                        - /bin/bash
                        - -lc
                        - |
                          set -euo pipefail
{opencl_bootstrap.rstrip()}
                          ROOT={output_dir}
                          mkdir -p "${{ROOT}}"

                          for MONTH in {months}; do
                            OUT="${{ROOT}}/month_${{MONTH}}"
                            if [ -f "${{OUT}}/metrics_by_year.csv" ]; then
                              echo "month ${{MONTH}}: metrics already exist, skipping ${{OUT}}"
                              continue
                            fi

                            echo
                            echo "running {run_id_value} shard month=${{MONTH}}"
                            echo "output_dir=${{OUT}}"

                            python {script} \\
                              --config {config_path.as_posix()} \\
                              --rolling-monthly \\
                              --train-months {train_months} \\
                              --test-start-month "${{MONTH}}" \\
                              --test-end-month "${{MONTH}}" \\
                              --output-dir "${{OUT}}"
                          done
                      volumeMounts:
                        - name: opening-strength-output
                          mountPath: {mount_path}
                      resources:
                        requests:
                          cpu: "{cpu_request}"
                          memory: {memory_request}
{gpu_resource_line.rstrip()}
                        limits:
                          cpu: "{cpu_limit}"
                          memory: {memory_limit}
{gpu_resource_line.rstrip()}
            """
        )
    test_start_year = _year_from_config(config, "test_start_date")
    test_end_year = _year_from_config(config, "test_end_date")
    years = " ".join(str(year) for year in range(test_start_year, test_end_year + 1))

    return textwrap.dedent(
        f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
        spec:
          backoffLimit: 0
          ttlSecondsAfterFinished: 86400
          template:
            spec:
              restartPolicy: Never
              imagePullSecrets:
                - name: {pull_secret}
              volumes:
                - name: opening-strength-output
                  persistentVolumeClaim:
                    claimName: {pvc}
{_scheduler_yaml(config, resources, indent=14).rstrip()}
              containers:
                - name: opening-strength-fit
                  image: {image}
                  imagePullPolicy: Always
{env_from}                  workingDir: /app/opening_strength_fit
                  command:
                    - /bin/bash
                    - -lc
                    - |
                      set -euo pipefail
{_gpu_opencl_bootstrap_yaml(resources, indent=22).rstrip()}
                      ROOT={output_dir}
                      mkdir -p "${{ROOT}}"

                      for YEAR in {years}; do
                        OUT="${{ROOT}}/year_${{YEAR}}"
                        if [ -f "${{OUT}}/metrics_by_year.csv" ]; then
                          echo "year ${{YEAR}}: metrics already exist, skipping ${{OUT}}"
                          continue
                        fi

                        echo
                        echo "running {run_id_value} shard year=${{YEAR}}"
                        echo "output_dir=${{OUT}}"

                        python {script} \\
                          --config {config_path.as_posix()} \\
                          --test-start-date "${{YEAR}}-01-01" \\
                          --test-end-date "${{YEAR}}-12-31" \\
                          --output-dir "${{OUT}}"
                      done
                  volumeMounts:
                    - name: opening-strength-output
                      mountPath: {mount_path}
                  resources:
                    requests:
                      cpu: "{cpu_request}"
                      memory: {memory_request}
{_gpu_resource_line(resources, indent=22).rstrip()}
                    limits:
                      cpu: "{cpu_limit}"
                      memory: {memory_limit}
{_gpu_resource_line(resources, indent=22).rstrip()}
        """
    )


def render_sharded_reader_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    job_name = f"opening-strength-read-{slug(run_id_value)}-sharded"
    namespace = get(config, "k8s", "namespace", "bizewu")
    pull_secret = get(config, "k8s", "image_pull_secret", "highfort")
    pvc = get(config, "k8s", "pvc", "bizewu-private-data")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    output_dir = get(
        config,
        "output",
        "k8s_dir",
        f"{mount_path}/opening_strength_fit/{run_id_value}",
    )
    if _window_mode(config) == "rolling_monthly":
        return textwrap.dedent(
            f"""\
            apiVersion: batch/v1
            kind: Job
            metadata:
              name: {job_name}
              namespace: {namespace}
            spec:
              backoffLimit: 0
              ttlSecondsAfterFinished: 3600
              template:
                spec:
                  restartPolicy: Never
                  imagePullSecrets:
                    - name: {pull_secret}
                  volumes:
                    - name: opening-strength-output
                      persistentVolumeClaim:
                        claimName: {pvc}
                  containers:
                    - name: reader
                      image: {image}
                      imagePullPolicy: Always
                      workingDir: /app/opening_strength_fit
                      command:
                        - /bin/bash
                        - -lc
                        - |
                          set -euo pipefail
                          ROOT={output_dir}
                          python - <<'PY'
                          from pathlib import Path
                          import pandas as pd
                          from opening_strength_fit.training import _metrics_by_year_from_windows

                          root = Path("{output_dir}")
                          metric_frames = []
                          prediction_frames = []
                          missing_metrics = []
                          missing_predictions = []

                          for shard in sorted(root.glob("month_*")):
                              month = shard.name.removeprefix("month_")
                              metrics_path = shard / "metrics_by_year.csv"
                              prediction_path = shard / f"predictions_{{month}}.parquet"
                              if not prediction_path.exists():
                                  prediction_path = shard / "predictions.parquet"

                              if metrics_path.exists():
                                  metric_frames.append(pd.read_csv(metrics_path))
                              else:
                                  missing_metrics.append(month)

                              if prediction_path.exists():
                                  prediction_frames.append(pd.read_parquet(prediction_path))
                              else:
                                  missing_predictions.append(month)

                          if metric_frames:
                              metrics_month = pd.concat(metric_frames, ignore_index=True).sort_values(["test_year", "test_month"])
                              metrics_month.to_csv(root / "metrics_by_month.csv", index=False)
                              metrics_month.to_parquet(root / "metrics_by_month.parquet", index=False)
                              metrics = _metrics_by_year_from_windows(metrics_month)
                              metrics.to_csv(root / "metrics_by_year.csv", index=False)
                              metrics.to_parquet(root / "metrics_by_year.parquet", index=False)
                              print(f"combined_monthly_metrics: {{root / 'metrics_by_month.csv'}}")
                              print(f"combined_metrics: {{root / 'metrics_by_year.csv'}}")
                          else:
                              print("no shard metrics found")

                          if prediction_frames:
                              predictions = pd.concat(prediction_frames, ignore_index=True)
                              sort_cols = [c for c in ["date", "symbol", "timestamp"] if c in predictions.columns]
                              if sort_cols:
                                  predictions = predictions.sort_values(sort_cols)
                              predictions.to_parquet(root / "predictions_all.parquet", index=False)
                              print(f"combined_predictions: {{root / 'predictions_all.parquet'}}")
                          else:
                              print("no shard predictions found")

                          if missing_metrics:
                              print(f"missing_metric_months: {{missing_metrics}}")
                          if missing_predictions:
                              print(f"missing_prediction_months: {{missing_predictions}}")
                          PY
                          python scripts/summarize_opening_results.py --input-dir "${{ROOT}}"
                      volumeMounts:
                        - name: opening-strength-output
                          mountPath: {mount_path}
            """
        )
    test_start_year = _year_from_config(config, "test_start_date")
    test_end_year = _year_from_config(config, "test_end_date")

    return textwrap.dedent(
        f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {job_name}
          namespace: {namespace}
        spec:
          backoffLimit: 0
          ttlSecondsAfterFinished: 3600
          template:
            spec:
              restartPolicy: Never
              imagePullSecrets:
                - name: {pull_secret}
              volumes:
                - name: opening-strength-output
                  persistentVolumeClaim:
                    claimName: {pvc}
              containers:
                - name: reader
                  image: {image}
                  imagePullPolicy: Always
                  workingDir: /app/opening_strength_fit
                  command:
                    - /bin/bash
                    - -lc
                    - |
                      set -euo pipefail
                      ROOT={output_dir}
                      python - <<'PY'
                      from pathlib import Path
                      import pandas as pd

                      root = Path("{output_dir}")
                      metric_frames = []
                      prediction_frames = []
                      missing_metrics = []
                      missing_predictions = []

                      for year in range({test_start_year}, {test_end_year + 1}):
                          shard = root / f"year_{{year}}"
                          metrics_path = shard / "metrics_by_year.csv"
                          prediction_path = shard / f"predictions_{{year}}.parquet"
                          if not prediction_path.exists():
                              prediction_path = shard / "predictions.parquet"

                          if metrics_path.exists():
                              metric_frames.append(pd.read_csv(metrics_path))
                          else:
                              missing_metrics.append(year)

                          if prediction_path.exists():
                              prediction_frames.append(pd.read_parquet(prediction_path))
                          else:
                              missing_predictions.append(year)

                      if metric_frames:
                          metrics = pd.concat(metric_frames, ignore_index=True).sort_values("test_year")
                          metrics.to_csv(root / "metrics_by_year.csv", index=False)
                          metrics.to_parquet(root / "metrics_by_year.parquet", index=False)
                          print(f"combined_metrics: {{root / 'metrics_by_year.csv'}}")
                      else:
                          print("no shard metrics found")

                      if prediction_frames:
                          predictions = pd.concat(prediction_frames, ignore_index=True)
                          sort_cols = [c for c in ["date", "symbol", "timestamp"] if c in predictions.columns]
                          if sort_cols:
                              predictions = predictions.sort_values(sort_cols)
                          predictions.to_parquet(root / "predictions_all.parquet", index=False)
                          print(f"combined_predictions: {{root / 'predictions_all.parquet'}}")
                      else:
                          print("no shard predictions found")

                      if missing_metrics:
                          print(f"missing_metric_years: {{missing_metrics}}")
                      if missing_predictions:
                          print(f"missing_prediction_years: {{missing_predictions}}")
                      PY
                      python scripts/summarize_opening_results.py --input-dir "${{ROOT}}"
                  volumeMounts:
                    - name: opening-strength-output
                      mountPath: {mount_path}
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="TOML run config.")
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Container image tag to run. Default: {DEFAULT_IMAGE}",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/jobs",
        help="Directory for rendered Job manifests.",
    )
    parser.add_argument(
        "--sharded",
        action="store_true",
        help="Render sequential per-year training and shard-combining reader jobs.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    run_id_value = run_id(config, config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_sharded" if args.sharded else ""
    training_path = output_dir / f"{run_id_value}{suffix}_job.yaml"
    reader_path = output_dir / f"{run_id_value}{suffix}_reader_job.yaml"
    if is_cache_run(config):
        if args.sharded:
            raise SystemExit("--sharded is only supported for training experiments")
        training_path.write_text(
            render_materialize_job(config_path, config, args.image),
            encoding="utf-8",
        )
        print("rendered_k8s_jobs:")
        print(f"  materialize: {training_path}")
        return
    if args.sharded:
        training_path.write_text(
            render_sharded_training_job(config_path, config, args.image),
            encoding="utf-8",
        )
        reader_path.write_text(
            render_sharded_reader_job(config_path, config, args.image),
            encoding="utf-8",
        )
    else:
        training_path.write_text(
            render_training_job(config_path, config, args.image),
            encoding="utf-8",
        )
        reader_path.write_text(
            render_reader_job(config_path, config, args.image),
            encoding="utf-8",
        )

    print("rendered_k8s_jobs:")
    print(f"  training: {training_path}")
    print(f"  reader: {reader_path}")


if __name__ == "__main__":
    main()
