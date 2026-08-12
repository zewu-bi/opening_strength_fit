import argparse
import os
import textwrap
from pathlib import Path

from opening_strength_fit import k8s_rendering_support as _render_support
from opening_strength_fit.commands.k8s_analysis_rendering import render_pool_internal_analysis_job
from opening_strength_fit.config import config_value as get
from opening_strength_fit.config import load_toml, run_id, slug
from opening_strength_fit.k8s import (
    KUBERNETES_NAME_LIMIT,
    render_config_for_mode,
    rendered_job_image,
)
from opening_strength_fit.k8s_builder_rendering import (
    render_indexed_builder_job,
    render_matrix_training_jobs,
    render_top1000_job,
)
from opening_strength_fit.pvc_layout import (
    output_layout,
    rolling_shard_dir_name,
    run_output_dir,
    yearly_shard_dir_name,
)

_k8s_job_name = _render_support.k8s_job_name
_training_support = _render_support

DEFAULT_IMAGE_ENV = "OPENING_STRENGTH_IMAGE"
_RUN_KIND_COMMANDS = {
    "alpha_conditioned_rolling_validation": "osf-run-alpha-conditioned-rolling-validation",
    "ask_level_attribution": "osf-ask-level-attribution",
    "cache_transform": "osf-build-target-label-cache",
    "capacity_acceptance": "osf-analyze-capacity-acceptance",
    "capacity_audit": "osf-audit-capacity",
    "clickhouse_labeled_cache": "osf-build-labeled-cache",
    "execution_context": "osf-extract-execution-context",
    "exposure_audit": "osf-audit-exposure",
    "exposure_input": "osf-build-exposure-input",
    "feature_audit": "osf-audit-feature-dependence",
    "feature_hygiene": "osf-audit-feature-hygiene",
    "gap_risk_attribution": "osf-run-gap-risk-attribution",
    "labeled_cache": "osf-build-labeled-cache",
    "learned_risk_layer": "osf-run-learned-risk-layer",
    "next_close_label_cache": "osf-build-next-close-labels",
    "score_risk_sweep": "osf-run-score-risk-sweep",
    "short_label_cache": "osf-build-short-labels",
    "strategy_acceptance": "osf-audit-strategy-acceptance",
    "target_cache": "osf-build-target-label-cache",
}


def resolve_render_image(
    image: str,
    *,
    fallback: object = "",
    allow_mutable: bool = False,
) -> str:
    resolved = (image or os.environ.get(DEFAULT_IMAGE_ENV, "") or str(fallback or "")).strip()
    if not resolved:
        raise SystemExit(
            f"missing container image: pass --image or set {DEFAULT_IMAGE_ENV} "
            "to an immutable tag or digest"
        )
    image_ref = resolved.rsplit("/", 1)[-1]
    if image_ref.endswith(":latest") and not allow_mutable:
        raise SystemExit("refusing mutable image tag ':latest'; pass an immutable tag or digest")
    return resolved


def training_command(config: dict) -> str:
    run_kind = str(get(config, "run", "kind", "experiment")).strip().lower()
    command = _RUN_KIND_COMMANDS.get(run_kind)
    if command:
        return command
    if run_kind not in {"experiment", "exploration"}:
        raise SystemExit(f"Unsupported run.kind for k8s rendering: {run_kind}")
    model_name = str(get(config, "model", "name", "ridge")).strip().lower()
    if model_name in set(
        "ridge gbm hist_gbm hist_gradient_boosting lightgbm lgbm torch_mlp mlp nn ensemble "
        "clock_segment_lightgbm clock_segment_lgbm segmented_lightgbm".split()
    ):
        return "osf-train"
    raise SystemExit(f"Unsupported model.name for k8s rendering: {model_name}")


def _command_config_path(config_path: Path, config: dict) -> Path:
    return Path(str(get(config, "k8s", "command_config_path", config_path.as_posix())))


def _shell_script(*parts: str, indent: int) -> str:
    return textwrap.indent("\n".join(part.rstrip() for part in parts), " " * indent)


def render_training_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    job_name = get(config, "k8s", "job_name", f"opening-strength-{slug(run_id_value)}")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    output_dir = run_output_dir(config, run_id_value, mount_path=str(mount_path))
    resources = config.get("k8s", {}).get("resources", {})
    scheduler_yaml = _training_support.scheduler_yaml(config, resources, indent=6)
    command = training_command(config)
    env_from = _training_support.k8s_env_from(config, indent=10)
    command_yaml = _training_support.training_command_yaml(
        command=command,
        config_path=_command_config_path(config_path, config),
        output_dir=output_dir,
        resources=resources,
        wait_for_paths=_training_support.wait_for_paths_yaml(config, indent=0),
        indent=10,
    )
    return _render_support.job_manifest_yaml(
        _render_support.configured_job_pod_header_yaml(config, job_name),
        scheduler_yaml or "\n",
        image,
        "",
        mount_path,
        _training_support.training_resources_yaml(resources, indent=10),
        env_from=env_from,
        env_before_workdir=True,
        config_volume=_render_support.training_config_map_volume_yaml(config, indent=8),
        extra_mounts=_render_support.training_config_map_mount_yaml(config, indent=12),
        command_yaml=command_yaml,
    )


def render_sharded_training_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    command_config_path = _command_config_path(config_path, config)
    explicit_job_name = str(get(config, "k8s", "job_name", "") or "").strip()
    job_name = explicit_job_name or _k8s_job_name("opening-strength", run_id_value, "sharded")
    mount_path = get(config, "k8s", "mount_path", "/mnt/output")
    layout = output_layout(config)
    output_dir = run_output_dir(config, run_id_value, mount_path=str(mount_path))
    resources = config.get("k8s", {}).get("resources", {})
    command = training_command(config)
    env_from = _training_support.k8s_env_from(config, indent=18)
    if _training_support.window_mode(config) == "rolling_monthly":
        scheduler_yaml = _training_support.scheduler_yaml(config, resources, indent=6)
        env_from = _training_support.k8s_env_from(config, indent=10)
        opencl_bootstrap = _training_support.gpu_opencl_bootstrap_yaml(resources, indent=0)
        wait_for_paths = _training_support.wait_for_paths_yaml(config, indent=0)
        config_map_volume = _render_support.training_config_map_volume_yaml(config, indent=8)
        config_map_mount = _render_support.training_config_map_mount_yaml(config, indent=12)
        resources_yaml = _training_support.training_resources_yaml(resources, indent=10)
        month_windows = _training_support.month_windows_from_config(config)
        test_starts = " ".join(start for start, _ in month_windows)
        test_ends = " ".join(end for _, end in month_windows)
        shard_job_mode = _training_support.shard_job_mode(config)
        index_suffix_chars = (2 if shard_job_mode == "separate" else 1) + len(
            str(len(month_windows) - 1)
        )
        if not explicit_job_name:
            job_name = _k8s_job_name(
                "opening-strength",
                run_id_value,
                "sharded",
                max_length=KUBERNETES_NAME_LIMIT - index_suffix_chars,
            )
        shard_parallelism = _training_support.shard_parallelism(config, resources)
        train_months = int(get(config, "window", "train_months", 12))
        test_months = int(get(config, "window", "test_months", 1) or 1)
        test_stride_months = int(
            get(config, "window", "test_stride_months", test_months) or test_months
        )
        completion_files = _training_support.rolling_completion_files(config, command)
        completion_check = _training_support.shell_file_check(completion_files)
        completion_label = ", ".join(completion_files)
        rolling_dir_expression = rolling_shard_dir_name("${TEST_START}", "${TEST_END}", layout)
        if bool(get(config, "k8s", "replace_existing_shard_output", False)):
            existing_output_action = textwrap.dedent(
                """\
                if [ -d "${OUT}" ]; then
                  echo "removing stale shard output ${OUT}"
                  rm -rf "${OUT}"
                fi
                """
            )
        else:
            existing_output_action = textwrap.dedent(
                f"""\
                if {completion_check}; then
                  echo "test window ${{TEST_START}}..${{TEST_END}}: required outputs already exist ({completion_label}), skipping ${{OUT}}"
                  exit 0
                fi
                """
            )
        per_index_backoff = get(config, "k8s", "backoff_limit_per_index", None)
        backoff_field = "backoffLimitPerIndex" if per_index_backoff is not None else "backoffLimit"
        backoff_value = int(per_index_backoff) if per_index_backoff is not None else 0
        if shard_job_mode == "separate":
            manifests: list[str] = []
            for index, (test_start, test_end) in enumerate(month_windows):
                suffix = f"-s{index}"
                if len(job_name) + len(suffix) > KUBERNETES_NAME_LIMIT:
                    raise SystemExit(
                        f"separate shard job name exceeds {KUBERNETES_NAME_LIMIT} characters: "
                        f"{job_name}{suffix}"
                    )
                shard_job_name = f"{job_name}{suffix}"
                rolling_dir = rolling_shard_dir_name(test_start, test_end, layout)
                script = _shell_script(
                    "set -euo pipefail",
                    opencl_bootstrap,
                    wait_for_paths,
                    textwrap.dedent(
                        f"""\
                        ROOT={output_dir}
                        mkdir -p "${{ROOT}}"
                        TEST_START={test_start}
                        TEST_END={test_end}
                        OUT="${{ROOT}}/{rolling_dir}"
                        if {completion_check}; then
                          echo "test window ${{TEST_START}}..${{TEST_END}}: required outputs already exist ({completion_label}), skipping ${{OUT}}"
                          exit 0
                        fi

                        echo
                        echo "running {run_id_value} shard test=${{TEST_START}}..${{TEST_END}} index={index}"
                        echo "output_dir=${{OUT}}"

                        {command} \\
                          --config {command_config_path.as_posix()} \\
                          --rolling-monthly \\
                          --train-months {train_months} \\
                          --test-months {test_months} \\
                          --test-stride-months {test_stride_months} \\
                          --test-start-month "${{TEST_START}}" \\
                          --test-end-month "${{TEST_END}}" \\
                        --output-dir "${{OUT}}"
                        """
                    ),
                    indent=14,
                )
                manifests.append(
                    _render_support.job_manifest_yaml(
                        _render_support.configured_job_pod_header_yaml(config, shard_job_name),
                        scheduler_yaml,
                        image,
                        script,
                        mount_path,
                        resources_yaml,
                        env_from=env_from,
                        config_volume=config_map_volume,
                        extra_mounts=config_map_mount,
                    ).rstrip()
                )
            return "\n---\n".join(manifests) + "\n"
        scheduler_yaml = _training_support.scheduler_yaml(config, resources, indent=18)
        env_from = _training_support.k8s_env_from(config, indent=22)
        opencl_bootstrap = _training_support.gpu_opencl_bootstrap_yaml(resources, indent=26)
        wait_for_paths = _training_support.wait_for_paths_yaml(config, indent=26)
        config_map_volume = _render_support.training_config_map_volume_yaml(config, indent=20)
        config_map_mount = _render_support.training_config_map_mount_yaml(config, indent=24)
        resources_yaml = _training_support.training_resources_yaml(resources, indent=22)
        job_header = _render_support.configured_job_pod_header_yaml(
            config,
            job_name,
            spec_lines=(
                f"{backoff_field}: {backoff_value}",
                "completionMode: Indexed",
                f"completions: {len(month_windows)}",
                f"parallelism: {shard_parallelism}",
            ),
            indent=12,
        )
        return textwrap.dedent(
            f"""\
{job_header.rstrip()}
{config_map_volume.rstrip()}
{scheduler_yaml.rstrip()}
                  containers:
                    - name: opening-strength-fit
                      image: {image}
                      imagePullPolicy: Always
{env_from}                      workingDir: /app/opening_strength_fit
                      env:
                        - name: JOB_COMPLETION_INDEX
                          valueFrom:
                            fieldRef:
                              fieldPath: "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                      command:
                        - /bin/bash
                        - -lc
                        - |
                          set -euo pipefail
{opencl_bootstrap.rstrip()}
{wait_for_paths.rstrip()}
                          ROOT={output_dir}
                          mkdir -p "${{ROOT}}"
                          TEST_STARTS=({test_starts})
                          TEST_ENDS=({test_ends})
                          INDEX="${{JOB_COMPLETION_INDEX:-}}"
                          if [ -z "${{INDEX}}" ]; then
                            echo "missing JOB_COMPLETION_INDEX for indexed shard job" >&2
                            exit 1
                          fi
                          if [ "${{INDEX}}" -lt 0 ] || [ "${{INDEX}}" -ge "${{#TEST_STARTS[@]}}" ]; then
                            echo "JOB_COMPLETION_INDEX out of range: ${{INDEX}}" >&2
                            exit 1
                          fi

                          TEST_START="${{TEST_STARTS[${{INDEX}}]}}"
                          TEST_END="${{TEST_ENDS[${{INDEX}}]}}"
                          OUT="${{ROOT}}/{rolling_dir_expression}"
{textwrap.indent(existing_output_action, " " * 26).rstrip()}

                          echo
                          echo "running {run_id_value} shard test=${{TEST_START}}..${{TEST_END}} index=${{INDEX}}"
                          echo "output_dir=${{OUT}}"

                          {command} \\
                            --config {command_config_path.as_posix()} \\
                            --rolling-monthly \\
                            --train-months {train_months} \\
                            --test-months {test_months} \\
                            --test-stride-months {test_stride_months} \\
                            --test-start-month "${{TEST_START}}" \\
                            --test-end-month "${{TEST_END}}" \\
                            --output-dir "${{OUT}}"
                      volumeMounts:
{_render_support.output_volume_mount_yaml(mount_path, 24).rstrip()}
{config_map_mount.rstrip()}
{resources_yaml.rstrip()}
            """
        )
    test_start_year = _training_support.year_from_config(config, "test_start_date")
    test_end_year = _training_support.year_from_config(config, "test_end_date")
    years = " ".join(str(year) for year in range(test_start_year, test_end_year + 1))
    yearly_dir_expression = yearly_shard_dir_name("${YEAR}", layout)
    job_header = _render_support.configured_job_pod_header_yaml(config, job_name, indent=8)
    resources_yaml = _training_support.training_resources_yaml(resources, indent=18)

    return textwrap.dedent(
        f"""\
{job_header.rstrip()}
{_training_support.scheduler_yaml(config, resources, indent=14).rstrip()}
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
{_training_support.gpu_opencl_bootstrap_yaml(resources, indent=22).rstrip()}
                      ROOT={output_dir}
                      mkdir -p "${{ROOT}}"

                      for YEAR in {years}; do
                        OUT="${{ROOT}}/{yearly_dir_expression}"
                        if [ -f "${{OUT}}/metrics_by_year.csv" ]; then
                          echo "year ${{YEAR}}: metrics already exist, skipping ${{OUT}}"
                          continue
                        fi

                        echo
                        echo "running {run_id_value} shard year=${{YEAR}}"
                        echo "output_dir=${{OUT}}"

                        {command} \\
                          --config {command_config_path.as_posix()} \\
                          --test-start-date "${{YEAR}}-01-01" \\
                          --test-end-date "${{YEAR}}-12-31" \\
                          --output-dir "${{OUT}}"
                      done
                  volumeMounts:
{_render_support.output_volume_mount_yaml(mount_path, 20).rstrip()}
{resources_yaml.rstrip()}
        """
    )


JOB_RENDERERS = {
    "analysis": render_pool_internal_analysis_job,
    "indexed": render_indexed_builder_job,
    "matrix": render_matrix_training_jobs,
    "sharded": render_sharded_training_job,
    "top1000": render_top1000_job,
    "training": render_training_job,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="TOML run config.")
    parser.add_argument(
        "--image",
        default="",
        help=f"Container image tag to run. Defaults to ${DEFAULT_IMAGE_ENV}.",
    )
    parser.add_argument(
        "--allow-mutable-image",
        action="store_true",
        help="Allow rendering a mutable image tag such as :latest.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/jobs",
        help="Directory for rendered Job manifests.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode_help = {
        "sharded": "Render a sequential per-year or per-month training job.",
        "indexed": "Render an Indexed annual data-builder job.",
        "matrix": "Render one Indexed training Job per configured matrix case.",
        "top1000": "Render a legacy Top1000 diagnostic job.",
        "analysis": "Render a cluster-side pool-internal analysis job for the configured run.",
    }
    for name, help_text in mode_help.items():
        mode.add_argument(f"--{name}", action="store_true", help=help_text)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_toml(config_path)
    render_mode = next(
        (name for name in mode_help if getattr(args, name)),
        "training",
    )
    image = resolve_render_image(
        args.image,
        fallback=rendered_job_image(config, render_mode),
        allow_mutable=args.allow_mutable_image,
    )
    config = render_config_for_mode(config, render_mode)
    run_id_value = run_id(config, config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = {
        "analysis": "_pool_internal_analysis",
        "matrix": "_matrix",
        "sharded": "_sharded",
    }.get(render_mode, "")
    output_path = output_dir / f"{run_id_value}{suffix}_job.yaml"
    output_path.write_text(
        JOB_RENDERERS[render_mode](config_path, config, image).rstrip() + "\n",
        encoding="utf-8",
    )
    print("rendered_k8s_jobs:")
    print(f"  {'analysis' if args.analysis else 'training'}: {output_path}")


if __name__ == "__main__":
    main()
