from __future__ import annotations

import shlex
import textwrap
from pathlib import Path

from opening_strength_fit.config import coerce_str_list, run_id
from opening_strength_fit.config import config_value as get
from opening_strength_fit.k8s import KUBERNETES_NAME_LIMIT
from opening_strength_fit.k8s_rendering_support import (
    configured_job_pod_header_yaml,
    container_resources_yaml,
    gpu_count,
    job_manifest_yaml,
    k8s_env_from,
    month_windows_from_config,
    scheduler_yaml,
    training_config_map_mount_yaml,
    training_config_map_volume_yaml,
    wait_for_specific_paths_yaml,
)
from opening_strength_fit.pvc_layout import run_output_dir

INDEXED_BUILDER_COMMANDS = dict(
    item.split("=", 1)
    for item in (
        "cache_transform=osf-build-target-label-cache labeled_cache=osf-build-labeled-cache "
        "long_horizon_label_split=osf-split-long-horizon-labels "
        "long_horizon_labels=osf-build-long-horizon-labels "
        "long_label_raw_source=osf-build-long-label-raw-source "
        "raw_source_cache=osf-build-raw-source-cache short_label_cache=osf-build-short-labels "
        "training_feature_dataset=osf-build-training-datasets"
    ).split()
)
INDEXED_BUILDER_KINDS = frozenset(INDEXED_BUILDER_COMMANDS)


def _years(config: dict) -> list[int]:
    for (
        section
    ) in "cache_shards target_cache_shards short_label_shards raw_source dataset k8s".split():
        values = get(config, section, "years", []) or []
        if values:
            years = [int(value) for value in values]
            if len(set(years)) != len(years):
                raise SystemExit(f"[{section}].years contains duplicates")
            return years
    raise SystemExit("indexed rendering requires an explicit annual years list")


def _year_path(value: object) -> str:
    return str(value or "").replace("{year}", "${YEAR}")


def _config_command(executable: str, config_path: Path, *arguments: str) -> str:
    return (" \\" + "\n").join((executable, f"--config {config_path.as_posix()}", *arguments))


def _target_cache_script(config_path: Path, config: dict, output_dir: str) -> str:
    shards = config.get("target_cache_shards", {})
    input_path = _year_path(shards.get("input_path_template"))
    output_path = _year_path(shards.get("output_path_template"))
    long_path = _year_path(shards.get("long_label_input_template"))
    short_path = _year_path(shards.get("short_label_input_template"))
    short_success = _year_path(shards.get("short_label_success_template"))
    if not input_path or not output_path:
        raise SystemExit("target-cache indexed rendering requires input/output path templates")

    dependencies = [
        input_path,
        f"{input_path}.manifest.json",
        f"{input_path}.lock.done",
        *((long_path,) if long_path else ()),
        *((short_path, f"{short_path}.manifest.json") if short_path else ()),
        *((short_success,) if short_path and short_success else ()),
    ]
    waits = " ".join(f'"{path}"' for path in dependencies)
    assignments = [
        f'INPUT="{input_path}"',
        f'OUTPUT="{output_path}"',
        *((f'SHORT_INPUT="{short_path}"',) if short_path else ()),
        *((f'LONG_INPUT="{long_path}"',) if long_path else ()),
        f'TRACE_DIR="{output_dir}/year=${{YEAR}}"',
    ]
    arguments = [
        '--input "${INPUT}"',
        '--output "${OUTPUT}"',
        '--output-dir "${TRACE_DIR}"',
        *(('--long-label-input "${LONG_INPUT}"',) if long_path else ()),
        *(
            (
                '--short-label-input "${SHORT_INPUT}"',
                f"--short-label-col {get(config, 'target_cache', 'short_label_col', 'label')}",
                f"--short-valid-col {get(config, 'target_cache', 'short_valid_col', 'valid_label')}",
            )
            if short_path
            else ()
        ),
        "--overwrite",
    ]
    command = _config_command("osf-build-target-label-cache", config_path, *arguments)
    lines = [
        *assignments,
        "WAIT_STARTED=${SECONDS}",
        f"for WAIT_PATH in {waits}; do",
        '  until [ -f "${WAIT_PATH}" ]; do',
        "    if [ $((SECONDS - WAIT_STARTED)) -ge "
        f"{int(get(config, 'k8s', 'wait_for_path_timeout_seconds', 86400))} ]; then",
        '      echo "timed out waiting for dependency file: ${WAIT_PATH}" >&2',
        "      exit 1",
        "    fi",
        '    echo "waiting for dependency file: ${WAIT_PATH}"',
        f"    sleep {int(get(config, 'k8s', 'wait_for_path_interval_seconds', 120))}",
        "  done",
        '  echo "dependency file is ready: ${WAIT_PATH}"',
        "done",
        'if [ -f "${OUTPUT}" ] && [ -f "${TRACE_DIR}/target_cache_trace.json" ] && [ -f "${TRACE_DIR}/_SUCCESS" ]; then',
        '  echo "mixed target shard already complete, skipping: ${OUTPUT}"',
        "  exit 0",
        "fi",
        'mkdir -p "$(dirname "${OUTPUT}")" "${TRACE_DIR}"',
        command,
        'touch "${TRACE_DIR}/_SUCCESS"',
    ]
    return "\n".join(lines)


def _indexed_config_script(config: dict, kind: str) -> str:
    run_id_template = str(get(config, "k8s", "indexed_run_id_template", "") or "")
    if not run_id_template:
        return ""
    if "{year}" not in run_id_template:
        raise SystemExit("k8s.indexed_run_id_template must contain {year}")
    if kind not in {"cache_transform", "labeled_cache"}:
        raise SystemExit("indexed run-id templates support cache_transform or labeled_cache")
    wait_paths = coerce_str_list(get(config, "k8s", "indexed_wait_for_paths", []))
    wait_script = (
        wait_for_specific_paths_yaml(
            [_year_path(path) for path in wait_paths],
            timeout_seconds=int(get(config, "k8s", "wait_for_path_timeout_seconds", 86400)),
            interval_seconds=int(get(config, "k8s", "wait_for_path_interval_seconds", 120)),
            indent=0,
        ).rstrip()
        if wait_paths
        else ""
    )
    run_id_value = run_id_template.replace("{year}", "${YEAR}")
    return "\n".join(
        [
            f'RUN_ID="{run_id_value}"',
            'CONFIG="experiments/runs/${RUN_ID}.toml"',
            'OUT="/mnt/output/opening_strength_fit/cache_builds/${RUN_ID}"',
            *((wait_script,) if wait_script else ()),
            f'exec {INDEXED_BUILDER_COMMANDS[kind]} --config "${{CONFIG}}" --output-dir "${{OUT}}"',
        ]
    )


def _short_label_script(config_path: Path, config: dict, output_dir: str) -> str:
    shards = config.get("short_label_shards", {})
    input_path = _year_path(shards.get("input_path_template"))
    output_path = _year_path(shards.get("output_path_template"))
    if not input_path or not output_path:
        raise SystemExit("short-label indexed rendering requires input/output path templates")
    trace_dir = f"{output_dir}/year=${{YEAR}}"
    waits = wait_for_specific_paths_yaml(
        [input_path, f"{input_path}.manifest.json"],
        timeout_seconds=int(get(config, "k8s", "wait_for_path_timeout_seconds", 259200)),
        interval_seconds=int(get(config, "k8s", "wait_for_path_interval_seconds", 120)),
        indent=0,
    ).rstrip()
    command = _config_command(
        "exec osf-build-short-labels",
        config_path,
        '--input "${INPUT}"',
        '--output "${OUTPUT}"',
        '--output-dir "${TRACE_DIR}"',
        "--overwrite",
    )
    return "\n".join(
        (
            f'INPUT="{input_path}"',
            f'OUTPUT="{output_path}"',
            f'TRACE_DIR="{trace_dir}"',
            waits,
            'if [ -f "${OUTPUT}" ] && [ -f "${OUTPUT}.manifest.json" ] && [ -f "${TRACE_DIR}/_SUCCESS" ]; then',
            '  echo "short-label shard already complete, skipping: ${OUTPUT}"',
            "  exit 0",
            "fi",
            command,
        )
    )


def _builder_script(config_path: Path, config: dict, output_dir: str) -> str:
    kind = str(get(config, "run", "kind", "")).strip().lower()
    if kind not in INDEXED_BUILDER_KINDS:
        allowed = ", ".join(sorted(INDEXED_BUILDER_KINDS))
        raise SystemExit(f"indexed rendering requires run.kind in: {allowed}")
    if indexed_config_script := _indexed_config_script(config, kind):
        return indexed_config_script
    if kind == "cache_transform":
        return _target_cache_script(config_path, config, output_dir)
    if kind == "short_label_cache":
        return _short_label_script(config_path, config, output_dir)
    if kind == "labeled_cache":
        return "\n".join(
            (
                f'OUT="{output_dir}/year=${{YEAR}}"',
                'mkdir -p "${OUT}"',
                _config_command(
                    "exec osf-build-labeled-cache", config_path, '--output-dir "${OUT}"'
                ),
            )
        )
    extra = " --kind features" if kind == "training_feature_dataset" else ""
    return (
        f"exec {INDEXED_BUILDER_COMMANDS[kind]} --config {config_path.as_posix()}{extra} "
        '--year "${YEAR}" --overwrite'
    )


def _spread_yaml(config: dict, job_name: str, indent: int) -> str:
    if not bool(get(config, "k8s", "spread_across_nodes", False)):
        return ""
    return textwrap.indent(
        f"""\
podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        topologyKey: kubernetes.io/hostname
        labelSelector:
          matchLabels:
            app: {job_name}
""",
        " " * indent,
    )


def _matrix_job_name(case_name: str, matrix_render: dict) -> str:
    tokens = case_name.split("_")
    if len(tokens) < 3 or not tokens[0].startswith("w") or not tokens[-1].startswith("h"):
        raise SystemExit(f"invalid matrix case name for job rendering: {case_name}")
    name = "-".join(
        (
            str(matrix_render.get("job_prefix", "opening-strength-matrix")).strip("-"),
            tokens[0],
            tokens[-1],
            str(matrix_render.get("job_version", "v1")).strip("-"),
        )
    )
    if len(name) > KUBERNETES_NAME_LIMIT:
        raise SystemExit(f"matrix job name exceeds {KUBERNETES_NAME_LIMIT} characters: {name}")
    return name


def render_matrix_training_jobs(config_path: Path, config: dict, image: str) -> str:
    matrix = config.get("matrix", {})
    cases = matrix.get("cases", []) if isinstance(matrix, dict) else []
    matrix_render = matrix.get("render", {}) if isinstance(matrix, dict) else {}
    if not cases or not isinstance(matrix_render, dict):
        raise SystemExit("matrix rendering requires [[matrix.cases]] and [matrix.render]")
    run_id_template = str(matrix_render.get("run_id_template", "") or "")
    input_years = [int(year) for year in matrix_render.get("input_years", [])]
    if "{case}" not in run_id_template or not input_years:
        raise SystemExit("matrix.render requires run_id_template with {case} and input_years")

    k8s = config.get("k8s", {})
    mount_path = str(k8s.get("mount_path", "/mnt/output"))
    resources = k8s.get("resources", {})
    output_root = run_output_dir(config, run_id(config, config_path), mount_path=mount_path)
    month_windows = month_windows_from_config(config)
    test_starts = " ".join(start for start, _ in month_windows)
    test_ends = " ".join(end for _, end in month_windows)
    scheduler = scheduler_yaml(config, resources, indent=6)
    env_from = k8s_env_from(config, indent=10)
    matrix_label = str(matrix_render.get("label", run_id(config, config_path)))
    parallelism = max(1, int(k8s.get("shard_parallelism", len(month_windows))))
    active_deadline = max(1, int(k8s.get("active_deadline_seconds", 1209600)))
    backoff = max(0, int(k8s.get("backoff_limit", 8)))
    ttl = max(0, int(k8s.get("ttl_seconds_after_finished", 86400)))
    suspend = str(bool(matrix_render.get("suspend", True))).lower()
    manifests = []

    for case in cases:
        if not isinstance(case, dict):
            raise SystemExit("matrix cases must be TOML tables")
        case_name = str(case.get("name", ""))
        feature_path = str(case.get("feature_path", ""))
        label_path = str(case.get("label_path", ""))
        if not case_name or not feature_path or not label_path:
            raise SystemExit("matrix case requires name, feature_path, and label_path")
        job_name = _matrix_job_name(case_name, matrix_render)
        case_window = case_name.removeprefix("w").rsplit("_h", 1)[0].replace("_", "-")
        case_run_id = run_id_template.format(case=case_name)
        output_dir = f"{output_root.rstrip('/')}/{case_name}/month_${{TEST_START}}"
        labels = {
            "app": job_name,
            "osf-matrix": matrix_label,
            "osf-window": case_window,
            "osf-case": case_name,
        }
        script = textwrap.indent(
            textwrap.dedent(
                f"""\
                set -euo pipefail
                TEST_STARTS=({test_starts})
                TEST_ENDS=({test_ends})
                INDEX="${{JOB_COMPLETION_INDEX:?missing JOB_COMPLETION_INDEX}}"
                if [ "${{INDEX}}" -lt 0 ] || [ "${{INDEX}}" -ge "${{#TEST_STARTS[@]}}" ]; then
                  echo "JOB_COMPLETION_INDEX out of range: ${{INDEX}}" >&2
                  exit 1
                fi
                TEST_START="${{TEST_STARTS[${{INDEX}}]}}"
                TEST_END="${{TEST_ENDS[${{INDEX}}]}}"
                FEATURE_ROOT={shlex.quote(feature_path)}
                LABEL_ROOT={shlex.quote(label_path)}
                OUT="{output_dir}"

                for YEAR in {" ".join(str(year) for year in input_years)}; do
                  test -f "${{FEATURE_ROOT}}/year=${{YEAR}}/_SUCCESS"
                  test -f "${{LABEL_ROOT}}/year=${{YEAR}}/_SUCCESS"
                done
                if [ -f "${{OUT}}/_SUCCESS" ] && [ -f "${{OUT}}/metrics_by_year.csv" ] && [ -f "${{OUT}}/predictions.parquet" ]; then
                  echo "output already complete, skipping ${{OUT}}"
                  exit 0
                fi

                exec osf-train \\
                  --config {config_path.as_posix()} \\
                  --feature-input "${{FEATURE_ROOT}}" \\
                  --label-input "${{LABEL_ROOT}}" \\
                  --run-id {shlex.quote(case_run_id)} \\
                  --rolling-monthly \\
                  --train-months {int(get(config, "window", "train_months", 36))} \\
                  --test-months {int(get(config, "window", "test_months", 6))} \\
                  --test-stride-months {int(get(config, "window", "test_stride_months", 6))} \\
                  --test-start-month "${{TEST_START}}" \\
                  --test-end-month "${{TEST_END}}" \\
                  --output-dir "${{OUT}}"
                """
            ),
            " " * 14,
        )
        job_header = configured_job_pod_header_yaml(
            config,
            job_name,
            labels=labels,
            spec_lines=(
                f"activeDeadlineSeconds: {active_deadline}",
                f"backoffLimit: {backoff}",
                "completionMode: Indexed",
                f"completions: {len(month_windows)}",
                f"parallelism: {parallelism}",
                f"suspend: {suspend}",
            ),
            pod_labels=labels,
            ttl_seconds_after_finished=ttl,
        )
        resources_yaml = container_resources_yaml(
            resources,
            indent=10,
            defaults=("8", "256Gi", "16", "384Gi"),
            gpu_count=gpu_count(resources),
        )
        manifests.append(
            job_manifest_yaml(
                job_header,
                scheduler,
                image,
                script,
                mount_path,
                resources_yaml,
                name="trainer",
                env_from=env_from,
                env_before_workdir=True,
                indexed=True,
            )
        )
    return "---\n".join(manifests)


def render_indexed_builder_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    k8s = config.get("k8s", {})
    job_name = str(k8s.get("job_name", f"opening-strength-{run_id_value}"))
    mount_path = str(k8s.get("mount_path", "/mnt/output"))
    resources = k8s.get("resources", {})
    years = _years(config)
    parallelism = max(1, int(k8s.get("shard_parallelism", 1)))
    backoff = max(0, int(k8s.get("backoff_limit", 0)))
    deadline = int(k8s.get("active_deadline_seconds", 0) or 0)
    output_dir = run_output_dir(config, run_id_value, mount_path=mount_path)
    command = textwrap.indent(_builder_script(config_path, config, output_dir), " " * 14)
    env_from = k8s_env_from(config, indent=10)
    scheduler = scheduler_yaml(config, resources, indent=6)
    config_map_volume = training_config_map_volume_yaml(config, indent=8)
    config_map_mount = training_config_map_mount_yaml(config, indent=12)
    spread = _spread_yaml(config, job_name, indent=8)
    if spread:
        if "      affinity:\n" not in scheduler:
            scheduler += "      affinity:\n"
        scheduler = scheduler.rstrip() + "\n" + spread

    run_id_template = str(k8s.get("indexed_run_id_template", "") or "")
    run_ids = ",".join(run_id_template.format(year=year) for year in years)
    annotations = {"opening-strength-fit/run-ids": f'"{run_ids}"'} if run_id_template else None
    spec_lines = (
        *((f"activeDeadlineSeconds: {deadline}",) if deadline else ()),
        f"backoffLimit: {backoff}",
        "completionMode: Indexed",
        f"completions: {len(years)}",
        f"parallelism: {parallelism}",
    )
    job_header = configured_job_pod_header_yaml(
        config,
        job_name,
        annotations=annotations,
        labels={"app": job_name},
        spec_lines=spec_lines,
        pod_labels={"app": job_name},
    )
    resources_yaml = container_resources_yaml(resources, indent=10)
    script = (
        textwrap.indent(
            textwrap.dedent(
                f"""\
            set -euo pipefail
            YEARS=({" ".join(str(year) for year in years)})
            INDEX="${{JOB_COMPLETION_INDEX:-}}"
            if [ -z "${{INDEX}}" ] || [ "${{INDEX}}" -lt 0 ] || [ "${{INDEX}}" -ge "${{#YEARS[@]}}" ]; then
              echo "invalid JOB_COMPLETION_INDEX=${{INDEX}}" >&2
              exit 1
            fi
            YEAR="${{YEARS[${{INDEX}}]}}"
            """
            ),
            " " * 14,
        )
        + command
    )
    return job_manifest_yaml(
        job_header,
        scheduler,
        image,
        script,
        mount_path,
        resources_yaml,
        env_from=env_from,
        indexed=True,
        config_volume=config_map_volume,
        extra_mounts=config_map_mount,
    )


def render_top1000_job(config_path: Path, config: dict, image: str) -> str:
    run_id_value = run_id(config, config_path)
    k8s = config.get("k8s", {})
    analysis = config.get("analysis", {}).get("top1000", {})
    output_dir = run_output_dir(config, run_id_value)
    job_name = str(k8s.get("job_name", f"opening-strength-{run_id_value}"))
    mount_path = str(k8s.get("mount_path", "/mnt/output"))
    script_map = str(k8s.get("analysis_config_map", "os-top1000-rank-bucket-script-v1"))
    resources = k8s.get("resources", {})
    scheduler = scheduler_yaml(config, resources, indent=6)
    env_from = k8s_env_from(config, indent=10)
    next_label_root = str(analysis.get("next_close_label_input", ""))
    arguments = [
        "python /opt/analysis/run_top1000_rank_bucket_diagnostics.py",
        f"--prediction-root {analysis.get('prediction_root', '/mnt/output/opening_strength_fit/nn')}",
        f"--next-label-root {next_label_root}",
        f"--pool-path {analysis.get('pool_path', '')}",
        f"--output-dir {output_dir}",
        f"--variant {analysis.get('variant', '')}",
        f"--run-id {analysis.get('source_run_id', '')}",
    ]
    if "histogram_bin_width_bps" in analysis:
        arguments += [
            "--top1000-bucket-return-histogram-only",
            f"--histogram-bin-width-bps {int(analysis['histogram_bin_width_bps'])}",
        ]
    command = (" \\" + "\n").join(arguments)
    if success_path := str(analysis.get("success_path", "") or ""):
        command += f"\ntouch {shlex.quote(success_path)}"
    compat_pattern = str(analysis.get("next_close_label_filename_template", "") or "")
    preamble = ""
    if compat_pattern:
        years = [int(value) for value in analysis.get("years", [])]
        if not years or "{year}" not in compat_pattern:
            raise SystemExit(
                "top1000 compatibility labels require years and a {year} filename template"
            )
        compat_root = "/tmp/next-close-labels"
        source = f"{next_label_root.rstrip('/')}/{compat_pattern}".replace("{year}", "${YEAR}")
        preamble = textwrap.dedent(
            f"""\
            COMPAT_LABEL_ROOT={compat_root}
            mkdir -p "${{COMPAT_LABEL_ROOT}}"
            for YEAR in {" ".join(str(year) for year in years)}; do
              ln -sf \
                "{source}" \
                "${{COMPAT_LABEL_ROOT}}/opening_${{YEAR}}_next_close_labels_v1.parquet"
            done
            """
        )
        command = command.replace(next_label_root, '"${COMPAT_LABEL_ROOT}"')
    script = textwrap.indent("set -euo pipefail\n" + preamble + command, " " * 14)
    resources_yaml = container_resources_yaml(
        resources,
        indent=10,
        defaults=("8", "192Gi", "16", "320Gi"),
    )

    job_header = configured_job_pod_header_yaml(
        config,
        job_name,
        volume_lines=(
            "        - name: analysis-script",
            "          configMap:",
            f"            name: {script_map}",
        ),
    )
    return job_manifest_yaml(
        job_header,
        scheduler,
        image,
        script,
        mount_path,
        resources_yaml,
        env_from=env_from,
        extra_mounts="""\
            - name: analysis-script
              mountPath: /opt/analysis
              readOnly: true
""",
    )
