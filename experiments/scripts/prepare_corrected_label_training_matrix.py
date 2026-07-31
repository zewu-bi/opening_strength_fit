from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
JOBS = ROOT / "experiments" / "jobs"
TRAINING_TEMPLATE = RUNS / (
    "nn_delay6_v6_decision_clock_state_36m_2022_2025_w0931_0940_"
    "auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1.toml"
)
IMAGE = (
    "registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260731-corrected-label-matrix"
)
YEARS = tuple(range(2019, 2026))


@dataclass(frozen=True, slots=True)
class Variant:
    window: str
    start_time: str
    end_time: str
    decision_times: tuple[str, ...]
    short_name: str
    hold_seconds: int
    base_root: str
    base_file: str
    short_root: str
    short_file: str
    short_trace_run: str
    long_root: str
    long_file: str

    @property
    def suffix(self) -> str:
        return f"w{self.window}_{self.short_name}_corrected_nextclose"

    @property
    def target_run_id(self) -> str:
        return f"build_target_v6_{self.suffix}"

    @property
    def training_run_id(self) -> str:
        return f"nn_v6_{self.suffix}_36m_grouped_gated_v2_mse"

    @property
    def target_root(self) -> str:
        return f"opening_2019_2025_target_v6_{self.suffix}"

    @property
    def target_file(self) -> str:
        return f"opening_{{year}}_target_v6_{self.suffix}.parquet"

    @property
    def uses_short_sidecar(self) -> bool:
        return self.hold_seconds == 180


VARIANTS = (
    Variant(
        window="0931_0940",
        start_time="09:31:00",
        end_time="09:40:00",
        decision_times=tuple(f"09:{minute:02d}:00" for minute in range(31, 41)),
        short_name="short1m",
        hold_seconds=60,
        base_root=("opening_2019_2025_label_v6_decision_clock_state_clock6_unique_base_mcap_lag1"),
        base_file=(
            "opening_{year}_label_v6_decision_clock_state_clock6_unique_base_mcap_lag1.parquet"
        ),
        short_root="",
        short_file="",
        short_trace_run="",
        long_root=("opening_2019_2025_next_close_decision_clock_state_clock6_0931_0940"),
        long_file=("opening_{year}_next_close_decision_clock_state_clock6_0931_0940.parquet"),
    ),
    Variant(
        window="0931_0940",
        start_time="09:31:00",
        end_time="09:40:00",
        decision_times=tuple(f"09:{minute:02d}:00" for minute in range(31, 41)),
        short_name="short3m",
        hold_seconds=180,
        base_root=("opening_2019_2025_label_v6_decision_clock_state_clock6_unique_base_mcap_lag1"),
        base_file=(
            "opening_{year}_label_v6_decision_clock_state_clock6_unique_base_mcap_lag1.parquet"
        ),
        short_root=("opening_2019_2025_short_label_v6_clock6_0931_0940_h180_vwap60_v1"),
        short_file=("opening_{year}_short_label_v6_clock6_0931_0940_h180_vwap60_v1.parquet"),
        short_trace_run="build_v6_w0931_0940_short_h180_vwap60_labels_v1",
        long_root=("opening_2019_2025_next_close_decision_clock_state_clock6_0931_0940"),
        long_file=("opening_{year}_next_close_decision_clock_state_clock6_0931_0940.parquet"),
    ),
    Variant(
        window="1001_1010",
        start_time="10:01:00",
        end_time="10:10:00",
        decision_times=tuple(f"10:{minute:02d}:00" for minute in range(1, 11)),
        short_name="short3m",
        hold_seconds=180,
        base_root=(
            "opening_2019_2025_label_v6_decision_clock_state_clock6_1001_1010_"
            "from_start_auction_reuse_mcap_lag1"
        ),
        base_file=(
            "opening_{year}_label_v6_decision_clock_state_clock6_1001_1010_"
            "from_start_auction_reuse_mcap_lag1.parquet"
        ),
        short_root=("opening_2019_2025_short_label_v6_clock6_1001_1010_h180_vwap60_v1"),
        short_file=("opening_{year}_short_label_v6_clock6_1001_1010_h180_vwap60_v1.parquet"),
        short_trace_run="build_v6_w1001_1010_short_h180_vwap60_labels_v1",
        long_root=("opening_2019_2025_next_close_decision_clock_state_clock6_1001_1010"),
        long_file=("opening_{year}_next_close_decision_clock_state_clock6_1001_1010.parquet"),
    ),
    Variant(
        window="1401_1410",
        start_time="14:01:00",
        end_time="14:10:00",
        decision_times=tuple(f"14:{minute:02d}:00" for minute in range(1, 11)),
        short_name="short3m",
        hold_seconds=180,
        base_root=(
            "opening_2019_2025_label_v6_decision_clock_state_clock6_1401_1410_"
            "from_start_auction_reuse_mcap_lag1"
        ),
        base_file=(
            "opening_{year}_label_v6_decision_clock_state_clock6_1401_1410_"
            "from_start_auction_reuse_mcap_lag1.parquet"
        ),
        short_root=("opening_2019_2025_short_label_v6_clock6_1401_1410_h180_vwap60_v1"),
        short_file=("opening_{year}_short_label_v6_clock6_1401_1410_h180_vwap60_v1.parquet"),
        short_trace_run="build_v6_w1401_1410_short_h180_vwap60_labels_v1",
        long_root=("opening_2019_2025_next_close_decision_clock_state_clock6_1401_1410"),
        long_file=("opening_{year}_next_close_decision_clock_state_clock6_1401_1410.parquet"),
    ),
)


def _cache_path(root: str, file_name: str, year: int | str) -> str:
    return f"/mnt/output/opening_strength_fit/cache/{root}/{file_name.format(year=year)}"


def _toml_list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(f'"{value}"' for value in values) + "]"


def render_target_config(variant: Variant) -> str:
    first_year = YEARS[0]
    first_input = _cache_path(variant.base_root, variant.base_file, first_year)
    first_output = _cache_path(variant.target_root, variant.target_file, first_year)
    first_long = _cache_path(variant.long_root, variant.long_file, first_year)
    short_fields = ""
    shard_short_field = ""
    if variant.uses_short_sidecar:
        short_fields = (
            f'short_label_input = "{_cache_path(variant.short_root, variant.short_file, first_year)}"\n'
            'short_label_col = "label"\n'
            'short_valid_col = "valid_label"\n'
        )
        shard_short_field = f'short_label_input_template = "{_cache_path(variant.short_root, variant.short_file, "{year}")}"\n'
    return f'''[run]
id = "{variant.target_run_id}"
kind = "cache_transform"
description = "Build seven annual v6 mixed-target shards for {variant.start_time}-{variant.end_time}: {variant.short_name} short component plus corrected same-window next-close long component."
status = "running"

[target_cache]
input_path = "{first_input}"
output_path = "{first_output}"
mode = "mixed"
group_cols = ["date", "decision_target_timestamp"]
label_col = "label"
target_col = "target_label"
raw_label_col = "label_raw"
min_group_size = 50
{short_fields}long_label_input = "{first_long}"
long_label_col = "alpha_return_next_close"
long_label_weight = 0.30
short_label_transform = "zscore"
long_label_transform = "zscore"

[target_cache_shards]
years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
input_path_template = "{_cache_path(variant.base_root, variant.base_file, "{year}")}"
output_path_template = "{_cache_path(variant.target_root, variant.target_file, "{year}")}"
{shard_short_field}long_label_input_template = "{_cache_path(variant.long_root, variant.long_file, "{year}")}"

[data]
source = "labeled_pvc"
labeled_path = "/mnt/output/opening_strength_fit/cache/{variant.base_root}"

[output]
local_dir = "output/artifacts/cache_builds/{variant.target_run_id}"
k8s_dir = "/mnt/output/opening_strength_fit/cache_builds/{variant.target_run_id}"

[k8s]
namespace = "bizewu"
job_name = "os-target-v6-{variant.window.replace("_", "-")}-{variant.short_name}"
image_pull_secret = "highfort"
pvc = "bizewu-private-data"
mount_path = "/mnt/output"
helper_image = "{IMAGE}"
avoid_nodes = ["node7", "node8"]
shard_parallelism = 7

[k8s.resources]
cpu_request = "8"
cpu_limit = "16"
memory_request = "256Gi"
memory_limit = "512Gi"
'''


def render_target_job(variant: Variant) -> str:
    job_name = f"os-target-v6-{variant.window.replace('_', '-')}-{variant.short_name}"
    base_path = _cache_path(variant.base_root, variant.base_file, "${YEAR}")
    output_path = _cache_path(variant.target_root, variant.target_file, "${YEAR}")
    long_path = _cache_path(variant.long_root, variant.long_file, "${YEAR}")
    short_setup = ""
    short_wait = ""
    short_args = ""
    if variant.uses_short_sidecar:
        short_path = _cache_path(variant.short_root, variant.short_file, "${YEAR}")
        short_setup = f'''              SHORT_INPUT="{short_path}"
              SHORT_TRACE="/mnt/output/opening_strength_fit/cache_builds/{variant.short_trace_run}/year=${{YEAR}}/_SUCCESS"
'''
        short_wait = ' "${SHORT_INPUT}" "${SHORT_INPUT}.manifest.json" "${SHORT_TRACE}"'
        short_args = """ \\
                --short-label-input "${SHORT_INPUT}" \\
                --short-label-col label \\
                --short-valid-col valid_label"""
    config_path = f"experiments/runs/{variant.target_run_id}.toml"
    return f'''apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: bizewu
  labels:
    app: {job_name}
spec:
  backoffLimit: 0
  completionMode: Indexed
  completions: 7
  parallelism: 7
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: {job_name}
    spec:
      restartPolicy: Never
      imagePullSecrets:
        - name: highfort
      volumes:
        - name: opening-strength-output
          persistentVolumeClaim:
            claimName: bizewu-private-data
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
                      - node7
                      - node8
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                topologyKey: kubernetes.io/hostname
                labelSelector:
                  matchLabels:
                    app: {job_name}
      containers:
        - name: opening-strength-fit
          image: {IMAGE}
          imagePullPolicy: Always
          workingDir: /app/opening_strength_fit
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
              YEARS=(2019 2020 2021 2022 2023 2024 2025)
              INDEX="${{JOB_COMPLETION_INDEX:-}}"
              if [ -z "${{INDEX}}" ] || [ "${{INDEX}}" -lt 0 ] || [ "${{INDEX}}" -ge "${{#YEARS[@]}}" ]; then
                echo "invalid JOB_COMPLETION_INDEX=${{INDEX}}" >&2
                exit 1
              fi
              YEAR="${{YEARS[$INDEX]}}"
              INPUT="{base_path}"
              OUTPUT="{output_path}"
              LONG_INPUT="{long_path}"
{short_setup}              TRACE_DIR="/mnt/output/opening_strength_fit/cache_builds/{variant.target_run_id}/year=${{YEAR}}"
              WAIT_STARTED=${{SECONDS}}
              for WAIT_PATH in "${{INPUT}}" "${{INPUT}}.manifest.json" "${{INPUT}}.lock.done" "${{LONG_INPUT}}"{short_wait}; do
                until [ -f "${{WAIT_PATH}}" ]; do
                  if [ $((SECONDS - WAIT_STARTED)) -ge 604800 ]; then
                    echo "timed out waiting for dependency file: ${{WAIT_PATH}}" >&2
                    exit 1
                  fi
                  echo "waiting for dependency file: ${{WAIT_PATH}}"
                  sleep 120
                done
                echo "dependency file is ready: ${{WAIT_PATH}}"
              done
              if [ -f "${{OUTPUT}}" ] && [ -f "${{TRACE_DIR}}/target_cache_trace.json" ] && [ -f "${{TRACE_DIR}}/_SUCCESS" ]; then
                echo "target shard already complete, skipping: ${{OUTPUT}}"
                exit 0
              fi
              mkdir -p "$(dirname "${{OUTPUT}}")" "${{TRACE_DIR}}"
              osf-build-target-label-cache \\
                --config {config_path} \\
                --input "${{INPUT}}" \\
                --output "${{OUTPUT}}" \\
                --output-dir "${{TRACE_DIR}}"{short_args} \\
                --long-label-input "${{LONG_INPUT}}" \\
                --overwrite
              touch "${{TRACE_DIR}}/_SUCCESS"
          volumeMounts:
            - name: opening-strength-output
              mountPath: /mnt/output
          resources:
            requests:
              cpu: "8"
              memory: 256Gi
            limits:
              cpu: "16"
              memory: 512Gi
'''


def _replace(text: str, old: str, new: str, *, count: int | None = None) -> str:
    occurrences = text.count(old)
    expected = occurrences if count is None else count
    if occurrences != expected or occurrences == 0:
        raise RuntimeError(
            f"template replacement mismatch: expected={expected}, actual={occurrences}, old={old!r}"
        )
    return text.replace(old, new, expected)


def render_training_config(template: str, variant: Variant) -> str:
    old_run_id = (
        "nn_delay6_v6_decision_clock_state_36m_2022_2025_w0931_0940_"
        "auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1"
    )
    old_root = "opening_2019_2025_label_v6_decision_clock_state_clock6_unique_mixed_w030_mcap_lag1"
    old_file = (
        "opening_{year}_label_v6_decision_clock_state_clock6_unique_mixed_w030_mcap_lag1.parquet"
    )
    text = template.replace(old_run_id, variant.training_run_id)
    text = _replace(
        text,
        'description = "Controlled rerun of the canonical v4 09:31-09:40 multiden incumbent on the corrected v6 decision-clock-state cache; target, features, model, pools, rolling OOS windows, and seed remain fixed."',
        f'description = "Canonical v6 {variant.start_time}-{variant.end_time} training with {variant.short_name} short component and corrected same-window next-close long component; all feature, model, rolling OOS, pool, resource, and seed settings match the latest v6 incumbent."',
        count=1,
    )
    text = _replace(text, 'status = "completed"', 'status = "running"', count=1)
    text = text.replace(old_root, variant.target_root)
    text = text.replace(old_file.format(year="2019"), variant.target_file.format(year=2019))
    for year in YEARS[1:]:
        text = text.replace(old_file.format(year=year), variant.target_file.format(year=year))
    old_sample = """start_time = "09:31:00"
end_time = "09:40:00"
mode = "decision_points"
decision_alignment = "clock_state"
decision_times = ["09:31:00", "09:32:00", "09:33:00", "09:34:00", "09:35:00", "09:36:00", "09:37:00", "09:38:00", "09:39:00", "09:40:00"]"""
    new_sample = f'''start_time = "{variant.start_time}"
end_time = "{variant.end_time}"
mode = "decision_points"
decision_alignment = "clock_state"
decision_times = {_toml_list(variant.decision_times)}'''
    text = _replace(text, old_sample, new_sample, count=1)
    text = _replace(text, "hold_seconds = 60", f"hold_seconds = {variant.hold_seconds}", count=1)
    text = _replace(
        text,
        'job_name = "os-nn-v6-w0931-0940-multiden-v1"',
        f'job_name = "os-nn-v6-{variant.window.replace("_", "-")}-{variant.short_name}"',
        count=1,
    )
    text = _replace(
        text,
        'helper_image = "registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260731-v6-w0931-rerun-v1"',
        f'helper_image = "{IMAGE}"',
        count=1,
    )
    text = _replace(
        text,
        "wait_for_path_timeout_seconds = 259200",
        "wait_for_path_timeout_seconds = 604800",
        count=1,
    )
    success_paths = "\n".join(
        f'  "/mnt/output/opening_strength_fit/cache_builds/{variant.target_run_id}/year={year}/_SUCCESS",'
        for year in YEARS
    )
    text = _replace(
        text,
        "]\nwait_for_path_timeout_seconds = 604800",
        f"{success_paths}\n]\nwait_for_path_timeout_seconds = 604800",
        count=1,
    )
    text = _replace(
        text,
        'job_name = "os-analyze-nn-v6-w0931-0940-multiden-v1"',
        f'job_name = "os-analyze-nn-v6-{variant.window.replace("_", "-")}-{variant.short_name}"',
        count=1,
    )
    text = _replace(
        text,
        'variant = "nn_v6_w0931_0940_auction_pruned_multiden_grouped_gated_v2_mech_v3_gelu_mse"',
        f'variant = "nn_v6_{variant.suffix}_grouped_gated_v2_mech_v3_gelu_mse"',
        count=1,
    )
    text = _replace(
        text,
        'plot_prefix = "nn_v6_w0931_0940_auction_pruned_multiden_grouped_gated_v2_mech_v3_gelu_mse"',
        f'plot_prefix = "nn_v6_{variant.suffix}_grouped_gated_v2_mech_v3_gelu_mse"',
        count=1,
    )
    text = _replace(
        text,
        'plot_variant_label = "v6 decision-state 09:31-09:40 + 25 multi-denominator ratios"',
        f'plot_variant_label = "v6 decision-state {variant.start_time}-{variant.end_time} {variant.short_name} + corrected next-close"',
        count=1,
    )
    text = _replace(
        text,
        'next_close_label_input = "/mnt/output/opening_strength_fit/cache/opening_2013_2025_next_close_labels_v1"',
        f'next_close_label_input = "/mnt/output/opening_strength_fit/cache/{variant.long_root}"',
        count=1,
    )
    return text


def main() -> None:
    template = TRAINING_TEMPLATE.read_text(encoding="utf-8")
    for variant in VARIANTS:
        target_config = RUNS / f"{variant.target_run_id}.toml"
        target_job = JOBS / f"{variant.target_run_id}_sharded_job.yaml"
        training_config = RUNS / f"{variant.training_run_id}.toml"
        target_config.write_text(render_target_config(variant), encoding="utf-8")
        target_job.write_text(render_target_job(variant), encoding="utf-8")
        training_config.write_text(
            render_training_config(template, variant),
            encoding="utf-8",
        )
        print(target_config.relative_to(ROOT))
        print(target_job.relative_to(ROOT))
        print(training_config.relative_to(ROOT))


if __name__ == "__main__":
    main()
