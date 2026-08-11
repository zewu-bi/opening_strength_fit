from __future__ import annotations

import csv
import json
from pathlib import Path

from opening_strength_fit.artifact_catalog import (
    artifact_file_manifest,
    copy_artifact_specs,
    copy_csv_columns,
    require_file,
)

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RUN_ID = (
    "nn_delay6_clock_state_36m_2022_2025_auction_pruned_multi_denominator_"
    "grouped_gated_v2_mech_v3_gelu_mse_v1"
)
BUNDLE_RELATIVE_DIR = Path("experiments/evidence/backtests") / CANONICAL_RUN_ID

OVERLAY_RELATIVE_DIR = Path(
    "experiments/results/backtests/"
    "optimization_overlay_acceptance_clock6_v4_control_multiden_vs_mech328_v2_2022_2025"
)
BUCKET_RELATIVE_DIR = Path(
    "experiments/results/backtests/"
    "top1000_rank_bucket_diag_auction_multiden_2022_2025_v1/rank_bucket"
)
HISTOGRAM_RELATIVE_DIR = Path(
    "experiments/results/backtests/"
    "top1000_score_bucket_return_histogram_auction_multiden_2022_2025_v1"
)
JOURNEY_RELATIVE_DIR = Path(
    "experiments/results/backtests/"
    "optimization_overlay_acceptance_lgbm328_mlp_base_gated_multiden_2022_2025"
)

COPY_SPECS = (
    (
        OVERLAY_RELATIVE_DIR / "optimization_directions_overlay_acceptance.svg",
        "01_signal_acceptance.svg",
    ),
    (
        OVERLAY_RELATIVE_DIR / "optimization_directions_overlay_acceptance_plot_data.csv",
        "01_signal_acceptance.csv",
    ),
    (
        OVERLAY_RELATIVE_DIR / "optimization_directions_net_alpha_cumulative.svg",
        "02_top100_cumulative.svg",
    ),
    (
        BUCKET_RELATIVE_DIR / "top1000_bucket_returns.svg",
        "03_top1000_bucket_curve.svg",
    ),
    (
        BUCKET_RELATIVE_DIR / "bucket_curve_plot_data.csv",
        "03_top1000_bucket_curve.csv",
    ),
    (
        BUCKET_RELATIVE_DIR / "pool_L_bucket_returns.svg",
        "03b_full_pool_bucket_curve.svg",
    ),
    (
        HISTOGRAM_RELATIVE_DIR / "top1000_score_bucket_return_100bps_counts.svg",
        "04_top1000_return_distribution.svg",
    ),
    (
        HISTOGRAM_RELATIVE_DIR / "top1000_score_bucket_return_100bps_counts.csv",
        "04_top1000_return_distribution.csv",
    ),
    (
        HISTOGRAM_RELATIVE_DIR / "top1000_score_bucket_return_100bps_counts_full_scale.svg",
        "04b_top1000_return_distribution_full_scale.svg",
    ),
    (
        JOURNEY_RELATIVE_DIR / "optimization_directions_net_alpha_cumulative.svg",
        "05_model_journey_cumulative.svg",
    ),
    (
        OVERLAY_RELATIVE_DIR / "optimization_directions_trace.json",
        "trace_optimization.json",
    ),
    (BUCKET_RELATIVE_DIR / "trace.json", "trace_top1000_bucket.json"),
    (HISTOGRAM_RELATIVE_DIR / "trace.json", "trace_top1000_distribution.json"),
    (
        JOURNEY_RELATIVE_DIR / "optimization_directions_trace.json",
        "trace_model_journey.json",
    ),
)

CUMULATIVE_SOURCE = (
    OVERLAY_RELATIVE_DIR / "optimization_directions_net_alpha_cumulative_plot_data.csv"
)
CUMULATIVE_OUTPUT = "02_top100_cumulative.csv"
FULL_POOL_BUCKET_SOURCE = BUCKET_RELATIVE_DIR / "bucket_curve_plot_data.csv"
FULL_POOL_BUCKET_OUTPUT = "03b_full_pool_bucket_curve.csv"
JOURNEY_CUMULATIVE_SOURCE = (
    JOURNEY_RELATIVE_DIR / "optimization_directions_net_alpha_cumulative_plot_data.csv"
)
JOURNEY_CUMULATIVE_OUTPUT = "05_model_journey_cumulative.csv"
CUMULATIVE_COLUMNS = (
    "pool",
    "pool_label",
    "week_start",
    "variant",
    "next_cumulative_net_return_bps",
    "next_cumulative_alpha_bps",
)


def _write_scope_rows(
    source: Path,
    destination: Path,
    *,
    scope: str,
) -> int:
    require_file(source)
    with source.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        if "scope" not in (reader.fieldnames or ()):
            raise ValueError(f"{source}: missing scope column")
        with destination.open("w", newline="", encoding="utf-8") as destination_handle:
            writer = csv.DictWriter(
                destination_handle,
                fieldnames=reader.fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            row_count = 0
            for row in reader:
                if row["scope"] != scope:
                    continue
                writer.writerow(row)
                row_count += 1
    if row_count == 0:
        raise ValueError(f"{source}: no rows found for scope={scope}")
    return row_count


def build_bundle(root: Path = ROOT) -> Path:
    destination = root / BUNDLE_RELATIVE_DIR
    destination.mkdir(parents=True, exist_ok=True)

    sources = copy_artifact_specs(root, destination, COPY_SPECS)

    cumulative_source = root / CUMULATIVE_SOURCE
    cumulative_rows = copy_csv_columns(
        cumulative_source,
        destination / CUMULATIVE_OUTPUT,
        CUMULATIVE_COLUMNS,
    )
    sources[CUMULATIVE_OUTPUT] = CUMULATIVE_SOURCE.as_posix()

    full_pool_bucket_source = root / FULL_POOL_BUCKET_SOURCE
    full_pool_bucket_rows = _write_scope_rows(
        full_pool_bucket_source,
        destination / FULL_POOL_BUCKET_OUTPUT,
        scope="pool_L",
    )
    sources[FULL_POOL_BUCKET_OUTPUT] = FULL_POOL_BUCKET_SOURCE.as_posix()

    journey_cumulative_source = root / JOURNEY_CUMULATIVE_SOURCE
    journey_cumulative_rows = copy_csv_columns(
        journey_cumulative_source,
        destination / JOURNEY_CUMULATIVE_OUTPUT,
        CUMULATIVE_COLUMNS,
    )
    sources[JOURNEY_CUMULATIVE_OUTPUT] = JOURNEY_CUMULATIVE_SOURCE.as_posix()

    manifest = {
        "schema_version": 1,
        "canonical_candidate": "clock6_v4_multiden",
        "candidate_run_id": CANONICAL_RUN_ID,
        "control_role": "ablation baseline retained in figures 1 and 2",
        "cumulative_compact_columns": list(CUMULATIVE_COLUMNS),
        "cumulative_rows": cumulative_rows,
        "full_pool_bucket_rows": full_pool_bucket_rows,
        "journey_cumulative_rows": journey_cumulative_rows,
        "files": artifact_file_manifest(destination, sources),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    destination = build_bundle()
    print(f"wrote four-figure evidence: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
