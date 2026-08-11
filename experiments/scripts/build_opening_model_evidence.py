from __future__ import annotations

import json
from pathlib import Path

from opening_strength_fit.artifact_catalog import (
    artifact_file_manifest,
    copy_artifact_specs,
    copy_csv_columns,
)

ROOT = Path(__file__).resolve().parents[2]
V4_RUN_ID = (
    "nn_delay6_clock_state_36m_2022_2025_auction_pruned_multi_denominator_"
    "grouped_gated_v2_mech_v3_gelu_mse_v1"
)
V6_RUN_ID = (
    "nn_delay6_v6_decision_clock_state_36m_2022_2025_w0931_0940_"
    "auction_pruned_multi_denominator_grouped_gated_v2_mech_v3_gelu_mse_v1"
)
BASELINE_NAME = "opening_model"
CACHE_NAME = "opening_cache"
COMPARISON_DIR = Path(
    "experiments/results/backtests/v6_decision_state_acceptance_w0931_vs_v4_2022_2025_v1"
)
V6_BACKTEST_DIR = Path("experiments/results/backtests") / V6_RUN_ID
BUCKET_DIR = Path(
    "experiments/results/backtests/"
    "top1000_rank_bucket_diag_v6_w0931_0940_multiden_2022_2025_v1/rank_bucket"
)
HISTOGRAM_DIR = Path(
    "experiments/results/backtests/"
    "top1000_score_bucket_return_histogram_v6_w0931_0940_multiden_2022_2025_v1"
)
BUNDLE_DIR = Path("experiments/evidence/backtests") / V6_RUN_ID

COPY_SPECS = (
    (
        COMPARISON_DIR / "optimization_directions_overlay_acceptance.svg",
        "01_signal_acceptance.svg",
    ),
    (
        COMPARISON_DIR / "optimization_directions_overlay_acceptance_plot_data.csv",
        "01_signal_acceptance.csv",
    ),
    (
        COMPARISON_DIR / "optimization_directions_net_alpha_cumulative.svg",
        "02_top100_cumulative.svg",
    ),
    (
        BUCKET_DIR / "top1000_bucket_returns.svg",
        "03_top1000_bucket_curve.svg",
    ),
    (
        BUCKET_DIR / "bucket_curve_plot_data.csv",
        "03_top1000_bucket_curve.csv",
    ),
    (
        HISTOGRAM_DIR / "top1000_score_bucket_return_100bps_counts.svg",
        "04_top1000_return_distribution.svg",
    ),
    (
        HISTOGRAM_DIR / "top1000_score_bucket_return_100bps_counts.csv",
        "04_top1000_return_distribution.csv",
    ),
    (
        V6_BACKTEST_DIR / "pool_internal_summary.csv",
        "pool_internal_summary.csv",
    ),
    (
        V6_BACKTEST_DIR / "pool_internal_halfyear_summary.csv",
        "pool_internal_halfyear_summary.csv",
    ),
    (
        HISTOGRAM_DIR / "top1000_score_bucket_distribution_summary.csv",
        "top1000_distribution_summary.csv",
    ),
    (
        COMPARISON_DIR / "optimization_directions_trace.json",
        "trace_optimization.json",
    ),
    (BUCKET_DIR / "trace.json", "trace_top1000_bucket.json"),
    (HISTOGRAM_DIR / "trace.json", "trace_top1000_distribution.json"),
)
CUMULATIVE_SOURCE = COMPARISON_DIR / "optimization_directions_net_alpha_cumulative_plot_data.csv"
CUMULATIVE_OUTPUT = "02_top100_cumulative.csv"
CUMULATIVE_COLUMNS = (
    "pool",
    "pool_label",
    "week_start",
    "variant",
    "next_cumulative_net_return_bps",
    "next_cumulative_alpha_bps",
)


def build_bundle(root: Path = ROOT) -> Path:
    destination = root / BUNDLE_DIR
    destination.mkdir(parents=True, exist_ok=True)
    sources = copy_artifact_specs(root, destination, COPY_SPECS)

    cumulative_rows = copy_csv_columns(
        root / CUMULATIVE_SOURCE,
        destination / CUMULATIVE_OUTPUT,
        CUMULATIVE_COLUMNS,
    )
    sources[CUMULATIVE_OUTPUT] = CUMULATIVE_SOURCE.as_posix()

    manifest = {
        "schema_version": 1,
        "baseline_name": BASELINE_NAME,
        "cache_name": CACHE_NAME,
        "baseline_run_id": V4_RUN_ID,
        "candidate_run_id": V6_RUN_ID,
        "source_run_id": V6_RUN_ID,
        "comparison": (
            "09:31-09:40 v4 forward-5s decision sampling versus corrected "
            "v6 last-known clock-state sampling"
        ),
        "cumulative_compact_columns": list(CUMULATIVE_COLUMNS),
        "cumulative_rows": cumulative_rows,
        "files": artifact_file_manifest(destination, sources),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    destination = build_bundle()
    print(f"wrote {BASELINE_NAME} four-figure evidence: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
