from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import equal_weighted_period_means, write_analysis_result

OUTCOMES = ("entry_to_close", "close_to_next_close", "entry_to_next_close")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    metrics = pd.read_parquet(args.input_dir / "limit_nextclose_group_metrics.parquet")
    for outcome in OUTCOMES:
        gross = f"{outcome}_limit_gross_contribution_bps"
        pool = f"{outcome}_limit_pool_contribution_bps"
        limit = f"{outcome}_limit_excess_contribution_bps"
        nonlimit = f"{outcome}_nonlimit_excess_contribution_bps"
        total = f"{outcome}_excess_bps"
        metrics[gross] = pd.to_numeric(metrics[gross], errors="coerce").fillna(0.0)
        metrics[pool] = pd.to_numeric(metrics[pool], errors="coerce").fillna(0.0)
        metrics[limit] = metrics[gross] - metrics[pool]
        residual = metrics[limit] + metrics[nonlimit] - metrics[total]
        max_abs_residual = float(residual.abs().max())
        print(
            f"reconcile outcome={outcome} max_abs_residual_bps={max_abs_residual:.12g}",
            flush=True,
        )
        if not np.isfinite(max_abs_residual) or max_abs_residual > 0.01:
            raise SystemExit(
                f"{outcome} repaired decomposition does not reconcile: "
                f"max_abs_residual_bps={max_abs_residual}"
            )

    columns = [
        "selected_limit_share_pct",
        "entry_to_close_excess_bps",
        "entry_to_close_selected_limit_mean_bps",
        "entry_to_close_limit_excess_contribution_bps",
        "entry_to_close_nonlimit_excess_contribution_bps",
        "close_to_next_close_excess_bps",
        "close_to_next_close_selected_limit_mean_bps",
        "close_to_next_close_limit_excess_contribution_bps",
        "close_to_next_close_nonlimit_excess_contribution_bps",
        "entry_to_next_close_excess_bps",
        "entry_to_next_close_selected_limit_mean_bps",
        "entry_to_next_close_limit_excess_contribution_bps",
        "entry_to_next_close_nonlimit_excess_contribution_bps",
        "reselected_nonlimit_next_excess_bps",
    ]
    summary = equal_weighted_period_means(
        metrics,
        by=["label_horizon"],
        period_column="quarter",
        value_columns=columns,
        count_name="groups",
    )

    trace = {
        "source": str(args.input_dir),
        "repair": "zero-fill selected-limit contribution when a Top100 cross-section contains no final-limit stock",
        "assertion": "limit plus non-limit excess contribution equals total excess within 0.01 bps for every cross-section and outcome",
    }
    write_analysis_result(
        args.output_dir,
        metrics,
        summary,
        metrics_filename="limit_nextclose_group_metrics.parquet",
        summary_filename="limit_nextclose_summary.csv",
        trace=trace,
    )


if __name__ == "__main__":
    main()
