from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import equal_weighted_period_means, write_analysis_result
from opening_strength_fit.ds350_holdout_analysis import iter_common_horizon_predictions

GROUP_KEYS = ["date", "decision_target_timestamp"]
HORIZONS = ("1m", "3m", "10m", "1h", "close")
OUTCOMES = ("entry_to_close", "close_to_next_close", "entry_to_next_close")
TOP_N = 100


def _group_metrics(work: pd.DataFrame) -> pd.DataFrame:
    ordered = work.sort_values(
        [*GROUP_KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    grouped = ordered.groupby(GROUP_KEYS, sort=False)
    top = grouped.head(TOP_N)
    limit = ordered["daily_closes_up_limit"]
    top_limit = top["daily_closes_up_limit"]

    out = grouped.agg(
        candidate_rows=("prediction", "size"),
        candidate_limit_rows=("daily_closes_up_limit", "sum"),
    )
    out = out.join(
        top.groupby(GROUP_KEYS, sort=False).agg(
            selected_rows=("prediction", "size"),
            selected_limit_rows=("daily_closes_up_limit", "sum"),
        )
    )

    for outcome in OUTCOMES:
        candidate_total = ordered.groupby(GROUP_KEYS, sort=False)[outcome].sum()
        candidate_limit = (
            ordered.loc[limit].groupby(GROUP_KEYS, sort=False)[outcome].agg(["sum", "mean"])
        )
        candidate_nonlimit = (
            ordered.loc[~limit].groupby(GROUP_KEYS, sort=False)[outcome].agg(["sum", "mean"])
        )
        selected_total = top.groupby(GROUP_KEYS, sort=False)[outcome].agg(["sum", "mean"])
        selected_limit = (
            top.loc[top_limit]
            .groupby(GROUP_KEYS, sort=False)[outcome]
            .agg(["sum", "mean", "count"])
        )
        selected_nonlimit = (
            top.loc[~top_limit]
            .groupby(GROUP_KEYS, sort=False)[outcome]
            .agg(["sum", "mean", "count"])
        )

        out[f"{outcome}_pool_mean_bps"] = candidate_total / out["candidate_rows"] * 10_000.0
        out[f"{outcome}_selected_mean_bps"] = selected_total["mean"] * 10_000.0
        out[f"{outcome}_excess_bps"] = (
            selected_total["mean"] - candidate_total / out["candidate_rows"]
        ) * 10_000.0
        out[f"{outcome}_selected_limit_mean_bps"] = selected_limit["mean"] * 10_000.0
        out[f"{outcome}_selected_nonlimit_mean_bps"] = selected_nonlimit["mean"] * 10_000.0
        selected_limit_sum = selected_limit["sum"].reindex(out.index).fillna(0.0)
        candidate_limit_sum = candidate_limit["sum"].reindex(out.index).fillna(0.0)
        selected_nonlimit_sum = selected_nonlimit["sum"].reindex(out.index).fillna(0.0)
        candidate_nonlimit_sum = candidate_nonlimit["sum"].reindex(out.index).fillna(0.0)
        out[f"{outcome}_limit_gross_contribution_bps"] = (
            selected_limit_sum / out["selected_rows"] * 10_000.0
        )
        out[f"{outcome}_limit_pool_contribution_bps"] = (
            candidate_limit_sum / out["candidate_rows"] * 10_000.0
        )
        out[f"{outcome}_limit_excess_contribution_bps"] = (
            out[f"{outcome}_limit_gross_contribution_bps"]
            - out[f"{outcome}_limit_pool_contribution_bps"]
        )
        out[f"{outcome}_nonlimit_excess_contribution_bps"] = (
            selected_nonlimit_sum / out["selected_rows"]
            - candidate_nonlimit_sum / out["candidate_rows"]
        ) * 10_000.0
        decomposed = (
            out[f"{outcome}_limit_excess_contribution_bps"]
            + out[f"{outcome}_nonlimit_excess_contribution_bps"]
        )
        if not np.allclose(
            decomposed.to_numpy(),
            out[f"{outcome}_excess_bps"].to_numpy(),
            rtol=1e-6,
            atol=1e-5,
            equal_nan=True,
        ):
            raise AssertionError(f"{outcome} contribution decomposition does not reconcile")

    nonlimit_ordered = ordered.loc[~limit]
    nonlimit_top = nonlimit_ordered.groupby(GROUP_KEYS, sort=False).head(TOP_N)
    nonlimit_pool_next = nonlimit_ordered.groupby(GROUP_KEYS, sort=False)[
        "entry_to_next_close"
    ].mean()
    nonlimit_selected_next = nonlimit_top.groupby(GROUP_KEYS, sort=False)[
        "entry_to_next_close"
    ].mean()
    out["reselected_nonlimit_next_excess_bps"] = (
        nonlimit_selected_next - nonlimit_pool_next
    ) * 10_000.0
    out["selected_limit_share_pct"] = out["selected_limit_rows"] / out["selected_rows"] * 100.0
    out["quarter"] = (
        pd.to_datetime(out.reset_index()["date"]).dt.to_period("Q").astype(str).to_numpy()
    )
    return out.reset_index()


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "selected_limit_share_pct",
        "entry_to_close_excess_bps",
        "entry_to_close_selected_limit_mean_bps",
        "entry_to_close_limit_excess_contribution_bps",
        "close_to_next_close_excess_bps",
        "close_to_next_close_selected_limit_mean_bps",
        "close_to_next_close_limit_excess_contribution_bps",
        "entry_to_next_close_excess_bps",
        "entry_to_next_close_selected_limit_mean_bps",
        "entry_to_next_close_limit_excess_contribution_bps",
        "entry_to_next_close_nonlimit_excess_contribution_bps",
        "reselected_nonlimit_next_excess_bps",
    ]
    return equal_weighted_period_means(
        metrics,
        by=["label_horizon"],
        period_column="quarter",
        value_columns=columns,
        count_name="groups",
    )


def _prepare_outcome(outcome: pd.DataFrame) -> pd.DataFrame:
    outcome = outcome.rename(
        columns={
            "label_short": "entry_to_close",
            "label_next_close": "entry_to_next_close",
        }
    )
    for column in ("entry_to_close", "entry_to_next_close"):
        outcome[column] = pd.to_numeric(outcome[column], errors="coerce")
    outcome["close_to_next_close"] = (1.0 + outcome["entry_to_next_close"]) / (
        1.0 + outcome["entry_to_close"]
    ) - 1.0
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, choices=["0931_0940", "1001_1010"])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    root = Path("/mnt/output/opening_strength_fit")
    model_root = root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"
    parts: list[pd.DataFrame] = []
    for fold_index, horizon, work in iter_common_horizon_predictions(
        model_root=model_root,
        raw_source_root=root / f"cache/opening_{args.window}_raw_source",
        window=args.window,
        horizons=HORIZONS,
        outcome_columns=["label_short", "label_next_close"],
        required_outcomes=OUTCOMES,
        prepare_outcome=_prepare_outcome,
    ):
        metrics = _group_metrics(work)
        metrics["label_horizon"] = horizon
        parts.append(metrics)
        print(
            f"progress window={args.window} fold={fold_index}/8 label={horizon} "
            f"common_rows={len(work)} groups={len(metrics)}",
            flush=True,
        )

    group_metrics = pd.concat(parts, ignore_index=True)
    summary = _summarize(group_metrics)
    trace = {
        "window": args.window,
        "selection": "Top100 within the common valid same-day Pool L universe",
        "event": "same-day final close at upper price limit",
        "entry_to_next_close": "entry+6s ask to next trading-day tick close",
        "close_to_next_close": "same-day tick close to next trading-day tick close",
        "aggregation": "quarter equal across 16 quarters",
        "contribution": "selected event contribution minus common-pool event contribution",
    }
    write_analysis_result(
        args.output_dir,
        group_metrics,
        summary,
        metrics_filename="limit_nextclose_group_metrics.parquet",
        summary_filename="limit_nextclose_summary.csv",
        trace=trace,
    )


if __name__ == "__main__":
    main()
