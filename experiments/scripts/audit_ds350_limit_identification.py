from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import equal_weighted_period_means, write_analysis_result
from opening_strength_fit.ds350_holdout_analysis import iter_common_horizon_predictions
from opening_strength_fit.schema import normalize_decision_keys_preserving_rows

KEYS = ["date", "decision_target_timestamp"]
HORIZONS = ("1m", "3m", "10m", "1h", "close")
TOP_N = 100


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_decision_keys_preserving_rows(frame)
    for column in ("prediction", "label"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _group_metrics(work: pd.DataFrame) -> pd.DataFrame:
    ordered = work.sort_values(
        [*KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    grouped = ordered.groupby(KEYS, sort=False)
    ordered["score_rank_ascending"] = grouped["prediction"].rank(method="average", ascending=True)
    top = ordered.groupby(KEYS, sort=False).head(TOP_N)

    base = grouped.agg(
        candidate_rows=("prediction", "size"),
        candidate_limit_rows=("daily_closes_up_limit", "sum"),
        pool_close_mean=("close_label", "mean"),
    )
    positive_rank = (
        ordered.loc[ordered["daily_closes_up_limit"]]
        .groupby(KEYS, sort=False)["score_rank_ascending"]
        .sum()
        .rename("positive_rank_sum")
    )
    selected = top.groupby(KEYS, sort=False).agg(
        selected_rows=("prediction", "size"),
        selected_limit_rows=("daily_closes_up_limit", "sum"),
        selected_close_mean=("close_label", "mean"),
    )
    selected_limit = (
        top.loc[top["daily_closes_up_limit"]]
        .groupby(KEYS, sort=False)
        .agg(
            selected_limit_close_sum=("close_label", "sum"),
            selected_limit_close_count=("close_label", "count"),
        )
    )
    out = (
        base.join(positive_rank, how="left")
        .join(selected, how="left")
        .join(selected_limit, how="left")
        .reset_index()
    )
    count_columns = [
        column for column in out if column.endswith("_rows") or column.endswith("_count")
    ]
    sum_columns = [column for column in out if column.endswith("_sum")]
    out[count_columns] = out[count_columns].fillna(0)
    out[sum_columns] = out[sum_columns].fillna(0.0)

    positive = out["candidate_limit_rows"].astype(float)
    negative = out["candidate_rows"].astype(float) - positive
    auc_numerator = out["positive_rank_sum"] - positive * (positive + 1.0) / 2.0
    out["limit_auc"] = auc_numerator / (positive * negative).replace(0, np.nan)
    out["candidate_limit_share_pct"] = positive / out["candidate_rows"] * 100.0
    out["top100_limit_precision_pct"] = out["selected_limit_rows"] / out["selected_rows"] * 100.0
    out["top100_limit_recall_pct"] = (
        out["selected_limit_rows"] / positive.replace(0, np.nan) * 100.0
    )
    out["top100_limit_lift_x"] = (
        out["top100_limit_precision_pct"] / out["candidate_limit_share_pct"]
    )
    out["top100_close_excess_bps"] = (
        out["selected_close_mean"] - out["pool_close_mean"]
    ) * 10_000.0
    out["selected_limit_close_mean_bps"] = (
        out["selected_limit_close_sum"]
        / out["selected_limit_close_count"].replace(0, np.nan)
        * 10_000.0
    )
    out["selected_limit_close_contribution_bps"] = (
        out["selected_limit_close_sum"] / out["selected_rows"] * 10_000.0
    )
    out["quarter"] = pd.to_datetime(out["date"]).dt.to_period("Q").astype(str)
    return out


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_limit_share_pct",
        "limit_auc",
        "top100_limit_precision_pct",
        "top100_limit_recall_pct",
        "top100_limit_lift_x",
        "top100_close_excess_bps",
        "selected_limit_close_mean_bps",
        "selected_limit_close_contribution_bps",
    ]
    return equal_weighted_period_means(
        metrics,
        by=["label_horizon"],
        period_column="quarter",
        value_columns=columns,
        count_name="groups",
    )


def _prepare_outcome(frame: pd.DataFrame) -> pd.DataFrame:
    frame["close_label"] = pd.to_numeric(frame.pop("label"), errors="coerce")
    return frame


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
        outcome_columns=["label"],
        required_outcomes=["close_label"],
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
        "labels": list(HORIZONS),
        "event": "same-day final close at upper price limit",
        "fixed_return": "entry+6s ask to same-day close",
        "comparison_universe": "intersection of valid same-day Pool L prediction keys across all five labels",
        "aggregation": "quarter equal across 16 quarters",
        "auc": "within-date-clock ROC AUC for model score vs final close-limit indicator",
    }
    write_analysis_result(
        args.output_dir,
        group_metrics,
        summary,
        metrics_filename="limit_identification_group_metrics.parquet",
        summary_filename="limit_identification_summary.csv",
        trace=trace,
    )


if __name__ == "__main__":
    main()
