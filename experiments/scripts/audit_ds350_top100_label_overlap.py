from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import equal_weighted_period_means, write_analysis_result
from opening_strength_fit.schema import normalize_text_series as _text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "decision_target_timestamp"]
HORIZONS = ("1m", "3m", "10m", "1h", "close")
TEST_STARTS = (
    "2022-01",
    "2022-07",
    "2023-01",
    "2023-07",
    "2024-01",
    "2024-07",
    "2025-01",
    "2025-07",
)


def _load_top100(path: Path, pool: pd.DataFrame) -> dict[tuple[object, object], set[str]]:
    frame = pd.read_parquet(
        path,
        columns=["date", "symbol", "decision_target_timestamp", "prediction"],
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["symbol"] = _text(frame["symbol"])
    frame["decision_target_timestamp"] = pd.to_datetime(
        frame["decision_target_timestamp"], errors="coerce"
    )
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame = frame.loc[frame[KEYS].notna().all(axis=1) & np.isfinite(frame["prediction"])].copy()
    frame = frame.loc[stock_pool_membership_mask(frame, pool, date_lag_sessions=0)].copy()
    ordered = frame.sort_values(
        [*KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top = ordered.groupby(KEYS, sort=False).head(100)
    return {key: set(part["symbol"]) for key, part in top.groupby(KEYS, sort=False, observed=True)}


def _pair_rows(
    selections: dict[str, dict[tuple[object, object], set[str]]],
    *,
    test_start: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(HORIZONS, 2):
        common_groups = sorted(set(selections[left]) & set(selections[right]))
        for date, timestamp in common_groups:
            left_names = selections[left][(date, timestamp)]
            right_names = selections[right][(date, timestamp)]
            intersection = len(left_names & right_names)
            union = len(left_names | right_names)
            rows.append(
                {
                    "test_start": test_start,
                    "date": date,
                    "decision_target_timestamp": timestamp,
                    "left_label": left,
                    "right_label": right,
                    "left_names": len(left_names),
                    "right_names": len(right_names),
                    "intersection_names": intersection,
                    "overlap_min": intersection / min(len(left_names), len(right_names)),
                    "jaccard": intersection / union,
                }
            )
    return rows


def _summarize(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["quarter"] = pd.to_datetime(work["date"]).dt.to_period("Q").astype(str)
    metrics = ["intersection_names", "overlap_min", "jaccard"]
    group_equal = work.groupby(["left_label", "right_label"], sort=False)[metrics].mean()
    quarter_equal = equal_weighted_period_means(
        work,
        by=["left_label", "right_label"],
        period_column="quarter",
        value_columns=metrics,
    ).set_index(["left_label", "right_label"])
    quantiles = (
        work.groupby(["left_label", "right_label"], sort=False)["intersection_names"]
        .quantile([0.1, 0.5, 0.9])
        .unstack()
        .rename(columns={0.1: "intersection_p10", 0.5: "intersection_p50", 0.9: "intersection_p90"})
    )
    groups = work.groupby(["left_label", "right_label"], sort=False).size().rename("groups")
    result = quarter_equal.add_prefix("quarter_equal_").join(group_equal.add_prefix("group_equal_"))
    result = result.join(quantiles).join(groups).reset_index()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, choices=["0931_0940", "1001_1010"])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    root = Path(
        "/mnt/output/opening_strength_fit/nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"
    )
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    rows: list[dict[str, object]] = []
    for test_start in TEST_STARTS:
        selections: dict[str, dict[tuple[object, object], set[str]]] = {}
        for horizon in HORIZONS:
            path = (
                root / f"w{args.window}_h{horizon}" / f"month_{test_start}" / "predictions.parquet"
            )
            if not path.is_file():
                raise SystemExit(f"missing prediction file: {path}")
            selections[horizon] = _load_top100(path, pool)
            print(
                f"progress window={args.window} test_start={test_start} "
                f"label={horizon} groups={len(selections[horizon])}",
                flush=True,
            )
        rows.extend(_pair_rows(selections, test_start=test_start))

    daily = pd.DataFrame(rows)
    summary = _summarize(daily)
    trace = {
        "window": args.window,
        "labels": list(HORIZONS),
        "universe": "same-day Pool L (date_lag_sessions=0)",
        "selection": "Top100 by prediction within date x decision_target_timestamp",
        "overlap_min": "intersection / min(left TopN size, right TopN size)",
        "jaccard": "intersection / union",
        "aggregation": "quarter_equal is mean within each quarter, then mean over 16 quarters",
        "prediction_root": str(root),
        "groups": int(len(daily)),
    }
    write_analysis_result(
        args.output_dir,
        daily,
        summary,
        metrics_filename="top100_label_overlap_daily.csv",
        summary_filename="top100_label_overlap_summary.csv",
        trace=trace,
    )


if __name__ == "__main__":
    main()
