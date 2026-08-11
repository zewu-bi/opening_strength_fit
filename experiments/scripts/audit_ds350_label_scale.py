from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.schema import normalize_text_series as _text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "decision_target_timestamp"]
HORIZONS = ("1m", "3m", "10m", "1h", "close")


def _group_corr(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    grouped = frame.groupby(KEYS, sort=False)
    left_centered = frame[left] - grouped[left].transform("mean")
    right_centered = frame[right] - grouped[right].transform("mean")
    covariance = (
        (left_centered * right_centered)
        .groupby([frame[column] for column in KEYS], sort=False)
        .mean()
    )
    left_scale = (
        (left_centered * left_centered)
        .groupby([frame[column] for column in KEYS], sort=False)
        .mean()
        .pow(0.5)
    )
    right_scale = (
        (right_centered * right_centered)
        .groupby([frame[column] for column in KEYS], sort=False)
        .mean()
        .pow(0.5)
    )
    return covariance / (left_scale * right_scale)


def _summarize_groups(frame: pd.DataFrame, *, include_selection: bool) -> pd.DataFrame:
    work = frame.dropna(
        subset=[*KEYS, "label_short", "label_next_close", "target_label", "prediction"]
    ).copy()
    numeric = ["label_short", "label_next_close", "target_label", "prediction"]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    finite = np.isfinite(work[numeric]).all(axis=1)
    work = work.loc[finite].copy()
    grouped = work.groupby(KEYS, sort=False)
    summary = grouped.agg(
        rows=("label_short", "size"),
        raw_mean=("label_short", "mean"),
        raw_std=("label_short", lambda values: values.std(ddof=0)),
        target_mean=("target_label", "mean"),
        target_std=("target_label", lambda values: values.std(ddof=0)),
    )
    quantiles = grouped["label_short"].quantile([0.01, 0.05, 0.50, 0.95, 0.99]).unstack()
    quantiles.columns = ["raw_p01", "raw_p05", "raw_p50", "raw_p95", "raw_p99"]
    summary = summary.join(quantiles)
    summary["short_next_pearson"] = _group_corr(work, "label_short", "label_next_close")
    summary["target_short_pearson"] = _group_corr(work, "target_label", "label_short")

    prediction_rank = grouped["prediction"].rank(method="average", pct=True)
    short_rank = grouped["label_short"].rank(method="average", pct=True)
    ranked = work.loc[:, KEYS].copy()
    ranked["prediction_rank"] = prediction_rank
    ranked["short_rank"] = short_rank
    summary["prediction_short_rank_ic"] = _group_corr(ranked, "prediction_rank", "short_rank")

    if include_selection:
        ordered_model = work.sort_values(
            [*KEYS, "prediction", "symbol"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        model_top = ordered_model.groupby(KEYS, sort=False).head(100)
        ordered_oracle = work.sort_values(
            [*KEYS, "label_short", "symbol"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        oracle_top = ordered_oracle.groupby(KEYS, sort=False).head(100)
        model_mean = model_top.groupby(KEYS, sort=False)["label_short"].mean()
        oracle_mean = oracle_top.groupby(KEYS, sort=False)["label_short"].mean()
        summary["model_excess"] = model_mean - summary["raw_mean"]
        summary["oracle_excess"] = oracle_mean - summary["raw_mean"]
        summary["model_excess_z"] = summary["model_excess"] / summary["raw_std"]
        summary["oracle_excess_z"] = summary["oracle_excess"] / summary["raw_std"]

    return summary.reset_index()


def _aggregate(groups: pd.DataFrame) -> dict[str, float | int]:
    groups = groups.copy()
    groups["quarter"] = pd.to_datetime(groups["date"]).dt.to_period("Q").astype(str)
    metric_columns = [
        column for column in groups.columns if column not in {*KEYS, "quarter", "rows"}
    ]
    group_equal = groups[metric_columns].mean()
    quarter_equal = groups.groupby("quarter", sort=False)[metric_columns].mean().mean()

    def converted(values: pd.Series) -> dict[str, float]:
        result: dict[str, float] = {}
        for column, value in values.items():
            scale = 10_000.0 if column.startswith("raw_") or column.endswith("_excess") else 1.0
            result[f"{column}_bps" if scale == 10_000.0 else column] = float(value * scale)
        return result

    return {
        "groups": int(len(groups)),
        "mean_rows": float(groups["rows"].mean()),
        "group_equal": converted(group_equal),
        "quarter_equal": converted(quarter_equal),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, choices=["0931_0940", "1001_1010"])
    parser.add_argument("--horizon", required=True, choices=HORIZONS)
    args = parser.parse_args()

    root = Path("/mnt/output/opening_strength_fit")
    case = f"w{args.window}_h{args.horizon}"
    prediction_root = root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1" / case
    files = sorted(prediction_root.glob("month_*/predictions.parquet"))
    if len(files) != 8:
        raise SystemExit(f"expected 8 prediction files for {case}, got {len(files)}")

    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    universe_parts: list[pd.DataFrame] = []
    pool_parts: list[pd.DataFrame] = []
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(
            path,
            columns=[
                "date",
                "symbol",
                "decision_target_timestamp",
                "prediction",
                "label_short",
                "label_next_close",
                "target_label",
            ],
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["symbol"] = _text(frame["symbol"])
        frame["decision_target_timestamp"] = pd.to_datetime(
            frame["decision_target_timestamp"], errors="coerce"
        )
        universe_parts.append(_summarize_groups(frame, include_selection=False))
        in_pool = stock_pool_membership_mask(frame, pool, date_lag_sessions=0)
        pool_parts.append(_summarize_groups(frame.loc[in_pool], include_selection=True))
        print(
            f"progress case={case} file={index}/{len(files)} rows={len(frame)} "
            f"pool_rows={int(in_pool.sum())}",
            flush=True,
        )

    result = {
        "case": case,
        "window": args.window,
        "horizon": args.horizon,
        "prediction_root": str(prediction_root),
        "definition": {
            "raw": "label_short (entry to requested horizon)",
            "target": "xs_zscore(label_short) + 0.3 * xs_zscore(label_next_close)",
            "selection": "Top100 by prediction within same-day Pool L",
            "oracle": "Top100 by realized label_short within same-day Pool L",
        },
        "universe": _aggregate(pd.concat(universe_parts, ignore_index=True)),
        "pool_L": _aggregate(pd.concat(pool_parts, ignore_index=True)),
    }
    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
