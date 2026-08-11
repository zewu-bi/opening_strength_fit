from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from opening_strength_fit.schema import normalize_date_series as normalize_date
from opening_strength_fit.schema import normalize_text_series as text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "decision_target_timestamp"]


def reference(root: Path, window: str, year: int) -> pd.DataFrame:
    raw = root / f"cache/opening_{window}_raw_source/year={year}"
    frame = pd.read_parquet(
        raw / "daily_reference.parquet",
        columns=["TradingDay", "Symbol", "ClosePrice", "UpdownLimitStatus"],
    ).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "ClosePrice": "daily_close",
            "UpdownLimitStatus": "updown_limit_status",
        }
    )
    frame["date"] = normalize_date(frame["date"])
    frame["symbol"] = text(frame["symbol"])
    frame = frame.loc[frame["date"].str.startswith(str(year), na=False)].copy()
    frame["daily_closes_up_limit"] = pd.to_numeric(
        frame["updown_limit_status"], errors="coerce"
    ).eq(1)
    close = pd.read_parquet(
        raw / "close_reference.parquet",
        columns=["TradingDay", "Symbol", "ClosePrice"],
    ).rename(columns={"TradingDay": "date", "Symbol": "symbol", "ClosePrice": "tick_close"})
    close["date"] = normalize_date(close["date"])
    close["symbol"] = text(close["symbol"])
    close = close.loc[close["date"].str.startswith(str(year), na=False)]
    out = frame[["date", "symbol", "daily_close", "daily_closes_up_limit"]].merge(
        close[["date", "symbol", "tick_close"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    out["daily_close"] = pd.to_numeric(out["daily_close"], errors="coerce")
    out["tick_close"] = pd.to_numeric(out["tick_close"], errors="coerce")
    return out.drop_duplicates(["date", "symbol"], keep="last")


def summarize(work: pd.DataFrame, lag: int) -> pd.DataFrame:
    ordered = work.sort_values(
        [*KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top = ordered.groupby(KEYS, sort=False).head(100)
    filtered = ordered.loc[~ordered["daily_closes_up_limit"]]
    reselected = filtered.groupby(KEYS, sort=False).head(100)

    base = ordered.groupby(KEYS, sort=False).agg(
        candidate_rows=("label", "size"),
        pool_mean=("label", "mean"),
        pool_official_close_mean=("label_official_close", "mean"),
        candidate_limit_rows=("daily_closes_up_limit", "sum"),
    )
    top_summary = top.groupby(KEYS, sort=False).agg(
        selected_rows=("label", "size"),
        selected_mean=("label", "mean"),
        selected_official_close_mean=("label_official_close", "mean"),
        selected_limit_rows=("daily_closes_up_limit", "sum"),
    )
    top_limit = (
        top.loc[top["daily_closes_up_limit"]]
        .groupby(KEYS, sort=False)
        .agg(selected_limit_sum=("label", "sum"), selected_limit_mean=("label", "mean"))
    )
    top_gt5 = (
        top.loc[top["label"].gt(0.05)]
        .groupby(KEYS, sort=False)
        .agg(selected_gt5_rows=("label", "size"), selected_gt5_sum=("label", "sum"))
    )
    filtered_summary = filtered.groupby(KEYS, sort=False).agg(filtered_pool_mean=("label", "mean"))
    reselected_summary = reselected.groupby(KEYS, sort=False).agg(
        reselected_rows=("label", "size"), reselected_mean=("label", "mean")
    )
    out = (
        base.join(top_summary, how="left")
        .join(top_limit, how="left")
        .join(top_gt5, how="left")
        .join(filtered_summary, how="left")
        .join(reselected_summary, how="left")
        .reset_index()
    )
    count_columns = [column for column in out if column.endswith("_rows")]
    out[count_columns] = out[count_columns].fillna(0)
    sum_columns = [column for column in out if column.endswith("_sum")]
    out[sum_columns] = out[sum_columns].fillna(0.0)
    out["pool_lag"] = lag
    out["quarter"] = out["date"].map(lambda value: str(pd.Timestamp(value).to_period("Q")))
    out["pool_mean_bps"] = out["pool_mean"] * 10_000.0
    out["selected_mean_bps"] = out["selected_mean"] * 10_000.0
    out["excess_bps"] = (out["selected_mean"] - out["pool_mean"]) * 10_000.0
    out["official_close_pool_mean_bps"] = out["pool_official_close_mean"] * 10_000.0
    out["official_close_selected_mean_bps"] = out["selected_official_close_mean"] * 10_000.0
    out["official_close_excess_bps"] = (
        out["selected_official_close_mean"] - out["pool_official_close_mean"]
    ) * 10_000.0
    out["tick_vs_official_excess_delta_bps"] = out["excess_bps"] - out["official_close_excess_bps"]
    out["candidate_limit_share_pct"] = out["candidate_limit_rows"] / out["candidate_rows"] * 100.0
    out["selected_limit_share_pct"] = out["selected_limit_rows"] / out["selected_rows"] * 100.0
    out["selected_limit_mean_bps"] = out["selected_limit_mean"] * 10_000.0
    out["selected_limit_contribution_bps"] = (
        out["selected_limit_sum"] / out["selected_rows"] * 10_000.0
    )
    out["selected_gt5_share_pct"] = out["selected_gt5_rows"] / out["selected_rows"] * 100.0
    out["selected_gt5_contribution_bps"] = out["selected_gt5_sum"] / out["selected_rows"] * 10_000.0
    out["reselected_no_limit_mean_bps"] = out["reselected_mean"] * 10_000.0
    out["reselected_no_limit_excess_vs_original_pool_bps"] = (
        out["reselected_mean"] - out["pool_mean"]
    ) * 10_000.0
    out["reselected_no_limit_excess_vs_filtered_pool_bps"] = (
        out["reselected_mean"] - out["filtered_pool_mean"]
    ) * 10_000.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, choices=["0931_0940", "1001_1010"])
    args = parser.parse_args()
    root = Path("/mnt/output/opening_strength_fit")
    predictions = (
        root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1" / f"w{args.window}_hclose"
    )
    files = sorted(predictions.glob("month_*/predictions.parquet"))
    if len(files) != 8:
        raise SystemExit(f"expected 8 prediction files, got {len(files)}")
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    refs: dict[int, pd.DataFrame] = {}
    parts: dict[int, list[pd.DataFrame]] = {0: [], 1: []}
    for path in files:
        frame = pd.read_parquet(
            path,
            columns=["date", "symbol", "decision_target_timestamp", "prediction", "label"],
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["symbol"] = text(frame["symbol"])
        frame["decision_target_timestamp"] = pd.to_datetime(
            frame["decision_target_timestamp"], errors="coerce"
        )
        year = int(frame["date"].iloc[0][:4])
        refs.setdefault(year, reference(root, args.window, year))
        for lag in (0, 1):
            work = frame.loc[stock_pool_membership_mask(frame, pool, date_lag_sessions=lag)].merge(
                refs[year], on=["date", "symbol"], how="left", validate="many_to_one"
            )
            work["daily_closes_up_limit"] = work["daily_closes_up_limit"].fillna(False).astype(bool)
            ratio = pd.to_numeric(work["daily_close"], errors="coerce") / pd.to_numeric(
                work["tick_close"], errors="coerce"
            )
            work["label_official_close"] = (1.0 + work["label"]) * ratio - 1.0
            parts[lag].append(summarize(work, lag))

    result: dict[str, object] = {"window": args.window, "root": str(predictions)}
    metric_columns = [
        "pool_mean_bps",
        "selected_mean_bps",
        "excess_bps",
        "official_close_pool_mean_bps",
        "official_close_selected_mean_bps",
        "official_close_excess_bps",
        "tick_vs_official_excess_delta_bps",
        "candidate_limit_share_pct",
        "selected_limit_share_pct",
        "selected_limit_mean_bps",
        "selected_limit_contribution_bps",
        "selected_gt5_share_pct",
        "selected_gt5_contribution_bps",
        "reselected_no_limit_mean_bps",
        "reselected_no_limit_excess_vs_original_pool_bps",
        "reselected_no_limit_excess_vs_filtered_pool_bps",
    ]
    for lag in (0, 1):
        metrics = pd.concat(parts[lag], ignore_index=True)
        quarter = metrics.groupby("quarter", as_index=False)[metric_columns].mean()
        result[f"lag_{lag}_quarter_equal"] = {
            column: float(quarter[column].mean()) for column in metric_columns
        }
        result[f"lag_{lag}_groups"] = int(len(metrics))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
