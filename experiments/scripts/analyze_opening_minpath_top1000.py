from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.opening_pool_analysis import (
    daily_pool_return_comparison,
    load_selected_market_outcomes,
    market_event_summary,
    read_standardized_score_path,
    return_series_summary,
)

ROOT = Path("/mnt/output/opening_strength_fit")
PREDICTION_ROOT = ROOT / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1/w0931_0940_h1m"
DAILY_ROOT = ROOT / "cache/opening_0931_0940_raw_source"
CLOSE_LABEL_ROOT = ROOT / "datasets/opening_0931_0940_labels_10m_1h_close_next"
MAX_POOL_PATH = (
    ROOT / "runs/analyses/opening_h1m_maxpath_deep_audit_v1/current_selected_members.parquet"
)
OUTPUT_DIR = ROOT / "runs/analyses/opening_h1m_minpath_top1000_v1"
CLOCKS = [f"09:{minute:02d}" for minute in range(31, 41)]


def select_prediction_shard(path: Path) -> pd.DataFrame:
    frame = read_standardized_score_path(
        path,
        clocks=CLOCKS,
        extra_numeric_columns=("label_short", "label_next_close"),
    )
    trough = (
        frame.sort_values(
            ["date", "symbol", "score_z", "clock"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
        .drop_duplicates(["date", "symbol"], keep="first")
        .rename(
            columns={
                "clock": "min_clock",
                "score_z": "min_z",
                "label_short": "min_label_short",
                "label_next_close": "min_label_next_close",
            }
        )
    )
    selected = (
        trough.sort_values(
            ["date", "min_z", "symbol"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .groupby("date", sort=False)
        .head(1_000)
        .copy()
    )
    return selected[
        [
            "date",
            "symbol",
            "min_clock",
            "min_z",
            "min_label_short",
            "min_label_next_close",
        ]
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_paths = sorted(PREDICTION_ROOT.glob("month_*/predictions.parquet"))
    if len(prediction_paths) != 8:
        raise SystemExit(f"expected 8 prediction shards, found {len(prediction_paths)}")
    selected_parts = []
    for index, path in enumerate(prediction_paths, start=1):
        print(f"selecting shard {index}/{len(prediction_paths)} {path.parent.name}", flush=True)
        selected_parts.append(select_prediction_shard(path))
    selected = pd.concat(selected_parts, ignore_index=True)
    if selected.groupby("date", sort=False).size().ne(1_000).any():
        raise SystemExit("min-path selector did not produce exactly 1000 names per day")

    members, market, full_a_daily = load_selected_market_outcomes(
        selected,
        daily_root=DAILY_ROOT,
        label_root=CLOSE_LABEL_ROOT,
        years=range(2022, 2026),
    )

    max_pool = pd.read_parquet(MAX_POOL_PATH, columns=["date", "symbol"])
    max_pool["date"] = max_pool["date"].astype(str)
    max_pool["symbol"] = max_pool["symbol"].astype(str)
    max_keys = set(zip(max_pool["date"], max_pool["symbol"], strict=True))
    members["also_in_max_path_pool"] = [
        (date, symbol) in max_keys
        for date, symbol in zip(members["date"], members["symbol"], strict=True)
    ]
    symbol = members["symbol"]
    members["board"] = np.select(
        [
            symbol.str.startswith("00"),
            symbol.str.startswith("30"),
            symbol.str.startswith("60"),
            symbol.str.startswith("68"),
        ],
        ["sz_main", "chinext", "sh_main", "star"],
        default="other",
    )
    members.to_parquet(OUTPUT_DIR / "selected_members.parquet", index=False)

    event_summary = market_event_summary(members, market)
    event_summary.to_csv(OUTPUT_DIR / "event_summary.csv", index=False)

    returns = daily_pool_return_comparison(members, full_a_daily)
    returns.to_csv(OUTPUT_DIR / "daily_returns.csv", index=False)
    return_summary = return_series_summary(
        returns,
        (
            ("min_path_top1000", "pool_return_bps"),
            ("full_a_equal_weight", "full_a_return_bps"),
            ("min_path_minus_full_a", "active_return_bps"),
        ),
    )
    return_summary.to_csv(OUTPUT_DIR / "return_summary.csv", index=False)

    returns["year"] = returns["date"].str[:4]
    return_by_year = (
        returns.groupby("year", sort=False)
        .agg(
            days=("date", "size"),
            pool_return_bps=("pool_return_bps", "mean"),
            full_a_return_bps=("full_a_return_bps", "mean"),
            active_return_bps=("active_return_bps", "mean"),
        )
        .reset_index()
    )
    return_by_year.to_csv(OUTPUT_DIR / "return_by_year.csv", index=False)

    overlap_daily = (
        members.groupby("date", sort=False)["also_in_max_path_pool"]
        .sum()
        .rename("intersection")
        .reset_index()
    )
    overlap_daily["overlap_pct"] = overlap_daily["intersection"].div(1_000).mul(100.0)
    overlap_daily.to_csv(OUTPUT_DIR / "overlap_with_max_path_daily.csv", index=False)
    adjacent_sets = {
        date: set(group["symbol"]) for date, group in members.groupby("date", sort=False)
    }
    dates = sorted(adjacent_sets)
    adjacent_intersections = [
        len(adjacent_sets[previous] & adjacent_sets[current])
        for previous, current in zip(dates[:-1], dates[1:], strict=True)
    ]
    pool_summary = pd.DataFrame(
        [
            {
                "days": len(dates),
                "selected_rows": len(members),
                "unique_symbols": members["symbol"].nunique(),
                "same_day_short_label_at_min_mean_bps": members["min_label_short"].mean()
                * 10_000.0,
                "same_day_short_label_at_min_median_bps": members["min_label_short"].median()
                * 10_000.0,
                "same_day_short_label_at_min_positive_pct": members["min_label_short"].gt(0).mean()
                * 100.0,
                "same_day_close_preclose_mean_bps": members["daily_return_bps"].mean(),
                "st_share_pct": members["st_flag"].mean() * 100.0,
                "overlap_with_max_path_mean_names": overlap_daily["intersection"].mean(),
                "overlap_with_max_path_mean_pct": overlap_daily["overlap_pct"].mean(),
                "adjacent_day_retention_mean_names": np.mean(adjacent_intersections),
                "adjacent_day_retention_mean_pct": np.mean(adjacent_intersections) / 10.0,
            }
        ]
    )
    pool_summary.to_csv(OUTPUT_DIR / "pool_summary.csv", index=False)

    board_summary = (
        members["board"].value_counts().rename_axis("board").rename("selected_rows").reset_index()
    )
    board_summary["selected_share_pct"] = (
        board_summary["selected_rows"].div(len(members)).mul(100.0)
    )
    board_summary.to_csv(OUTPUT_DIR / "board_summary.csv", index=False)
    clock_summary = (
        members["min_clock"]
        .value_counts()
        .sort_index()
        .rename_axis("min_clock")
        .rename("rows")
        .reset_index()
    )
    clock_summary["share_pct"] = clock_summary["rows"].div(len(members)).mul(100.0)
    clock_summary.to_csv(OUTPUT_DIR / "min_clock_summary.csv", index=False)
    (OUTPUT_DIR / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    print(event_summary.to_string(index=False), flush=True)
    print(return_summary.to_string(index=False), flush=True)
    print(pool_summary.to_string(index=False), flush=True)
    print(f"wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
