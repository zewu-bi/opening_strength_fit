from __future__ import annotations

from pathlib import Path

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
CLOCKS = [f"09:{minute:02d}" for minute in range(31, 41)]

OUTPUT_DIR = ROOT / "runs/analyses/opening_h1m_meanabs_top1000_v1"


def select_prediction_shard(path: Path) -> pd.DataFrame:
    frame = read_standardized_score_path(path, clocks=CLOCKS)
    frame["abs_z"] = frame["score_z"].abs()
    path = (
        frame.groupby(["date", "symbol"], sort=False)["abs_z"]
        .mean()
        .rename("mean_abs_z")
        .reset_index()
    )
    return (
        path.sort_values(
            ["date", "mean_abs_z", "symbol"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("date", sort=False)
        .head(1_000)
        .copy()
    )


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
        raise SystemExit("mean-abs selector did not produce exactly 1000 names per day")

    members, market, full_a = load_selected_market_outcomes(
        selected,
        daily_root=DAILY_ROOT,
        label_root=CLOSE_LABEL_ROOT,
        years=range(2022, 2026),
    )
    members.to_parquet(OUTPUT_DIR / "selected_members.parquet", index=False)

    event_summary = market_event_summary(members, market)
    event_summary.to_csv(OUTPUT_DIR / "event_summary.csv", index=False)

    returns = daily_pool_return_comparison(members, full_a)
    returns.to_csv(OUTPUT_DIR / "daily_returns.csv", index=False)
    return_summary = return_series_summary(
        returns,
        (
            ("mean_abs_top1000", "pool_return_bps"),
            ("full_a_equal_weight", "full_a_return_bps"),
            ("mean_abs_minus_full_a", "active_return_bps"),
        ),
    )
    return_summary.to_csv(OUTPUT_DIR / "return_summary.csv", index=False)
    (OUTPUT_DIR / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    print(event_summary.to_string(index=False), flush=True)
    print(return_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
