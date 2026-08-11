from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import newey_west_mean_ci as nw_ci
from opening_strength_fit.feature_utils import finite_numeric as finite
from opening_strength_fit.labels import read_clock_return_labels

SELECTED_PATH = Path(
    "/mnt/output/opening_strength_fit/runs/analyses/"
    "opening_h1m_maxpath_deep_audit_v1/current_selected_members.parquet"
)
LABEL_ROOT = Path(
    "/mnt/output/opening_strength_fit/datasets/opening_0931_0940_labels_10m_1h_close_next"
)
OUTPUT_DIR = Path(
    "/mnt/output/opening_strength_fit/runs/analyses/opening_h1m_top1000_remainder_day_v1"
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_parquet(
        SELECTED_PATH,
        columns=[
            "date",
            "symbol",
            "limit_up",
            "limit_down",
            "limit_event",
            "board",
            "industry",
        ],
    )
    selected["date"] = selected["date"].astype(str)
    selected["symbol"] = selected["symbol"].astype(str)
    selected_parts: list[pd.DataFrame] = []
    for year in range(2022, 2026):
        print(f"reading same-day-close labels for {year}", flush=True)
        labels = read_clock_return_labels(
            LABEL_ROOT,
            year,
            clock="09:40",
            label_column="label_same_day_close",
            valid_column="valid_same_day_close",
            output_column="remainder_return_bps",
            multiplier=10_000.0,
        )
        keys = selected.loc[selected["date"].str.startswith(str(year))]
        selected_parts.append(
            keys.merge(labels, on=["date", "symbol"], how="left", validate="one_to_one")
        )
    members = pd.concat(selected_parts, ignore_index=True)
    members.to_parquet(OUTPUT_DIR / "selected_0940_to_close.parquet", index=False)
    returns = finite(members["remainder_return_bps"])

    quantiles = returns.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    stock_summary = pd.DataFrame(
        [
            {
                "selected_rows": len(members),
                "valid_rows": int(returns.notna().sum()),
                "valid_pct": returns.notna().mean() * 100.0,
                "mean_bps": returns.mean(),
                "median_bps": returns.median(),
                "positive_pct": returns.gt(0).sum() / returns.notna().sum() * 100.0,
                "zero_pct": returns.eq(0).sum() / returns.notna().sum() * 100.0,
                "negative_pct": returns.lt(0).sum() / returns.notna().sum() * 100.0,
                "std_bps": returns.std(),
                **{f"p{int(q * 100):02d}_bps": value for q, value in quantiles.items()},
            }
        ]
    )
    stock_summary.to_csv(OUTPUT_DIR / "stock_return_summary.csv", index=False)

    threshold_rows = []
    valid_count = returns.notna().sum()
    for threshold in (10, 25, 50, 100, 200, 300, 500, 800, 1_000):
        threshold_rows.append(
            {
                "threshold_bps": threshold,
                "gain_at_least_pct": returns.ge(threshold).sum() / valid_count * 100.0,
                "loss_at_least_pct": returns.le(-threshold).sum() / valid_count * 100.0,
            }
        )
    pd.DataFrame(threshold_rows).to_csv(OUTPUT_DIR / "return_tail_thresholds.csv", index=False)

    daily = (
        members.groupby("date", sort=False)["remainder_return_bps"]
        .agg(valid_names="count", pool_mean_bps="mean", stock_median_bps="median")
        .reset_index()
    )
    ci_low, ci_high = nw_ci(daily["pool_mean_bps"])
    daily_summary = pd.DataFrame(
        [
            {
                "days": len(daily),
                "valid_names_mean": daily["valid_names"].mean(),
                "daily_pool_mean_bps": daily["pool_mean_bps"].mean(),
                "daily_pool_median_bps": daily["pool_mean_bps"].median(),
                "positive_days_pct": daily["pool_mean_bps"].gt(0).mean() * 100.0,
                "daily_pool_std_bps": daily["pool_mean_bps"].std(),
                "daily_pool_p05_bps": daily["pool_mean_bps"].quantile(0.05),
                "daily_pool_p10_bps": daily["pool_mean_bps"].quantile(0.10),
                "daily_pool_p25_bps": daily["pool_mean_bps"].quantile(0.25),
                "daily_pool_p75_bps": daily["pool_mean_bps"].quantile(0.75),
                "daily_pool_p90_bps": daily["pool_mean_bps"].quantile(0.90),
                "daily_pool_p95_bps": daily["pool_mean_bps"].quantile(0.95),
                "nw5_mean_ci_low_bps": ci_low,
                "nw5_mean_ci_high_bps": ci_high,
            }
        ]
    )
    daily.to_csv(OUTPUT_DIR / "daily_pool_returns.csv", index=False)
    daily_summary.to_csv(OUTPUT_DIR / "daily_pool_return_summary.csv", index=False)

    working = members.assign(
        year=members["date"].str[:4],
        month=members["date"].str[:7],
    )
    by_year = (
        working.groupby("year", sort=False)["remainder_return_bps"]
        .agg(valid_rows="count", mean_bps="mean", median_bps="median")
        .reset_index()
    )
    by_year_positive = working.groupby("year", sort=False)["remainder_return_bps"].apply(
        lambda values: values.gt(0).sum() / values.notna().sum() * 100.0
    )
    by_year["positive_pct"] = by_year["year"].map(by_year_positive)
    daily_with_period = daily.assign(year=daily["date"].str[:4], month=daily["date"].str[:7])
    year_daily = (
        daily_with_period.groupby("year", sort=False)["pool_mean_bps"]
        .agg(daily_mean_bps="mean", daily_median_bps="median", daily_std_bps="std")
        .reset_index()
    )
    year_daily["positive_days_pct"] = (
        daily_with_period.assign(positive=daily_with_period["pool_mean_bps"].gt(0))
        .groupby("year", sort=False)["positive"]
        .mean()
        .mul(100.0)
        .to_numpy()
    )
    by_year = by_year.merge(year_daily, on="year", validate="one_to_one")
    by_year.to_csv(OUTPUT_DIR / "return_by_year.csv", index=False)

    by_month = (
        daily_with_period.groupby("month", sort=False)["pool_mean_bps"]
        .agg(days="size", mean_bps="mean", median_bps="median", std_bps="std")
        .reset_index()
    )
    by_month["positive_days_pct"] = (
        daily_with_period.assign(positive=daily_with_period["pool_mean_bps"].gt(0))
        .groupby("month", sort=False)["positive"]
        .mean()
        .mul(100.0)
        .to_numpy()
    )
    by_month.to_csv(OUTPUT_DIR / "return_by_month.csv", index=False)

    outcome_parts = []
    for outcome, mask in (
        ("final_limit_up", members["limit_up"]),
        ("final_limit_down", members["limit_down"]),
        ("final_non_limit", ~members["limit_event"]),
    ):
        part = members.loc[mask]
        part_return = finite(part["remainder_return_bps"])
        outcome_parts.append(
            {
                "outcome": outcome,
                "rows": len(part),
                "pool_share_pct": len(part) / len(members) * 100.0,
                "valid_rows": int(part_return.notna().sum()),
                "mean_bps": part_return.mean(),
                "median_bps": part_return.median(),
                "positive_pct": part_return.gt(0).sum() / part_return.notna().sum() * 100.0,
                "p10_bps": part_return.quantile(0.10),
                "p90_bps": part_return.quantile(0.90),
                "pool_mean_contribution_bps": part_return.sum() / returns.notna().sum(),
            }
        )
    pd.DataFrame(outcome_parts).to_csv(OUTPUT_DIR / "return_by_final_outcome.csv", index=False)

    for group_column, filename in (
        ("board", "return_by_board.csv"),
        ("industry", "return_by_industry.csv"),
    ):
        grouped = (
            working.assign(**{group_column: working[group_column].fillna("missing")})
            .groupby(group_column, sort=False)["remainder_return_bps"]
            .agg(rows="size", valid_rows="count", mean_bps="mean", median_bps="median")
            .reset_index()
        )
        positive = (
            working.assign(**{group_column: working[group_column].fillna("missing")})
            .groupby(group_column, sort=False)["remainder_return_bps"]
            .apply(lambda values: values.gt(0).sum() / values.notna().sum() * 100.0)
        )
        grouped["positive_pct"] = grouped[group_column].map(positive)
        grouped["pool_share_pct"] = grouped["rows"].div(len(working)).mul(100.0)
        grouped.to_csv(OUTPUT_DIR / filename, index=False)

    cost_rows = []
    gross = daily["pool_mean_bps"]
    for total_cost_bps in (0, 5, 10, 20):
        net = gross - total_cost_bps
        cost_rows.append(
            {
                "total_round_trip_cost_bps": total_cost_bps,
                "net_mean_bps": net.mean(),
                "net_median_bps": net.median(),
                "positive_days_pct": net.gt(0).mean() * 100.0,
            }
        )
    pd.DataFrame(cost_rows).to_csv(OUTPUT_DIR / "cost_sensitivity.csv", index=False)
    (OUTPUT_DIR / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    print(f"wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
