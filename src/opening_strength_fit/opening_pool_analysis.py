"""Shared data preparation and summaries for opening-pool research."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import newey_west_mean_ci
from opening_strength_fit.feature_utils import finite_numeric
from opening_strength_fit.labels import read_clock_return_labels
from opening_strength_fit.raw_source import read_daily_market_events
from opening_strength_fit.schema import normalize_decision_keys_preserving_rows

EVENT_LABELS = (
    ("limit_up", "涨停"),
    ("limit_down", "跌停"),
    ("limit_event", "涨跌停合计"),
)


def read_standardized_score_path(
    path: Path,
    *,
    clocks: Sequence[str],
    extra_numeric_columns: Sequence[str] = (),
) -> pd.DataFrame:
    columns = ["date", "symbol", "decision_target_timestamp", "prediction"]
    columns.extend(extra_numeric_columns)
    frame = normalize_decision_keys_preserving_rows(pd.read_parquet(path, columns=columns))
    frame["clock"] = frame["decision_target_timestamp"].dt.strftime("%H:%M")
    for column in ("prediction", *extra_numeric_columns):
        frame[column] = finite_numeric(frame[column])
    frame = frame.loc[frame["clock"].isin(clocks) & frame["prediction"].notna()].copy()
    cross = frame.groupby("decision_target_timestamp", sort=False)["prediction"]
    score_std = cross.transform(lambda values: values.std(ddof=0))
    frame["score_z"] = (
        frame["prediction"].sub(cross.transform("mean")).div(score_std.where(score_std.gt(0)))
    )
    return frame.dropna(subset=["score_z"])


def load_selected_market_outcomes(
    selected: pd.DataFrame,
    *,
    daily_root: Path,
    label_root: Path,
    years: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    member_parts = []
    market_parts = []
    full_a_parts = []
    for year in years:
        print(f"joining outcomes and close labels for {year}", flush=True)
        labels = read_clock_return_labels(
            label_root,
            year,
            clock="09:40",
            label_column="label_same_day_close",
            valid_column="valid_same_day_close",
            output_column="remainder_return_bps",
            multiplier=10_000.0,
        )
        dates = selected.loc[selected["date"].str.startswith(str(year)), "date"].unique()
        labels = labels.loc[labels["date"].isin(dates)].copy()
        market = read_daily_market_events(daily_root, year)
        market = market.loc[market["date"].isin(dates)].copy()
        keys = selected.loc[selected["date"].str.startswith(str(year))]
        members = keys.merge(market, on=["date", "symbol"], how="left", validate="one_to_one")
        members = members.merge(labels, on=["date", "symbol"], how="left", validate="one_to_one")
        member_parts.append(members)
        market_parts.append(market)
        full_a_parts.append(
            labels.groupby("date", sort=False)["remainder_return_bps"]
            .agg(full_a_valid_names="count", full_a_return_bps="mean")
            .reset_index()
        )
    return (
        pd.concat(member_parts, ignore_index=True),
        pd.concat(market_parts, ignore_index=True),
        pd.concat(full_a_parts, ignore_index=True),
    )


def market_event_summary(members: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    days = members["date"].nunique()
    rows = []
    for event, label in EVENT_LABELS:
        market_count = int(market[event].sum())
        selected_count = int(members[event].sum())
        market_density = market_count / len(market)
        selected_density = selected_count / len(members)
        rows.append(
            {
                "event": event,
                "label": label,
                "market_count": market_count,
                "selected_count": selected_count,
                "days": days,
                "market_per_day": market_count / days,
                "selected_per_day": selected_count / days,
                "coverage_pct": selected_count / market_count * 100.0,
                "market_density_pct": market_density * 100.0,
                "selected_density_pct": selected_density * 100.0,
                "density_enrichment": selected_density / market_density,
            }
        )
    return pd.DataFrame(rows)


def daily_pool_return_comparison(members: pd.DataFrame, full_a_daily: pd.DataFrame) -> pd.DataFrame:
    pool = (
        members.groupby("date", sort=False)["remainder_return_bps"]
        .agg(pool_valid_names="count", pool_return_bps="mean")
        .reset_index()
    )
    out = pool.merge(full_a_daily, on="date", validate="one_to_one")
    out["active_return_bps"] = out["pool_return_bps"].sub(out["full_a_return_bps"])
    return out


def return_series_summary(
    returns: pd.DataFrame,
    series: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for name, column in series:
        values = finite_numeric(returns[column]).dropna()
        ci_low, ci_high = newey_west_mean_ci(values)
        rows.append(
            {
                "series": name,
                "days": len(values),
                "mean_bps": values.mean(),
                "median_bps": values.median(),
                "std_bps": values.std(),
                "positive_days_pct": values.gt(0).mean() * 100.0,
                "p10_bps": values.quantile(0.10),
                "p90_bps": values.quantile(0.90),
                "nw5_ci_low_bps": ci_low,
                "nw5_ci_high_bps": ci_high,
            }
        )
    return pd.DataFrame(rows)
