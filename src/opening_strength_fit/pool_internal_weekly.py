from __future__ import annotations

import pandas as pd

from opening_strength_fit.pool_internal_plots import PLOT_POOLS

POOL_CHOICES = ("universe", "S", "M", "L", "pool_S", "pool_M", "pool_L")
EXCESS_COLUMNS = ("short_internal_excess_bps", "next_internal_excess_bps")
ROLLING_COLUMNS = (
    "pool_short_mean_bps",
    "selected_short_mean_bps",
    "short_internal_excess_bps",
    "pool_next_mean_bps",
    "selected_next_mean_bps",
    "next_internal_excess_bps",
    "short_rank_ic",
    "next_rank_ic",
)
REQUIRED_GROUP_COLUMNS = {
    "pool",
    "test_month",
    "date",
    "clock",
    "candidate_rows",
    "selected_rows",
    *ROLLING_COLUMNS,
}


def normalize_pool(value: str) -> str:
    if value == "universe" or value.startswith("pool_"):
        return value
    return f"pool_{value}"


def normalize_pools(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return PLOT_POOLS
    pools = tuple(normalize_pool(value) for value in values)
    unknown = sorted(set(pools) - set(PLOT_POOLS))
    if unknown:
        raise ValueError(f"unknown pools: {unknown}")
    return pools


def build_weekly_pool_internal_summaries(
    group_metrics: pd.DataFrame,
    *,
    pools: tuple[str, ...] = PLOT_POOLS,
    rolling_weeks: int = 4,
    min_rolling_weeks: int | None = None,
    top_worst: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rolling_weeks <= 0:
        raise ValueError("rolling_weeks must be positive")
    min_rolling_weeks = rolling_weeks if min_rolling_weeks is None else min_rolling_weeks
    if min_rolling_weeks <= 0 or min_rolling_weeks > rolling_weeks:
        raise ValueError("min_rolling_weeks must be in [1, rolling_weeks]")

    daily = build_daily_summary(group_metrics, pools=pools)
    weekly = build_weekly_summary(daily)
    weekly = add_trading_day_equal_rolling(
        weekly,
        rolling_weeks=rolling_weeks,
        min_rolling_weeks=min_rolling_weeks,
    )
    overall = build_overall_summary(weekly, rolling_weeks=rolling_weeks)
    worst = build_worst_windows(weekly, rolling_weeks=rolling_weeks, top_worst=top_worst)
    return daily, weekly, overall, worst


def build_daily_summary(group_metrics: pd.DataFrame, *, pools: tuple[str, ...]) -> pd.DataFrame:
    missing = sorted(REQUIRED_GROUP_COLUMNS - set(group_metrics.columns))
    if missing:
        raise ValueError(f"group_metrics missing columns: {missing}")

    frame = group_metrics.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame = frame.loc[frame["pool"].isin(pools)].copy()
    if frame.empty:
        raise ValueError("group_metrics has no rows for selected pools")
    pool_order = {pool: index for index, pool in enumerate(pools)}
    frame["_pool_order"] = frame["pool"].map(pool_order)
    frame = frame.sort_values(["_pool_order", "date"])

    daily = (
        frame.groupby(["pool", "date"], sort=False)
        .agg(
            test_month=("test_month", "first"),
            decision_groups=("short_internal_excess_bps", "size"),
            clocks=("clock", "nunique"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            pool_short_mean_bps=("pool_short_mean_bps", "mean"),
            selected_short_mean_bps=("selected_short_mean_bps", "mean"),
            short_internal_excess_bps=("short_internal_excess_bps", "mean"),
            pool_next_mean_bps=("pool_next_mean_bps", "mean"),
            selected_next_mean_bps=("selected_next_mean_bps", "mean"),
            next_internal_excess_bps=("next_internal_excess_bps", "mean"),
            short_rank_ic=("short_rank_ic", "mean"),
            next_rank_ic=("next_rank_ic", "mean"),
        )
        .reset_index()
    )
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="D")
    daily["week_label"] = daily["week_start"].dt.strftime("%Y-%m-%d")
    return daily


def build_weekly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        daily.groupby(["pool", "week_start"], sort=False)
        .agg(
            week_start_date=("date", "min"),
            week_end_date=("date", "max"),
            trading_days=("date", "nunique"),
            decision_groups=("decision_groups", "sum"),
            mean_daily_clocks=("clocks", "mean"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            pool_short_mean_bps=("pool_short_mean_bps", "mean"),
            selected_short_mean_bps=("selected_short_mean_bps", "mean"),
            short_internal_excess_bps=("short_internal_excess_bps", "mean"),
            pool_next_mean_bps=("pool_next_mean_bps", "mean"),
            selected_next_mean_bps=("selected_next_mean_bps", "mean"),
            next_internal_excess_bps=("next_internal_excess_bps", "mean"),
            short_rank_ic=("short_rank_ic", "mean"),
            next_rank_ic=("next_rank_ic", "mean"),
        )
        .reset_index()
    )
    weekly["week_label"] = weekly["week_start"].dt.strftime("%Y-%m-%d")
    for column in EXCESS_COLUMNS:
        horizon = column.split("_", maxsplit=1)[0]
        weekly[f"{horizon}_positive_week"] = weekly[column] > 0
    return weekly


def add_trading_day_equal_rolling(
    weekly: pd.DataFrame,
    *,
    rolling_weeks: int,
    min_rolling_weeks: int,
) -> pd.DataFrame:
    records = []
    start_col = f"rolling_{rolling_weeks}w_start_week"
    weeks_col = f"rolling_{rolling_weeks}w_weeks"
    days_col = f"rolling_{rolling_weeks}w_trading_days"
    for _, pool_frame in weekly.groupby("pool", sort=False):
        item = pool_frame.sort_values("week_start").copy().reset_index(drop=True)
        start_weeks: list[object] = []
        week_counts: list[int | None] = []
        day_counts: list[int | None] = []
        rolling_values = {f"{column}_rolling_{rolling_weeks}w": [] for column in ROLLING_COLUMNS}

        for index in range(len(item)):
            start = max(0, index - rolling_weeks + 1)
            window = item.iloc[start : index + 1]
            enough = len(window) >= min_rolling_weeks
            start_weeks.append(window["week_start"].iloc[0] if enough else pd.NaT)
            week_counts.append(int(len(window)) if enough else None)
            day_counts.append(int(window["trading_days"].sum()) if enough else None)
            for column in ROLLING_COLUMNS:
                out_col = f"{column}_rolling_{rolling_weeks}w"
                value = (
                    _weighted_mean(window[column], window["trading_days"])
                    if enough
                    else float("nan")
                )
                rolling_values[out_col].append(value)

        item[start_col] = start_weeks
        item[weeks_col] = week_counts
        item[days_col] = day_counts
        for column, values in rolling_values.items():
            item[column] = values
        for column in EXCESS_COLUMNS:
            horizon = column.split("_", maxsplit=1)[0]
            rolling_col = f"{column}_rolling_{rolling_weeks}w"
            item[f"{horizon}_rolling_{rolling_weeks}w_positive"] = item[rolling_col] > 0
        records.append(item)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def build_overall_summary(weekly: pd.DataFrame, *, rolling_weeks: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for pool, item in weekly.groupby("pool", sort=False):
        record: dict[str, object] = {
            "pool": pool,
            "weeks": int(len(item)),
            "trading_days": int(item["trading_days"].sum()),
            "weighting": "trading_day_equal",
        }
        for column in EXCESS_COLUMNS:
            horizon = column.split("_", maxsplit=1)[0]
            rolling_col = f"{column}_rolling_{rolling_weeks}w"
            rolling_values = pd.to_numeric(item[rolling_col], errors="coerce").dropna()
            record[column] = _weighted_mean(item[column], item["trading_days"])
            record[f"{horizon}_positive_weeks"] = int((item[column] > 0).sum())
            record[f"{horizon}_positive_week_ratio"] = float((item[column] > 0).mean())
            record[f"{horizon}_worst_week_bps"] = float(item[column].min())
            record[f"{horizon}_rolling_{rolling_weeks}w_windows"] = int(len(rolling_values))
            record[f"{horizon}_rolling_{rolling_weeks}w_positive_windows"] = int(
                (rolling_values > 0).sum()
            )
            record[f"{horizon}_rolling_{rolling_weeks}w_positive_ratio"] = (
                float((rolling_values > 0).mean()) if len(rolling_values) else float("nan")
            )
            record[f"{horizon}_rolling_{rolling_weeks}w_worst_bps"] = (
                float(rolling_values.min()) if len(rolling_values) else float("nan")
            )
        records.append(record)
    return pd.DataFrame(records)


def build_worst_windows(
    weekly: pd.DataFrame,
    *,
    rolling_weeks: int,
    top_worst: int,
) -> pd.DataFrame:
    if top_worst <= 0:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for pool, item in weekly.groupby("pool", sort=False):
        item = item.sort_values("week_start").reset_index(drop=True)
        for column in EXCESS_COLUMNS:
            horizon = column.split("_", maxsplit=1)[0]
            single = item.dropna(subset=[column]).nsmallest(top_worst, column)
            for rank, row in enumerate(single.itertuples(index=False), start=1):
                records.append(
                    {
                        "pool": pool,
                        "horizon": horizon,
                        "window_type": "single_week",
                        "rank": rank,
                        "start_week": _date_text(row.week_start),
                        "end_week": _date_text(row.week_start),
                        "start_date": _date_text(row.week_start_date),
                        "end_date": _date_text(row.week_end_date),
                        "weeks": 1,
                        "trading_days": int(row.trading_days),
                        "value_bps": float(getattr(row, column)),
                        "weekly_values_bps": _window_values_text(item, column, row.week_start),
                    }
                )

            rolling_col = f"{column}_rolling_{rolling_weeks}w"
            start_col = f"rolling_{rolling_weeks}w_start_week"
            days_col = f"rolling_{rolling_weeks}w_trading_days"
            weeks_col = f"rolling_{rolling_weeks}w_weeks"
            rolling = item.dropna(subset=[rolling_col]).nsmallest(top_worst, rolling_col)
            for rank, row in enumerate(rolling.itertuples(index=False), start=1):
                start_week = getattr(row, start_col)
                end_week = row.week_start
                records.append(
                    {
                        "pool": pool,
                        "horizon": horizon,
                        "window_type": f"{rolling_weeks}w_rolling",
                        "rank": rank,
                        "start_week": _date_text(start_week),
                        "end_week": _date_text(end_week),
                        "start_date": _date_text(
                            item.loc[item["week_start"].eq(start_week), "week_start_date"].min()
                        ),
                        "end_date": _date_text(row.week_end_date),
                        "weeks": int(getattr(row, weeks_col)),
                        "trading_days": int(getattr(row, days_col)),
                        "value_bps": float(getattr(row, rolling_col)),
                        "weekly_values_bps": _window_values_text(
                            item,
                            column,
                            start_week,
                            end_week=end_week,
                        ),
                    }
                )
    return pd.DataFrame(records)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_series = pd.to_numeric(values, errors="coerce")
    weight_series = pd.to_numeric(weights, errors="coerce")
    mask = value_series.notna() & weight_series.notna() & weight_series.gt(0)
    if not bool(mask.any()):
        return float("nan")
    valid_values = value_series.loc[mask]
    valid_weights = weight_series.loc[mask]
    return float((valid_values * valid_weights).sum() / valid_weights.sum())


def _window_values_text(
    weekly: pd.DataFrame,
    column: str,
    start_week: object,
    *,
    end_week: object | None = None,
) -> str:
    start = pd.to_datetime(start_week, errors="coerce")
    end = start if end_week is None else pd.to_datetime(end_week, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return ""
    item = weekly.loc[weekly["week_start"].between(start, end)]
    return "; ".join(
        f"{row.week_start:%Y-%m-%d}:{float(getattr(row, column)):+.2f}"
        for row in item.itertuples(index=False)
    )


def _date_text(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d")

