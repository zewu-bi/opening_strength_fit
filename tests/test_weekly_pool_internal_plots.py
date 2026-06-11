from __future__ import annotations

import pandas as pd
import pytest

from opening_strength_fit.pool_internal_weekly import (
    build_weekly_pool_internal_summaries,
    normalize_pools,
)


def _group_metric_row(date: str, *, short_bps: float, next_bps: float = 0.0) -> dict[str, object]:
    return {
        "pool": "universe",
        "test_month": date[:7],
        "date": date,
        "clock": "09:35",
        "candidate_rows": 5000,
        "selected_rows": 100,
        "pool_short_mean_bps": 0.0,
        "selected_short_mean_bps": short_bps,
        "short_internal_excess_bps": short_bps,
        "pool_next_mean_bps": 0.0,
        "selected_next_mean_bps": next_bps,
        "next_internal_excess_bps": next_bps,
        "short_rank_ic": 0.1,
        "next_rank_ic": 0.0,
    }


def test_rolling_windows_are_trading_day_equal() -> None:
    rows = [
        _group_metric_row("2024-01-02", short_bps=10.0),
        _group_metric_row("2024-01-03", short_bps=10.0),
        _group_metric_row("2024-01-04", short_bps=10.0),
        _group_metric_row("2024-01-05", short_bps=10.0),
        _group_metric_row("2024-01-06", short_bps=10.0),
        _group_metric_row("2024-01-08", short_bps=100.0),
    ]

    _, weekly, overall, worst = build_weekly_pool_internal_summaries(
        pd.DataFrame(rows),
        pools=("universe",),
        rolling_weeks=2,
        top_worst=1,
    )

    week_two = weekly.loc[weekly["week_start"].eq(pd.Timestamp("2024-01-08"))].iloc[0]
    assert week_two["trading_days"] == 1
    assert week_two["short_internal_excess_bps_rolling_2w"] == pytest.approx(25.0)
    assert week_two["rolling_2w_trading_days"] == 6

    assert overall["short_internal_excess_bps"].item() == pytest.approx(25.0)
    assert worst.loc[
        worst["window_type"].eq("2w_rolling") & worst["horizon"].eq("short"),
        "value_bps",
    ].item() == pytest.approx(25.0)


def test_normalize_pools_accepts_short_names() -> None:
    assert normalize_pools(["universe", "S", "pool_M"]) == ("universe", "pool_S", "pool_M")
