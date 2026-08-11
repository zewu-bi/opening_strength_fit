from __future__ import annotations

import pandas as pd

from opening_strength_fit.opening_pool_analysis import (
    daily_pool_return_comparison,
    market_event_summary,
    return_series_summary,
)


def test_opening_pool_summaries_use_observed_days_and_shared_return_schema() -> None:
    members = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "limit_up": [True, False, False, False],
            "limit_down": [False, False, True, False],
            "limit_event": [True, False, True, False],
            "remainder_return_bps": [10.0, 20.0, -10.0, 0.0],
        }
    )
    market = pd.DataFrame(
        {
            "limit_up": [True, False, False, False, False, False],
            "limit_down": [False, False, True, False, False, False],
            "limit_event": [True, False, True, False, False, False],
        }
    )
    events = market_event_summary(members, market)
    assert events["days"].tolist() == [2, 2, 2]
    assert events.set_index("event").loc["limit_event", "coverage_pct"] == 100.0

    full_a = pd.DataFrame({"date": ["2025-01-02", "2025-01-03"], "full_a_return_bps": [5.0, -2.0]})
    returns = daily_pool_return_comparison(members, full_a)
    summary = return_series_summary(returns, (("active", "active_return_bps"),))
    assert returns["active_return_bps"].tolist() == [10.0, -3.0]
    assert {"std_bps", "p10_bps", "p90_bps"}.issubset(summary.columns)
