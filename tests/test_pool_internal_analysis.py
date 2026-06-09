from __future__ import annotations

import pandas as pd
import pytest

from opening_strength_fit.commands.pool_internal_analysis import (
    halfyear_summary,
    prediction_files,
    year_summary,
)


def _month_row(month: str, short_bps: float, next_bps: float) -> dict[str, object]:
    return {
        "pool": "pool_S",
        "test_month": month,
        "candidate_rows": 1000.0,
        "selected_rows": 100.0,
        "pool_short_mean_bps": 0.0,
        "selected_short_mean_bps": short_bps,
        "short_internal_excess_bps": short_bps,
        "pool_next_mean_bps": 0.0,
        "selected_next_mean_bps": next_bps,
        "next_internal_excess_bps": next_bps,
        "short_rank_ic": 0.10,
        "next_rank_ic": 0.01,
    }


def test_pool_internal_halfyear_and_year_summaries() -> None:
    month_summary = pd.DataFrame(
        [
            _month_row("2025-01", 4.0, -1.0),
            _month_row("2025-02", 8.0, 3.0),
            _month_row("2025-07", 2.0, 5.0),
        ]
    )

    halfyear = halfyear_summary(month_summary)
    yearly = year_summary(month_summary)

    h1 = halfyear.loc[halfyear["half"].eq("H1")].iloc[0]
    assert h1["months"] == 2
    assert h1["short_internal_excess_bps"] == pytest.approx(6.0)
    assert h1["next_positive_months"] == 1

    year = yearly.iloc[0]
    assert year["year"] == 2025
    assert year["months"] == 3
    assert year["short_positive_months"] == 3
    assert year["next_internal_excess_bps"] == pytest.approx(7.0 / 3.0)


def test_prediction_files_can_read_k8s_shard_layout(tmp_path) -> None:
    first = tmp_path / "month_2022-01" / "predictions.parquet"
    second = tmp_path / "month_2022-07" / "predictions_2022.parquet"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    paths = prediction_files(tmp_path)

    assert paths == [first, second]
