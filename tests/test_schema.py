from __future__ import annotations

import pandas as pd
import pytest

from opening_strength_fit.schema import (
    frame_clock_series,
    normalize_clock_series,
    normalize_decision_keys,
)


def test_normalize_clock_series_accepts_embedded_hour_minute_values() -> None:
    values = pd.Series(["9:31", "decision=09:32:03", "", None])

    assert normalize_clock_series(values).tolist() == ["09:31:00", "09:32:03", "", ""]


def test_frame_clock_series_prefers_decision_time() -> None:
    frame = pd.DataFrame(
        {
            "decision_time": ["09:31", "09:32"],
            "decision_target_timestamp": ["2026-01-01 10:00", "2026-01-01 10:01"],
        }
    )

    assert frame_clock_series(frame).tolist() == ["09:31:00", "09:32:00"]


def test_frame_clock_series_falls_back_to_timestamp() -> None:
    frame = pd.DataFrame({"decision_target_timestamp": ["2026-01-01 09:31:02", "not-a-timestamp"]})

    assert frame_clock_series(frame).tolist() == ["09:31:02", ""]


def test_frame_clock_series_requires_a_clock_source() -> None:
    with pytest.raises(ValueError, match="frame has no clock column"):
        frame_clock_series(pd.DataFrame({"date": ["2026-01-01"]}))


def test_normalize_decision_keys_canonicalizes_join_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026/01/02", "bad-date"],
            "symbol": [1, "000002.SZ"],
            "decision_target_timestamp": ["2026-01-02 09:31", None],
            "score": [0.1, 0.2],
        }
    )

    out = normalize_decision_keys(frame)

    assert out["date"].tolist() == ["2026-01-02"]
    assert out["symbol"].tolist() == ["1"]
    assert out["decision_target_timestamp"].dt.strftime("%H:%M").tolist() == ["09:31"]
