import pandas as pd
import pytest

from opening_strength_fit.trading_sessions import (
    parse_trading_sessions,
    shift_by_trading_seconds,
)


def test_trading_seconds_skip_a_share_lunch_break() -> None:
    shifted = shift_by_trading_seconds("2025-01-02 11:29:09", 300)

    assert shifted == pd.Timestamp("2025-01-02 13:04:09")


def test_final_close_is_representable_but_past_close_is_not() -> None:
    at_close = shift_by_trading_seconds("2025-01-02 14:59:00", 60)
    past_close = shift_by_trading_seconds("2025-01-02 14:59:00", 61)

    assert at_close == pd.Timestamp("2025-01-02 15:00:00")
    assert pd.isna(past_close)


def test_sessions_must_not_overlap() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        parse_trading_sessions(["09:30:00-11:30:00", "11:00:00-15:00:00"])
