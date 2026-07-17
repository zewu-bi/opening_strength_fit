from __future__ import annotations

import pandas as pd
import pytest

from opening_strength_fit.clickhouse_daily_reference import (
    attach_daily_market_reference,
    query_lagged_daily_market_reference,
)


class _FakeClient:
    def __init__(self, results: list[pd.DataFrame]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict]] = []

    def query_df(self, sql: str, *, parameters: dict) -> pd.DataFrame:
        self.calls.append((sql, parameters))
        return self.results.pop(0)


def test_query_market_reference_is_strictly_lagged_and_unit_normalized() -> None:
    client = _FakeClient(
        [
            pd.DataFrame({"reference_day": [pd.Timestamp("2025-06-30")]}),
            pd.DataFrame(
                {
                    "Symbol": ["000001.SZ", "600000.SH"],
                    "TotalMarketValue": [23_000_000.0, 0.0],
                    "TotalFloatMarketValue": [20_000_000.0, 30_000_000.0],
                    "TotalShareToday": [1_900_000.0, 3_000_000.0],
                    "FloatAShare": [1_800_000.0, 2_900_000.0],
                    "FreeShareToday": [1_700_000.0, 2_800_000.0],
                }
            ),
        ]
    )

    out = query_lagged_daily_market_reference(
        client,
        trading_day="2025-07-01",
        symbols=["600000.SH", "000001.SZ", "000001.SZ"],
    )

    session_sql, session_parameters = client.calls[0]
    assert "TradingDay < {trading_day:Date}" in session_sql
    assert "limit 1 offset 0" in session_sql
    assert session_parameters == {"trading_day": "2025-07-01"}
    _, value_parameters = client.calls[1]
    assert value_parameters == {
        "reference_day": "2025-06-30",
        "symbols": ["000001.SZ", "600000.SH"],
    }
    first = out.loc[out["symbol"] == "000001.SZ"].iloc[0]
    assert first["date"] == "2025-07-01"
    assert first["market_cap_reference_date"] == pd.Timestamp("2025-06-30")
    assert first["market_cap_reference_lag_sessions"] == 1
    assert first["total_market_cap"] == 230_000_000_000.0
    assert first["total_shares"] == 19_000_000_000.0
    assert pd.isna(out.loc[out["symbol"] == "600000.SH", "total_market_cap"]).all()


def test_query_market_reference_rejects_same_day_reference() -> None:
    client = _FakeClient([pd.DataFrame({"reference_day": [pd.Timestamp("2025-07-01")]})])

    with pytest.raises(RuntimeError, match="strictly earlier"):
        query_lagged_daily_market_reference(
            client,
            trading_day="2025-07-01",
            symbols=["000001.SZ"],
        )


def test_attach_market_reference_is_many_to_one_and_preserves_missing_symbols() -> None:
    ticks = pd.DataFrame(
        {
            "date": ["2025-07-01"] * 3,
            "symbol": ["000001.SZ", "000001.SZ", "600000.SH"],
            "last_price": [12.0, 12.1, 13.8],
        }
    )
    reference = pd.DataFrame(
        {
            "date": ["2025-07-01"],
            "symbol": ["000001.SZ"],
            "total_market_cap": [230_000_000_000.0],
            "float_market_cap": [200_000_000_000.0],
            "total_shares": [19_000_000_000.0],
            "float_shares": [18_000_000_000.0],
            "free_float_shares": [17_000_000_000.0],
            "market_cap_reference_date": [pd.Timestamp("2025-06-30")],
            "market_cap_reference_lag_sessions": [1],
        }
    )

    out = attach_daily_market_reference(ticks, reference)

    assert out.loc[out["symbol"] == "000001.SZ", "total_market_cap"].notna().all()
    assert out.loc[out["symbol"] == "600000.SH", "total_market_cap"].isna().all()
    assert out["last_price"].tolist() == ticks["last_price"].tolist()
