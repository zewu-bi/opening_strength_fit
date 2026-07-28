from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opening_strength_fit.backward_price_paths import (
    HORIZON_MINUTES,
    STATE_COUNT,
    assemble_backward_price_sequence,
    return_clock_seconds,
    state_endpoint_offsets_us,
    state_endpoint_seconds,
)
from opening_strength_fit.commands.backward_price_path_cache import (
    endpoint_price_state_sql,
)
from opening_strength_fit.temporal_analysis import TARGET_COLUMN


def _states(symbol: str, *, scale: float = 1.0) -> pd.DataFrame:
    endpoints = state_endpoint_offsets_us()
    prices = scale * (100.0 + np.arange(STATE_COUNT, dtype=np.float64))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "state_index": np.arange(STATE_COUNT),
            "price": prices,
            "status": "TRADE",
            "source_offset_us": endpoints - 1,
        }
    )


def test_endpoint_clocks_span_trading_minutes_and_skip_lunch() -> None:
    endpoints = state_endpoint_seconds()
    clocks = return_clock_seconds()
    assert len(endpoints) == 241
    assert len(clocks) == 240
    assert endpoints[0] == 9 * 3600 + 30 * 60
    assert endpoints[120] == 11 * 3600 + 30 * 60
    assert endpoints[121] == 13 * 3600 + 60
    assert endpoints[-1] == 15 * 3600


def test_backward_returns_share_the_same_endpoint() -> None:
    states = pd.concat(
        [
            _states("000001.SZ"),
            _states("000002.SZ", scale=2.0),
        ],
        ignore_index=True,
    )
    labels = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            TARGET_COLUMN: [0.01, 0.02],
        }
    )
    arrays = assemble_backward_price_sequence(
        states,
        labels,
        pool_symbols={"000002.SZ"},
    )

    assert arrays["values"].shape == (2, 3, 240)
    assert arrays["clock_seconds"].shape == (240,)
    assert arrays["pool_member"].tolist() == [False, True]
    for channel, horizon in enumerate(HORIZON_MINUTES):
        endpoint_index = 100
        expected = (100.0 + endpoint_index + 1) / (100.0 + endpoint_index + 1 - horizon) - 1.0
        assert arrays["values"][0, channel, endpoint_index] == pytest.approx(expected)
        assert not arrays["valid"][0, channel, : horizon - 1].any()
        assert arrays["valid"][0, channel, horizon - 1 :].all()


def test_invalid_status_masks_returns_using_either_endpoint() -> None:
    states = _states("000001.SZ")
    states.loc[states["state_index"].eq(100), "status"] = "SUSP"
    labels = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            TARGET_COLUMN: [0.01],
        }
    )
    arrays = assemble_backward_price_sequence(states, labels)
    for channel, horizon in enumerate(HORIZON_MINUTES):
        assert not arrays["valid"][0, channel, 99]
        if 100 + horizon <= 240:
            assert not arrays["valid"][0, channel, 99 + horizon]


def test_future_source_tick_is_rejected() -> None:
    states = _states("000001.SZ")
    states.loc[states["state_index"].eq(10), "source_offset_us"] += 2
    labels = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            TARGET_COLUMN: [0.01],
        }
    )
    with pytest.raises(RuntimeError, match="future tick"):
        assemble_backward_price_sequence(states, labels)


def test_clickhouse_sql_uses_endpoint_buckets_and_point_in_time_argmax() -> None:
    sql = endpoint_price_state_sql("stock.tick")
    assert "argMax(LastPrice" in sql
    assert "arrayMax(mapValues(LocalTimeStamp))" in sql
    assert "UNION ALL" in sql
    assert "state_index" in sql
