from __future__ import annotations

import pandas as pd

from opening_strength_fit.features import build_preopen_features


def _auction_ticks() -> pd.DataFrame:
    rows = [
        ("09:15:00", 0.0, 9.90, 9.90, 0.10, 0.0, 0.0),
        ("09:20:00", 0.0, 10.10, 10.10, 0.20, 0.0, 0.0),
        ("09:24:57", 0.0, 10.00, 10.00, 0.30, 0.0, 0.0),
        ("09:25:00", 10.05, 10.04, 10.06, -0.70, 1_000.0, 10_050.0),
        ("09:25:03", 10.05, 10.03, 10.07, -0.80, 1_000.0, 10_050.0),
    ]
    return pd.DataFrame(
        [
            {
                "date": "2025-07-01",
                "symbol": "000001.SZ",
                "timestamp": pd.Timestamp(f"2025-07-01 {clock}"),
                "last_price": last_price,
                "bid_price_1": bid1,
                "ask_price_1": ask1,
                "depth_imbalance_10": imbalance,
                "volume": volume,
                "turnover": turnover,
            }
            for clock, last_price, bid1, ask1, imbalance, volume, turnover in rows
        ]
    )


def test_indicative_preopen_mode_separates_path_match_and_prematch_book() -> None:
    out = build_preopen_features(
        _auction_ticks(),
        price_mode="indicative_quote_v2",
    ).iloc[0]

    assert out["preopen_price_min"] == 9.90
    assert out["preopen_price_max"] == 10.10
    assert out["preopen_last_price"] == 10.05
    assert out["preopen_depth_imbalance_10"] == 0.30
    assert out["preopen_volume"] == 1_000.0
    assert out["preopen_turnover"] == 10_050.0


def test_legacy_preopen_mode_remains_default_for_old_configs() -> None:
    out = build_preopen_features(_auction_ticks()).iloc[0]

    assert out["preopen_price_min"] == 0.0
    assert out["preopen_price_max"] == 10.05
    assert out["preopen_last_price"] == 10.05
    assert out["preopen_depth_imbalance_10"] == -0.80


def test_indicative_preopen_mode_uses_positive_single_side_fallback() -> None:
    ticks = _auction_ticks()
    ticks.loc[0, "ask_price_1"] = 0.0
    ticks.loc[0, "bid_price_1"] = 9.88

    out = build_preopen_features(
        ticks,
        price_mode="indicative_quote_v2",
    ).iloc[0]

    assert out["preopen_price_min"] == 9.88
