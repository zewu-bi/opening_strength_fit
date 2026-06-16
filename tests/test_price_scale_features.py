from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.features import add_price_scale_features


def test_price_scale_features_make_tick_size_explicit() -> None:
    frame = pd.DataFrame(
        {
            "ask_price_1": [2.0, 50.0],
            "bid_price_1": [1.99, 49.99],
            "ask_price_2": [2.01, 50.02],
            "bid_price_2": [1.98, 49.98],
            "spread_bps": [50.0, 2.0],
            "ask_depth_10": [1000.0, 1000.0],
        }
    )

    out = add_price_scale_features(
        frame,
        bucket_edges=(5.0, 20.0),
        interaction_columns=("spread_bps",),
    )

    assert np.isclose(out.loc[0, "price_scale_tick_bps"], 50.0)
    assert np.isclose(out.loc[1, "price_scale_tick_bps"], 2.0)
    assert np.isclose(out.loc[0, "price_scale_spread_ticks"], 1.0)
    assert np.isclose(out.loc[0, "price_scale_ask_gap_2_ticks"], 1.0)
    assert np.isclose(out.loc[1, "price_scale_bid_gap_2_ticks"], 1.0)
    assert out.loc[0, "price_scale_bucket_cheap"] == 1
    assert out.loc[1, "price_scale_bucket_expensive"] == 1
    assert np.isclose(out.loc[0, "price_scale_spread_bps_x_cheap"], 50.0)
    assert np.isclose(out.loc[1, "price_scale_spread_bps_x_expensive"], 2.0)
