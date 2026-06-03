from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.model import feature_columns
from opening_strength_fit.feature_config import feature_filters_from_config
from opening_strength_fit.training import _feature_filters_from_config


class FeatureFilterTest(unittest.TestCase):
    def test_feature_columns_support_include_and_drop_filters(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"],
                "symbol": ["000001.SZ"],
                "label": [0.01],
                "ask_volume_1": [100.0],
                "postopen_ask_volume_1_diff_1m": [10.0],
                "postopen_v2_trade_vwap_vs_mid_10t_bps": [1.0],
                "postopen_v2_weak_feature": [2.0],
                "preopen_volume": [3.0],
                "volume": [4.0],
                "other": [5.0],
            }
        )

        features = feature_columns(
            frame,
            include_prefixes=("ask_volume_", "postopen_v2_trade_vwap_vs_"),
            include_patterns=(r"^postopen_(?!v2_)",),
            drop_columns=("volume",),
            drop_prefixes=("preopen_",),
        )

        self.assertEqual(
            features,
            [
                "ask_volume_1",
                "postopen_ask_volume_1_diff_1m",
                "postopen_v2_trade_vwap_vs_mid_10t_bps",
            ],
        )

    def test_feature_filters_from_config_maps_toml_keys(self) -> None:
        config = {
            "features": {
                "include_feature_columns": ["spread_bps"],
                "include_feature_prefixes": ["postopen_v2_trade_vwap_vs_"],
                "include_feature_regexes": [r"^postopen_(?!v2_)"],
                "drop_feature_columns": ["volume"],
                "drop_feature_prefixes": ["preopen_"],
                "drop_feature_regexes": [r"_debug$"],
            }
        }
        filters = feature_filters_from_config(config)

        self.assertEqual(filters["include_columns"], ("spread_bps",))
        self.assertEqual(
            filters["include_prefixes"],
            ("postopen_v2_trade_vwap_vs_",),
        )
        self.assertEqual(filters["include_patterns"], (r"^postopen_(?!v2_)",))
        self.assertEqual(filters["drop_columns"], ("volume",))
        self.assertEqual(filters["drop_prefixes"], ("preopen_",))
        self.assertEqual(filters["drop_patterns"], (r"_debug$",))
        self.assertEqual(_feature_filters_from_config(config), filters)


if __name__ == "__main__":
    unittest.main()
