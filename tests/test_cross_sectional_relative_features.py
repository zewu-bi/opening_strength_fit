from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.features import add_cross_sectional_relative_features


class CrossSectionalRelativeFeatureTest(unittest.TestCase):
    def test_relative_features_are_computed_within_decision_cross_section(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04", "2022-01-04", "2022-01-04", "2022-01-04"],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00"),
                    pd.Timestamp("2022-01-04 09:31:00"),
                    pd.Timestamp("2022-01-04 09:32:00"),
                    pd.Timestamp("2022-01-04 09:32:00"),
                ],
                "symbol": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
                "spread_bps": [10.0, 30.0, 100.0, 140.0],
            }
        )

        out = add_cross_sectional_relative_features(
            frame,
            columns=("spread_bps",),
            modes=("demean", "zscore", "rank_centered"),
        )

        self.assertEqual(out["xs_rel_spread_bps_demean"].tolist(), [-10.0, 10.0, -20.0, 20.0])
        self.assertAlmostEqual(out.loc[0, "xs_rel_spread_bps_zscore"], -0.70710678, places=6)
        self.assertAlmostEqual(out.loc[1, "xs_rel_spread_bps_zscore"], 0.70710678, places=6)
        self.assertEqual(out["xs_rel_spread_bps_rank_centered"].tolist(), [0.0, 0.5, 0.0, 0.5])


if __name__ == "__main__":
    unittest.main()
