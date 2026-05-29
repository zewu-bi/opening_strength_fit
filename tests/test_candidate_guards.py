from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from opening_strength_fit.candidates import (
    filter_opening_candidates,
    opening_candidate_mask,
)
from opening_strength_fit.model import feature_columns, fit_lightgbm_frame
from opening_strength_fit.training import (
    _apply_guard_features_from_config,
    _apply_sample_weight_from_config,
)


class CandidateGuardTest(unittest.TestCase):
    def test_candidate_filter_supports_rank_ranges(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 5,
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 5,
                "symbol": [f"00000{idx}.SZ" for idx in range(5)],
                "heat": [1, 2, 3, 4, 5],
            }
        )

        filtered = filter_opening_candidates(
            frame,
            rank_min_values={"heat": 0.4},
            rank_max_values={"heat": 0.8},
            rank_method="average",
        )

        self.assertEqual(filtered["heat"].tolist(), [2, 3, 4])

    def test_candidate_filter_treats_inf_as_missing(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 3,
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 3,
                "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "return_10t": [0.01, np.inf, 0.02],
            }
        )

        mask = opening_candidate_mask(frame, min_values={"return_10t": 0.0})

        self.assertEqual(mask.tolist(), [True, False, True])

    def test_sample_weight_config_uses_candidate_mask(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 4,
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 4,
                "symbol": [f"00000{idx}.SZ" for idx in range(4)],
                "heat": [1, 2, 3, 4],
                "label": [0.01, 0.02, 0.03, 0.04],
            }
        )
        config = {
            "sample_weight": {
                "enabled": True,
                "pass_weight": 1.0,
                "fail_weight": 0.25,
                "rank_method": "average",
                "rank_min": {"heat": 0.5},
                "rank_max": {"heat": 0.75},
            }
        }

        weighted = _apply_sample_weight_from_config(frame, config)

        self.assertEqual(weighted["sample_weight"].tolist(), [0.25, 1.0, 1.0, 0.25])
        self.assertTrue(
            opening_candidate_mask(
                frame,
                rank_min_values={"heat": 0.5},
                rank_max_values={"heat": 0.75},
                rank_method="average",
            ).equals(weighted["sample_weight"].eq(1.0))
        )

    def test_lightgbm_uses_sample_weight_without_feature_leakage(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 8,
                "symbol": [f"00000{idx}.SZ" for idx in range(8)],
                "label": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "sample_weight": [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0],
            }
        )

        model, stats = fit_lightgbm_frame(
            frame,
            sample_weight_col="sample_weight",
            n_estimators=3,
            num_leaves=3,
            min_child_samples=1,
        )

        self.assertNotIn("sample_weight", feature_columns(frame))
        self.assertNotIn("sample_weight", model.features)
        self.assertAlmostEqual(stats["sample_weight_mean"], 0.4375)

    def test_guard_features_add_rank_columns_and_pass_flag(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 4,
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 4,
                "symbol": [f"00000{idx}.SZ" for idx in range(4)],
                "heat": [1, 2, 3, 4],
                "depth": [4, 3, 2, 1],
                "label": [0.01, 0.02, 0.03, 0.04],
            }
        )
        config = {
            "guard_features": {
                "enabled": True,
                "rank_method": "average",
                "rank_columns": ["heat", "depth"],
                "rank_min": {"heat": 0.5},
                "rank_max": {"heat": 0.75},
            }
        }

        out = _apply_guard_features_from_config(frame, config)

        self.assertEqual(
            out["guard_heat_rank_pct"].round(2).tolist(),
            [0.25, 0.50, 0.75, 1.00],
        )
        self.assertEqual(
            out["guard_depth_rank_pct"].round(2).tolist(),
            [1.00, 0.75, 0.50, 0.25],
        )
        self.assertEqual(out["guard_pass"].tolist(), [0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
