from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.alpha_conditioning import add_alpha_conditioned_risk_targets


class AlphaConditioningTest(unittest.TestCase):
    def test_alpha_conditioned_risk_targets_share_candidate_and_weights(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 4,
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 4,
                "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "candidate_alpha_rank": [0.25, 0.50, 0.75, 1.00],
                "alpha_return_next_close": [0.40, 0.10, -0.20, 0.30],
            }
        )
        config = {
            "risk_layer": {
                "candidate_alpha_rank_min": 0.75,
                "gap_next_rank_max": 0.50,
                "binary_next_rank_max": 0.40,
                "candidate_weight": 1.0,
                "non_candidate_weight": 0.1,
            }
        }

        out = add_alpha_conditioned_risk_targets(frame, config)

        self.assertEqual(
            out["target_alpha_conditioned_candidate"].tolist(),
            [False, False, True, True],
        )
        self.assertEqual(
            out["target_alpha_conditioned_gap_risk"].tolist(),
            [0.0, 0.0, 0.5, 0.0],
        )
        self.assertEqual(
            out["target_alpha_conditioned_binary_risk"].tolist(),
            [0.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(out["risk_sample_weight"].tolist(), [0.1, 0.1, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
