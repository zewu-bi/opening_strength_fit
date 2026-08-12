from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.alpha_conditioning import add_alpha_conditioned_risk_targets
from opening_strength_fit.commands.learned_risk_layer import manual_dirty_risk
from opening_strength_fit.commands.score_risk_sweep import (
    RISK_COLUMNS,
    add_risk_scores,
)
from opening_strength_fit.risk_labels import rank_risk_components, short_next_ranks


def risk_frame() -> pd.DataFrame:
    values = [1.0, 2.0, 3.0, 4.0]
    return pd.DataFrame(
        {
            "run_id": "alpha",
            "date": "2022-01-04",
            "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
            "label": values[::-1],
            "alpha_return_next_close": values,
            "alpha_score": values[::-1],
            **{column: values for column in (*RISK_COLUMNS, "low_tail", "high_tail")},
        }
    )


class AlphaConditioningTest(unittest.TestCase):
    def test_shared_cross_sectional_risk_ranks(self) -> None:
        components = rank_risk_components(
            risk_frame(),
            rank_min={"low_tail": 0.5},
            rank_max={"high_tail": 0.5},
        )
        short_rank, next_rank = short_next_ranks(risk_frame())

        self.assertEqual(components["low_tail"].tolist(), [0.5, 0.0, 0.0, 0.0])
        self.assertEqual(components["high_tail"].tolist(), [0.0, 0.0, 0.5, 1.0])
        self.assertEqual(short_rank.tolist(), [1.0, 0.75, 0.5, 0.25])
        self.assertEqual(next_rank.tolist(), [0.25, 0.5, 0.75, 1.0])

    def test_manual_and_sweep_risk_share_components(self) -> None:
        frame = risk_frame()

        components = rank_risk_components(frame)
        sweep = add_risk_scores(frame)

        pd.testing.assert_series_equal(
            manual_dirty_risk(frame, {}),
            components.mean(axis=1),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            sweep["dirty_risk"],
            components.mean(axis=1),
            check_names=False,
        )
        self.assertEqual(
            sweep["next_flip_guard_10t_pass"].tolist(),
            components.eq(0.0).all(axis=1).tolist(),
        )

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
