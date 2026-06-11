from __future__ import annotations

import argparse
import unittest

import numpy as np
import pandas as pd

from opening_strength_fit.model import predict_frame
from opening_strength_fit.training_modeling import fit_prediction_model


class ModelEnsembleTest(unittest.TestCase):
    def test_ensemble_model_fits_members_and_emits_prediction(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 6,
                "symbol": [f"00000{i}.SZ" for i in range(6)],
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 6,
                "label": [0.1, 0.2, 0.05, -0.1, 0.0, 0.3],
                "target_label": [0.2, 0.4, 0.1, -0.2, 0.0, 0.6],
                "valid_label": [True] * 6,
                "feature_a": [1.0, 2.0, 1.5, -1.0, 0.0, 3.0],
                "feature_b": [0.0, 1.0, 0.5, 2.0, -0.5, 1.5],
            }
        )
        config = {
            "model": {
                "name": "ensemble",
                "target_col": "target_label",
                "combine_mode": "rank_centered",
                "rank_group_cols": ["date", "decision_target_timestamp"],
                "members": [
                    {"name": "ridge", "weight": 0.4, "alpha": 1.0},
                    {
                        "name": "gbm",
                        "weight": 0.6,
                        "max_iter": 5,
                        "learning_rate": 0.1,
                        "max_leaf_nodes": 3,
                        "l2_regularization": 0.0,
                        "random_state": 7,
                    },
                ],
            }
        }

        model, stats = fit_prediction_model(
            frame,
            args=argparse.Namespace(feature_limit=None),
            config=config,
            alpha=1.0,
        )
        predictions = predict_frame(model, frame)

        self.assertEqual(len(model.models), 2)
        self.assertEqual(stats["features"], 2)
        self.assertIn("ensemble_rank_centered", model.model_name)
        self.assertTrue(np.isfinite(predictions["prediction"]).all())


if __name__ == "__main__":
    unittest.main()
