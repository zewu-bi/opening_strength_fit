from __future__ import annotations

import argparse
import importlib.util
import unittest

import numpy as np
import pandas as pd

from opening_strength_fit.model import predict_frame
from opening_strength_fit.training_modeling import fit_prediction_model


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchMLPTest(unittest.TestCase):
    def _toy_frame(self) -> pd.DataFrame:
        rows = 32
        return pd.DataFrame(
            {
                "date": ["2022-01-04"] * rows,
                "symbol": [f"000{i:03d}.SZ" for i in range(rows)],
                "decision_target_timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * rows,
                "label": np.linspace(-0.2, 0.2, rows),
                "target_label": np.linspace(-0.4, 0.4, rows),
                "valid_label": [True] * rows,
                "feature_a": np.linspace(-1.0, 1.0, rows),
                "feature_b": np.sin(np.linspace(0.0, 2.0, rows)),
                "unused_text": ["x"] * rows,
            }
        )

    def test_torch_mlp_fits_and_predicts_target_label(self) -> None:
        frame = self._toy_frame()
        config = {
            "features": {"include_feature_columns": ["feature_a", "feature_b"]},
            "model": {
                "name": "torch_mlp",
                "target_col": "target_label",
                "hidden_layers": [8, 4],
                "dropout": 0.0,
                "batch_size": 8,
                "predict_batch_size": 8,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "max_epochs": 2,
                "validation_fraction": 0.25,
                "validation_max_rows": 8,
                "early_stopping_patience": 1,
                "device": "cpu",
                "random_state": 7,
            },
        }

        model, stats = fit_prediction_model(
            frame,
            args=argparse.Namespace(feature_limit=None),
            config=config,
            alpha=1.0,
        )
        predictions = predict_frame(model, frame)

        self.assertEqual(model.target_col, "target_label")
        self.assertEqual(stats["features"], 2)
        self.assertIn("target_label", predictions.columns)
        self.assertTrue(np.isfinite(predictions["prediction"]).all())

    def test_torch_mlp_wide_deep_residual_architecture(self) -> None:
        frame = self._toy_frame()
        config = {
            "features": {"include_feature_columns": ["feature_a", "feature_b"]},
            "model": {
                "name": "torch_mlp",
                "target_col": "target_label",
                "architecture": "wide_deep_residual",
                "hidden_layers": [4],
                "dropout": 0.0,
                "batch_size": 8,
                "predict_batch_size": 8,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "max_epochs": 2,
                "validation_fraction": 0.25,
                "validation_max_rows": 8,
                "early_stopping_patience": 1,
                "device": "cpu",
                "random_state": 7,
            },
        }

        model, stats = fit_prediction_model(
            frame,
            args=argparse.Namespace(feature_limit=None),
            config=config,
            alpha=1.0,
        )
        predictions = predict_frame(model, frame)

        self.assertEqual(stats["features"], 2)
        self.assertTrue(np.isfinite(predictions["prediction"]).all())


if __name__ == "__main__":
    unittest.main()
