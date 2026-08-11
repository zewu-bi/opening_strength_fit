from __future__ import annotations

import argparse
import importlib.util
import unittest

import numpy as np
import pandas as pd

from opening_strength_fit.model import predict_frame
from opening_strength_fit.torch_model.training import (
    _resolve_training_tensor_storage,
    _torch_loss,
    _torch_random_sampler_order,
)
from opening_strength_fit.training_modeling import fit_prediction_model


def test_training_tensor_storage_resolution() -> None:
    gib = 1024**3
    assert (
        _resolve_training_tensor_storage(
            "auto",
            device="cuda",
            required_bytes=40 * gib,
            free_bytes=80 * gib,
            reserve_bytes=12 * gib,
        )
        == "cuda_resident"
    )
    assert (
        _resolve_training_tensor_storage(
            "auto",
            device="cuda",
            required_bytes=40 * gib,
            free_bytes=48 * gib,
            reserve_bytes=12 * gib,
        )
        == "host_vectorized"
    )
    assert (
        _resolve_training_tensor_storage(
            "auto",
            device="cpu",
            required_bytes=40 * gib,
            free_bytes=0,
            reserve_bytes=12 * gib,
        )
        == "host_vectorized"
    )


def test_required_cuda_resident_storage_rejects_insufficient_memory() -> None:
    gib = 1024**3
    with unittest.TestCase().assertRaisesRegex(SystemExit, "does not fit"):
        _resolve_training_tensor_storage(
            "cuda_resident",
            device="cuda",
            required_bytes=40 * gib,
            free_bytes=48 * gib,
            reserve_bytes=12 * gib,
        )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchMLPTest(unittest.TestCase):
    def test_huber_mse_blend_loss(self) -> None:
        import torch

        prediction = torch.tensor([0.0, 2.0])
        target = torch.zeros(2)
        huber = _torch_loss("huber", torch, huber_beta=1.0)(prediction, target)
        blended = _torch_loss(
            "huber_mse_blend",
            torch,
            huber_beta=1.0,
            mse_blend_weight=0.2,
        )(prediction, target)

        torch.testing.assert_close(huber, torch.tensor([0.0, 1.5]))
        torch.testing.assert_close(blended, torch.tensor([0.0, 2.0]))

    def test_huber_mse_blend_rejects_invalid_settings(self) -> None:
        import torch

        with self.assertRaisesRegex(SystemExit, "huber_beta"):
            _torch_loss("huber", torch, huber_beta=0.0)
        with self.assertRaisesRegex(SystemExit, "mse_blend_weight"):
            _torch_loss("huber_mse_blend", torch, mse_blend_weight=1.1)

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
        self.assertEqual(stats["training_tensor_storage"], "host_vectorized")
        self.assertTrue(stats["vectorized_index_batches"])
        self.assertIn("target_label", predictions.columns)
        self.assertTrue(np.isfinite(predictions["prediction"]).all())

    def test_vectorized_shuffle_matches_torch_dataloader_order(self) -> None:
        import torch

        dataset = torch.utils.data.TensorDataset(torch.arange(20))
        train_indices = np.array([0, 2, 3, 6, 7, 11, 12, 15, 18], dtype=np.int64)
        subset = torch.utils.data.Subset(dataset, train_indices)

        torch.manual_seed(91)
        loader = torch.utils.data.DataLoader(subset, batch_size=4, shuffle=True)
        loader_values = torch.cat([batch[0] for batch in loader])

        torch.manual_seed(91)
        order = _torch_random_sampler_order(torch, len(train_indices))
        vectorized_values = torch.from_numpy(train_indices).index_select(0, order)

        torch.testing.assert_close(vectorized_values, loader_values)

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

    def test_torch_mlp_grouped_architectures(self) -> None:
        frame = self._toy_frame()
        frame = frame.rename(
            columns={
                "feature_a": "turnover_diff_1t",
                "feature_b": "hist_surprise_volume_diff_1t_20d_zscore",
            }
        )
        for architecture in [
            "grouped_residual",
            "grouped_gated",
            "grouped_cross",
            "group_token_transformer",
        ]:
            with self.subTest(architecture=architecture):
                config = {
                    "features": {
                        "include_feature_columns": [
                            "turnover_diff_1t",
                            "hist_surprise_volume_diff_1t_20d_zscore",
                        ]
                    },
                    "model": {
                        "name": "torch_mlp",
                        "target_col": "target_label",
                        "architecture": architecture,
                        "hidden_layers": [8, 4],
                        "group_embedding_dim": 8,
                        "fusion_dim": 16,
                        "block_hidden_dim": 32,
                        "num_blocks": 1,
                        "transformer_heads": 2,
                        "dropout": 0.0,
                        "batch_size": 8,
                        "predict_batch_size": 8,
                        "learning_rate": 0.01,
                        "weight_decay": 0.0,
                        "max_epochs": 1,
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
