from __future__ import annotations

import unittest

import pandas as pd

from opening_strength_fit.model import feature_columns, fit_ridge_frame, predict_frame
from opening_strength_fit.targets import add_cross_sectional_target_label


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2022-01-04",
                "symbol": "000001.SZ",
                "decision_target_timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "label": 0.01,
                "valid_label": True,
            },
            {
                "date": "2022-01-04",
                "symbol": "000002.SZ",
                "decision_target_timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "timestamp": pd.Timestamp("2022-01-04 09:30:00"),
                "label": 0.03,
                "valid_label": True,
            },
            {
                "date": "2022-01-04",
                "symbol": "000003.SZ",
                "decision_target_timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                "timestamp": pd.Timestamp("2022-01-04 09:31:00"),
                "label": 0.05,
                "valid_label": True,
            },
        ]
    )


class TargetLabelTest(unittest.TestCase):
    def test_demean_replaces_label_and_preserves_raw_label(self) -> None:
        out = add_cross_sectional_target_label(_frame(), mode="demean")

        self.assertIn("label_raw", out.columns)
        self.assertAlmostEqual(out.loc[0, "label_raw"], 0.01)
        self.assertAlmostEqual(out.loc[1, "label_raw"], 0.03)
        self.assertAlmostEqual(out.loc[0, "label"], -0.01)
        self.assertAlmostEqual(out.loc[1, "label"], 0.01)
        self.assertTrue(pd.isna(out.loc[2, "label"]))
        self.assertFalse(bool(out.loc[2, "valid_label"]))

    def test_zscore_uses_cross_sectional_standard_deviation(self) -> None:
        out = add_cross_sectional_target_label(_frame(), mode="zscore")

        self.assertAlmostEqual(out.loc[0, "label"], -1.0)
        self.assertAlmostEqual(out.loc[1, "label"], 1.0)
        self.assertAlmostEqual(out.loc[0, "label_xs_std"], 0.01)

    def test_rank_centered_is_groupwise(self) -> None:
        out = add_cross_sectional_target_label(_frame(), mode="rank_centered")

        self.assertAlmostEqual(out.loc[0, "label"], -0.25)
        self.assertAlmostEqual(out.loc[1, "label"], 0.25)
        self.assertTrue(pd.isna(out.loc[2, "label"]))

    def test_target_label_is_not_a_feature_and_can_train_model(self) -> None:
        frame = _frame().iloc[:2].copy()
        frame["feature"] = [1.0, 2.0]
        out = add_cross_sectional_target_label(
            frame,
            mode="demean",
            target_col="target_label",
        )

        self.assertEqual(out["label"].tolist(), [0.01, 0.03])
        self.assertEqual(out["target_label"].round(6).tolist(), [-0.01, 0.01])
        self.assertNotIn("target_label", feature_columns(out))
        self.assertNotIn("label_xs_mean", feature_columns(out))

        model, _ = fit_ridge_frame(out, target_col="target_label")
        predictions = predict_frame(model, out)

        self.assertIn("label", predictions.columns)
        self.assertIn("target_label", predictions.columns)
        self.assertEqual(model.target_col, "target_label")

    def test_heat_neutral_target_removes_configured_exposure(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 5,
                "symbol": [f"00000{idx}.SZ" for idx in range(5)],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00")
                ]
                * 5,
                "timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 5,
                "heat": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "label": [-0.21, -0.09, 0.02, 0.11, 0.19],
                "valid_label": [True] * 5,
            }
        )

        out = add_cross_sectional_target_label(
            frame,
            mode="heat_neutral",
            target_col="target_label",
            neutralize_cols=("heat",),
            neutralization_transform="zscore",
            neutralization_ridge_alpha=0.0,
        )

        self.assertEqual(out["label"].tolist(), frame["label"].tolist())
        self.assertIn("label_xs_heat_fitted", out.columns)
        self.assertLess(abs(out["target_label"].corr(out["heat"])), 1e-12)
        self.assertEqual(out["label_xs_heat_exposure_count"].dropna().unique().tolist(), [1.0])

    def test_heat_neutral_strength_shrinks_instead_of_full_residual(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 5,
                "symbol": [f"00000{idx}.SZ" for idx in range(5)],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00")
                ]
                * 5,
                "timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 5,
                "heat": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "label": [-0.21, -0.09, 0.02, 0.11, 0.19],
                "valid_label": [True] * 5,
            }
        )

        full = add_cross_sectional_target_label(
            frame,
            mode="heat_neutral",
            target_col="target_label",
            neutralize_cols=("heat",),
            neutralization_transform="zscore",
            neutralization_ridge_alpha=0.0,
            neutralization_strength=1.0,
        )
        half = add_cross_sectional_target_label(
            frame,
            mode="heat_neutral",
            target_col="target_label",
            neutralize_cols=("heat",),
            neutralization_transform="zscore",
            neutralization_ridge_alpha=0.0,
            neutralization_strength=0.5,
        )
        demeaned = frame["label"] - frame["label"].mean()

        self.assertLess(abs(full["target_label"].corr(full["heat"])), 1e-12)
        self.assertGreater(abs(half["target_label"].corr(half["heat"])), 0.1)
        expected = (demeaned + full["target_label"]) / 2.0
        self.assertEqual(half["target_label"].round(12).tolist(), expected.round(12).tolist())

    def test_guard_shrunk_target_only_penalizes_dirty_positive_excess(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 3,
                "symbol": [f"00000{idx}.SZ" for idx in range(3)],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00")
                ]
                * 3,
                "timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 3,
                "label": [-8.0, -8.0, 32.0],
                "guard_pass": [False, True, False],
                "valid_label": [True] * 3,
            }
        )

        out = add_cross_sectional_target_label(
            frame,
            mode="guard_shrunk",
            target_col="target_label",
            guard_shrink_penalty=0.75,
            guard_pass_col="guard_pass",
        )

        self.assertEqual(out["label"].tolist(), frame["label"].tolist())
        self.assertEqual(out["label_xs_median"].tolist(), [-8.0, -8.0, -8.0])
        self.assertEqual(out["target_label"].tolist(), [-8.0, -8.0, 2.0])
        self.assertEqual(out["label_xs_guard_shrink"].tolist(), [0.0, 0.0, 30.0])

    def test_guard_shrunk_target_can_compute_dirty_from_rank_guard(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 3,
                "symbol": [f"00000{idx}.SZ" for idx in range(3)],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00")
                ]
                * 3,
                "timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 3,
                "heat": [1.0, 2.0, 3.0],
                "label": [-8.0, -8.0, 32.0],
                "valid_label": [True] * 3,
            }
        )

        out = add_cross_sectional_target_label(
            frame,
            mode="guard_shrunk",
            target_col="target_label",
            guard_shrink_penalty=0.50,
            guard_pass_col="",
            guard_rank_max_values={"heat": 0.70},
            guard_rank_method="average",
        )

        self.assertEqual(out["label_xs_guard_pass"].tolist(), [1, 1, 0])
        self.assertEqual(out["target_label"].tolist(), [-8.0, -8.0, 12.0])

    def test_guard_risk_shrunk_target_scales_positive_excess_by_risk(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 3,
                "symbol": [f"00000{idx}.SZ" for idx in range(3)],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00")
                ]
                * 3,
                "timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 3,
                "spread": [1.0, 2.0, 3.0],
                "label": [-8.0, -8.0, 32.0],
                "valid_label": [True] * 3,
            }
        )

        out = add_cross_sectional_target_label(
            frame,
            mode="guard_risk_shrunk",
            target_col="target_label",
            guard_risk_lambda=0.75,
            guard_risk_rank_max_values={"spread": 0.75},
            guard_rank_method="average",
        )

        self.assertEqual(out["label"].tolist(), frame["label"].tolist())
        self.assertEqual(out["label_xs_median"].tolist(), [-8.0, -8.0, -8.0])
        self.assertEqual(out["label_xs_guard_risk"].round(6).tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(out["target_label"].tolist(), [-8.0, -8.0, 2.0])

    def test_guard_risk_shrunk_target_averages_components(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2022-01-04"] * 3,
                "symbol": [f"00000{idx}.SZ" for idx in range(3)],
                "decision_target_timestamp": [
                    pd.Timestamp("2022-01-04 09:31:00")
                ]
                * 3,
                "timestamp": [pd.Timestamp("2022-01-04 09:31:00")] * 3,
                "spread": [1.0, 2.0, 3.0],
                "depth": [3.0, 2.0, 1.0],
                "label": [-8.0, -8.0, 32.0],
                "valid_label": [True] * 3,
            }
        )

        out = add_cross_sectional_target_label(
            frame,
            mode="guard_risk_shrunk",
            target_col="target_label",
            guard_risk_lambda=1.0,
            guard_risk_rank_min_values={"depth": 0.50},
            guard_risk_rank_max_values={"spread": 0.75},
            guard_rank_method="average",
            guard_risk_normalization="mean",
        )

        self.assertEqual(out["label_xs_guard_risk_component_count"].tolist(), [0.0, 0.0, 2.0])
        self.assertEqual(out["label_xs_guard_risk"].round(6).tolist(), [0.0, 0.0, 0.666667])
        self.assertEqual(out["target_label"].round(6).tolist(), [-8.0, -8.0, 5.333333])


if __name__ == "__main__":
    unittest.main()
