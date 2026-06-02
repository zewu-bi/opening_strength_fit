from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_alpha_conditioned_rolling_validation import score_variants  # noqa: E402
from run_learned_risk_layer import (  # noqa: E402
    normalize_next_close_labels as normalize_learned_next_close_labels,
)
from run_score_risk_sweep import (  # noqa: E402
    normalize_next_close_labels as normalize_score_risk_next_close_labels,
)
from run_score_tail_guards import load_next_close_labels  # noqa: E402
from plot_signal_baseline_panels import (  # noqa: E402
    normalize_next_close_labels as normalize_panel_next_close_labels,
)
from opening_strength_fit.labels import safe_price_return  # noqa: E402


class NextCloseCleaningTest(unittest.TestCase):
    def _next_close_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": ["2021-08-02", "2021-08-02", "2021-08-02"],
                "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "decision_target_timestamp": [
                    pd.Timestamp("2021-08-02 09:31:00"),
                    pd.Timestamp("2021-08-02 09:31:00"),
                    pd.Timestamp("2021-08-02 09:31:00"),
                ],
                "alpha_return_next_close": [0.01, np.inf, -np.inf],
            }
        )

    def test_normalize_next_close_labels_drops_non_finite_returns(self) -> None:
        frame = self._next_close_frame()

        labels = normalize_learned_next_close_labels(frame)

        self.assertEqual(labels["symbol"].tolist(), ["000001.SZ"])
        self.assertTrue(np.isfinite(labels["alpha_return_next_close"]).all())

    def test_safe_price_return_masks_non_positive_buy_price(self) -> None:
        returns = safe_price_return(
            pd.Series([11.0, 12.0, 0.0, np.inf]),
            pd.Series([10.0, 0.0, 10.0, 10.0]),
        )

        self.assertAlmostEqual(returns.iloc[0], 0.1)
        self.assertTrue(np.isnan(returns.iloc[1]))
        self.assertTrue(np.isnan(returns.iloc[2]))
        self.assertTrue(np.isnan(returns.iloc[3]))
        self.assertFalse(np.isinf(returns).any())

    def test_next_close_consumers_drop_non_finite_returns(self) -> None:
        frame = self._next_close_frame()

        for normalize in (
            normalize_panel_next_close_labels,
            normalize_score_risk_next_close_labels,
        ):
            with self.subTest(normalize=normalize.__module__):
                labels = normalize(frame)
                self.assertEqual(labels["symbol"].tolist(), ["000001.SZ"])
                self.assertTrue(np.isfinite(labels["alpha_return_next_close"]).all())

    def test_tail_guard_label_loader_drops_non_finite_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "next_close.parquet"
            self._next_close_frame().to_parquet(path)

            labels = load_next_close_labels(path)

        self.assertEqual(labels["symbol"].tolist(), ["000001.SZ"])
        self.assertTrue(np.isfinite(labels["alpha_return_next_close"]).all())

    def test_score_variants_ignores_non_finite_next_close_in_excess(self) -> None:
        timestamp = pd.Timestamp("2021-08-02 09:31:00")
        frame = pd.DataFrame(
            {
                "date": ["2021-08-02"] * 4,
                "decision_target_timestamp": [timestamp] * 4,
                "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "label": [0.01, 0.02, 0.03, 0.04],
                "alpha_return_next_close": [0.10, np.inf, -0.10, 0.00],
                "candidate_alpha_rank": [0.40, 0.80, 1.00, 0.60],
                "gap_risk_rank": [0.1, 0.2, 0.3, 0.4],
                "binary_risk_rank": [0.1, 0.2, 0.3, 0.4],
            }
        )

        metrics = score_variants(
            frame,
            month="2021-08",
            variants=[
                {
                    "variant": "alpha_rank",
                    "risk_model": "",
                    "penalty": 0.0,
                    "candidate_alpha_rank_min": 0.0,
                }
            ],
            top_n=2,
        )

        self.assertTrue(np.isfinite(metrics.loc[0, "next_top_mean_bps"]))
        self.assertTrue(np.isfinite(metrics.loc[0, "next_top_excess_bps"]))

    def test_score_variants_can_select_only_from_stock_pool(self) -> None:
        timestamp = pd.Timestamp("2021-08-02 09:31:00")
        frame = pd.DataFrame(
            {
                "date": ["2021-08-02"] * 4,
                "decision_target_timestamp": [timestamp] * 4,
                "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "label": [0.01, 0.08, 0.03, 0.04],
                "alpha_return_next_close": [0.01, 0.08, 0.03, 0.04],
                "candidate_alpha_rank": [0.90, 1.00, 0.80, 0.70],
                "gap_risk_rank": [0.1, 0.2, 0.3, 0.4],
                "binary_risk_rank": [0.1, 0.2, 0.3, 0.4],
                "stock_pool_member": [1, 0, 1, 0],
            }
        )

        metrics = score_variants(
            frame,
            month="2021-08",
            variants=[
                {
                    "variant": "alpha_rank",
                    "risk_model": "",
                    "penalty": 0.0,
                    "candidate_alpha_rank_min": 0.0,
                }
            ],
            top_n=2,
            selection_mask_col="stock_pool_member",
        )

        self.assertEqual(int(metrics.loc[0, "alpha_candidate_rows"]), 4)
        self.assertEqual(int(metrics.loc[0, "stock_pool_candidate_rows"]), 2)
        self.assertEqual(int(metrics.loc[0, "candidate_rows"]), 2)
        self.assertEqual(int(metrics.loc[0, "selected_rows"]), 2)
        self.assertEqual(int(metrics.loc[0, "selected_stock_pool_rows"]), 2)
        self.assertAlmostEqual(float(metrics.loc[0, "short_top_mean_bps"]), 200.0)


if __name__ == "__main__":
    unittest.main()
