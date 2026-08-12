from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from opening_strength_fit.legacy.multiscale_bucket_diag import spearman_rank_ic
from opening_strength_fit.legacy.rank_bucket_reaudit import (
    build_bucket_diagnostics,
    build_group_ic_diagnostics,
    finalize_curve_data,
    numpy_average_ranks,
    numpy_rank_ic,
)


def test_rank_ic_implementations_match_with_ties() -> None:
    scores = np.array([4.0, 4.0, 3.0, 2.0, 1.0])
    outcomes = np.array([2.0, 1.0, 2.0, 0.0, 0.0])

    expected = spearmanr(scores, outcomes).statistic

    assert spearman_rank_ic(scores, outcomes) == pytest.approx(expected)
    assert numpy_rank_ic(scores, outcomes) == pytest.approx(expected)
    assert numpy_average_ranks(scores) == pytest.approx(np.array([4.5, 4.5, 3.0, 2.0, 1.0]))


def test_group_and_bucket_audit_preserve_direction() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-02"] * 1000,
            "decision_target_timestamp": ["2025-01-02 09:31:00"] * 1000,
            "month": ["2025-01"] * 1000,
            "symbol": [f"{index:06d}.SZ" for index in range(1000)],
            "prediction": np.arange(1000, 0, -1, dtype="float64"),
            "alpha_return_next_close": np.arange(1000, 0, -1, dtype="float64"),
            "excess_bps": np.arange(1000, 0, -1, dtype="float64"),
            "score_rank": np.arange(1, 1001),
            "group_size": [1000] * 1000,
        }
    )
    group_ic = build_group_ic_diagnostics(frame, variant="test")
    bucket_ic, curve_parts = build_bucket_diagnostics(frame, variant="test")
    curves = finalize_curve_data(curve_parts)

    assert group_ic["rank_ic"].to_numpy() == pytest.approx(1.0)
    assert group_ic["independent_rank_ic"].to_numpy() == pytest.approx(1.0)
    assert group_ic["reverse_score_rank_ic"].to_numpy() == pytest.approx(-1.0)
    assert bucket_ic["bucket_rank_ic"].to_numpy() == pytest.approx(1.0)
    assert curves["mean_excess_bps"].notna().all()
    assert curves["mean_within_group_outcome_rank_pct"].notna().all()
