from __future__ import annotations

import numpy as np
import pytest

from opening_strength_fit.temporal_dataset import (
    daily_rank_ic,
    eligible_feature_mask,
    prepare_day_inputs,
    rolling_date_bounds,
    target_rank,
    top_n_excess,
    universe_mask,
)


def _arrays() -> dict[str, np.ndarray]:
    values = np.array(
        [
            [[0.01, 0.03], [0.02, 0.04], [0.03, 0.05]],
            [[0.02, 0.02], [0.04, 0.03], [0.06, 0.04]],
            [[0.03, np.nan], [0.06, 0.02], [0.09, 0.03]],
        ],
        dtype=np.float32,
    )
    return {
        "symbols": np.array(["a", "b", "c"]),
        "clock_seconds": np.array([13 * 3600 + 48 * 60, 14 * 3600 + 47 * 60]),
        "values": values,
        "valid": np.isfinite(values),
        "target": np.array([-0.01, 0.00, 0.02], dtype=np.float32),
        "pool_member": np.array([False, True, True]),
    }


def test_feature_cutoffs_are_horizon_specific() -> None:
    valid = eligible_feature_mask(_arrays())
    assert valid[:, 0, 1].tolist() == [True, True, False]
    assert not valid[:, 1, 1].any()
    assert not valid[:, 2, 1].any()


def test_prepare_cross_section_rank_adds_valid_channels() -> None:
    inputs, time_valid = prepare_day_inputs(
        _arrays(),
        value_mode="cross_section_rank",
    )
    assert inputs.shape == (3, 6, 2)
    assert time_valid.shape == (3, 2)
    assert np.isfinite(inputs).all()
    np.testing.assert_allclose(inputs[:, 0, 0], [-1.0, 0.0, 1.0], atol=1e-7)


def test_targets_universes_and_metrics() -> None:
    arrays = _arrays()
    pool = universe_mask(arrays, "pool_l")
    ranked = target_rank(arrays["target"], universe_mask=pool)
    assert np.isnan(ranked[0])
    np.testing.assert_allclose(ranked[1:], [-1.0, 1.0])
    scores = np.array([-1.0, 0.0, 1.0])
    assert daily_rank_ic(scores, arrays["target"], np.ones(3, dtype=bool)) == pytest.approx(1.0)
    assert (
        top_n_excess(
            scores,
            arrays["target"],
            np.ones(3, dtype=bool),
            top_n=1,
        )
        > 0
    )


def test_rolling_bounds_reserve_validation_tail() -> None:
    bounds = rolling_date_bounds(
        test_start_month="2022-01",
        test_end_month="2022-06",
        train_months=36,
        validation_months=3,
    )
    assert bounds == {
        "train_start_date": "2019-01-01",
        "train_end_date": "2021-09-30",
        "validation_start_date": "2021-10-01",
        "validation_end_date": "2021-12-31",
        "test_start_date": "2022-01-01",
        "test_end_date": "2022-06-30",
    }
