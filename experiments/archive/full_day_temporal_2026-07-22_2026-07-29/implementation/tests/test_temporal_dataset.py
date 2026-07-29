from __future__ import annotations

import numpy as np
import pytest

from opening_strength_fit.temporal_dataset import (
    aligned_sequence_validity,
    daily_rank_ic,
    eligible_feature_mask,
    prepare_day_inputs,
    rolling_date_bounds,
    target_rank,
    target_values,
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


def test_cross_sequence_mask_swaps_only_mask_channels_by_symbol_and_slot() -> None:
    arrays = _arrays()
    mask_values = np.full((2, 3, 2), np.nan, dtype=np.float32)
    mask_values[0, :, 1] = 1.0
    mask_values[1, :, 0] = 1.0
    mask_source = {
        "symbols": np.array(["c", "a"]),
        "clock_seconds": np.array([13 * 3600 + 49 * 60, 14 * 3600 + 48 * 60]),
        "values": mask_values,
    }
    aligned = aligned_sequence_validity(arrays, mask_source)
    assert aligned[0, 0].tolist() == [True, False]
    assert not aligned[1].any()
    assert aligned[2, 0].tolist() == [False, True]

    latest = {"1m": "23:59", "10m": "23:59", "60m": "23:59"}
    native_inputs, _ = prepare_day_inputs(
        arrays,
        value_mode="cross_section_rank",
        latest_clocks=latest,
    )
    crossed_inputs, crossed_time_valid = prepare_day_inputs(
        arrays,
        value_mode="cross_section_rank",
        latest_clocks=latest,
        input_valid_override=aligned,
    )
    np.testing.assert_array_equal(crossed_inputs[:, :3], native_inputs[:, :3])
    np.testing.assert_array_equal(crossed_inputs[:, 3:].astype(bool), aligned)
    assert crossed_time_valid[0].tolist() == [True, False]
    assert not crossed_time_valid[1].any()


def test_prepare_relative_tanh_demeans_each_clock_without_ranking() -> None:
    inputs, _ = prepare_day_inputs(
        _arrays(),
        value_mode="relative_tanh",
        raw_scales={"1m": 0.01, "10m": 0.02, "60m": 0.03},
    )
    expected = np.tanh(np.array([-0.01, 0.0, 0.01]) / 0.01)
    np.testing.assert_allclose(inputs[:, 0, 0], expected, atol=1e-6)


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


def test_market_relative_target_is_centered_and_uses_fixed_winsor_bounds() -> None:
    target = np.array([-0.02, 0.01, 0.07], dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    relative = target_values(target, universe_mask=mask, mode="market_relative")
    assert float(relative.mean()) == pytest.approx(0.0, abs=1e-8)
    winsorized = target_values(
        target,
        universe_mask=mask,
        mode="market_relative_winsor",
        winsor_bounds=(-0.01, 0.02),
    )
    np.testing.assert_allclose(winsorized, [-0.01, -0.01, 0.02], atol=1e-7)


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
