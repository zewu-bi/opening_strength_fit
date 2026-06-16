from __future__ import annotations

import pytest

from opening_strength_fit.optimization_acceptance_plots import (
    default_plot_directions,
    ensure_plot_colors,
    validate_plot_directions,
)
from opening_strength_fit.optimization_direction_data import DEFAULT_REALIZED_FEE_BPS, DirectionSpec
from opening_strength_fit.pool_internal_plot_svg import PLOT_COLORS


def _direction(key: str) -> DirectionSpec:
    return DirectionSpec(key=key, label=key, run_id=f"{key}_run")


def test_default_plot_directions_selects_current_fixed_models() -> None:
    selected = default_plot_directions()

    assert [item.key for item in selected] == ["hist_surprise", "path_shape"]


def test_default_realized_fee_uses_all_in_a_share_round_trip_estimate() -> None:
    assert DEFAULT_REALIZED_FEE_BPS == 8.0


def test_validate_plot_directions_accepts_two_or_three_models() -> None:
    two = validate_plot_directions((_direction("a"), _direction("b")))
    three = validate_plot_directions((_direction("a"), _direction("b"), _direction("c")))

    assert [item.key for item in two] == ["a", "b"]
    assert [item.key for item in three] == ["a", "b", "c"]


def test_validate_plot_directions_rejects_bad_counts_and_reserved_keys() -> None:
    with pytest.raises(ValueError, match="2-3 comparison models"):
        validate_plot_directions((_direction("a"),))

    with pytest.raises(ValueError, match="reserved"):
        validate_plot_directions((_direction("baseline"), _direction("b")))


def test_ensure_plot_colors_assigns_unknown_model_key() -> None:
    PLOT_COLORS.pop("new_model", None)

    ensure_plot_colors(("new_model",))

    assert "new_model" in PLOT_COLORS
