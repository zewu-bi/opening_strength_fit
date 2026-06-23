from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from opening_strength_fit.optimization_acceptance_plots import (
    AUTO_COLOR_SEQUENCE,
    add_background_cumulative_data,
    add_cumulative_baseline_relative_data,
    add_cumulative_market_relative_data,
    add_cumulative_percent_display_columns,
    add_market_cumulative_data,
    apply_display_labels,
    combine_net_alpha_cumulative_data,
    default_plot_directions,
    ensure_plot_colors,
    validate_plot_directions,
)
from opening_strength_fit.optimization_direction_data import (
    DEFAULT_POOL_FEE_MODE,
    DEFAULT_REALIZED_FEE_BPS,
    NEXT_CLOSE_CAPITAL_DIVISOR,
    DirectionSpec,
    load_realized_cumulative_plot_data,
)
from opening_strength_fit.pool_internal_plot_svg import PLOT_COLORS


def _direction(key: str) -> DirectionSpec:
    return DirectionSpec(key=key, label=key, run_id=f"{key}_run")


def test_default_plot_directions_selects_current_fixed_models() -> None:
    selected = default_plot_directions()

    assert [item.key for item in selected] == ["hist_surprise", "path_shape"]


def test_hist_path_display_labels_use_deviation_language() -> None:
    data = pd.DataFrame(
        {
            "pool": ["hist_path", "hist_path_zscore", "rank_centered"],
            "pool_label": ["hist_path", "hist_path_zscore", "rank_centered"],
            "variant": ["hist_path", "hist_path_zscore", "rank_centered"],
        }
    )

    out = apply_display_labels(data)

    assert out["pool_label"].tolist() == [
        "deviation+path",
        "deviation+path zscore",
        "deviation+path rank",
    ]
    assert out["variant"].tolist() == out["pool_label"].tolist()


def test_default_realized_fee_and_pool_fee_mode_match_acceptance_assumptions() -> None:
    assert DEFAULT_REALIZED_FEE_BPS == 8.0
    assert DEFAULT_POOL_FEE_MODE == "stock_pool_membership"
    assert NEXT_CLOSE_CAPITAL_DIVISOR == 2.0


def test_validate_plot_directions_accepts_one_to_three_models() -> None:
    one = validate_plot_directions((_direction("a"),))
    two = validate_plot_directions((_direction("a"), _direction("b")))
    three = validate_plot_directions((_direction("a"), _direction("b"), _direction("c")))

    assert [item.key for item in one] == ["a"]
    assert [item.key for item in two] == ["a", "b"]
    assert [item.key for item in three] == ["a", "b", "c"]


def test_validate_plot_directions_rejects_bad_counts_and_reserved_keys() -> None:
    with pytest.raises(ValueError, match="1-3 comparison models"):
        validate_plot_directions(())

    with pytest.raises(ValueError, match="reserved"):
        validate_plot_directions((_direction("market"), _direction("b")))


def test_ensure_plot_colors_assigns_comparison_models_by_order() -> None:
    PLOT_COLORS["first_model"] = "#000000"
    PLOT_COLORS["second_model"] = "#000000"

    ensure_plot_colors(("baseline", "first_model", "first_model", "second_model"))

    assert PLOT_COLORS["first_model"] == AUTO_COLOR_SEQUENCE[0]
    assert PLOT_COLORS["second_model"] == AUTO_COLOR_SEQUENCE[1]


def test_realized_cumulative_uses_capital_adjusted_cumsum_and_round_trip_pool_fee(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "baseline_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L"],
            "date": ["2024-01-02", "2024-01-03"],
            "candidate_rows": [1000.0, 1000.0],
            "selected_rows": [100.0, 100.0],
            "pool_short_mean_bps": [0.0, 0.0],
            "selected_short_mean_bps": [0.0, 0.0],
            "short_internal_excess_bps": [0.0, 0.0],
            "pool_next_mean_bps": [10.0, 20.0],
            "selected_next_mean_bps": [100.0, 100.0],
            "next_internal_excess_bps": [90.0, 80.0],
        }
    ).to_csv(run_dir / "daily_pool_internal_summary.csv", index=False)

    realized = load_realized_cumulative_plot_data(
        backtests_root=tmp_path,
        directions=(),
        pool="pool_L",
        include_baseline_pool=True,
        include_baseline_universe=False,
        baseline_run_id="baseline_run",
        fee_bps=8.0,
        pool_fee_mode="round_trip",
    )

    first = realized.iloc[0]
    second = realized.iloc[1]
    assert first["pool_turnover"] == pytest.approx(1.0)
    assert first["pool_fee_bps"] == pytest.approx(8.0)
    assert second["pool_turnover"] == pytest.approx(1.0)
    assert second["pool_fee_bps"] == pytest.approx(8.0)
    assert first["next_net_return_bps"] == pytest.approx(92.0)
    assert first["next_capital_net_return_bps"] == pytest.approx(46.0)
    assert second["next_cumulative_net_return_bps"] == pytest.approx(92.0)
    assert second["next_cumulative_internal_excess_return_bps"] == pytest.approx(85.0)

    cumulative = combine_net_alpha_cumulative_data(realized)
    with_background = add_background_cumulative_data(
        cumulative,
        baseline_key="baseline_pool_l",
    )
    background = with_background.loc[with_background["pool"].eq("background")].iloc[0]
    assert background["pool_turnover_source"] == "daily_label_round_trip"
    assert background["fee_bps"] == pytest.approx(8.0)
    assert background["next_net_return_bps"] == pytest.approx(2.0)
    assert background["next_capital_net_return_bps"] == pytest.approx(
        first["pool_next_capital_net_return_bps"]
    )
    assert background["next_cumulative_net_return_bps"] == pytest.approx(
        first["pool_next_cumulative_net_return_bps"]
    )
    assert background["next_cumulative_internal_excess_return_bps"] == pytest.approx(0.0)


def test_baseline_relative_curve_uses_capital_adjusted_cumulative_difference() -> None:
    data = pd.DataFrame(
        {
            "pool": ["baseline_pool_l", "baseline_pool_l", "model", "model"],
            "week_start": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
            "next_net_return_bps": [1000.0, 1000.0, 2000.0, 0.0],
            "next_cumulative_net_return_bps": [1000.0, 2000.0, 2000.0, 2000.0],
            "next_capital_internal_excess_bps": [100.0, 100.0, 200.0, 0.0],
            "next_cumulative_internal_excess_return_bps": [100.0, 200.0, 200.0, 200.0],
        }
    )

    out = add_cumulative_baseline_relative_data(
        data,
        baseline_key="baseline_pool_l",
        comparison_keys=("model",),
    )

    model = out.loc[out["pool"].eq("model")].sort_values("week_start")
    assert model["next_vs_baseline_bps"].tolist() == pytest.approx([1000.0, -1000.0])
    assert model["next_cumulative_vs_baseline_bps"].tolist() == pytest.approx([1000.0, 0.0])
    assert model["next_internal_excess_vs_baseline_bps"].tolist() == pytest.approx([100.0, -100.0])
    assert model["next_cumulative_internal_excess_vs_baseline_bps"].tolist() == pytest.approx(
        [100.0, 0.0]
    )


def test_market_cumulative_uses_universe_average_without_fee() -> None:
    data = pd.DataFrame(
        {
            "pool": ["baseline_universe", "baseline_universe", "baseline_pool_l"],
            "pool_label": ["baseline universe", "baseline universe", "baseline"],
            "variant": ["baseline universe", "baseline universe", "baseline"],
            "week_start": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "pool_next_mean_bps": [10.0, 30.0, 5.0],
            "pool_next_net_return_bps": [9.0, 29.0, 4.0],
            "pool_next_capital_net_return_bps": [4.5, 14.5, 2.0],
            "pool_next_cumulative_net_return_bps": [4.5, 19.0, 2.0],
            "selected_next_mean_bps": [50.0, 60.0, 20.0],
            "selected_turnover": [1.0, 1.0, 1.0],
            "selected_fee_bps": [8.0, 8.0, 8.0],
            "pool_turnover": [0.1, 0.1, 0.1],
            "pool_turnover_source": ["estimated", "estimated", "estimated"],
            "pool_fee_bps": [1.0, 1.0, 1.0],
            "fee_bps": [8.0, 8.0, 8.0],
            "next_internal_excess_bps": [40.0, 30.0, 15.0],
            "next_net_return_bps": [42.0, 52.0, 12.0],
            "next_capital_net_return_bps": [21.0, 26.0, 6.0],
            "next_cumulative_net_return_bps": [21.0, 47.0, 6.0],
            "next_alpha_bps": [33.0, 23.0, 8.0],
            "next_capital_alpha_bps": [16.5, 11.5, 4.0],
            "next_cumulative_alpha_bps": [16.5, 28.0, 4.0],
            "next_capital_internal_excess_bps": [20.0, 15.0, 7.5],
            "next_cumulative_internal_excess_return_bps": [20.0, 35.0, 7.5],
        }
    )

    out = add_market_cumulative_data(data)

    assert "baseline_universe" not in out["pool"].tolist()
    market = out.loc[out["pool"].eq("market")].sort_values("week_start")
    assert market["pool_label"].tolist() == ["market", "market"]
    assert market["fee_bps"].tolist() == pytest.approx([0.0, 0.0])
    assert market["next_net_return_bps"].tolist() == pytest.approx([10.0, 30.0])
    assert market["next_capital_net_return_bps"].tolist() == pytest.approx([5.0, 15.0])
    assert market["next_cumulative_net_return_bps"].tolist() == pytest.approx([5.0, 20.0])


def test_market_relative_alpha_uses_full_market_cumulative_difference() -> None:
    data = pd.DataFrame(
        {
            "pool": ["market", "market", "background", "background", "baseline_pool_l"],
            "week_start": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-02",
                "2024-01-03",
                "2024-01-02",
            ],
            "next_net_return_bps": [10.0, 30.0, 8.0, 34.0, 20.0],
            "next_capital_net_return_bps": [5.0, 15.0, 4.0, 17.0, 10.0],
            "next_cumulative_net_return_bps": [5.0, 20.0, 4.0, 21.0, 10.0],
        }
    )

    out = add_cumulative_market_relative_data(
        data,
        market_key="market",
        comparison_keys=("background", "baseline_pool_l"),
    )

    background = out.loc[out["pool"].eq("background")].sort_values("week_start")
    baseline = out.loc[out["pool"].eq("baseline_pool_l")]
    market = out.loc[out["pool"].eq("market")]
    assert background["next_alpha_vs_market_bps"].tolist() == pytest.approx([-2.0, 4.0])
    assert background["next_capital_alpha_vs_market_bps"].tolist() == pytest.approx([-1.0, 2.0])
    assert background["next_cumulative_alpha_vs_market_bps"].tolist() == pytest.approx([-1.0, 1.0])
    assert baseline["next_cumulative_alpha_vs_market_bps"].tolist() == pytest.approx([5.0])
    assert market["next_cumulative_alpha_vs_market_bps"].isna().all()


def test_cumulative_percent_display_columns_preserve_source_bps() -> None:
    data = pd.DataFrame(
        {
            "next_cumulative_net_return_bps": [1234.0, -50.0],
            "next_cumulative_vs_baseline_bps": [321.0, -25.0],
            "next_cumulative_alpha_vs_market_bps": [111.0, -10.0],
        }
    )

    out = add_cumulative_percent_display_columns(data)

    assert out["next_cumulative_net_return_bps"].tolist() == pytest.approx([1234.0, -50.0])
    assert out["next_cumulative_vs_baseline_bps"].tolist() == pytest.approx([321.0, -25.0])
    assert out["next_cumulative_net_return_pct"].tolist() == pytest.approx([12.34, -0.5])
    assert out["next_cumulative_vs_baseline_pct"].tolist() == pytest.approx([3.21, -0.25])
    assert out["next_cumulative_alpha_vs_market_pct"].tolist() == pytest.approx([1.11, -0.1])


def test_realized_cumulative_can_use_stock_pool_membership_fee(tmp_path: Path) -> None:
    run_dir = tmp_path / "baseline_run"
    run_dir.mkdir()
    pool_path = tmp_path / "pool_L.parquet"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "000001.SZ": [True, True, False],
            "000002.SZ": [True, True, True],
            "000003.SZ": [False, False, True],
        }
    ).to_parquet(pool_path, index=False)
    pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L"],
            "date": ["2024-01-02", "2024-01-03"],
            "candidate_rows": [1000.0, 1000.0],
            "selected_rows": [100.0, 100.0],
            "pool_short_mean_bps": [0.0, 0.0],
            "selected_short_mean_bps": [0.0, 0.0],
            "short_internal_excess_bps": [0.0, 0.0],
            "pool_next_mean_bps": [10.0, 20.0],
            "selected_next_mean_bps": [100.0, 100.0],
            "next_internal_excess_bps": [90.0, 80.0],
        }
    ).to_csv(run_dir / "daily_pool_internal_summary.csv", index=False)

    realized = load_realized_cumulative_plot_data(
        backtests_root=tmp_path,
        directions=(),
        pool="pool_L",
        include_baseline_pool=True,
        include_baseline_universe=False,
        baseline_run_id="baseline_run",
        fee_bps=8.0,
        pool_turnover_path=pool_path,
        pool_fee_mode="stock_pool_membership",
    )

    first = realized.iloc[0]
    second = realized.iloc[1]
    assert first["pool_turnover_source"] == "stock_pool_membership"
    assert first["pool_fee_bps"] == pytest.approx(0.0)
    assert second["pool_turnover"] == pytest.approx(0.5)
    assert second["pool_fee_bps"] == pytest.approx(4.0)
