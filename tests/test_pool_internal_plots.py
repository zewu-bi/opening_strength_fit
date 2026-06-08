from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from opening_strength_fit.pool_internal_plots import (
    month_major_plot_data,
    write_universe_sml_pool_internal_plots,
    write_weekly_pool_internal_cumulative_plot,
    write_weekly_pool_internal_rolling_plot,
)


def _month_summary() -> pd.DataFrame:
    rows = []
    for month_index, test_month in enumerate(["2024-01", "2024-02"], start=1):
        for pool_index, pool in enumerate(["universe", "pool_S", "pool_M", "pool_L"], start=1):
            rows.append(
                {
                    "pool": pool,
                    "test_month": test_month,
                    "short_internal_excess_bps": float(month_index * pool_index),
                    "next_internal_excess_bps": float(month_index - pool_index),
                    "short_rank_ic": 0.10 + month_index * 0.01 + pool_index * 0.001,
                    "next_rank_ic": -0.02 + month_index * 0.01 - pool_index * 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_month_major_plot_data_appends_mean() -> None:
    plot_data = month_major_plot_data(
        _month_summary(),
        value_cols=["short_internal_excess_bps", "next_internal_excess_bps"],
        variant_label="baseline",
    )

    assert len(plot_data) == 12
    assert plot_data["test_month"].head(4).tolist() == ["2024-01"] * 4
    assert plot_data["pool"].head(4).tolist() == ["universe", "pool_S", "pool_M", "pool_L"]
    assert plot_data["test_month"].tail(4).tolist() == ["Mean"] * 4
    universe_mean = plot_data.loc[
        plot_data["test_month"].eq("Mean") & plot_data["pool"].eq("universe"),
        "short_internal_excess_bps",
    ].item()
    assert universe_mean == 1.5


def test_write_universe_sml_pool_internal_plots(tmp_path) -> None:
    paths = write_universe_sml_pool_internal_plots(
        _month_summary(),
        tmp_path,
        output_prefix="baseline",
        variant_label="baseline",
    )

    for path in paths.values():
        assert Path(path).exists()

    excess_figure = (
        tmp_path
        / "baseline_universe_sml_pool_internal_with_mean"
        / ("baseline_universe_sml_top100_pool_internal_with_mean.svg")
    )
    rank_figure = (
        tmp_path
        / "baseline_universe_sml_rank_ic_with_mean"
        / ("baseline_universe_sml_rank_ic_with_mean.svg")
    )
    assert excess_figure.exists()
    assert rank_figure.exists()
    assert "\u6c60\u5185\u8d85\u989d" in excess_figure.read_text(encoding="utf-8")
    assert "Rank IC" in rank_figure.read_text(encoding="utf-8")

    trace_path = (
        tmp_path
        / "baseline_universe_sml_rank_ic_with_mean"
        / ("baseline_universe_sml_rank_ic_with_mean_trace.json")
    )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["included_months"] == ["2024-01", "2024-02"]


def test_write_universe_only_pool_internal_plots(tmp_path) -> None:
    month_summary = _month_summary().loc[lambda frame: frame["pool"].eq("universe")]

    paths = write_universe_sml_pool_internal_plots(
        month_summary,
        tmp_path,
        output_prefix="pre2020",
        variant_label="baseline pre2020 universe",
        pools=("universe",),
    )

    for path in paths.values():
        assert Path(path).exists()

    excess_figure = (
        tmp_path
        / "pre2020_universe_pool_internal_with_mean"
        / "pre2020_universe_top100_pool_internal_with_mean.svg"
    )
    assert excess_figure.exists()
    assert "baseline pre2020 universe: universe Top 100" in excess_figure.read_text(
        encoding="utf-8"
    )

    trace_path = (
        tmp_path
        / "pre2020_universe_rank_ic_with_mean"
        / "pre2020_universe_rank_ic_with_mean_trace.json"
    )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["series"] == ["universe"]


def test_write_weekly_pool_internal_rolling_plot(tmp_path) -> None:
    weekly_summary = pd.DataFrame(
        {
            "pool": ["universe", "universe", "pool_S", "pool_S"],
            "week_start": ["2024-01-01", "2024-01-08", "2024-01-01", "2024-01-08"],
            "short_internal_excess_bps_rolling_4w": [10.0, 12.0, 8.0, 9.0],
            "next_internal_excess_bps_rolling_4w": [1.0, -2.0, 3.0, 4.0],
        }
    )

    paths = write_weekly_pool_internal_rolling_plot(
        weekly_summary,
        tmp_path,
        output_prefix="baseline",
        variant_label="baseline weekly",
        pools=("universe", "pool_S"),
        rolling_weeks=4,
    )

    for path in paths.values():
        assert Path(path).exists()

    figure = Path(paths["weekly_rolling_figure"])
    assert "baseline weekly: universe / S Top 100" in figure.read_text(encoding="utf-8")

    trace = json.loads(Path(paths["weekly_rolling_trace"]).read_text(encoding="utf-8"))
    assert trace["rolling_weeks"] == 4
    assert trace["series"] == ["universe", "pool_S"]


def test_write_weekly_pool_internal_cumulative_plot(tmp_path) -> None:
    weekly_summary = pd.DataFrame(
        {
            "pool": ["universe", "universe", "pool_S", "pool_S"],
            "week_start": ["2024-01-01", "2024-01-08", "2024-01-01", "2024-01-08"],
            "short_internal_excess_bps": [10.0, -2.0, 8.0, 3.0],
            "next_internal_excess_bps": [1.0, 4.0, -5.0, 6.0],
            "trading_days": [5, 5, 5, 5],
        }
    )

    paths = write_weekly_pool_internal_cumulative_plot(
        weekly_summary,
        tmp_path,
        output_prefix="baseline",
        output_name="baseline_weekly_cumulative",
        variant_label="baseline cumulative",
        pools=("universe", "pool_S"),
    )

    for path in paths.values():
        assert Path(path).exists()

    plot_data = pd.read_csv(paths["weekly_cumulative_plot_data"])
    universe = plot_data.loc[plot_data["pool"].eq("universe")]
    assert universe["short_cumulative_internal_excess_bps"].tolist() == [10.0, 8.0]
    assert universe["next_cumulative_internal_excess_bps"].tolist() == [1.0, 5.0]

    figure = Path(paths["weekly_cumulative_figure"])
    assert "baseline cumulative: universe / S Top 100" in figure.read_text(encoding="utf-8")

    trace = json.loads(Path(paths["weekly_cumulative_trace"]).read_text(encoding="utf-8"))
    assert trace["metric"] == "weekly_pool_internal_excess_cumulative_sum"
    assert trace["series"] == ["universe", "pool_S"]


def test_write_cumulative_plot_can_label_years_only(tmp_path) -> None:
    daily_summary = pd.DataFrame(
        {
            "pool": ["pool_L", "pool_L", "pool_L"],
            "week_start": ["2022-01-04", "2023-01-03", "2025-12-31"],
            "short_internal_excess_bps": [10.0, 12.0, 8.0],
            "next_internal_excess_bps": [1.0, -2.0, 3.0],
        }
    )

    paths = write_weekly_pool_internal_cumulative_plot(
        daily_summary,
        tmp_path,
        output_name="daily_cumulative",
        variant_label="daily cumulative",
        pools=("pool_L",),
        x_label_mode="years_only",
    )

    figure_text = Path(paths["weekly_cumulative_figure"]).read_text(encoding="utf-8")
    assert "2022-01-04" not in figure_text
    assert "2025-12-31" not in figure_text
    assert ">2022<" in figure_text
    assert ">2025<" in figure_text
