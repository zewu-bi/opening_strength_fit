from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from opening_strength_fit.pool_internal_plots import (
    month_major_plot_data,
    write_universe_sml_pool_internal_plots,
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
