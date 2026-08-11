from __future__ import annotations

import shutil
from pathlib import Path


def _plot_trace_path(plot_data_path: Path) -> Path:
    name = plot_data_path.name
    if name.endswith("_plot_data.csv"):
        return plot_data_path.with_name(name.replace("_plot_data.csv", "_trace.json"))
    return plot_data_path.with_suffix(".json")


def record_pool_internal_outputs(
    *,
    output_dir: Path,
    records_dir: Path,
    record_prefix: str,
    report_plots: dict[str, str],
    record_subdir: str = "",
) -> list[Path]:
    backtests_dir = records_dir / "backtests" / record_subdir

    def record_name(suffix: str) -> str:
        return suffix if record_subdir else f"{record_prefix}_{suffix}"

    records = [
        (output_dir / "pool_internal_summary.csv", record_name("pool_internal_summary.csv")),
        (
            output_dir / "pool_internal_quarter_summary.csv",
            record_name("pool_internal_quarter_summary.csv"),
        ),
        (
            output_dir / "daily_pool_internal_summary.csv",
            record_name("daily_pool_internal_summary.csv"),
        ),
        (
            output_dir / "pool_internal_month_summary.csv",
            record_name("pool_internal_month_summary.csv"),
        ),
        (
            output_dir / "pool_internal_clock_summary.csv",
            record_name("pool_internal_clock_summary.csv"),
        ),
        (
            output_dir / "pool_internal_group_metrics.csv",
            record_name("pool_internal_group_metrics.csv"),
        ),
        (
            output_dir / "pool_internal_halfyear_summary.csv",
            record_name("pool_internal_halfyear_summary.csv"),
        ),
        (
            output_dir / "pool_internal_year_summary.csv",
            record_name("pool_internal_year_summary.csv"),
        ),
        (output_dir / "pool_internal_trace.json", record_name("pool_internal_trace.json")),
    ]
    plot_records = {
        "pool_internal_plot_data": record_name("pool_internal_plot_data.csv"),
        "pool_internal_figure": record_name("pool_internal_with_mean.svg"),
        "rank_ic_plot_data": record_name("rank_ic_plot_data.csv"),
        "rank_ic_figure": record_name("rank_ic_with_mean.svg"),
        "short_excess_rank_ic_plot_data": record_name("short_excess_rank_ic_plot_data.csv"),
        "short_excess_rank_ic_figure": record_name("short_excess_rank_ic_with_mean.svg"),
        "next_excess_rank_ic_plot_data": record_name("next_excess_rank_ic_plot_data.csv"),
        "next_excess_rank_ic_figure": record_name("next_excess_rank_ic_with_mean.svg"),
        "daily_cumulative_plot_data": record_name("daily_cumulative_plot_data.csv"),
        "daily_cumulative_figure": record_name("daily_cumulative.svg"),
        "company_backtest_plot_data": record_name("company_backtest_plot_data.csv"),
        "company_backtest_figure": record_name("company_backtest.svg"),
    }
    for key, name in plot_records.items():
        if key in report_plots:
            records.append((Path(report_plots[key]), name))
    for key, name in plot_records.items():
        if not key.endswith("_plot_data") or key not in report_plots:
            continue
        trace_path = _plot_trace_path(Path(report_plots[key]))
        if trace_path.exists():
            trace_name = name.replace("_plot_data.csv", "_trace.json")
            if trace_name == "pool_internal_trace.json" or trace_name.endswith(
                "_pool_internal_trace.json"
            ):
                trace_name = record_name("pool_internal_with_mean_trace.json")
            records.append((trace_path, trace_name))

    copied: list[Path] = []
    for source, name in records:
        if not source.exists():
            continue
        destination = backtests_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied
