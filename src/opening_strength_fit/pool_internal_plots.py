from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import write_json
from opening_strength_fit.pool_internal_plot_metadata import (
    pool_group_slug as _pool_group_slug,
)
from opening_strength_fit.pool_internal_plot_metadata import (
    pool_group_title as _pool_group_title,
)
from opening_strength_fit.pool_internal_plot_metadata import slug_label as slug_label
from opening_strength_fit.pool_internal_plot_metadata import (
    summary_months as _summary_months,
)
from opening_strength_fit.pool_internal_plot_metadata import (
    write_plot_trace as _write_plot_trace,
)
from opening_strength_fit.pool_internal_plot_svg import FONT_FAMILY as FONT_FAMILY
from opening_strength_fit.pool_internal_plot_svg import PLOT_COLORS as PLOT_COLORS
from opening_strength_fit.pool_internal_plot_svg import PLOT_POOLS
from opening_strength_fit.pool_internal_plot_svg import nice_line_axis as _nice_line_axis
from opening_strength_fit.pool_internal_plot_svg import (
    write_two_panel_bar_svg as _write_two_panel_bar_svg,
)
from opening_strength_fit.pool_internal_plot_svg import (
    write_two_panel_line_svg as _write_two_panel_line_svg,
)


def month_major_plot_data(
    month_summary: pd.DataFrame,
    *,
    value_cols: list[str],
    pools: tuple[str, ...] = PLOT_POOLS,
    variant_label: str = "baseline",
) -> pd.DataFrame:
    required = {"pool", "test_month", *value_cols}
    missing = sorted(required - set(month_summary.columns))
    if missing:
        raise ValueError(f"month_summary missing columns: {missing}")

    months = sorted(month_summary["test_month"].astype(str).unique())
    records: list[dict[str, object]] = []
    for test_month in months:
        item = month_summary.loc[month_summary["test_month"].astype(str).eq(test_month)]
        item = item.set_index("pool")
        for pool in pools:
            if pool not in item.index:
                raise ValueError(f"month_summary missing pool {pool!r} for {test_month}")
            record: dict[str, object] = {
                "test_month": test_month,
                "variant": variant_label,
                "pool": pool,
                "pool_label": pool,
            }
            for column in value_cols:
                record[column] = float(item.loc[pool, column])
            records.append(record)

    for pool in pools:
        item = month_summary.loc[month_summary["pool"].eq(pool)]
        if item.empty:
            raise ValueError(f"month_summary missing pool {pool!r}")
        record = {
            "test_month": "Mean",
            "variant": variant_label,
            "pool": pool,
            "pool_label": pool,
        }
        for column in value_cols:
            record[column] = float(item[column].mean())
        records.append(record)
    return pd.DataFrame(records)


def write_universe_sml_pool_internal_plots(
    month_summary: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path | None = None,
    output_prefix: str = "baseline",
    variant_label: str = "baseline",
    pools: tuple[str, ...] = PLOT_POOLS,
) -> dict[str, str]:
    output_prefix = slug_label(output_prefix)
    pools = tuple(pools)
    pool_group_slug = _pool_group_slug(pools)
    pool_group_title = _pool_group_title(pools)
    output_dir.mkdir(parents=True, exist_ok=True)

    excess_data = month_major_plot_data(
        month_summary,
        value_cols=["short_internal_excess_bps", "next_internal_excess_bps"],
        pools=pools,
        variant_label=variant_label,
    ).rename(
        columns={
            "short_internal_excess_bps": "pool_internal_short_excess_bps",
            "next_internal_excess_bps": "pool_internal_next_excess_bps",
        }
    )
    excess_dir = output_dir / f"{output_prefix}_{pool_group_slug}_pool_internal_with_mean"
    excess_plot_data = (
        excess_dir / f"{output_prefix}_{pool_group_slug}_pool_internal_with_mean_plot_data.csv"
    )
    excess_figure = (
        excess_dir / f"{output_prefix}_{pool_group_slug}_top100_pool_internal_with_mean.svg"
    )
    excess_dir.mkdir(parents=True, exist_ok=True)
    excess_data.to_csv(excess_plot_data, index=False, float_format="%.6f")
    _write_two_panel_bar_svg(
        excess_data,
        title=f"{variant_label}: {pool_group_title} Top 100 \u6c60\u5185\u8d85\u989d",
        panels=[
            {
                "title": "\u77ed\u671f\u6536\u76ca",
                "ylabel": "bps",
                "column": "pool_internal_short_excess_bps",
                "default_ylim": (-5.0, 40.0),
                "tick_step": 5.0,
                "tick_decimals": None,
                "label_decimals": 1,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 6,
                "min_tick_step": 5.0,
            },
            {
                "title": "\u9694\u591c\u6536\u76ca",
                "ylabel": "bps",
                "column": "pool_internal_next_excess_bps",
                "default_ylim": (-80.0, 100.0),
                "tick_step": 10.0,
                "tick_decimals": None,
                "label_decimals": 1,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 6,
                "min_tick_step": 5.0,
            },
        ],
        output_path=excess_figure,
        pools=pools,
    )
    _write_plot_trace(
        excess_dir / f"{output_prefix}_{pool_group_slug}_pool_internal_with_mean_trace.json",
        input_path=input_path,
        plot_data=excess_plot_data,
        figure=excess_figure,
        variant_label=variant_label,
        included_months=_summary_months(month_summary),
        metric="pool_internal_top100_excess_bps",
        pools=pools,
    )

    rank_data = month_major_plot_data(
        month_summary,
        value_cols=["short_rank_ic", "next_rank_ic"],
        pools=pools,
        variant_label=variant_label,
    )
    rank_dir = output_dir / f"{output_prefix}_{pool_group_slug}_rank_ic_with_mean"
    rank_plot_data = rank_dir / f"{output_prefix}_{pool_group_slug}_rank_ic_with_mean_plot_data.csv"
    rank_figure = rank_dir / f"{output_prefix}_{pool_group_slug}_rank_ic_with_mean.svg"
    rank_dir.mkdir(parents=True, exist_ok=True)
    rank_data.to_csv(rank_plot_data, index=False, float_format="%.6f")
    _write_two_panel_bar_svg(
        rank_data,
        title=f"{variant_label}: {pool_group_title} Rank IC",
        panels=[
            {
                "title": "\u77ed\u671f Rank IC",
                "ylabel": "IC",
                "column": "short_rank_ic",
                "default_ylim": (0.0, 0.22),
                "tick_step": 0.02,
                "tick_decimals": 2,
                "label_decimals": 3,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 10,
                "min_tick_step": 0.01,
            },
            {
                "title": "\u9694\u591c Rank IC",
                "ylabel": "IC",
                "column": "next_rank_ic",
                "default_ylim": (-0.08, 0.10),
                "tick_step": 0.02,
                "tick_decimals": 2,
                "label_decimals": 3,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 10,
                "min_tick_step": 0.01,
            },
        ],
        output_path=rank_figure,
        pools=pools,
    )
    _write_plot_trace(
        rank_dir / f"{output_prefix}_{pool_group_slug}_rank_ic_with_mean_trace.json",
        input_path=input_path,
        plot_data=rank_plot_data,
        figure=rank_figure,
        variant_label=variant_label,
        included_months=_summary_months(month_summary),
        metric="rank_ic",
        pools=pools,
    )

    horizon_plots = _write_horizon_excess_rank_ic_plots(
        month_summary,
        output_dir,
        input_path=input_path,
        output_prefix=output_prefix,
        variant_label=variant_label,
        pools=pools,
        pool_group_slug=pool_group_slug,
        pool_group_title=pool_group_title,
    )

    return {
        "pool_internal_plot_data": str(excess_plot_data),
        "pool_internal_figure": str(excess_figure),
        "rank_ic_plot_data": str(rank_plot_data),
        "rank_ic_figure": str(rank_figure),
        **horizon_plots,
    }


def write_weekly_pool_internal_rolling_plot(
    weekly_summary: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path | None = None,
    output_prefix: str = "baseline",
    variant_label: str = "baseline",
    pools: tuple[str, ...] = PLOT_POOLS,
    rolling_weeks: int = 4,
) -> dict[str, str]:
    required = {
        "pool",
        "week_start",
        f"short_internal_excess_bps_rolling_{rolling_weeks}w",
        f"next_internal_excess_bps_rolling_{rolling_weeks}w",
    }
    missing = sorted(required - set(weekly_summary.columns))
    if missing:
        raise ValueError(f"weekly_summary missing columns: {missing}")

    output_prefix = slug_label(output_prefix)
    pools = tuple(pools)
    pool_group_slug = _pool_group_slug(pools)
    pool_group_title = _pool_group_title(pools)
    output_dir.mkdir(parents=True, exist_ok=True)

    short_col = f"short_internal_excess_bps_rolling_{rolling_weeks}w"
    next_col = f"next_internal_excess_bps_rolling_{rolling_weeks}w"
    plot_data = weekly_summary.loc[
        weekly_summary["pool"].isin(pools),
        ["pool", "week_start", short_col, next_col],
    ].copy()
    plot_data = plot_data.rename(
        columns={
            short_col: "short_internal_excess_bps_rolling",
            next_col: "next_internal_excess_bps_rolling",
        }
    )
    plot_data["variant"] = variant_label
    plot_data["week_start"] = pd.to_datetime(plot_data["week_start"]).dt.strftime("%Y-%m-%d")

    chart_dir = output_dir / f"{output_prefix}_{pool_group_slug}_weekly_rolling_{rolling_weeks}w"
    plot_data_path = (
        chart_dir
        / f"{output_prefix}_{pool_group_slug}_weekly_rolling_{rolling_weeks}w_plot_data.csv"
    )
    figure = chart_dir / f"{output_prefix}_{pool_group_slug}_weekly_rolling_{rolling_weeks}w.svg"
    trace = (
        chart_dir / f"{output_prefix}_{pool_group_slug}_weekly_rolling_{rolling_weeks}w_trace.json"
    )
    chart_dir.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(plot_data_path, index=False, float_format="%.6f")

    _write_two_panel_line_svg(
        plot_data,
        title=(
            f"{variant_label}: {pool_group_title} Top 100 \u6c60\u5185\u8d85\u989d "
            f"{rolling_weeks}\u5468\u6eda\u52a8"
        ),
        panels=[
            {
                "title": f"\u77ed\u671f\u6536\u76ca {rolling_weeks}\u5468\u6eda\u52a8",
                "ylabel": "bps",
                "column": "short_internal_excess_bps_rolling",
                "default_ylim": (-10.0, 50.0),
                "tick_step": 10.0,
                "tick_decimals": None,
            },
            {
                "title": f"\u9694\u591c\u6536\u76ca {rolling_weeks}\u5468\u6eda\u52a8",
                "ylabel": "bps",
                "column": "next_internal_excess_bps_rolling",
                "default_ylim": (-120.0, 160.0),
                "tick_step": 40.0,
                "tick_decimals": None,
            },
        ],
        output_path=figure,
        pools=pools,
    )
    write_json(
        trace,
        {
            "input": str(input_path) if input_path is not None else None,
            "plot_data": str(plot_data_path),
            "figure": str(figure),
            "variant_label": variant_label,
            "series": list(pools),
            "included_weeks": sorted(plot_data["week_start"].dropna().astype(str).unique()),
            "rolling_weeks": rolling_weeks,
            "weighting": "weekly_summary precomputed by caller; intended CLI uses trading-day equal rolling windows",
            "metric": "weekly_pool_internal_excess_rolling",
            "style": "manual svg two-panel weekly line figure for selected pools",
        },
        ensure_ascii=True,
    )
    return {
        "weekly_rolling_plot_data": str(plot_data_path),
        "weekly_rolling_figure": str(figure),
        "weekly_rolling_trace": str(trace),
    }


def write_weekly_pool_internal_cumulative_plot(
    weekly_summary: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path | None = None,
    output_prefix: str = "baseline",
    output_name: str = "",
    variant_label: str = "baseline",
    pools: tuple[str, ...] = PLOT_POOLS,
    x_label_mode: str = "dates_at_ends",
) -> dict[str, str]:
    return _write_cumulative_pool_internal_plot(
        weekly_summary,
        output_dir,
        input_path=input_path,
        output_prefix=output_prefix,
        output_name=output_name,
        variant_label=variant_label,
        pools=pools,
        x_label_mode=x_label_mode,
        date_column="week_start",
        error_label="weekly_summary",
        optional_columns=("trading_days",),
        default_stem_suffix="weekly_cumulative",
        result_prefix="weekly_cumulative",
        title_suffix="\u7d2f\u548c",
        included_key="included_weeks",
        metric="weekly_pool_internal_excess_cumulative_sum",
        style="manual svg two-panel cumulative weekly line figure for selected pools",
        cumulative_definition=(
            "per-pool cumulative sum of weekly short/next internal excess bps; "
            "weekly rows are expected to be precomputed by the caller"
        ),
    )


def write_daily_pool_internal_cumulative_plot(
    daily_summary: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path | None = None,
    output_prefix: str = "baseline",
    output_name: str = "",
    variant_label: str = "baseline",
    pools: tuple[str, ...] = PLOT_POOLS,
    x_label_mode: str = "years_only",
) -> dict[str, str]:
    return _write_cumulative_pool_internal_plot(
        daily_summary,
        output_dir,
        input_path=input_path,
        output_prefix=output_prefix,
        output_name=output_name,
        variant_label=variant_label,
        pools=pools,
        x_label_mode=x_label_mode,
        date_column="date",
        error_label="daily_summary",
        round_inputs=True,
        default_stem_suffix="daily_cumulative",
        result_prefix="daily_cumulative",
        title_suffix="\u65e5\u5ea6\u7d2f\u548c",
        included_key="included_dates",
        metric="daily_pool_internal_excess_cumulative_sum",
        style="manual svg two-panel cumulative daily line figure for selected pools",
        cumulative_definition=(
            "per-pool cumulative sum of daily short/next internal excess bps; "
            "daily rows are averaged across decision clocks before plotting"
        ),
    )


def _write_company_backtest_plot(
    plot_data: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path | None,
    output_prefix: str,
    output_name: str,
    variant_label: str,
    pools: tuple[str, ...] | None,
    series_colors: dict[str, str] | None,
    x_label_mode: str,
    second_column: str,
    error_label: str,
    palette: tuple[str, ...],
    default_stem_suffix: str,
    title: str,
    second_panel_title: str,
    result_prefix: str,
    metric: str,
    style: str,
    cumulative_definition: str,
) -> dict[str, str]:
    required = {"pool", "week_start", "profit_cumulative_bps", second_column}
    missing = sorted(required - set(plot_data.columns))
    if missing:
        raise ValueError(f"{error_label} missing columns: {missing}")

    output_prefix = slug_label(output_prefix)
    data = plot_data.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], errors="coerce")
    data = data.dropna(subset=["week_start"]).sort_values(["pool", "week_start"])
    if data.empty:
        raise ValueError(f"{error_label.removesuffix(' missing columns:')} is empty")

    pools = tuple(pools or data["pool"].dropna().astype(str).drop_duplicates().tolist())
    series_colors = dict(series_colors or {})
    for index, pool in enumerate(pools):
        PLOT_COLORS.setdefault(pool, series_colors.get(pool, palette[index % len(palette)]))

    (profit_ylim, profit_step) = _nice_line_axis(
        data["profit_cumulative_bps"],
        include_zero=True,
        target_ticks=9,
    )
    (second_ylim, second_step) = _nice_line_axis(
        data[second_column],
        include_zero=True,
        target_ticks=9,
    )

    file_stem = slug_label(output_name) if output_name else f"{output_prefix}_{default_stem_suffix}"
    chart_dir = output_dir if output_name else output_dir / file_stem
    plot_data_path = chart_dir / f"{file_stem}_plot_data.csv"
    figure = chart_dir / f"{file_stem}.svg"
    trace = chart_dir / f"{file_stem}_trace.json"
    chart_dir.mkdir(parents=True, exist_ok=True)
    csv_data = data.copy()
    csv_data["week_start"] = csv_data["week_start"].dt.strftime("%Y-%m-%d")
    csv_data.to_csv(plot_data_path, index=False, float_format="%.6f")

    _write_two_panel_line_svg(
        csv_data,
        title=f"{variant_label} {title}",
        panels=[
            {
                "title": "收益累和",
                "ylabel": "bps",
                "column": "profit_cumulative_bps",
                "default_ylim": profit_ylim,
                "tick_step": profit_step,
                "tick_decimals": None,
                "fixed_ylim": True,
            },
            {
                "title": second_panel_title,
                "ylabel": "bps",
                "column": second_column,
                "default_ylim": second_ylim,
                "tick_step": second_step,
                "tick_decimals": None,
                "fixed_ylim": True,
            },
        ],
        output_path=figure,
        pools=pools,
        x_label_mode=x_label_mode,
    )
    write_json(
        trace,
        {
            "input": str(input_path) if input_path is not None else None,
            "plot_data": str(plot_data_path),
            "figure": str(figure),
            "variant_label": variant_label,
            "series": list(pools),
            "included_dates": sorted(csv_data["week_start"].dropna().astype(str).unique()),
            "metric": metric,
            "style": style,
            "cumulative_definition": cumulative_definition,
            "x_label_mode": x_label_mode,
        },
        ensure_ascii=True,
    )
    return {
        f"{result_prefix}_plot_data": str(plot_data_path),
        f"{result_prefix}_figure": str(figure),
        f"{result_prefix}_trace": str(trace),
    }


def write_company_backtest_cumulative_plot(
    plot_data: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path | None = None,
    output_prefix: str = "company_backtest",
    output_name: str = "",
    variant_label: str = "company API",
    pools: tuple[str, ...] | None = None,
    series_colors: dict[str, str] | None = None,
    x_label_mode: str = "years_only",
) -> dict[str, str]:
    return _write_company_backtest_plot(
        plot_data,
        output_dir,
        input_path=input_path,
        output_prefix=output_prefix,
        output_name=output_name,
        variant_label=variant_label,
        pools=pools,
        series_colors=series_colors,
        x_label_mode=x_label_mode,
        second_column="alpha_cumulative_bps",
        error_label="company backtest plot data",
        palette=("#e49413", "#009e73", "#2f6796", "#5d6674", "#d55e00", "#7a68a6"),
        default_stem_suffix="company_backtest",
        title="公司API回测累和",
        second_panel_title="Alpha累和",
        result_prefix="company_backtest",
        metric="company_backtest_cumulative_profit_alpha",
        style="manual svg two-panel line figure for company API profit and alpha",
        cumulative_definition="cumulative sum of company API daily profit and alpha, displayed in bps",
    )


def write_company_backtest_neutral_comparison_plot(
    plot_data: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path | None = None,
    output_prefix: str = "company_backtest_neutral",
    output_name: str = "",
    variant_label: str = "company API",
    pools: tuple[str, ...] | None = None,
    series_colors: dict[str, str] | None = None,
    x_label_mode: str = "years_only",
) -> dict[str, str]:
    return _write_company_backtest_plot(
        plot_data,
        output_dir,
        input_path=input_path,
        output_prefix=output_prefix,
        output_name=output_name,
        variant_label=variant_label,
        pools=pools,
        series_colors=series_colors,
        x_label_mode=x_label_mode,
        second_column="incremental_cumulative_bps",
        error_label="company neutral comparison plot data",
        palette=("#e49413", "#009e73", "#5d6674", "#2f6796", "#7a68a6", "#d55e00"),
        default_stem_suffix="neutral_comparison",
        title="公司API neutral baseline",
        second_panel_title="相对 neutral_pool 增量",
        result_prefix="company_neutral_comparison",
        metric="company_backtest_neutral_comparison",
        style="manual svg two-panel line figure for company API neutral baseline",
        cumulative_definition=(
            "top panel: cumulative company API profit for model and neutral pool; "
            "bottom panel: cumulative model profit minus neutral-pool profit, in bps"
        ),
    )


def _write_cumulative_pool_internal_plot(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path | None,
    output_prefix: str,
    output_name: str,
    variant_label: str,
    pools: tuple[str, ...],
    x_label_mode: str,
    date_column: str,
    error_label: str,
    default_stem_suffix: str,
    result_prefix: str,
    title_suffix: str,
    included_key: str,
    metric: str,
    style: str,
    cumulative_definition: str,
    optional_columns: tuple[str, ...] = (),
    round_inputs: bool = False,
) -> dict[str, str]:
    required = {"pool", date_column, "short_internal_excess_bps", "next_internal_excess_bps"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"{error_label} missing columns: {missing}")

    output_prefix = slug_label(output_prefix)
    pools = tuple(pools)
    pool_group_slug = _pool_group_slug(pools)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_data, short_ylim, short_step, next_ylim, next_step = _cumulative_plot_data(
        summary,
        date_column=date_column,
        error_label=error_label,
        pools=pools,
        variant_label=variant_label,
        optional_columns=optional_columns,
        round_inputs=round_inputs,
    )
    file_stem = (
        slug_label(output_name)
        if output_name
        else f"{output_prefix}_{pool_group_slug}_{default_stem_suffix}"
    )
    chart_dir = output_dir if output_name else output_dir / file_stem
    plot_data_path = chart_dir / f"{file_stem}_plot_data.csv"
    figure = chart_dir / f"{file_stem}.svg"
    trace = chart_dir / f"{file_stem}_trace.json"
    chart_dir.mkdir(parents=True, exist_ok=True)
    csv_data.to_csv(plot_data_path, index=False, float_format="%.6f")
    _write_two_panel_line_svg(
        csv_data,
        title=(
            f"{variant_label}: {_pool_group_title(pools)} Top 100 "
            f"\u6c60\u5185\u8d85\u989d{title_suffix}"
        ),
        panels=_cumulative_panels(short_ylim, short_step, next_ylim, next_step),
        output_path=figure,
        pools=pools,
        x_label_mode=x_label_mode,
    )
    write_json(
        trace,
        {
            "input": str(input_path) if input_path is not None else None,
            "plot_data": str(plot_data_path),
            "figure": str(figure),
            "variant_label": variant_label,
            "series": list(pools),
            included_key: sorted(csv_data["week_start"].dropna().astype(str).unique()),
            "metric": metric,
            "style": style,
            "cumulative_definition": cumulative_definition,
            "x_label_mode": x_label_mode,
        },
        ensure_ascii=True,
    )
    return {
        f"{result_prefix}_plot_data": str(plot_data_path),
        f"{result_prefix}_figure": str(figure),
        f"{result_prefix}_trace": str(trace),
    }


def _cumulative_plot_data(
    summary: pd.DataFrame,
    *,
    date_column: str,
    error_label: str,
    pools: tuple[str, ...],
    variant_label: str,
    optional_columns: tuple[str, ...] = (),
    round_inputs: bool = False,
) -> tuple[pd.DataFrame, tuple[float, float], float, tuple[float, float], float]:
    columns = [
        "pool",
        date_column,
        "short_internal_excess_bps",
        "next_internal_excess_bps",
        *[column for column in optional_columns if column in summary.columns],
    ]
    plot_data = summary.loc[summary["pool"].isin(pools), columns].copy()
    plot_data[date_column] = pd.to_datetime(plot_data[date_column], errors="coerce")
    plot_data = plot_data.dropna(subset=[date_column]).sort_values(["pool", date_column])
    if plot_data.empty:
        raise ValueError(f"{error_label} has no rows for selected pools")

    for column in ("short_internal_excess_bps", "next_internal_excess_bps"):
        values = pd.to_numeric(plot_data[column], errors="coerce")
        plot_data[column] = values.astype("float64").round(6) if round_inputs else values
    plot_data["variant"] = variant_label
    for pool in pools:
        mask = plot_data["pool"].eq(pool)
        plot_data.loc[mask, "short_cumulative_internal_excess_bps"] = (
            plot_data.loc[mask, "short_internal_excess_bps"].fillna(0.0).cumsum()
        )
        plot_data.loc[mask, "next_cumulative_internal_excess_bps"] = (
            plot_data.loc[mask, "next_internal_excess_bps"].fillna(0.0).cumsum()
        )

    short_ylim, short_step = _nice_line_axis(
        plot_data["short_cumulative_internal_excess_bps"],
        include_zero=True,
        target_ticks=9,
    )
    next_ylim, next_step = _nice_line_axis(
        plot_data["next_cumulative_internal_excess_bps"],
        include_zero=True,
        target_ticks=9,
    )
    csv_data = plot_data.rename(columns={date_column: "week_start"}).copy()
    csv_data["week_start"] = csv_data["week_start"].dt.strftime("%Y-%m-%d")
    return csv_data, short_ylim, short_step, next_ylim, next_step


def _cumulative_panels(
    short_ylim: tuple[float, float],
    short_step: float,
    next_ylim: tuple[float, float],
    next_step: float,
) -> list[dict[str, object]]:
    return [
        {
            "title": "\u77ed\u671f\u6536\u76ca\u7d2f\u548c",
            "ylabel": "",
            "column": "short_cumulative_internal_excess_bps",
            "default_ylim": short_ylim,
            "tick_step": short_step,
            "tick_decimals": None,
            "fixed_ylim": True,
        },
        {
            "title": "\u9694\u591c\u6536\u76ca\u7d2f\u548c",
            "ylabel": "",
            "column": "next_cumulative_internal_excess_bps",
            "default_ylim": next_ylim,
            "tick_step": next_step,
            "tick_decimals": None,
            "fixed_ylim": True,
        },
    ]


def _write_horizon_excess_rank_ic_plots(
    month_summary: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path | None,
    output_prefix: str,
    variant_label: str,
    pools: tuple[str, ...],
    pool_group_slug: str,
    pool_group_title: str,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    horizons = (
        {
            "slug": "short",
            "label": "\u77ed\u671f",
            "excess_col": "short_internal_excess_bps",
            "rank_ic_col": "short_rank_ic",
            "excess_default_ylim": (-5.0, 40.0),
            "rank_default_ylim": (0.0, 0.22),
        },
        {
            "slug": "next",
            "label": "\u9694\u591c",
            "excess_col": "next_internal_excess_bps",
            "rank_ic_col": "next_rank_ic",
            "excess_default_ylim": (-80.0, 100.0),
            "rank_default_ylim": (-0.08, 0.10),
        },
    )
    for horizon in horizons:
        slug = str(horizon["slug"])
        data = month_major_plot_data(
            month_summary,
            value_cols=[str(horizon["excess_col"]), str(horizon["rank_ic_col"])],
            pools=pools,
            variant_label=variant_label,
        )
        chart_dir = (
            output_dir / f"{output_prefix}_{pool_group_slug}_{slug}_excess_rank_ic_with_mean"
        )
        plot_data = (
            chart_dir
            / f"{output_prefix}_{pool_group_slug}_{slug}_excess_rank_ic_with_mean_plot_data.csv"
        )
        figure = (
            chart_dir / f"{output_prefix}_{pool_group_slug}_{slug}_excess_rank_ic_with_mean.svg"
        )
        chart_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(plot_data, index=False, float_format="%.6f")
        _write_two_panel_bar_svg(
            data,
            title=f"{variant_label}: {pool_group_title} {horizon['label']} excess / Rank IC",
            panels=[
                {
                    "title": f"{horizon['label']} Top 100 \u6c60\u5185\u8d85\u989d",
                    "ylabel": "bps",
                    "column": str(horizon["excess_col"]),
                    "default_ylim": horizon["excess_default_ylim"],
                    "tick_step": 5.0 if slug == "short" else 10.0,
                    "tick_decimals": None,
                    "label_decimals": 1,
                    "adaptive_ylim": True,
                    "include_zero": True,
                    "target_ticks": 6,
                    "min_tick_step": 5.0,
                },
                {
                    "title": f"{horizon['label']} Rank IC",
                    "ylabel": "IC",
                    "column": str(horizon["rank_ic_col"]),
                    "default_ylim": horizon["rank_default_ylim"],
                    "tick_step": 0.02,
                    "tick_decimals": 2,
                    "label_decimals": 3,
                    "adaptive_ylim": True,
                    "include_zero": True,
                    "target_ticks": 10,
                    "min_tick_step": 0.01,
                },
            ],
            output_path=figure,
            pools=pools,
        )
        _write_plot_trace(
            chart_dir
            / f"{output_prefix}_{pool_group_slug}_{slug}_excess_rank_ic_with_mean_trace.json",
            input_path=input_path,
            plot_data=plot_data,
            figure=figure,
            variant_label=variant_label,
            included_months=_summary_months(month_summary),
            metric=f"{slug}_pool_internal_excess_rank_ic",
            pools=pools,
        )
        outputs[f"{slug}_excess_rank_ic_plot_data"] = str(plot_data)
        outputs[f"{slug}_excess_rank_ic_figure"] = str(figure)
    return outputs
