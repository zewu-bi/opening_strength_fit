from __future__ import annotations

import html
import math
import re
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import write_json

PLOT_POOLS = ("universe", "pool_S", "pool_M", "pool_L")
PLOT_COLORS = {
    "universe": "#5d6674",
    "pool_S": "#2f6796",
    "pool_M": "#2ca091",
    "pool_L": "#e49413",
}
FONT_FAMILY = (
    "Inter, 'Noto Sans CJK SC', 'Microsoft YaHei', 'PingFang SC', "
    "'Hiragino Sans GB', Arial, sans-serif"
)


def slug_label(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return text or "pool_internal"


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
    required = {
        "pool",
        "week_start",
        "short_internal_excess_bps",
        "next_internal_excess_bps",
    }
    missing = sorted(required - set(weekly_summary.columns))
    if missing:
        raise ValueError(f"weekly_summary missing columns: {missing}")

    output_prefix = slug_label(output_prefix)
    pools = tuple(pools)
    pool_group_slug = _pool_group_slug(pools)
    pool_group_title = _pool_group_title(pools)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = ["pool", "week_start", "short_internal_excess_bps", "next_internal_excess_bps"]
    if "trading_days" in weekly_summary.columns:
        columns.append("trading_days")
    plot_data = weekly_summary.loc[weekly_summary["pool"].isin(pools), columns].copy()
    plot_data["week_start"] = pd.to_datetime(plot_data["week_start"], errors="coerce")
    plot_data = plot_data.dropna(subset=["week_start"]).sort_values(["pool", "week_start"])
    if plot_data.empty:
        raise ValueError("weekly_summary has no rows for selected pools")

    for column in ("short_internal_excess_bps", "next_internal_excess_bps"):
        plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
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

    if output_name:
        chart_dir = output_dir
        file_stem = slug_label(output_name)
    else:
        file_stem = f"{output_prefix}_{pool_group_slug}_weekly_cumulative"
        chart_dir = output_dir / file_stem
    chart_dir.mkdir(parents=True, exist_ok=True)
    plot_data_path = chart_dir / f"{file_stem}_plot_data.csv"
    figure = chart_dir / f"{file_stem}.svg"
    trace = chart_dir / f"{file_stem}_trace.json"

    csv_data = plot_data.copy()
    csv_data["week_start"] = csv_data["week_start"].dt.strftime("%Y-%m-%d")
    csv_data.to_csv(plot_data_path, index=False, float_format="%.6f")

    _write_two_panel_line_svg(
        csv_data,
        title=f"{variant_label}: {pool_group_title} Top 100 \u6c60\u5185\u8d85\u989d\u7d2f\u548c",
        panels=[
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
            "included_weeks": sorted(csv_data["week_start"].dropna().astype(str).unique()),
            "metric": "weekly_pool_internal_excess_cumulative_sum",
            "style": "manual svg two-panel cumulative weekly line figure for selected pools",
            "cumulative_definition": (
                "per-pool cumulative sum of weekly short/next internal excess bps; "
                "weekly rows are expected to be precomputed by the caller"
            ),
            "x_label_mode": x_label_mode,
        },
        ensure_ascii=True,
    )
    return {
        "weekly_cumulative_plot_data": str(plot_data_path),
        "weekly_cumulative_figure": str(figure),
        "weekly_cumulative_trace": str(trace),
    }


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


def _pool_group_slug(pools: tuple[str, ...]) -> str:
    if pools == PLOT_POOLS:
        return "universe_sml"
    return slug_label("_".join(pools))


def _pool_group_title(pools: tuple[str, ...]) -> str:
    if pools == PLOT_POOLS:
        return "universe / S / M / L"
    labels = {
        "pool_S": "S",
        "pool_M": "M",
        "pool_L": "L",
    }
    return " / ".join(labels.get(pool, pool) for pool in pools)


def _summary_months(month_summary: pd.DataFrame) -> list[str]:
    return sorted(month_summary["test_month"].astype(str).unique())


def _write_plot_trace(
    path: Path,
    *,
    input_path: Path | None,
    plot_data: Path,
    figure: Path,
    variant_label: str,
    included_months: list[str],
    metric: str,
    pools: tuple[str, ...] = PLOT_POOLS,
) -> None:
    write_json(
        path,
        {
            "input": str(input_path) if input_path is not None else None,
            "plot_data": str(plot_data),
            "figure": str(figure),
            "variant_label": variant_label,
            "series": list(pools),
            "included_months": included_months,
            "mean": "simple average across monthly summary rows",
            "metric": metric,
            "style": "manual svg two-panel figure for selected pools",
        },
        ensure_ascii=True,
    )


def _write_two_panel_bar_svg(
    plot_data: pd.DataFrame,
    *,
    title: str,
    panels: list[dict[str, object]],
    output_path: Path,
    pools: tuple[str, ...] = PLOT_POOLS,
) -> None:
    categories = [item for item in plot_data["test_month"].astype(str).unique() if item != "Mean"]
    categories.append("Mean")
    pools = tuple(pools)
    pool_count = len(pools)
    if pool_count <= 0:
        raise ValueError("at least one pool is required for plotting")

    width = 1600
    height = 900
    left = 86.0
    right = 1562.0
    panel_tops = (145.0, 550.0)
    panel_height = 310.0
    chart_width = right - left
    group_step = chart_width / len(categories)
    centers = [left + group_step * (index + 0.5) for index in range(len(categories))]
    if pools == PLOT_POOLS:
        bar_width = min(24.0, group_step * 0.17)
        bar_gap = min(7.0, group_step * 0.05)
    else:
        bar_gap = min(7.0, group_step * 0.05)
        total_group_width = min(110.0, group_step * 0.72)
        bar_width = min(34.0, (total_group_width - bar_gap * (pool_count - 1)) / pool_count)
    offsets = tuple(
        (index - (pool_count - 1) / 2.0) * (bar_width + bar_gap) for index in range(pool_count)
    )
    data_index = plot_data.set_index(["test_month", "pool"])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fbfaf7"/>',
        _svg_text(width / 2, 42, title, size=34, weight=800, anchor="middle"),
    ]
    legend_item_width = 230.0
    legend_start = (
        438.0 if pools == PLOT_POOLS else width / 2 - (legend_item_width * pool_count) / 2
    )
    legend_y = 80.0
    for index, pool in enumerate(pools):
        x = legend_start + legend_item_width * index
        lines.append(
            f'<rect x="{x:.1f}" y="{legend_y - 12:.1f}" width="34" height="18" '
            f'fill="{PLOT_COLORS[pool]}"/>'
        )
        lines.append(_svg_text(x + 46.0, legend_y + 3.0, pool, size=19, fill="#262626"))

    for panel_index, panel in enumerate(panels):
        top = panel_tops[panel_index]
        bottom = top + panel_height
        column = str(panel["column"])
        ymin, ymax, tick_step = _panel_axis(
            plot_data[column],
            panel=panel,
        )
        tick_values = _ticks(ymin, ymax, tick_step)
        tick_decimals = panel["tick_decimals"]
        label_decimals = int(panel["label_decimals"])

        def ymap(
            value: float,
            *,
            bottom: float = bottom,
            ymin: float = ymin,
            ymax: float = ymax,
        ) -> float:
            return bottom - (value - ymin) / (ymax - ymin) * panel_height

        lines.append(_svg_text(left, top - 22.0, str(panel["title"]), size=28, weight=800))
        for tick in tick_values:
            y = ymap(tick)
            is_zero = abs(tick) < 1e-12
            lines.append(
                f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
                f'stroke="{"#b8b2a8" if is_zero else "#dedbd4"}" '
                f'stroke-width="{1.3 if is_zero else 1.0}"/>'
            )
            lines.append(
                _svg_text(
                    left - 14.0,
                    y + 5.0,
                    _format_tick(tick, tick_decimals),
                    size=16,
                    fill="#3a3a3a",
                    anchor="end",
                )
            )
        lines.append(
            f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" '
            'stroke="#928d84" stroke-width="1"/>'
        )
        lines.append(
            f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" '
            'stroke="#928d84" stroke-width="1"/>'
        )
        lines.append(_svg_text(24.0, top + panel_height / 2 + 6.0, str(panel["ylabel"]), size=18))
        mean_separator_x = (centers[-2] + centers[-1]) / 2
        lines.append(
            f'<line x1="{mean_separator_x:.1f}" y1="{top:.1f}" x2="{mean_separator_x:.1f}" '
            f'y2="{bottom:.1f}" stroke="#bdb7ad" stroke-width="1.2" stroke-dasharray="6 7"/>'
        )

        zero_y = ymap(0.0)
        for category_index, category in enumerate(categories):
            center_x = centers[category_index]
            if panel_index == 1:
                lines.append(
                    _svg_text(
                        center_x,
                        bottom + 35.0,
                        category,
                        size=19,
                        fill="#3a3a3a",
                        anchor="middle",
                    )
                )
            for pool_index, pool in enumerate(pools):
                value = float(data_index.loc[(category, pool), column])
                x = center_x + offsets[pool_index] - bar_width / 2
                value_y = ymap(value)
                bar_y = min(value_y, zero_y)
                bar_height = max(abs(zero_y - value_y), 1.2)
                lines.append(
                    f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                    f'height="{bar_height:.1f}" fill="{PLOT_COLORS[pool]}"/>'
                )
                label_y = bar_y - 7.0 if value >= 0 else bar_y + bar_height + 14.0
                lines.append(
                    _svg_text(
                        x + bar_width / 2,
                        label_y,
                        _format_value(value, label_decimals),
                        size=12,
                        weight=700,
                        fill=PLOT_COLORS[pool],
                        anchor="middle",
                    )
                )

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_two_panel_line_svg(
    plot_data: pd.DataFrame,
    *,
    title: str,
    panels: list[dict[str, object]],
    output_path: Path,
    pools: tuple[str, ...] = PLOT_POOLS,
    x_label_mode: str = "dates_at_ends",
) -> None:
    pools = tuple(pools)
    if not pools:
        raise ValueError("at least one pool is required for plotting")
    data = plot_data.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], errors="coerce")
    data = data.dropna(subset=["week_start"]).sort_values(["pool", "week_start"])
    if data.empty:
        raise ValueError("weekly plot data is empty")

    width = 1600
    height = 900
    left = 86.0
    right = 1562.0
    panel_tops = (145.0, 550.0)
    panel_height = 310.0
    chart_width = right - left
    min_date = data["week_start"].min()
    max_date = data["week_start"].max()
    span_days = max((max_date - min_date).days, 1)

    def xmap(value: pd.Timestamp) -> float:
        return left + ((value - min_date).days / span_days) * chart_width

    year_ticks = _line_x_ticks(min_date, max_date, mode=x_label_mode)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fbfaf7"/>',
        _svg_text(width / 2, 42, title, size=34, weight=800, anchor="middle"),
    ]
    legend_item_width = 230.0
    legend_start = (
        438.0 if pools == PLOT_POOLS else width / 2 - (legend_item_width * len(pools)) / 2
    )
    legend_y = 80.0
    for index, pool in enumerate(pools):
        x = legend_start + legend_item_width * index
        lines.append(
            f'<line x1="{x:.1f}" y1="{legend_y - 3:.1f}" x2="{x + 34.0:.1f}" '
            f'y2="{legend_y - 3:.1f}" stroke="{PLOT_COLORS[pool]}" stroke-width="5"/>'
        )
        lines.append(
            f'<circle cx="{x + 17.0:.1f}" cy="{legend_y - 3:.1f}" r="4.5" '
            f'fill="{PLOT_COLORS[pool]}"/>'
        )
        lines.append(_svg_text(x + 46.0, legend_y + 3.0, pool, size=19, fill="#262626"))

    for panel_index, panel in enumerate(panels):
        top = panel_tops[panel_index]
        bottom = top + panel_height
        column = str(panel["column"])
        ymin, ymax, tick_step = _panel_axis(
            data[column],
            panel=panel,
        )
        tick_values = _ticks(ymin, ymax, tick_step)
        tick_decimals = panel["tick_decimals"]

        def ymap(
            value: float,
            *,
            bottom: float = bottom,
            ymin: float = ymin,
            ymax: float = ymax,
        ) -> float:
            return bottom - (value - ymin) / (ymax - ymin) * panel_height

        lines.append(_svg_text(left, top - 22.0, str(panel["title"]), size=28, weight=800))
        for tick in tick_values:
            y = ymap(tick)
            is_zero = abs(tick) < 1e-12
            lines.append(
                f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
                f'stroke="{"#b8b2a8" if is_zero else "#dedbd4"}" '
                f'stroke-width="{1.3 if is_zero else 1.0}"/>'
            )
            lines.append(
                _svg_text(
                    left - 14.0,
                    y + 5.0,
                    _format_tick(tick, tick_decimals),
                    size=16,
                    fill="#3a3a3a",
                    anchor="end",
                )
            )
        for tick in year_ticks:
            x = xmap(tick)
            lines.append(
                f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
                'stroke="#ebe7de" stroke-width="1"/>'
            )
            if panel_index == 1:
                label = _line_x_tick_label(
                    tick,
                    min_date=min_date,
                    max_date=max_date,
                    mode=x_label_mode,
                )
                lines.append(
                    _svg_text(
                        x,
                        bottom + 35.0,
                        label,
                        size=18,
                        fill="#3a3a3a",
                        anchor="middle",
                    )
                )
        lines.append(
            f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" '
            'stroke="#928d84" stroke-width="1"/>'
        )
        lines.append(
            f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" '
            'stroke="#928d84" stroke-width="1"/>'
        )
        ylabel = str(panel.get("ylabel", ""))
        if ylabel:
            lines.append(_svg_text(24.0, top + panel_height / 2 + 6.0, ylabel, size=18))

        for pool in pools:
            item = data.loc[data["pool"].eq(pool)].dropna(subset=[column])
            if item.empty:
                continue
            points = [
                f"{xmap(row.week_start):.1f},{ymap(float(getattr(row, column))):.1f}"
                for row in item.itertuples(index=False)
            ]
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{PLOT_COLORS[pool]}" stroke-width="3.0" stroke-linejoin="round" '
                'stroke-linecap="round"/>'
            )
            for row in item.iloc[:: max(1, len(item) // 24)].itertuples(index=False):
                lines.append(
                    f'<circle cx="{xmap(row.week_start):.1f}" '
                    f'cy="{ymap(float(getattr(row, column))):.1f}" r="2.6" '
                    f'fill="{PLOT_COLORS[pool]}"/>'
                )

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _line_x_ticks(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    *,
    mode: str,
) -> list[pd.Timestamp]:
    if mode == "years_only":
        ticks = [min_date]
        ticks.extend(
            pd.Timestamp(year=year, month=1, day=1)
            for year in range(min_date.year + 1, max_date.year)
        )
        if max_date.year != min_date.year:
            ticks.append(max_date)
        return ticks

    ticks = [
        tick
        for tick in pd.date_range(
            pd.Timestamp(year=min_date.year, month=1, day=1),
            pd.Timestamp(year=max_date.year + 1, month=1, day=1),
            freq="YS",
        )
        if min_date <= tick <= max_date
    ]
    if min_date not in ticks:
        ticks = [min_date, *ticks]
    if max_date not in ticks:
        ticks = [*ticks, max_date]
    return ticks


def _line_x_tick_label(
    tick: pd.Timestamp,
    *,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    mode: str,
) -> str:
    if mode == "years_only":
        return tick.strftime("%Y")
    if tick == min_date or tick == max_date:
        return tick.strftime("%Y-%m-%d")
    return tick.strftime("%Y")


def _nice_ylim(
    values: pd.Series, *, default: tuple[float, float], step: float
) -> tuple[float, float]:
    ymin, ymax = default
    finite = values.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if not finite.empty:
        observed_min = float(finite.min())
        observed_max = float(finite.max())
        ymin = min(ymin, math.floor((observed_min - step) / step) * step)
        ymax = max(ymax, math.ceil((observed_max + step) / step) * step)
    return ymin, ymax


def _panel_axis(values: pd.Series, *, panel: dict[str, object]) -> tuple[float, float, float]:
    step = float(panel["tick_step"])
    if bool(panel.get("fixed_ylim", False)):
        ymin, ymax = panel["default_ylim"]  # type: ignore[misc]
        return float(ymin), float(ymax), step
    if not bool(panel.get("adaptive_ylim", False)):
        ymin, ymax = _nice_ylim(
            pd.to_numeric(values, errors="coerce"),
            default=panel["default_ylim"],  # type: ignore[arg-type]
            step=step,
        )
        return ymin, ymax, step
    (ymin, ymax), step = _nice_adaptive_axis(
        values,
        default=panel["default_ylim"],  # type: ignore[arg-type]
        include_zero=bool(panel.get("include_zero", True)),
        target_ticks=int(panel.get("target_ticks", 7)),
        min_step=float(panel.get("min_tick_step", 0.0)),
    )
    return ymin, ymax, step


def _nice_adaptive_axis(
    values: pd.Series,
    *,
    default: tuple[float, float],
    include_zero: bool = False,
    target_ticks: int = 7,
    min_step: float = 0.0,
    pad_fraction: float = 0.35,
) -> tuple[tuple[float, float], float]:
    finite = pd.to_numeric(values, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
    finite = finite.dropna()
    if finite.empty:
        default_span = max(default[1] - default[0], 1e-9)
        step = max(_nice_step(default_span / max(target_ticks, 1)), min_step)
        return default, step

    raw_min = float(finite.min())
    raw_max = float(finite.max())
    observed_min = min(raw_min, 0.0) if include_zero else raw_min
    observed_max = max(raw_max, 0.0) if include_zero else raw_max
    span = max(observed_max - observed_min, 1e-9)
    step = max(_nice_step(span / max(target_ticks, 1)), min_step)
    lower = observed_min - step * pad_fraction
    upper = observed_max + step * pad_fraction
    if include_zero and raw_min >= 0:
        lower = 0.0
    if include_zero and raw_max <= 0:
        upper = 0.0
    ymin = math.floor(lower / step) * step
    ymax = math.ceil(upper / step) * step
    if include_zero and raw_min >= 0:
        ymin = 0.0
    if include_zero and raw_max <= 0:
        ymax = 0.0
    if ymin == ymax:
        ymax = ymin + step
    return (float(ymin), float(ymax)), float(step)


def _nice_line_axis(
    values: pd.Series,
    *,
    include_zero: bool = False,
    target_ticks: int = 8,
) -> tuple[tuple[float, float], float]:
    return _nice_adaptive_axis(
        values,
        default=(0.0, 1.0),
        include_zero=include_zero,
        target_ticks=target_ticks,
        pad_fraction=0.25,
    )


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0 or not math.isfinite(raw_step):
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    base = 10**exponent
    normalized = raw_step / base
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= multiplier:
            return multiplier * base
    return 10.0 * base


def _ticks(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + step / 10:
        values.append(round(current, 10))
        current += step
    return values


def _format_tick(value: float, decimals: object) -> str:
    if decimals is None:
        return f"{value:g}"
    return f"{value:.{int(decimals)}f}"


def _format_value(value: float, decimals: int) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: int | str = 400,
    fill: str = "#222222",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT_FAMILY}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}">{html.escape(text)}</text>'
    )
