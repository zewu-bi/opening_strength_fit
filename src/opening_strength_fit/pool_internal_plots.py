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
) -> dict[str, str]:
    output_prefix = slug_label(output_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    excess_data = month_major_plot_data(
        month_summary,
        value_cols=["short_internal_excess_bps", "next_internal_excess_bps"],
        variant_label=variant_label,
    ).rename(
        columns={
            "short_internal_excess_bps": "pool_internal_short_excess_bps",
            "next_internal_excess_bps": "pool_internal_next_excess_bps",
        }
    )
    excess_dir = output_dir / f"{output_prefix}_universe_sml_pool_internal_with_mean"
    excess_plot_data = (
        excess_dir / f"{output_prefix}_universe_sml_pool_internal_with_mean_plot_data.csv"
    )
    excess_figure = excess_dir / f"{output_prefix}_universe_sml_top100_pool_internal_with_mean.svg"
    excess_dir.mkdir(parents=True, exist_ok=True)
    excess_data.to_csv(excess_plot_data, index=False, float_format="%.6f")
    _write_two_panel_bar_svg(
        excess_data,
        title=f"{variant_label}: universe / S / M / L Top 100 \u6c60\u5185\u8d85\u989d",
        panels=[
            {
                "title": "\u77ed\u671f\u6536\u76ca",
                "ylabel": "bps",
                "column": "pool_internal_short_excess_bps",
                "default_ylim": (-5.0, 40.0),
                "tick_step": 5.0,
                "tick_decimals": None,
                "label_decimals": 1,
            },
            {
                "title": "\u9694\u591c\u6536\u76ca",
                "ylabel": "bps",
                "column": "pool_internal_next_excess_bps",
                "default_ylim": (-80.0, 100.0),
                "tick_step": 20.0,
                "tick_decimals": None,
                "label_decimals": 1,
            },
        ],
        output_path=excess_figure,
    )
    _write_plot_trace(
        excess_dir / f"{output_prefix}_universe_sml_pool_internal_with_mean_trace.json",
        input_path=input_path,
        plot_data=excess_plot_data,
        figure=excess_figure,
        variant_label=variant_label,
        included_months=_summary_months(month_summary),
        metric="pool_internal_top100_excess_bps",
    )

    rank_data = month_major_plot_data(
        month_summary,
        value_cols=["short_rank_ic", "next_rank_ic"],
        variant_label=variant_label,
    )
    rank_dir = output_dir / f"{output_prefix}_universe_sml_rank_ic_with_mean"
    rank_plot_data = rank_dir / f"{output_prefix}_universe_sml_rank_ic_with_mean_plot_data.csv"
    rank_figure = rank_dir / f"{output_prefix}_universe_sml_rank_ic_with_mean.svg"
    rank_dir.mkdir(parents=True, exist_ok=True)
    rank_data.to_csv(rank_plot_data, index=False, float_format="%.6f")
    _write_two_panel_bar_svg(
        rank_data,
        title=f"{variant_label}: universe / S / M / L Rank IC",
        panels=[
            {
                "title": "\u77ed\u671f Rank IC",
                "ylabel": "IC",
                "column": "short_rank_ic",
                "default_ylim": (0.0, 0.22),
                "tick_step": 0.02,
                "tick_decimals": 2,
                "label_decimals": 3,
            },
            {
                "title": "\u9694\u591c Rank IC",
                "ylabel": "IC",
                "column": "next_rank_ic",
                "default_ylim": (-0.08, 0.10),
                "tick_step": 0.02,
                "tick_decimals": 2,
                "label_decimals": 3,
            },
        ],
        output_path=rank_figure,
    )
    _write_plot_trace(
        rank_dir / f"{output_prefix}_universe_sml_rank_ic_with_mean_trace.json",
        input_path=input_path,
        plot_data=rank_plot_data,
        figure=rank_figure,
        variant_label=variant_label,
        included_months=_summary_months(month_summary),
        metric="rank_ic",
    )

    return {
        "pool_internal_plot_data": str(excess_plot_data),
        "pool_internal_figure": str(excess_figure),
        "rank_ic_plot_data": str(rank_plot_data),
        "rank_ic_figure": str(rank_figure),
    }


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
) -> None:
    write_json(
        path,
        {
            "input": str(input_path) if input_path is not None else None,
            "plot_data": str(plot_data),
            "figure": str(figure),
            "variant_label": variant_label,
            "series": list(PLOT_POOLS),
            "included_months": included_months,
            "mean": "simple average across monthly summary rows",
            "metric": metric,
            "style": "manual svg matching mentor-facing universe/S/M/L two-panel figure",
        },
        ensure_ascii=True,
    )


def _write_two_panel_bar_svg(
    plot_data: pd.DataFrame,
    *,
    title: str,
    panels: list[dict[str, object]],
    output_path: Path,
) -> None:
    categories = [item for item in plot_data["test_month"].astype(str).unique() if item != "Mean"]
    categories.append("Mean")

    width = 1600
    height = 900
    left = 86.0
    right = 1562.0
    panel_tops = (145.0, 550.0)
    panel_height = 310.0
    chart_width = right - left
    group_step = chart_width / len(categories)
    centers = [left + group_step * (index + 0.5) for index in range(len(categories))]
    bar_width = min(24.0, group_step * 0.17)
    bar_gap = min(7.0, group_step * 0.05)
    offsets = (
        -(1.5 * bar_width + 1.5 * bar_gap),
        -(0.5 * bar_width + 0.5 * bar_gap),
        0.5 * bar_width + 0.5 * bar_gap,
        1.5 * bar_width + 1.5 * bar_gap,
    )
    data_index = plot_data.set_index(["test_month", "pool"])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fbfaf7"/>',
        _svg_text(width / 2, 42, title, size=34, weight=800, anchor="middle"),
    ]
    legend_start = 438.0
    legend_y = 80.0
    for index, pool in enumerate(PLOT_POOLS):
        x = legend_start + 230.0 * index
        lines.append(
            f'<rect x="{x:.1f}" y="{legend_y - 12:.1f}" width="34" height="18" '
            f'fill="{PLOT_COLORS[pool]}"/>'
        )
        lines.append(_svg_text(x + 46.0, legend_y + 3.0, pool, size=19, fill="#262626"))

    for panel_index, panel in enumerate(panels):
        top = panel_tops[panel_index]
        bottom = top + panel_height
        column = str(panel["column"])
        ymin, ymax = _nice_ylim(
            plot_data[column].astype(float),
            default=panel["default_ylim"],  # type: ignore[arg-type]
            step=float(panel["tick_step"]),
        )
        tick_values = _ticks(ymin, ymax, float(panel["tick_step"]))
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
            for pool_index, pool in enumerate(PLOT_POOLS):
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
