from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

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


def write_two_panel_bar_svg(
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


def write_two_panel_line_svg(
    plot_data: pd.DataFrame,
    *,
    title: str,
    panels: list[dict[str, object]],
    output_path: Path,
    pools: tuple[str, ...] = PLOT_POOLS,
    x_label_mode: str = "dates_at_ends",
    line_width: float = 3.0,
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
            f'y2="{legend_y - 3:.1f}" stroke="{PLOT_COLORS[pool]}" '
            f'stroke-width="{max(line_width + 2.0, 3.0):.1f}"/>'
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
                f'stroke="{PLOT_COLORS[pool]}" stroke-width="{line_width:.1f}" '
                'stroke-linejoin="round" '
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


def nice_line_axis(
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
