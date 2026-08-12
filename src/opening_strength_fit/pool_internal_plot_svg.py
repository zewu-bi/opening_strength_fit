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
SVG_WIDTH, SVG_HEIGHT = 1600, 900
PLOT_LEFT, PLOT_RIGHT = 86.0, 1562.0


def _y_mapper(bottom: float, ymin: float, ymax: float, height: float):
    def scale(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * height

    return scale


def _svg_header(title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        f'<rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="#fbfaf7"/>',
        _svg_text(SVG_WIDTH / 2, 42, title, size=34, weight=800, anchor="middle"),
    ]


def _append_legend(
    lines: list[str],
    plot_data: pd.DataFrame,
    pools: tuple[str, ...],
    marker: str,
    line_width: float = 3.0,
) -> None:
    labels = _legend_labels(plot_data, pools)
    for pool, x, y in _legend_layout(labels, pools):
        if marker == "bar":
            lines.append(
                f'<rect x="{x:.1f}" y="{y - 12:.1f}" width="34" height="18" '
                f'fill="{PLOT_COLORS[pool]}"/>'
            )
        else:
            lines.extend(
                (
                    f'<line x1="{x:.1f}" y1="{y - 3:.1f}" x2="{x + 34.0:.1f}" '
                    f'y2="{y - 3:.1f}" stroke="{PLOT_COLORS[pool]}" '
                    f'stroke-width="{max(line_width + 2.0, 3.0):.1f}"/>',
                    f'<circle cx="{x + 17.0:.1f}" cy="{y - 3:.1f}" r="4.5" '
                    f'fill="{PLOT_COLORS[pool]}"/>',
                )
            )
        lines.append(_svg_text(x + 46.0, y + 3.0, labels[pool], size=19, fill="#262626"))


def _append_y_grid(
    lines: list[str],
    values: pd.Series,
    panel: dict[str, object],
    top: float,
    bottom: float,
) -> tuple[str, float, float, object]:
    column = str(panel["column"])
    ymin, ymax, tick_step = _panel_axis(values, panel=panel)
    ymap = _y_mapper(bottom, ymin, ymax, bottom - top)
    panel_title = str(panel.get("title", ""))
    if panel_title:
        lines.append(
            _svg_text(
                PLOT_LEFT,
                top - 22.0,
                panel_title,
                size=int(panel.get("title_size", 28)),
                weight=800,
            )
        )
    for tick in _ticks(ymin, ymax, tick_step):
        y = ymap(tick)
        is_zero = abs(tick) < 1e-12
        lines.extend(
            (
                f'<line x1="{PLOT_LEFT:.1f}" y1="{y:.1f}" x2="{PLOT_RIGHT:.1f}" y2="{y:.1f}" '
                f'stroke="{"#b8b2a8" if is_zero else "#dedbd4"}" '
                f'stroke-width="{1.3 if is_zero else 1.0}"/>',
                _svg_text(
                    PLOT_LEFT - 14.0,
                    y + 5.0,
                    _format_tick(tick, panel["tick_decimals"]),
                    size=16,
                    fill="#3a3a3a",
                    anchor="end",
                ),
            )
        )
    return column, ymin, ymax, ymap


def _append_panel_border(
    lines: list[str],
    panel: dict[str, object],
    top: float,
    bottom: float,
) -> None:
    lines.extend(
        (
            f'<line x1="{PLOT_LEFT:.1f}" y1="{top:.1f}" x2="{PLOT_LEFT:.1f}" y2="{bottom:.1f}" '
            'stroke="#928d84" stroke-width="1"/>',
            f'<line x1="{PLOT_LEFT:.1f}" y1="{bottom:.1f}" x2="{PLOT_RIGHT:.1f}" y2="{bottom:.1f}" '
            'stroke="#928d84" stroke-width="1"/>',
        )
    )
    ylabel = str(panel.get("ylabel", ""))
    if ylabel:
        lines.append(_svg_text(24.0, (top + bottom) / 2 + 6.0, ylabel, size=18))


def _write_svg(path: Path, lines: list[str]) -> None:
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    left, right = PLOT_LEFT, PLOT_RIGHT
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

    lines = _svg_header(title)
    _append_legend(lines, plot_data, pools, "bar")

    for panel_index, panel in enumerate(panels):
        top = panel_tops[panel_index]
        bottom = top + panel_height
        column, ymin, ymax, ymap = _append_y_grid(
            lines, plot_data[str(panel["column"])], panel, top, bottom
        )
        label_decimals = int(panel["label_decimals"])
        value_labels = str(panel.get("value_labels", "all"))
        _append_panel_border(lines, panel, top, bottom)
        mean_separator_x = (centers[-2] + centers[-1]) / 2
        lines.append(
            f'<line x1="{mean_separator_x:.1f}" y1="{top:.1f}" x2="{mean_separator_x:.1f}" '
            f'y2="{bottom:.1f}" stroke="#bdb7ad" stroke-width="1.2" stroke-dasharray="6 7"/>'
        )

        baseline_value = min(max(0.0, ymin), ymax)
        baseline_y = ymap(baseline_value)
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
                try:
                    raw_value = data_index.loc[(category, pool), column]
                except KeyError:
                    continue
                if pd.isna(raw_value):
                    continue
                value = float(raw_value)
                if not math.isfinite(value):
                    continue
                x = center_x + offsets[pool_index] - bar_width / 2
                value_y = ymap(value)
                bar_y = min(value_y, baseline_y)
                bar_height = max(abs(baseline_y - value_y), 1.2)
                lines.append(
                    f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                    f'height="{bar_height:.1f}" fill="{PLOT_COLORS[pool]}"/>'
                )
                if not _should_label_bar(value_labels, category):
                    continue
                label_y = value_y - 7.0 if value >= baseline_value else value_y + 14.0
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

    _write_svg(output_path, lines)


def write_two_panel_line_svg(
    plot_data: pd.DataFrame,
    *,
    title: str,
    panels: list[dict[str, object]],
    output_path: Path,
    pools: tuple[str, ...] = PLOT_POOLS,
    x_label_mode: str = "dates_at_ends",
    line_width: float = 3.0,
    line_marker_count: int = 24,
) -> None:
    pools = tuple(pools)
    if not pools:
        raise ValueError("at least one pool is required for plotting")
    data = plot_data.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], errors="coerce")
    data = data.dropna(subset=["week_start"]).sort_values(["pool", "week_start"])
    if data.empty:
        raise ValueError("weekly plot data is empty")
    left, right = PLOT_LEFT, PLOT_RIGHT
    panel_count = len(panels)
    if panel_count == 1:
        panel_tops = (112.0,)
        panel_height = 720.0
    elif panel_count == 2:
        panel_tops = (145.0, 550.0)
        panel_height = 310.0
    else:
        raise ValueError("write_two_panel_line_svg supports one or two panels")
    chart_width = right - left
    data_min_date = data["week_start"].min()
    data_max_date = data["week_start"].max()
    if x_label_mode == "years_only":
        min_date = pd.Timestamp(year=data_min_date.year, month=1, day=1)
        max_date = pd.Timestamp(year=data_max_date.year + 1, month=1, day=1)
    else:
        min_date = data_min_date
        max_date = data_max_date
    span_days = max((max_date - min_date).days, 1)

    def xmap(value: pd.Timestamp) -> float:
        return left + ((value - min_date).days / span_days) * chart_width

    year_ticks = _line_x_ticks(min_date, max_date, mode=x_label_mode)

    lines = _svg_header(title)
    _append_legend(lines, data, pools, "line", line_width)

    for panel_index, panel in enumerate(panels):
        top = panel_tops[panel_index]
        bottom = top + panel_height
        column, _, _, ymap = _append_y_grid(lines, data[str(panel["column"])], panel, top, bottom)
        for tick in year_ticks:
            x = xmap(tick)
            lines.append(
                f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
                'stroke="#ebe7de" stroke-width="1"/>'
            )
            if panel_index == panel_count - 1:
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
        _append_panel_border(lines, panel, top, bottom)

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
            if line_marker_count > 0:
                marker_step = max(1, len(item) // line_marker_count)
                for row in item.iloc[::marker_step].itertuples(index=False):
                    lines.append(
                        f'<circle cx="{xmap(row.week_start):.1f}" '
                        f'cy="{ymap(float(getattr(row, column))):.1f}" r="2.6" '
                        f'fill="{PLOT_COLORS[pool]}"/>'
                    )

    _write_svg(output_path, lines)


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


def _legend_labels(plot_data: pd.DataFrame, pools: tuple[str, ...]) -> dict[str, str]:
    if "pool_label" not in plot_data.columns:
        return {pool: pool for pool in pools}
    labels: dict[str, str] = {}
    for pool in pools:
        item = plot_data.loc[plot_data["pool"].astype(str).eq(pool), "pool_label"]
        labels[pool] = str(item.iloc[0]) if not item.empty else pool
    return labels


def _legend_layout(
    legend_labels: dict[str, str],
    pools: tuple[str, ...],
) -> list[tuple[str, float, float]]:
    marker_and_gap = 46.0
    trailing_gap = 34.0
    font_size = 19.0
    min_item_width = 110.0
    row_gap = 28.0
    available_width = max(PLOT_RIGHT - PLOT_LEFT, SVG_WIDTH * 0.6)
    item_widths = {
        pool: max(
            min_item_width,
            marker_and_gap + _svg_text_width(legend_labels[pool], font_size) + trailing_gap,
        )
        for pool in pools
    }
    rows: list[tuple[list[str], float]] = []
    current: list[str] = []
    current_width = 0.0
    for pool in pools:
        item_width = item_widths[pool]
        if current and current_width + item_width > available_width:
            rows.append((current, current_width))
            current = []
            current_width = 0.0
        current.append(pool)
        current_width += item_width
    if current:
        rows.append((current, current_width))

    start_y = 80.0 - (len(rows) - 1) * row_gap / 2.0
    positions: list[tuple[str, float, float]] = []
    for row_index, (row, row_width) in enumerate(rows):
        x = max(PLOT_LEFT, SVG_WIDTH / 2.0 - row_width / 2.0)
        row_y = start_y + row_gap * row_index
        for pool in row:
            positions.append((pool, x, row_y))
            x += item_widths[pool]
    return positions


def _svg_text_width(text: str, size: float) -> float:
    units = 0.0
    for char in text:
        if char.isspace():
            units += 0.35
        elif ord(char) < 128:
            units += 0.54
        else:
            units += 0.95
    return units * size


def _should_label_bar(mode: str, category: str) -> bool:
    if mode == "none":
        return False
    if mode == "mean_only":
        return category == "Mean"
    return True


def _line_x_ticks(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    *,
    mode: str,
) -> list[pd.Timestamp]:
    if mode == "years_only":
        end_year = max_date.year
        if max_date.month != 1 or max_date.day != 1:
            end_year += 1
        return [
            pd.Timestamp(year=year, month=1, day=1) for year in range(min_date.year, end_year + 1)
        ]

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
    return tick.strftime("%Y-%m-%d" if tick in (min_date, max_date) else "%Y")


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
