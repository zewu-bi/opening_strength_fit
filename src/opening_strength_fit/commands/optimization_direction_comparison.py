from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import write_json
from opening_strength_fit.pool_internal_plot_svg import (
    PLOT_COLORS,
    write_two_panel_bar_svg,
    write_two_panel_line_svg,
)

DIRECTION_COLORS = {
    "baseline_pool_l": "#1f2937",
    "baseline_universe": "#9aa0a6",
    "xs_relative": "#0072b2",
    "hist_surprise": "#d55e00",
    "path_shape": "#009e73",
    "clock_segment": "#cc79a7",
}
CUMULATIVE_DECISION_NORMALIZER = 1000.0


@dataclass(frozen=True)
class DirectionSpec:
    key: str
    run_id: str
    label: str


DEFAULT_DIRECTIONS = (
    DirectionSpec(
        key="xs_relative",
        run_id="lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1",
        label="xs_relative",
    ),
    DirectionSpec(
        key="hist_surprise",
        run_id="lgbm_delay2_36m_2022_2025_fullxs_hist_same_minute_surprise_v1",
        label="hist_same_minute_surprise",
    ),
    DirectionSpec(
        key="path_shape",
        run_id="lgbm_delay2_36m_2022_2025_fullxs_path_shape_confirm_v1",
        label="path_shape_confirm",
    ),
    DirectionSpec(
        key="clock_segment",
        run_id="lgbm_delay2_36m_2022_2025_fullxs_clock_segment_lgbm_v1",
        label="clock_segment_lgbm",
    ),
)


def parse_direction_spec(value: str) -> DirectionSpec:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--direction must use key=label=run_id, for example "
            "xs_relative=xs_relative=lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1"
        )
    key, label, run_id = (part.strip() for part in parts)
    return DirectionSpec(key=key, label=label, run_id=run_id)


def write_optimization_direction_plots(
    *,
    backtests_root: Path,
    output_dir: Path,
    directions: tuple[DirectionSpec, ...] = DEFAULT_DIRECTIONS,
    pool: str = "pool_L",
    include_baseline_pool_cumulative: bool = True,
    include_baseline_universe_cumulative: bool = False,
    baseline_run_id: str = "baseline_2022_2025_cluster",
    title_prefix: str = "2022-2025 pool_L optimization directions",
) -> dict[str, str]:
    if not directions:
        raise ValueError("at least one direction is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    series = tuple(direction.key for direction in directions)
    PLOT_COLORS.update(DIRECTION_COLORS)

    short_data = _load_horizon_plot_data(
        backtests_root=backtests_root,
        directions=directions,
        pool=pool,
        horizon="short",
    )
    next_data = _load_horizon_plot_data(
        backtests_root=backtests_root,
        directions=directions,
        pool=pool,
        horizon="next",
    )
    cumulative_data = _load_cumulative_plot_data(
        backtests_root=backtests_root,
        directions=directions,
        pool=pool,
        include_baseline_pool=include_baseline_pool_cumulative,
        include_baseline_universe=include_baseline_universe_cumulative,
        baseline_run_id=baseline_run_id,
    )
    relative_cumulative_data = _relative_to_baseline_cumulative_data(
        cumulative_data,
        directions=directions,
        baseline_key="baseline_pool_l",
    )
    relative_year_data = _relative_to_baseline_year_data(
        backtests_root=backtests_root,
        directions=directions,
        pool=pool,
        baseline_run_id=baseline_run_id,
    )
    cumulative_series = tuple(
        [
            *(["baseline_pool_l"] if include_baseline_pool_cumulative else []),
            *series,
            *(["baseline_universe"] if include_baseline_universe_cumulative else []),
        ]
    )

    short_csv = output_dir / "optimization_directions_short_excess_rank_ic_plot_data.csv"
    short_svg = output_dir / "optimization_directions_short_excess_rank_ic_with_mean.svg"
    next_csv = output_dir / "optimization_directions_next_excess_rank_ic_plot_data.csv"
    next_svg = output_dir / "optimization_directions_next_excess_rank_ic_with_mean.svg"
    cumulative_csv = output_dir / "optimization_directions_daily_cumulative_plot_data.csv"
    cumulative_svg = output_dir / "optimization_directions_daily_cumulative.svg"
    relative_cumulative_csv = (
        output_dir / "optimization_directions_relative_baseline_daily_cumulative_plot_data.csv"
    )
    relative_cumulative_svg = (
        output_dir / "optimization_directions_relative_baseline_daily_cumulative.svg"
    )
    relative_year_csv = (
        output_dir / "optimization_directions_relative_baseline_yearly_mean_plot_data.csv"
    )
    relative_year_svg = output_dir / "optimization_directions_relative_baseline_yearly_mean.svg"
    trace_path = output_dir / "optimization_directions_trace.json"

    short_data.to_csv(short_csv, index=False, float_format="%.6f")
    write_two_panel_bar_svg(
        short_data,
        title=f"{title_prefix}: short excess / Rank IC",
        panels=[
            {
                "title": "Short Top 100 internal excess",
                "ylabel": "bps",
                "column": "short_internal_excess_bps",
                "default_ylim": (-5.0, 30.0),
                "tick_step": 5.0,
                "tick_decimals": None,
                "label_decimals": 1,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 6,
                "min_tick_step": 5.0,
            },
            {
                "title": "Short Rank IC",
                "ylabel": "IC",
                "column": "short_rank_ic",
                "default_ylim": (0.0, 0.18),
                "tick_step": 0.02,
                "tick_decimals": 2,
                "label_decimals": 3,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 9,
                "min_tick_step": 0.01,
            },
        ],
        output_path=short_svg,
        pools=series,
    )

    next_data.to_csv(next_csv, index=False, float_format="%.6f")
    write_two_panel_bar_svg(
        next_data,
        title=f"{title_prefix}: next-close excess / Rank IC",
        panels=[
            {
                "title": "Next Top 100 internal excess",
                "ylabel": "bps",
                "column": "next_internal_excess_bps",
                "default_ylim": (-20.0, 30.0),
                "tick_step": 10.0,
                "tick_decimals": None,
                "label_decimals": 1,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 6,
                "min_tick_step": 5.0,
            },
            {
                "title": "Next Rank IC",
                "ylabel": "IC",
                "column": "next_rank_ic",
                "default_ylim": (-0.03, 0.04),
                "tick_step": 0.01,
                "tick_decimals": 2,
                "label_decimals": 3,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 8,
                "min_tick_step": 0.005,
            },
        ],
        output_path=next_svg,
        pools=series,
    )

    cumulative_data.to_csv(cumulative_csv, index=False, float_format="%.6f")
    write_two_panel_line_svg(
        cumulative_data,
        title=f"{title_prefix}: daily cumulative excess",
        panels=[
            {
                "title": "Short cumulative avg excess",
                "ylabel": "bps",
                "column": "short_cumulative_internal_excess_bps",
                "default_ylim": _line_axis(cumulative_data["short_cumulative_internal_excess_bps"]),
                "tick_step": _line_step(cumulative_data["short_cumulative_internal_excess_bps"]),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
            {
                "title": "Next cumulative avg excess",
                "ylabel": "bps",
                "column": "next_cumulative_internal_excess_bps",
                "default_ylim": _line_axis(cumulative_data["next_cumulative_internal_excess_bps"]),
                "tick_step": _line_step(cumulative_data["next_cumulative_internal_excess_bps"]),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
        ],
        output_path=cumulative_svg,
        pools=cumulative_series,
        x_label_mode="years_only",
        line_width=2.1,
    )

    relative_cumulative_data.to_csv(
        relative_cumulative_csv,
        index=False,
        float_format="%.6f",
    )
    write_two_panel_line_svg(
        relative_cumulative_data,
        title=f"{title_prefix}: cumulative excess vs baseline",
        panels=[
            {
                "title": "Short cumulative avg excess vs baseline",
                "ylabel": "bps",
                "column": "short_cumulative_relative_excess_bps",
                "default_ylim": _line_axis(
                    relative_cumulative_data["short_cumulative_relative_excess_bps"]
                ),
                "tick_step": _line_step(
                    relative_cumulative_data["short_cumulative_relative_excess_bps"]
                ),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
            {
                "title": "Next cumulative avg excess vs baseline",
                "ylabel": "bps",
                "column": "next_cumulative_relative_excess_bps",
                "default_ylim": _line_axis(
                    relative_cumulative_data["next_cumulative_relative_excess_bps"]
                ),
                "tick_step": _line_step(
                    relative_cumulative_data["next_cumulative_relative_excess_bps"]
                ),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
        ],
        output_path=relative_cumulative_svg,
        pools=series,
        x_label_mode="years_only",
        line_width=2.1,
    )

    relative_year_data.to_csv(relative_year_csv, index=False, float_format="%.6f")
    write_two_panel_bar_svg(
        relative_year_data,
        title=f"{title_prefix}: yearly avg excess vs baseline",
        panels=[
            {
                "title": "Short yearly avg excess vs baseline",
                "ylabel": "bps",
                "column": "short_relative_excess_bps",
                "default_ylim": (-1.0, 1.0),
                "tick_step": 0.5,
                "tick_decimals": 1,
                "label_decimals": 2,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 7,
                "min_tick_step": 0.1,
            },
            {
                "title": "Next yearly avg excess vs baseline",
                "ylabel": "bps",
                "column": "next_relative_excess_bps",
                "default_ylim": (-1.0, 1.0),
                "tick_step": 0.5,
                "tick_decimals": 1,
                "label_decimals": 2,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 7,
                "min_tick_step": 0.1,
            },
        ],
        output_path=relative_year_svg,
        pools=series,
    )

    trace = {
            "backtests_root": str(backtests_root),
            "output_dir": str(output_dir),
            "pool": pool,
            "cumulative_decision_normalizer": CUMULATIVE_DECISION_NORMALIZER,
            "baseline_pool_cumulative": {
                "enabled": include_baseline_pool_cumulative,
                "run_id": baseline_run_id,
                "pool": pool,
                "key": "baseline_pool_l",
            },
            "directions": [
                {"key": item.key, "label": item.label, "run_id": item.run_id}
                for item in directions
            ],
            "figures": {
                "short_excess_rank_ic": str(short_svg),
                "next_excess_rank_ic": str(next_svg),
                "daily_cumulative": str(cumulative_svg),
                "relative_baseline_daily_cumulative": str(relative_cumulative_svg),
                "relative_baseline_yearly_mean": str(relative_year_svg),
            },
            "plot_data": {
                "short_excess_rank_ic": str(short_csv),
                "next_excess_rank_ic": str(next_csv),
                "daily_cumulative": str(cumulative_csv),
                "relative_baseline_daily_cumulative": str(relative_cumulative_csv),
                "relative_baseline_yearly_mean": str(relative_year_csv),
            },
            "source_files": _source_files(backtests_root, directions),
    }
    if include_baseline_universe_cumulative:
        trace["baseline_universe_cumulative"] = {
            "enabled": True,
            "run_id": baseline_run_id,
            "pool": "universe",
            "key": "baseline_universe",
            "panels": ["next"],
        }
    write_json(trace_path, trace, ensure_ascii=True)

    return {
        "short_excess_rank_ic_plot_data": str(short_csv),
        "short_excess_rank_ic_figure": str(short_svg),
        "next_excess_rank_ic_plot_data": str(next_csv),
        "next_excess_rank_ic_figure": str(next_svg),
        "daily_cumulative_plot_data": str(cumulative_csv),
        "daily_cumulative_figure": str(cumulative_svg),
        "relative_baseline_daily_cumulative_plot_data": str(relative_cumulative_csv),
        "relative_baseline_daily_cumulative_figure": str(relative_cumulative_svg),
        "relative_baseline_yearly_mean_plot_data": str(relative_year_csv),
        "relative_baseline_yearly_mean_figure": str(relative_year_svg),
        "trace": str(trace_path),
    }


def _load_horizon_plot_data(
    *,
    backtests_root: Path,
    directions: tuple[DirectionSpec, ...],
    pool: str,
    horizon: str,
) -> pd.DataFrame:
    value_columns = {
        "short": ("short_internal_excess_bps", "short_rank_ic"),
        "next": ("next_internal_excess_bps", "next_rank_ic"),
    }
    if horizon not in value_columns:
        raise ValueError(f"unknown horizon: {horizon}")
    required = {"test_month", "pool", *value_columns[horizon]}

    frames = []
    for direction in directions:
        path = backtests_root / direction.run_id / f"{horizon}_excess_rank_ic_plot_data.csv"
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        item = frame.loc[frame["pool"].astype(str).eq(pool)].copy()
        if item.empty:
            raise ValueError(f"{path} has no rows for pool {pool!r}")
        item["pool"] = direction.key
        item["pool_label"] = direction.label
        item["variant"] = direction.label
        frames.append(item[["test_month", "variant", "pool", "pool_label", *value_columns[horizon]]])
    combined = pd.concat(frames, ignore_index=True)
    return _sort_month_major(combined)


def _load_cumulative_plot_data(
    *,
    backtests_root: Path,
    directions: tuple[DirectionSpec, ...],
    pool: str,
    include_baseline_pool: bool,
    include_baseline_universe: bool,
    baseline_run_id: str,
) -> pd.DataFrame:
    required = {
        "pool",
        "week_start",
        "short_internal_excess_bps",
        "next_internal_excess_bps",
        "short_cumulative_internal_excess_bps",
        "next_cumulative_internal_excess_bps",
    }
    frames = []
    if include_baseline_pool:
        frames.append(
            _load_one_cumulative_plot_data(
                path=backtests_root / baseline_run_id / "daily_cumulative_plot_data.csv",
                source_pool=pool,
                key="baseline_pool_l",
                label="baseline pool_L",
                required=required,
            )
        )
    if include_baseline_universe:
        frames.append(
            _load_one_cumulative_plot_data(
                path=backtests_root / baseline_run_id / "daily_cumulative_plot_data.csv",
                source_pool="universe",
                key="baseline_universe",
                label="baseline universe",
                required=required,
                next_only=True,
            )
        )
    for direction in directions:
        path = backtests_root / direction.run_id / "daily_cumulative_plot_data.csv"
        frames.append(
            _load_one_cumulative_plot_data(
                path=path,
                source_pool=pool,
                key=direction.key,
                label=direction.label,
                required=required,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"], errors="coerce")
    combined = combined.dropna(subset=["week_start"]).sort_values(["pool", "week_start"])
    combined = _normalize_cumulative_decision_bps(combined)
    combined["week_start"] = combined["week_start"].dt.strftime("%Y-%m-%d")
    return combined


def _load_one_cumulative_plot_data(
    *,
    path: Path,
    source_pool: str,
    key: str,
    label: str,
    required: set[str],
    next_only: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    item = frame.loc[frame["pool"].astype(str).eq(source_pool)].copy()
    if item.empty:
        raise ValueError(f"{path} has no rows for pool {source_pool!r}")
    item["pool"] = key
    item["pool_label"] = label
    item["variant"] = label
    if next_only:
        item["short_internal_excess_bps"] = pd.NA
        item["short_cumulative_internal_excess_bps"] = pd.NA
    return item[
        [
            "pool",
            "pool_label",
            "week_start",
            "short_internal_excess_bps",
            "next_internal_excess_bps",
            "variant",
            "short_cumulative_internal_excess_bps",
            "next_cumulative_internal_excess_bps",
        ]
    ]


def _normalize_cumulative_decision_bps(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    columns = [
        "short_internal_excess_bps",
        "next_internal_excess_bps",
        "short_cumulative_internal_excess_bps",
        "next_cumulative_internal_excess_bps",
    ]
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce") / CUMULATIVE_DECISION_NORMALIZER
    return data


def _relative_to_baseline_cumulative_data(
    cumulative_data: pd.DataFrame,
    *,
    directions: tuple[DirectionSpec, ...],
    baseline_key: str,
) -> pd.DataFrame:
    data = cumulative_data.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], errors="coerce")
    data = data.dropna(subset=["week_start"])
    baseline = data.loc[data["pool"].astype(str).eq(baseline_key)].copy()
    if baseline.empty:
        raise ValueError(f"cumulative data has no baseline rows for {baseline_key!r}")

    baseline = baseline[
        [
            "week_start",
            "short_internal_excess_bps",
            "next_internal_excess_bps",
        ]
    ].rename(
        columns={
            "short_internal_excess_bps": "baseline_short_internal_excess_bps",
            "next_internal_excess_bps": "baseline_next_internal_excess_bps",
        }
    )

    frames = []
    for direction in directions:
        item = data.loc[data["pool"].astype(str).eq(direction.key)].copy()
        if item.empty:
            raise ValueError(f"cumulative data has no rows for direction {direction.key!r}")
        item = item.merge(baseline, on="week_start", how="inner")
        if item.empty:
            raise ValueError(
                f"direction {direction.key!r} has no dates in common with baseline {baseline_key!r}"
            )
        item["short_relative_excess_bps"] = (
            item["short_internal_excess_bps"] - item["baseline_short_internal_excess_bps"]
        )
        item["next_relative_excess_bps"] = (
            item["next_internal_excess_bps"] - item["baseline_next_internal_excess_bps"]
        )
        item = item.sort_values("week_start")
        item["short_cumulative_relative_excess_bps"] = item["short_relative_excess_bps"].cumsum()
        item["next_cumulative_relative_excess_bps"] = item["next_relative_excess_bps"].cumsum()
        item["pool"] = direction.key
        item["pool_label"] = direction.label
        item["variant"] = direction.label
        frames.append(
            item[
                [
                    "pool",
                    "pool_label",
                    "week_start",
                    "short_relative_excess_bps",
                    "next_relative_excess_bps",
                    "variant",
                    "short_cumulative_relative_excess_bps",
                    "next_cumulative_relative_excess_bps",
                ]
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    return combined


def _relative_to_baseline_year_data(
    *,
    backtests_root: Path,
    directions: tuple[DirectionSpec, ...],
    pool: str,
    baseline_run_id: str,
) -> pd.DataFrame:
    baseline_year = _load_year_summary(
        backtests_root / baseline_run_id / "pool_internal_year_summary.csv",
        pool=pool,
    )
    baseline_mean = _load_summary_row(
        backtests_root / baseline_run_id / "pool_internal_summary.csv",
        pool=pool,
    )
    frames = []
    for direction in directions:
        year = _load_year_summary(
            backtests_root / direction.run_id / "pool_internal_year_summary.csv",
            pool=pool,
        )
        merged = year.merge(
            baseline_year,
            on="test_month",
            how="inner",
            suffixes=("", "_baseline"),
        )
        if merged.empty:
            raise ValueError(f"{direction.run_id} has no yearly rows in common with baseline")
        item = pd.DataFrame(
            {
                "test_month": merged["test_month"],
                "pool": direction.key,
                "pool_label": direction.label,
                "variant": direction.label,
                "short_relative_excess_bps": (
                    merged["short_internal_excess_bps"]
                    - merged["short_internal_excess_bps_baseline"]
                ),
                "next_relative_excess_bps": (
                    merged["next_internal_excess_bps"]
                    - merged["next_internal_excess_bps_baseline"]
                ),
            }
        )
        direction_mean = _load_summary_row(
            backtests_root / direction.run_id / "pool_internal_summary.csv",
            pool=pool,
        )
        mean_item = pd.DataFrame(
            {
                "test_month": ["Mean"],
                "pool": [direction.key],
                "pool_label": [direction.label],
                "variant": [direction.label],
                "short_relative_excess_bps": [
                    direction_mean["short_internal_excess_bps"]
                    - baseline_mean["short_internal_excess_bps"]
                ],
                "next_relative_excess_bps": [
                    direction_mean["next_internal_excess_bps"]
                    - baseline_mean["next_internal_excess_bps"]
                ],
            }
        )
        frames.append(pd.concat([item, mean_item], ignore_index=True))
    return _sort_month_major(pd.concat(frames, ignore_index=True))


def _load_year_summary(path: Path, *, pool: str) -> pd.DataFrame:
    required = {"pool", "year", "short_internal_excess_bps", "next_internal_excess_bps"}
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    item = frame.loc[frame["pool"].astype(str).eq(pool)].copy()
    if item.empty:
        raise ValueError(f"{path} has no rows for pool {pool!r}")
    item["test_month"] = item["year"].astype(str)
    return item[["test_month", "short_internal_excess_bps", "next_internal_excess_bps"]]


def _load_summary_row(path: Path, *, pool: str) -> pd.Series:
    required = {"pool", "short_internal_excess_bps", "next_internal_excess_bps"}
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    item = frame.loc[frame["pool"].astype(str).eq(pool)].copy()
    if item.empty:
        raise ValueError(f"{path} has no rows for pool {pool!r}")
    return item.iloc[0]


def _line_axis(values: pd.Series) -> tuple[float, float]:
    from opening_strength_fit.pool_internal_plot_svg import nice_line_axis

    axis, _ = nice_line_axis(values, include_zero=True, target_ticks=9)
    return axis


def _line_step(values: pd.Series) -> float:
    from opening_strength_fit.pool_internal_plot_svg import nice_line_axis

    _, step = nice_line_axis(values, include_zero=True, target_ticks=9)
    return step


def _sort_month_major(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["_mean_order"] = data["test_month"].astype(str).eq("Mean").astype("int8")
    data = data.sort_values(["_mean_order", "test_month", "pool"]).drop(columns="_mean_order")
    return data.reset_index(drop=True)


def _source_files(backtests_root: Path, directions: tuple[DirectionSpec, ...]) -> dict[str, dict[str, str]]:
    return {
        direction.key: {
            "short": str(backtests_root / direction.run_id / "short_excess_rank_ic_plot_data.csv"),
            "next": str(backtests_root / direction.run_id / "next_excess_rank_ic_plot_data.csv"),
            "daily_cumulative": str(
                backtests_root / direction.run_id / "daily_cumulative_plot_data.csv"
            ),
        }
        for direction in directions
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render 2022-2025 pool_L comparison plots for optimization directions "
            "using existing pool-internal plot data."
        )
    )
    parser.add_argument(
        "--backtests-root",
        type=Path,
        default=Path("experiments/results/backtests"),
        help="Root containing per-run pool-internal backtest directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/backtests/optimization_direction_comparison_2022_2025"),
        help="Directory for comparison SVGs, CSV plot data, and trace JSON.",
    )
    parser.add_argument(
        "--pool",
        default="pool_L",
        help="Pool slice to compare across directions.",
    )
    parser.add_argument(
        "--title-prefix",
        default="2022-2025 pool_L optimization directions",
        help="Figure title prefix.",
    )
    parser.add_argument(
        "--baseline-run-id",
        default="baseline_2022_2025_cluster",
        help="Backtest directory used for the baseline pool reference.",
    )
    parser.add_argument(
        "--no-baseline-pool-cumulative",
        action="store_true",
        help="Do not add baseline pool_L as a reference line on the cumulative plot.",
    )
    parser.add_argument(
        "--include-baseline-universe-cumulative",
        action="store_true",
        help="Also add baseline universe as a next-panel reference line on the cumulative plot.",
    )
    parser.add_argument("--no-baseline-universe-cumulative", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--direction",
        action="append",
        type=parse_direction_spec,
        help=(
            "Override default directions. Repeat as key=label=run_id. "
            "When provided, defaults are replaced."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    directions = tuple(args.direction) if args.direction else DEFAULT_DIRECTIONS
    outputs = write_optimization_direction_plots(
        backtests_root=args.backtests_root,
        output_dir=args.output_dir,
        directions=directions,
        pool=args.pool,
        include_baseline_pool_cumulative=not args.no_baseline_pool_cumulative,
        include_baseline_universe_cumulative=(
            args.include_baseline_universe_cumulative and not args.no_baseline_universe_cumulative
        ),
        baseline_run_id=args.baseline_run_id,
        title_prefix=args.title_prefix,
    )
    print("optimization_direction_comparison_outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
