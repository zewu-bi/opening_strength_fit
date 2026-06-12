from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import write_json
from opening_strength_fit.optimization_direction_data import (
    CUMULATIVE_DECISION_NORMALIZER,
    DEFAULT_DIRECTIONS,
    DEFAULT_REALIZED_FEE_BPS,
    DirectionSpec,
    line_axis,
    line_step,
    load_horizon_plot_data,
    load_realized_cumulative_plot_data,
    load_yearly_net_alpha_data,
    source_files,
)
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


def parse_direction_spec(value: str) -> DirectionSpec:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--direction must use key=label=run_id, for example "
            "xs_relative=xs_relative=lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1"
        )
    key, label, run_id = (part.strip() for part in parts)
    return DirectionSpec(key=key, label=label, run_id=run_id)


def combine_short_next_excess_data(
    short_data: pd.DataFrame,
    next_data: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["test_month", "variant", "pool", "pool_label"]
    return short_data[key_columns + ["short_internal_excess_bps"]].merge(
        next_data[key_columns + ["next_internal_excess_bps"]],
        on=key_columns,
        how="inner",
    )


def combine_net_alpha_cumulative_data(
    realized_cumulative_output: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["pool", "pool_label", "week_start", "variant"]
    combined = realized_cumulative_output[
        key_columns
        + [
            "pool_next_mean_bps",
            "selected_next_mean_bps",
            "next_internal_excess_bps",
            "fee_bps",
            "next_net_return_bps",
            "next_cumulative_net_return_bps",
        ]
    ].copy()
    combined["next_alpha_bps"] = combined["next_net_return_bps"] - combined["pool_next_mean_bps"]
    combined["next_cumulative_alpha_bps"] = (
        combined.groupby("pool", sort=False)["next_alpha_bps"].cumsum()
    )
    return combined


def write_optimization_direction_plots(
    *,
    backtests_root: Path,
    output_dir: Path,
    directions: tuple[DirectionSpec, ...] = DEFAULT_DIRECTIONS,
    pool: str = "pool_L",
    include_baseline_pool_cumulative: bool = True,
    include_baseline_universe_cumulative: bool = False,
    baseline_run_id: str = "baseline_2022_2025_cluster",
    realized_fee_bps: float = DEFAULT_REALIZED_FEE_BPS,
    title_prefix: str = "2022-2025 pool_L optimization directions",
) -> dict[str, str]:
    if not directions:
        raise ValueError("at least one direction is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    series = tuple(direction.key for direction in directions)
    PLOT_COLORS.update(DIRECTION_COLORS)

    short_data = load_horizon_plot_data(
        backtests_root=backtests_root,
        directions=directions,
        pool=pool,
        horizon="short",
    )
    next_data = load_horizon_plot_data(
        backtests_root=backtests_root,
        directions=directions,
        pool=pool,
        horizon="next",
    )
    realized_cumulative_data = load_realized_cumulative_plot_data(
        backtests_root=backtests_root,
        directions=directions,
        pool=pool,
        include_baseline_pool=include_baseline_pool_cumulative,
        include_baseline_universe=include_baseline_universe_cumulative,
        baseline_run_id=baseline_run_id,
        fee_bps=realized_fee_bps,
    )
    yearly_net_alpha_data = load_yearly_net_alpha_data(
        backtests_root=backtests_root,
        directions=directions,
        pool=pool,
        include_baseline_pool=include_baseline_pool_cumulative,
        baseline_run_id=baseline_run_id,
        fee_bps=realized_fee_bps,
    )
    cumulative_series = tuple(
        [
            *(["baseline_pool_l"] if include_baseline_pool_cumulative else []),
            *series,
            *(["baseline_universe"] if include_baseline_universe_cumulative else []),
        ]
    )
    yearly_series = tuple(
        [
            *(["baseline_pool_l"] if include_baseline_pool_cumulative else []),
            *series,
        ]
    )
    realized_cumulative_output = realized_cumulative_data.drop(
        columns=[
            "pool_short_mean_bps",
            "selected_short_mean_bps",
            "short_internal_excess_bps",
            "short_net_return_bps",
            "short_cumulative_net_return_bps",
        ],
        errors="ignore",
    )
    short_next_data = combine_short_next_excess_data(short_data, next_data)
    net_alpha_cumulative_data = combine_net_alpha_cumulative_data(
        realized_cumulative_output,
    )
    short_next_csv = output_dir / "optimization_directions_short_next_excess_plot_data.csv"
    short_next_svg = output_dir / "optimization_directions_short_next_excess.svg"
    net_alpha_cumulative_csv = (
        output_dir / "optimization_directions_net_alpha_cumulative_plot_data.csv"
    )
    net_alpha_cumulative_svg = output_dir / "optimization_directions_net_alpha_cumulative.svg"
    yearly_net_alpha_csv = output_dir / "optimization_directions_yearly_net_alpha_plot_data.csv"
    yearly_net_alpha_svg = output_dir / "optimization_directions_yearly_net_alpha.svg"
    trace_path = output_dir / "optimization_directions_trace.json"

    short_next_data.to_csv(short_next_csv, index=False, float_format="%.6f")
    write_two_panel_bar_svg(
        short_next_data,
        title=f"{title_prefix}: Top 100 internal excess",
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
        ],
        output_path=short_next_svg,
        pools=series,
    )

    net_alpha_cumulative_data.to_csv(
        net_alpha_cumulative_csv,
        index=False,
        float_format="%.6f",
    )
    write_two_panel_line_svg(
        net_alpha_cumulative_data,
        title="2022-2025 pool L Top 100选股累和与 alpha",
        panels=[
            {
                "title": "Next selected net return, fee-adjusted",
                "ylabel": "bps",
                "column": "next_cumulative_net_return_bps",
                "default_ylim": line_axis(
                    net_alpha_cumulative_data["next_cumulative_net_return_bps"]
                ),
                "tick_step": line_step(
                    net_alpha_cumulative_data["next_cumulative_net_return_bps"]
                ),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
            {
                "title": "Cumulative alpha vs pool background",
                "ylabel": "bps",
                "column": "next_cumulative_alpha_bps",
                "default_ylim": line_axis(
                    net_alpha_cumulative_data["next_cumulative_alpha_bps"]
                ),
                "tick_step": line_step(
                    net_alpha_cumulative_data["next_cumulative_alpha_bps"]
                ),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
        ],
        output_path=net_alpha_cumulative_svg,
        pools=cumulative_series,
        x_label_mode="years_only",
        line_width=2.1,
    )

    yearly_net_alpha_data.to_csv(yearly_net_alpha_csv, index=False, float_format="%.6f")
    write_two_panel_bar_svg(
        yearly_net_alpha_data,
        title=f"{title_prefix}: yearly next net / alpha",
        panels=[
            {
                "title": "Next yearly selected net return",
                "ylabel": "bps",
                "column": "next_net_return_bps",
                "default_ylim": (-10.0, 30.0),
                "tick_step": 5.0,
                "tick_decimals": None,
                "label_decimals": 1,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 7,
                "min_tick_step": 2.5,
            },
            {
                "title": "Next yearly alpha vs pool background",
                "ylabel": "bps",
                "column": "next_alpha_bps",
                "default_ylim": (-10.0, 20.0),
                "tick_step": 5.0,
                "tick_decimals": None,
                "label_decimals": 1,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 7,
                "min_tick_step": 2.5,
            },
        ],
        output_path=yearly_net_alpha_svg,
        pools=yearly_series,
    )

    trace = {
        "backtests_root": str(backtests_root),
        "output_dir": str(output_dir),
        "pool": pool,
        "cumulative_decision_normalizer": CUMULATIVE_DECISION_NORMALIZER,
        "realized_fee_bps": realized_fee_bps,
        "daily_cumulative_semantics": (
            "next net selected return = pool_L background + internal excess - fee; "
            "daily values are divided by cumulative_decision_normalizer"
        ),
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
            "short_next_excess": str(short_next_svg),
            "net_alpha_cumulative": str(net_alpha_cumulative_svg),
            "yearly_net_alpha": str(yearly_net_alpha_svg),
        },
        "plot_data": {
            "short_next_excess": str(short_next_csv),
            "net_alpha_cumulative": str(net_alpha_cumulative_csv),
            "yearly_net_alpha": str(yearly_net_alpha_csv),
        },
        "cumulative_acceptance": {
            "panels": ["next_absolute_net", "next_pool_benchmark_alpha"],
            "reason": "short cumulative is omitted because this workflow cannot trade T+0",
            "normalizer": CUMULATIVE_DECISION_NORMALIZER,
            "unit": "bps",
            "fee_bps_per_trade": realized_fee_bps,
            "absolute_definition": "pool_L selected next return minus fee",
            "alpha_definition": (
                "API-style profit minus benchmark; local benchmark is the pool_L "
                "background next return, so alpha = selected next return - fee - pool_L mean"
            ),
        },
        "source_files": source_files(backtests_root, directions),
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
        "short_next_excess_plot_data": str(short_next_csv),
        "short_next_excess_figure": str(short_next_svg),
        "net_alpha_cumulative_plot_data": str(net_alpha_cumulative_csv),
        "net_alpha_cumulative_figure": str(net_alpha_cumulative_svg),
        "yearly_net_alpha_plot_data": str(yearly_net_alpha_csv),
        "yearly_net_alpha_figure": str(yearly_net_alpha_svg),
        "trace": str(trace_path),
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
        "--realized-fee-bps",
        type=float,
        default=DEFAULT_REALIZED_FEE_BPS,
        help="Per-trade fee bps subtracted from selected returns in the absolute cumulative plot.",
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
        realized_fee_bps=args.realized_fee_bps,
        title_prefix=args.title_prefix,
    )
    print("optimization_direction_comparison_outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
