from __future__ import annotations

import argparse
from pathlib import Path

from opening_strength_fit.optimization_acceptance_plots import (
    DEFAULT_PLOT_DIRECTION_KEYS,
    default_plot_directions,
    write_optimization_direction_plots,
)
from opening_strength_fit.optimization_direction_data import (
    DEFAULT_DIRECTIONS,
    DEFAULT_REALIZED_FEE_BPS,
    DirectionSpec,
)


def parse_direction_spec(value: str) -> DirectionSpec:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--direction must use key=label=run_id, for example "
            "hist_surprise=histsurprise=lgbm_delay2_36m_2022_2025_fullxs_hist_same_minute_surprise_v1"
        )
    key, label, run_id = (part.strip() for part in parts)
    return DirectionSpec(key=key, label=label, run_id=run_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render 2022-2025 fixed opening-strength acceptance plots for "
            "baseline plus 2-3 comparison model results."
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
        default=Path("experiments/results/backtests/optimization_overlay_acceptance_2022_2025"),
        help="Directory for comparison SVGs, CSV plot data, and trace JSON.",
    )
    parser.add_argument(
        "--pool",
        default="pool_L",
        help="Pool slice to compare across directions.",
    )
    parser.add_argument(
        "--title-prefix",
        default="2022-2025",
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
        help="Per-trade fee bps subtracted from selected returns in the cumulative plot.",
    )
    parser.add_argument(
        "--include-baseline-universe-cumulative",
        action="store_true",
        help="Also add baseline universe as a next-panel reference line on the cumulative plot.",
    )
    parser.add_argument(
        "--direction",
        action="append",
        type=parse_direction_spec,
        help=(
            "Comparison model to plot besides baseline. Repeat 2-3 times as "
            "key=label=run_id. If omitted, defaults are "
            f"{'/'.join(DEFAULT_PLOT_DIRECTION_KEYS)}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    directions = (
        tuple(args.direction)
        if args.direction
        else default_plot_directions(DEFAULT_DIRECTIONS)
    )
    outputs = write_optimization_direction_plots(
        backtests_root=args.backtests_root,
        output_dir=args.output_dir,
        directions=directions,
        pool=args.pool,
        include_baseline_universe_cumulative=args.include_baseline_universe_cumulative,
        baseline_run_id=args.baseline_run_id,
        realized_fee_bps=args.realized_fee_bps,
        title_prefix=args.title_prefix,
    )
    print("optimization_direction_comparison_outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
