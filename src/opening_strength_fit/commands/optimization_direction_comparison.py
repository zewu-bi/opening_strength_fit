from __future__ import annotations

import argparse
from pathlib import Path

from opening_strength_fit.capacity_acceptance import (
    DEFAULT_CAPACITY_SLICES,
    DEFAULT_CAPACITY_TOTAL_NOTIONAL,
)
from opening_strength_fit.optimization_acceptance_plots import (
    CUMULATIVE_MODE_TOP100,
    CUMULATIVE_MODES,
    DEFAULT_PLOT_DIRECTION_KEYS,
    default_plot_directions,
    write_optimization_direction_plots,
)
from opening_strength_fit.optimization_direction_data import (
    DEFAULT_DIRECTIONS,
    DEFAULT_POOL_FEE_MODE,
    DEFAULT_REALIZED_FEE_BPS,
    POOL_FEE_MODES,
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


def parse_key_run_id(value: str) -> tuple[str, str]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--capacity-direction must use key=run_id, for example "
            "mlp_base=capacity_acceptance_nn_mlp_base_split20_v1"
        )
    key, run_id = (part.strip() for part in parts)
    return key, run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render 2022-2025 fixed opening-strength acceptance plots for "
            "baseline plus 1-3 comparison model results."
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
        "--baseline-label",
        default="baseline",
        help="Display label for the baseline run in both acceptance figures.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Selection count label used in plot titles and trace metadata.",
    )
    parser.add_argument(
        "--realized-fee-bps",
        type=float,
        default=DEFAULT_REALIZED_FEE_BPS,
        help="Per-trade fee bps subtracted from selected returns in the cumulative plot.",
    )
    parser.add_argument(
        "--pool-turnover-path",
        default="auto",
        help=(
            "Stock-pool parquet path used to compute equal-weight pool turnover fees. "
            "Used only with --pool-fee-mode stock_pool_membership. Use 'auto' to infer from --pool."
        ),
    )
    parser.add_argument(
        "--pool-fee-mode",
        choices=POOL_FEE_MODES,
        default=DEFAULT_POOL_FEE_MODE,
        help=(
            "Pool background fee turnover mode. stock_pool_membership charges equal-weight "
            "membership changes; summary_estimate falls back to selected_rows/candidate_rows; "
            "round_trip charges the full pool every day."
        ),
    )
    parser.add_argument(
        "--capacity-total-notional",
        type=float,
        default=0.0,
        help=(
            "Optional total strategy capital for cumulative plot scaling. "
            f"Use {DEFAULT_CAPACITY_TOTAL_NOTIONAL:g} for the 1bn split20 convention."
        ),
    )
    parser.add_argument(
        "--capacity-decision-notional",
        type=float,
        default=0.0,
        help=(
            "Optional notional per date x decision_time group. If omitted with "
            "--capacity-total-notional, it is derived as total / --capacity-slices."
        ),
    )
    parser.add_argument(
        "--capacity-slices",
        type=float,
        default=DEFAULT_CAPACITY_SLICES,
        help="Execution slices used to derive per-decision notional from total capital.",
    )
    parser.add_argument(
        "--cumulative-mode",
        choices=CUMULATIVE_MODES,
        default=CUMULATIVE_MODE_TOP100,
        help="Cumulative plot source: TopN pool-internal summaries or capacity acceptance summaries.",
    )
    parser.add_argument(
        "--capacity-baseline-run-id",
        default="",
        help="Capacity acceptance backtest directory for the baseline model in capacity mode.",
    )
    parser.add_argument(
        "--capacity-direction",
        action="append",
        type=parse_key_run_id,
        help="Capacity acceptance run for a comparison model, as key=run_id. Repeat as needed.",
    )
    parser.add_argument(
        "--realistic-baseline-run-id",
        default="",
        help="Realistic acceptance backtest directory for the baseline model in realistic mode.",
    )
    parser.add_argument(
        "--realistic-direction",
        action="append",
        type=parse_key_run_id,
        help="Realistic acceptance run for a comparison model, as key=run_id. Repeat as needed.",
    )
    parser.add_argument(
        "--include-baseline-universe-cumulative",
        action="store_true",
        help=(
            "Compatibility flag retained in trace metadata; the cumulative plot now always "
            "loads baseline universe data to derive the full-market reference line."
        ),
    )
    parser.add_argument(
        "--direction",
        action="append",
        type=parse_direction_spec,
        help=(
            "Comparison model to plot besides baseline. Repeat 1-3 times as "
            "key=label=run_id. If omitted, defaults are "
            f"{'/'.join(DEFAULT_PLOT_DIRECTION_KEYS)}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    directions = (
        tuple(args.direction) if args.direction else default_plot_directions(DEFAULT_DIRECTIONS)
    )
    capacity_total_notional = (
        float(args.capacity_total_notional) if args.capacity_total_notional > 0 else None
    )
    capacity_decision_notional = (
        float(args.capacity_decision_notional)
        if args.capacity_decision_notional > 0
        else (
            float(args.capacity_total_notional) / float(args.capacity_slices)
            if args.capacity_total_notional > 0 and args.capacity_slices > 0
            else None
        )
    )
    outputs = write_optimization_direction_plots(
        backtests_root=args.backtests_root,
        output_dir=args.output_dir,
        directions=directions,
        pool=args.pool,
        include_baseline_universe_cumulative=args.include_baseline_universe_cumulative,
        baseline_run_id=args.baseline_run_id,
        baseline_label=args.baseline_label,
        realized_fee_bps=args.realized_fee_bps,
        pool_turnover_path=args.pool_turnover_path,
        pool_fee_mode=args.pool_fee_mode,
        cumulative_mode=args.cumulative_mode,
        capacity_total_notional=capacity_total_notional,
        capacity_decision_notional=capacity_decision_notional,
        capacity_baseline_run_id=args.capacity_baseline_run_id or None,
        capacity_run_ids=dict(args.capacity_direction or []),
        realistic_baseline_run_id=args.realistic_baseline_run_id or None,
        realistic_run_ids=dict(args.realistic_direction or []),
        title_prefix=args.title_prefix,
        top_n=args.top_n,
    )
    print("optimization_direction_comparison_outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
