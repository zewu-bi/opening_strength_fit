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
    source_files,
)
from opening_strength_fit.pool_internal_plot_svg import (
    PLOT_COLORS,
    write_two_panel_bar_svg,
    write_two_panel_line_svg,
)

DIRECTION_COLORS = {
    "baseline": "#1f2937",
    "baseline_pool_l": "#1f2937",
    "baseline_universe": "#9aa0a6",
    "background": "#6b7280",
    "xs_relative": "#0072b2",
    "hist_surprise": "#d55e00",
    "path_shape": "#009e73",
    "clock_segment": "#cc79a7",
}

DISPLAY_LABELS = {
    "baseline": "baseline",
    "baseline_pool_l": "baseline",
    "background": "background",
    "xs_relative": "xsrelative",
    "hist_surprise": "histsurprise",
    "path_shape": "pathshape",
    "clock_segment": "clocksegment",
}

DEFAULT_PLOT_DIRECTION_KEYS = ("hist_surprise", "path_shape")


def parse_direction_spec(value: str) -> DirectionSpec:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--direction must use key=label=run_id, for example "
            "xs_relative=xs_relative=lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1"
        )
    key, label, run_id = (part.strip() for part in parts)
    return DirectionSpec(key=key, label=label, run_id=run_id)


def combine_overlay_acceptance_data(
    short_universe_data: pd.DataFrame,
    next_pool_data: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["test_month", "variant", "pool", "pool_label"]
    return short_universe_data[key_columns + ["short_rank_ic"]].merge(
        next_pool_data[key_columns + ["next_internal_excess_bps"]],
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


def add_background_cumulative_data(
    cumulative_data: pd.DataFrame,
    *,
    baseline_key: str,
) -> pd.DataFrame:
    baseline = cumulative_data.loc[cumulative_data["pool"].astype(str).eq(baseline_key)].copy()
    if baseline.empty:
        raise ValueError(f"cumulative data has no baseline rows for {baseline_key!r}")
    baseline["week_start"] = pd.to_datetime(baseline["week_start"], errors="coerce")
    baseline = baseline.dropna(subset=["week_start"]).sort_values("week_start")
    background = baseline.copy()
    background["pool"] = "background"
    background["pool_label"] = "background"
    background["variant"] = "background"
    background["selected_next_mean_bps"] = pd.NA
    background["next_internal_excess_bps"] = pd.NA
    background["fee_bps"] = 0.0
    background["next_net_return_bps"] = background["pool_next_mean_bps"]
    background["next_cumulative_net_return_bps"] = background["next_net_return_bps"].cumsum()
    background["next_alpha_bps"] = pd.NA
    background["next_cumulative_alpha_bps"] = pd.NA
    background["next_cumulative_vs_baseline_bps"] = pd.NA
    out = cumulative_data.copy()
    if "next_cumulative_vs_baseline_bps" not in out.columns:
        out["next_cumulative_vs_baseline_bps"] = pd.NA
    return pd.concat([out, background], ignore_index=True)


def add_cumulative_baseline_relative_data(
    cumulative_data: pd.DataFrame,
    *,
    baseline_key: str,
    comparison_keys: tuple[str, ...],
) -> pd.DataFrame:
    data = cumulative_data.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], errors="coerce")
    data = data.dropna(subset=["week_start"])
    baseline = data.loc[data["pool"].astype(str).eq(baseline_key), [
        "week_start",
        "next_cumulative_net_return_bps",
    ]].rename(columns={"next_cumulative_net_return_bps": "baseline_cumulative_net_bps"})
    if baseline.empty:
        raise ValueError(f"cumulative data has no baseline rows for {baseline_key!r}")
    data["next_cumulative_vs_baseline_bps"] = pd.NA
    for key in comparison_keys:
        item = data.loc[data["pool"].astype(str).eq(key), ["week_start"]].merge(
            baseline,
            on="week_start",
            how="left",
        )
        index = data.index[data["pool"].astype(str).eq(key)]
        if len(index) != len(item):
            raise ValueError(f"cannot align cumulative baseline rows for {key!r}")
        data.loc[index, "next_cumulative_vs_baseline_bps"] = (
            data.loc[index, "next_cumulative_net_return_bps"].to_numpy()
            - item["baseline_cumulative_net_bps"].to_numpy()
        )
    data["week_start"] = data["week_start"].dt.strftime("%Y-%m-%d")
    return data


def apply_display_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "pool" not in out.columns:
        return out
    labels = out["pool"].astype(str).map(DISPLAY_LABELS)
    if "pool_label" in out.columns:
        out["pool_label"] = labels.fillna(out["pool_label"])
    if "variant" in out.columns:
        out["variant"] = labels.fillna(out["variant"])
    return out


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
    title_prefix: str = "2022-2025",
) -> dict[str, str]:
    if not directions:
        raise ValueError("at least one direction is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_directions = tuple(
        direction for direction in directions if direction.key in DEFAULT_PLOT_DIRECTION_KEYS
    )
    missing_plot_keys = sorted(
        set(DEFAULT_PLOT_DIRECTION_KEYS) - {direction.key for direction in plot_directions}
    )
    if missing_plot_keys:
        raise ValueError(f"missing default plot directions: {missing_plot_keys}")
    series = tuple(direction.key for direction in plot_directions)
    acceptance_directions = (
        DirectionSpec(key="baseline", label="baseline", run_id=baseline_run_id),
        *plot_directions,
    )
    acceptance_series = tuple(direction.key for direction in acceptance_directions)
    PLOT_COLORS.update(DIRECTION_COLORS)

    short_universe_data = apply_display_labels(
        load_horizon_plot_data(
            backtests_root=backtests_root,
            directions=acceptance_directions,
            pool="universe",
            horizon="short",
        )
    )
    next_pool_data = apply_display_labels(
        load_horizon_plot_data(
            backtests_root=backtests_root,
            directions=acceptance_directions,
            pool=pool,
            horizon="next",
        )
    )
    realized_cumulative_data = load_realized_cumulative_plot_data(
        backtests_root=backtests_root,
        directions=plot_directions,
        pool=pool,
        include_baseline_pool=include_baseline_pool_cumulative,
        include_baseline_universe=include_baseline_universe_cumulative,
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
    realized_cumulative_output = apply_display_labels(
        realized_cumulative_data.drop(
            columns=[
                "pool_short_mean_bps",
                "selected_short_mean_bps",
                "short_internal_excess_bps",
                "short_net_return_bps",
                "short_cumulative_net_return_bps",
            ],
            errors="ignore",
        )
    )
    overlay_acceptance_data = combine_overlay_acceptance_data(
        short_universe_data,
        next_pool_data,
    )
    net_alpha_cumulative_data = combine_net_alpha_cumulative_data(
        realized_cumulative_output,
    )
    net_alpha_cumulative_data = add_cumulative_baseline_relative_data(
        net_alpha_cumulative_data,
        baseline_key="baseline_pool_l",
        comparison_keys=series,
    )
    net_alpha_cumulative_data = add_background_cumulative_data(
        net_alpha_cumulative_data,
        baseline_key="baseline_pool_l",
    )
    overlay_acceptance_csv = (
        output_dir / "optimization_directions_overlay_acceptance_plot_data.csv"
    )
    overlay_acceptance_svg = output_dir / "optimization_directions_overlay_acceptance.svg"
    net_alpha_cumulative_csv = (
        output_dir / "optimization_directions_net_alpha_cumulative_plot_data.csv"
    )
    net_alpha_cumulative_svg = output_dir / "optimization_directions_net_alpha_cumulative.svg"
    trace_path = output_dir / "optimization_directions_trace.json"

    overlay_acceptance_data.to_csv(overlay_acceptance_csv, index=False, float_format="%.6f")
    write_two_panel_bar_svg(
        overlay_acceptance_data,
        title=f"{title_prefix} short rank IC和next pool_L 超额",
        panels=[
            {
                "title": "short universe rank IC",
                "ylabel": "Rank IC",
                "column": "short_rank_ic",
                "default_ylim": (0.12, 0.17),
                "tick_step": 0.01,
                "tick_decimals": 3,
                "label_decimals": 3,
                "adaptive_ylim": True,
                "include_zero": False,
                "target_ticks": 6,
                "min_tick_step": 0.005,
            },
            {
                "title": "next pool_L Top 100 excess",
                "ylabel": "bps",
                "column": "next_internal_excess_bps",
                "default_ylim": (0.0, 12.0),
                "tick_step": 2.0,
                "tick_decimals": None,
                "label_decimals": 1,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 6,
                "min_tick_step": 1.0,
            },
        ],
        output_path=overlay_acceptance_svg,
        pools=acceptance_series,
    )

    net_alpha_cumulative_data.to_csv(
        net_alpha_cumulative_csv,
        index=False,
        float_format="%.6f",
    )
    write_two_panel_line_svg(
        net_alpha_cumulative_data,
        title=f"{title_prefix} 池内Top100隔夜收益累和",
        panels=[
            {
                "title": "累计总收益",
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
                "title": "相对baseline",
                "ylabel": "bps",
                "column": "next_cumulative_vs_baseline_bps",
                "default_ylim": line_axis(
                    net_alpha_cumulative_data["next_cumulative_vs_baseline_bps"]
                ),
                "tick_step": line_step(
                    net_alpha_cumulative_data["next_cumulative_vs_baseline_bps"]
                ),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
        ],
        output_path=net_alpha_cumulative_svg,
        pools=(*cumulative_series, "background"),
        x_label_mode="years_only",
        line_width=2.1,
    )

    trace = {
        "backtests_root": str(backtests_root),
        "output_dir": str(output_dir),
        "pool": pool,
        "cumulative_decision_normalizer": CUMULATIVE_DECISION_NORMALIZER,
        "realized_fee_bps": realized_fee_bps,
        "daily_cumulative_semantics": (
            "overnight total return = pool_L background return + internal excess - fee; "
            "daily cumulative points are kept; values are divided by "
            "cumulative_decision_normalizer"
        ),
        "overlay_acceptance": {
            "figure_title": f"{title_prefix} short rank IC和next pool_L 超额",
            "panels": [
                "short universe rank IC",
                f"next {pool} Top 100 excess",
            ],
            "reason": (
                "short pool excess is omitted because A-share T+1 makes short-horizon "
                "cash PnL non-tradable; next IC is omitted because this model is not "
                "trained to rank next-day returns directly"
            ),
            "baseline_run_id": baseline_run_id,
        },
        "baseline_pool_cumulative": {
            "enabled": include_baseline_pool_cumulative,
            "run_id": baseline_run_id,
            "pool": pool,
            "key": "baseline_pool_l",
        },
        "directions": [
            {"key": item.key, "label": item.label, "run_id": item.run_id}
            for item in plot_directions
        ],
        "plotted_series": ["baseline", *series],
        "figures": {
            "overlay_acceptance": str(overlay_acceptance_svg),
            "net_alpha_cumulative": str(net_alpha_cumulative_svg),
        },
        "plot_data": {
            "overlay_acceptance": str(overlay_acceptance_csv),
            "net_alpha_cumulative": str(net_alpha_cumulative_csv),
        },
        "cumulative_acceptance": {
            "figure_title": f"{title_prefix} 池内Top100隔夜收益累和",
            "panels": ["累计总收益", "相对baseline"],
            "background_series": "pool_L background overnight return",
            "reason": "short cumulative is omitted because this workflow cannot trade T+0",
            "normalizer": CUMULATIVE_DECISION_NORMALIZER,
            "unit": "bps",
            "fee_bps_per_trade": realized_fee_bps,
            "absolute_definition": "pool_L selected overnight return minus fee",
            "background_definition": (
                "pool_L background overnight return; no company backtest API wrapper was "
                "found in this repository"
            ),
            "relative_to_baseline_definition": (
                "hist_surprise/path_shape cumulative selected overnight net return minus "
                "baseline cumulative selected overnight net return"
            ),
        },
        "source_files": source_files(backtests_root, plot_directions),
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
        "overlay_acceptance_plot_data": str(overlay_acceptance_csv),
        "overlay_acceptance_figure": str(overlay_acceptance_svg),
        "net_alpha_cumulative_plot_data": str(net_alpha_cumulative_csv),
        "net_alpha_cumulative_figure": str(net_alpha_cumulative_svg),
        "trace": str(trace_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render 2022-2025 fixed opening-strength acceptance plots for "
            "optimization directions using existing pool-internal plot data."
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
