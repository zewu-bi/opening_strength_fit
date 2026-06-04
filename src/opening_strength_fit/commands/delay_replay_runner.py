from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opening_strength_fit.commands.delay_replay_validation import delay_entry_ticks
from opening_strength_fit.commands.opening_intraday_backtest import (
    build_summary,
    parse_status_values,
    plot_contact_sheet,
    plot_daily_curves,
    run_backtest,
)

DEFAULT_DELAYS = ("delay0", "delay1", "delay2")
DEFAULT_SCENARIOS = (
    "proxy_top20",
    "cost_10bps",
    "tradable_cost",
    "liquidity_cost",
    "capacity_l3_1m",
    "capacity_l5_2m",
)
SCENARIO_PLOT_LABELS = {
    "proxy_top20": "Proxy\nTop20",
    "cost_10bps": "Cost\n10bps",
    "tradable_cost": "Tradable",
    "liquidity_cost": "Liquidity",
    "capacity_l3_1m": "Capacity\nL3 / 1m",
    "capacity_l5_2m": "Capacity\nL5 / 2m",
    "limit_up_room_10s": "Limit-up\nroom",
}
RUN_PLOT_LABELS = {
    "universe": "Universe",
    "strong": "Strong",
}
RUN_PLOT_COLORS = {
    "Universe": "#1f77b4",
    "Strong": "#d17a22",
}


def write_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def scenario_capital_per_cycle(
    scenario: object,
    args: argparse.Namespace,
) -> float:
    if scenario.capital_per_cycle is not None:
        return float(scenario.capital_per_cycle)
    return float(args.capital_per_cycle)


def run_branch(run: object) -> str:
    return RUN_PLOT_LABELS["strong"] if "strong" in str(run) else RUN_PLOT_LABELS["universe"]


def summary_plot_delays(
    summary: pd.DataFrame,
    requested: list[str] | None,
) -> list[str]:
    present = [delay for delay in DEFAULT_DELAYS if delay in set(summary["delay"].astype(str))]
    if not requested or "all" in requested:
        return present
    return [delay for delay in requested if delay in present]


def plot_scenario_summary(
    summary: pd.DataFrame,
    *,
    output_root: Path,
    requested_delays: list[str] | None,
) -> list[Path]:
    if summary.empty:
        return []
    work = summary.copy()
    work["branch"] = work["run"].map(run_branch)
    scenario_order = {
        name: index
        for index, name in enumerate(
            [name for name in DEFAULT_SCENARIOS if name in set(work["scenario"])]
            + [
                name
                for name in work["scenario"].astype(str).drop_duplicates()
                if name not in DEFAULT_SCENARIOS
            ]
        )
    }
    work["scenario_order"] = work["scenario"].map(scenario_order)
    paths = []
    for delay in summary_plot_delays(work, requested_delays):
        delay_frame = work.loc[work["delay"].astype(str) == delay].copy()
        if delay_frame.empty:
            continue
        scenarios = [
            name
            for name, _index in sorted(
                scenario_order.items(),
                key=lambda item: item[1],
            )
            if name in set(delay_frame["scenario"].astype(str))
        ]
        x = np.arange(len(scenarios), dtype="float64")
        width = 0.34
        fig, ax = plt.subplots(figsize=(11.0, 5.4))
        for offset, branch in ((-width / 2.0, "Universe"), (width / 2.0, "Strong")):
            branch_frame = delay_frame.loc[delay_frame["branch"] == branch].set_index("scenario")
            if branch_frame.empty:
                continue
            values = [
                float(branch_frame.loc[scenario, "mean_cycle_return_bps"])
                if scenario in branch_frame.index
                else np.nan
                for scenario in scenarios
            ]
            ax.bar(
                x + offset,
                values,
                width=width,
                color=RUN_PLOT_COLORS.get(branch, "#333333"),
                label=branch,
            )
            for xpos, value in zip(x + offset, values, strict=True):
                if not np.isfinite(value):
                    continue
                ax.annotate(
                    f"{value:.1f}",
                    (xpos, value),
                    textcoords="offset points",
                    xytext=(0, 4 if value >= 0 else -12),
                    ha="center",
                    fontsize=7.5,
                )
        ax.axhline(0.0, color="#555555", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_PLOT_LABELS.get(name, name) for name in scenarios])
        ax.set_ylabel("Mean cycle return (bps)")
        ax.set_title(f"{delay} replay under execution constraints")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        output_path = output_root / f"replay_l3_l5_single_tradable_{delay}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        paths.append(output_path)
    return paths


def replay_scenario(
    *,
    delay: str,
    scenario: object,
    runs: list[tuple[str, Path]],
    output_dir: Path,
    args: argparse.Namespace,
    entry_times: list[str],
    context_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cycles, selected = run_backtest(
        runs,
        entry_times=entry_times,
        context_frame=context_frame,
        context_label_mode=args.context_label_mode,
        cycle_minutes=args.cycle_minutes,
        top_n=args.top_n,
        fee_bps=scenario.fee_bps,
        slippage_bps=scenario.slippage_bps,
        tradable_statuses=parse_status_values(list(scenario.tradable_statuses)),
        require_entry_status=scenario.require_entry_status,
        max_decision_lag_seconds=scenario.max_decision_lag_seconds,
        max_entry_tick_gap_seconds=scenario.max_entry_tick_gap_seconds,
        max_spread_bps=scenario.max_spread_bps,
        min_limit_up_room_bps=scenario.min_limit_up_room_bps,
        min_ask_volume_1=scenario.min_ask_volume_1,
        min_bid_volume_1=scenario.min_bid_volume_1,
        capacity_notional_col=args.capacity_notional_col,
        capacity_volume_col=args.capacity_volume_col,
        capacity_price_col=args.capacity_price_col,
        min_capacity_notional=scenario.min_capacity_notional,
        max_participation_rate=scenario.max_participation_rate,
        capital_per_cycle=scenario_capital_per_cycle(scenario, args),
        ask_depth_levels=scenario.ask_depth_levels,
        ask_depth_participation_rate=scenario.ask_depth_participation_rate,
        ask_depth_fill_mode=scenario.ask_depth_fill_mode,
        allow_decision_depth_fallback=args.allow_decision_depth_fallback,
        max_symbol_trades_per_day=scenario.max_symbol_trades_per_day,
        symbol_cooldown_minutes=scenario.symbol_cooldown_minutes,
        max_symbol_weight=scenario.max_symbol_weight,
        missing_policy=args.missing_constraint,
        score_col="prediction",
        label_col="label",
    )
    daily, summary = build_summary(cycles)
    cycles.to_csv(output_dir / "intraday_cycles.csv", index=False)
    selected.to_csv(output_dir / "intraday_selected_trades.csv", index=False)
    daily.to_csv(output_dir / "intraday_daily_summary.csv", index=False)
    summary.to_csv(output_dir / "intraday_summary.csv", index=False)

    if not args.skip_plots:
        plot_daily_curves(
            cycles,
            output_dir=output_dir,
            entry_times=entry_times,
            cycle_minutes=args.cycle_minutes,
        )
        plot_contact_sheet(cycles, output_dir / "daily_curves_contact_sheet.png")

    write_trace(
        output_dir / "intraday_trace.json",
        {
            "delay": delay,
            "scenario": asdict(scenario),
            "runs": [{"label": label, "path": str(path)} for label, path in runs],
            "top_n": args.top_n,
            "entry_times": entry_times,
            "cycle_minutes": args.cycle_minutes,
            "capital_per_cycle": args.capital_per_cycle,
            "scenario_capital_per_cycle": scenario_capital_per_cycle(scenario, args),
            "context_input": args.context_input,
            "context_kind": args.context_kind,
            "context_entry_tick_delay": delay_entry_ticks(delay) if args.context_input else None,
            "context_entry_max_gap_seconds": args.context_entry_max_gap_seconds,
            "context_decision_max_lag_seconds": args.context_decision_max_lag_seconds,
            "context_label_mode": args.context_label_mode,
            "capacity_notional_col": args.capacity_notional_col,
            "capacity_volume_col": args.capacity_volume_col,
            "capacity_price_col": args.capacity_price_col,
            "allow_decision_depth_fallback": args.allow_decision_depth_fallback,
            "missing_constraint": args.missing_constraint,
            "skip_plots": args.skip_plots,
        },
    )
    summary.insert(0, "scenario", scenario.name)
    summary.insert(0, "delay", delay)
    summary.insert(2, "scenario_description", scenario.description)
    return summary
