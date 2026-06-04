from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from opening_strength_fit.commands.opening_backtest_constraints import (
    apply_static_constraints,
    clock_timestamp,
    enrich_predictions_with_context,
    load_predictions,
    load_replay_context,
    slot_weight,
)

DEFAULT_RUNS = (
    ("gbm", "output/predictions/gbm_opening_1y_next_month/predictions_all.parquet"),
    (
        "gbm_strong",
        "output/predictions/gbm_opening_1y_next_month_strong/predictions_all.parquet",
    ),
)
DEFAULT_ENTRY_TIMES = ("09:30:00", "09:32:00", "09:34:00", "09:36:00", "09:38:00")


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an opening-window intraday top-N backtest directly on tick prediction "
            "parquet files. Each day starts with capital 1.0, selects eligible top "
            "scores at non-overlapping entry times, and compounds weighted realized "
            "label returns."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        help=(
            "Prediction parquet formatted as label=path. Defaults to the GBM and "
            "GBM strong 1y next-month runs if omitted."
        ),
    )
    parser.add_argument("--output-dir", default="output/reports/opening_intraday_top20_1y")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--entry-time",
        action="append",
        dest="entry_times",
        help=(
            "Entry decision time in HH:MM or HH:MM:SS. May be repeated. "
            "Defaults to 09:30/32/34/36/38."
        ),
    )
    parser.add_argument(
        "--cycle-minutes",
        type=int,
        default=2,
        help="Minutes from entry to realized exit point for plotting.",
    )
    parser.add_argument(
        "--context-input",
        default="",
        help=(
            "Optional raw tick or labeled research parquet/csv root used to enrich "
            "prediction rows with replay-only execution context."
        ),
    )
    parser.add_argument(
        "--context-kind",
        choices=["auto", "raw_ticks", "labeled"],
        default="auto",
        help="Whether --context-input is raw ticks or an already labeled research dataset.",
    )
    parser.add_argument(
        "--context-entry-tick-delay",
        type=int,
        default=0,
        help="Entry tick delay used when deriving replay context from raw ticks.",
    )
    parser.add_argument(
        "--context-entry-max-gap-seconds",
        type=int,
        default=None,
        help=(
            "Maximum adjacent tick gap on the decision-to-entry path when deriving "
            "context from raw ticks."
        ),
    )
    parser.add_argument(
        "--context-decision-max-lag-seconds",
        type=int,
        default=5,
        help="Decision point sampling lag when deriving context from raw ticks.",
    )
    parser.add_argument(
        "--context-label-mode",
        choices=["keep", "fill", "replace"],
        default="replace",
        help=(
            "How to use label from --context-input: keep prediction label, fill only "
            "missing labels, or replace prediction label for replay PnL."
        ),
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=0.0,
        help="Extra per-cycle return haircut in bps. Current labels already use config fee_bps.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Extra per-cycle execution slippage haircut in bps.",
    )
    parser.add_argument(
        "--tradable-status",
        action="append",
        dest="tradable_statuses",
        help="Allowed status value. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--require-entry-status",
        action="store_true",
        help="Require entry_status to exist and pass --tradable-status when statuses are set.",
    )
    parser.add_argument("--max-decision-lag-seconds", type=float, default=None)
    parser.add_argument(
        "--max-entry-tick-gap-seconds",
        type=float,
        default=None,
        help=(
            "Maximum adjacent tick gap on the decision-to-entry path. This is "
            "entry freshness and is independent of --context-entry-tick-delay."
        ),
    )
    parser.add_argument("--max-spread-bps", type=float, default=None)
    parser.add_argument("--min-limit-up-room-bps", type=float, default=None)
    parser.add_argument("--min-ask-volume-1", type=float, default=None)
    parser.add_argument("--min-bid-volume-1", type=float, default=None)
    parser.add_argument(
        "--capacity-notional-col",
        default="turnover_diff_30t",
        help="Column used as visible per-row notional capacity.",
    )
    parser.add_argument(
        "--capacity-volume-col",
        default="",
        help="Fallback visible volume capacity column; multiplied by --capacity-price-col.",
    )
    parser.add_argument(
        "--capacity-price-col",
        default="ask_price_1",
        help="Price column used with --capacity-volume-col.",
    )
    parser.add_argument("--min-capacity-notional", type=float, default=0.0)
    parser.add_argument("--max-participation-rate", type=float, default=0.0)
    parser.add_argument(
        "--capital-per-cycle",
        type=float,
        default=0.0,
        help="Capital in CNY for per-name participation checks.",
    )
    parser.add_argument(
        "--ask-depth-levels",
        type=int,
        default=0,
        help=(
            "Number of entry ask levels to use for displayed sell-side depth. "
            "0 disables ask-depth execution checks."
        ),
    )
    parser.add_argument(
        "--ask-depth-participation-rate",
        type=float,
        default=1.0,
        help=(
            "Usable fraction of displayed ask-side depth when --ask-depth-levels "
            "is set; must be in (0, 1]."
        ),
    )
    parser.add_argument(
        "--ask-depth-fill-mode",
        choices=["filter", "scale", "sweep"],
        default="filter",
        help=(
            "filter drops rows without enough ask depth; scale partially fills and leaves "
            "cash; sweep requires enough depth and adjusts returns by sweep VWAP."
        ),
    )
    parser.add_argument(
        "--allow-decision-depth-fallback",
        action="store_true",
        help=(
            "Use decision-time ask_price/ask_volume columns when entry_ask_* columns "
            "are missing. Prefer leaving this off for delay experiments."
        ),
    )
    parser.add_argument(
        "--max-symbol-trades-per-day",
        type=int,
        default=1,
        help="Maximum times a symbol can be selected in one day. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--symbol-cooldown-minutes",
        type=int,
        default=0,
        help="Minutes after a selected cycle exit before the same symbol can be selected again.",
    )
    parser.add_argument(
        "--max-symbol-weight",
        type=float,
        default=0.0,
        help="Per-symbol capital weight cap. 0 means the normal 1/top_n slot weight.",
    )
    parser.add_argument(
        "--missing-constraint",
        choices=["error", "warn", "ignore"],
        default="error",
        help="What to do when a requested constraint column is absent.",
    )
    parser.add_argument("--score-col", default="prediction")
    parser.add_argument("--label-col", default="label")
    return parser.parse_args()


def normalize_time(value: object) -> str:
    text = str(value)
    if len(text) == 5:
        text = f"{text}:00"
    return text


def parse_status_values(values: list[str] | None) -> set[str]:
    statuses: set[str] = set()
    for value in values or []:
        for part in str(value).replace(",", " ").split():
            part = part.strip()
            if part:
                statuses.add(part.upper())
    return statuses


def exit_time(entry_time: str, cycle_minutes: int) -> str:
    timestamp = pd.Timestamp(f"2000-01-01 {entry_time}") + pd.Timedelta(minutes=cycle_minutes)
    return timestamp.strftime("%H:%M:%S")


def run_backtest(
    runs: list[tuple[str, Path]],
    *,
    entry_times: list[str],
    context_frame: pd.DataFrame | None,
    context_label_mode: str,
    cycle_minutes: int,
    top_n: int,
    fee_bps: float,
    slippage_bps: float,
    tradable_statuses: set[str],
    require_entry_status: bool,
    max_decision_lag_seconds: float | None,
    max_entry_tick_gap_seconds: float | None,
    max_spread_bps: float | None,
    min_limit_up_room_bps: float | None,
    min_ask_volume_1: float | None,
    min_bid_volume_1: float | None,
    capacity_notional_col: str,
    capacity_volume_col: str,
    capacity_price_col: str,
    min_capacity_notional: float,
    max_participation_rate: float,
    capital_per_cycle: float,
    ask_depth_levels: int,
    ask_depth_participation_rate: float,
    ask_depth_fill_mode: str,
    allow_decision_depth_fallback: bool,
    max_symbol_trades_per_day: int,
    symbol_cooldown_minutes: int,
    max_symbol_weight: float,
    missing_policy: str,
    score_col: str,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_rows = []
    selected_rows = []
    cost_return = (float(fee_bps) + float(slippage_bps)) / 10_000.0
    per_symbol_weight = slot_weight(top_n=top_n, max_symbol_weight=max_symbol_weight)

    for label, path in runs:
        predictions = load_predictions(path, score_col=score_col, label_col=label_col)
        if context_frame is not None and not context_frame.empty:
            predictions = enrich_predictions_with_context(
                predictions,
                context_frame,
                run_label=label,
                label_col=label_col,
                context_label_mode=context_label_mode,
            )
        predictions = predictions.loc[predictions["entry_time"].isin(entry_times)].copy()
        if predictions.empty:
            raise SystemExit(f"{label}: no prediction rows after filtering entry times")
        constrained = apply_static_constraints(
            predictions,
            run_label=label,
            top_n=top_n,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            tradable_statuses=tradable_statuses,
            require_entry_status=require_entry_status,
            max_decision_lag_seconds=max_decision_lag_seconds,
            max_entry_tick_gap_seconds=max_entry_tick_gap_seconds,
            max_spread_bps=max_spread_bps,
            min_limit_up_room_bps=min_limit_up_room_bps,
            min_ask_volume_1=min_ask_volume_1,
            min_bid_volume_1=min_bid_volume_1,
            capacity_notional_col=capacity_notional_col,
            capacity_volume_col=capacity_volume_col,
            capacity_price_col=capacity_price_col,
            min_capacity_notional=min_capacity_notional,
            max_participation_rate=max_participation_rate,
            capital_per_cycle=capital_per_cycle,
            ask_depth_levels=ask_depth_levels,
            ask_depth_participation_rate=ask_depth_participation_rate,
            ask_depth_fill_mode=ask_depth_fill_mode,
            allow_decision_depth_fallback=allow_decision_depth_fallback,
            max_symbol_weight=max_symbol_weight,
            missing_policy=missing_policy,
        )

        for trading_day, raw_day_frame in predictions.groupby("date", sort=True):
            day_frame = constrained.loc[constrained["date"] == trading_day]
            capital = 1.0
            selected_by_symbol: dict[str, int] = {}
            cooldown_until: dict[str, pd.Timestamp] = {}
            for entry in entry_times:
                raw_group = raw_day_frame.loc[raw_day_frame["entry_time"] == entry]
                group = day_frame.loc[day_frame["entry_time"] == entry]
                if max_symbol_trades_per_day and max_symbol_trades_per_day > 0:
                    group = group.loc[
                        group["symbol"].map(
                            lambda symbol, selected_by_symbol=selected_by_symbol, max_symbol_trades_per_day=int(max_symbol_trades_per_day): (
                                selected_by_symbol.get(str(symbol), 0) < max_symbol_trades_per_day
                            )
                        )
                    ]
                if symbol_cooldown_minutes and symbol_cooldown_minutes > 0:
                    entry_ts = clock_timestamp(entry)
                    group = group.loc[
                        group["symbol"].map(
                            lambda symbol, cooldown_until=cooldown_until, entry_ts=entry_ts: (
                                cooldown_until.get(
                                    str(symbol),
                                    pd.Timestamp.min,
                                )
                                <= entry_ts
                            )
                        )
                    ]

                if group.empty:
                    cycle_return = 0.0
                    selected_count = 0
                    median_return = float("nan")
                    win_rate = float("nan")
                    cash_weight = 1.0
                else:
                    selected = (
                        group.sort_values([score_col, "symbol"], ascending=[False, True])
                        .head(top_n)
                        .copy()
                    )
                    selected["rank"] = range(1, len(selected) + 1)
                    selected["weight"] = per_symbol_weight
                    if "_depth_fill_ratio" in selected.columns:
                        selected["weight"] = selected["weight"] * pd.to_numeric(
                            selected["_depth_fill_ratio"],
                            errors="coerce",
                        ).fillna(1.0).clip(lower=0.0, upper=1.0)
                    selected["execution_label"] = pd.to_numeric(
                        selected[label_col],
                        errors="coerce",
                    )
                    if "_sweep_buy_price" in selected.columns and "buy_price" in selected.columns:
                        buy_price = pd.to_numeric(selected["buy_price"], errors="coerce")
                        sweep_buy_price = pd.to_numeric(
                            selected["_sweep_buy_price"],
                            errors="coerce",
                        )
                        valid_sweep = buy_price.gt(0) & sweep_buy_price.gt(0)
                        selected.loc[valid_sweep, "execution_label"] = (
                            1.0 + selected.loc[valid_sweep, label_col].astype("float64")
                        ) * buy_price.loc[valid_sweep] / sweep_buy_price.loc[valid_sweep] - 1.0
                    selected["net_label"] = selected["execution_label"] - cost_return
                    selected_count = int(len(selected))
                    cycle_return = float((selected["net_label"] * selected["weight"]).sum())
                    median_return = float(selected["net_label"].median())
                    win_rate = float((selected["net_label"] > 0).mean())
                    cash_weight = max(0.0, 1.0 - float(selected["weight"].sum()))
                    cooldown_release = clock_timestamp(entry) + pd.Timedelta(
                        minutes=cycle_minutes + int(symbol_cooldown_minutes)
                    )
                    for _, row in selected.iterrows():
                        symbol = str(row["symbol"])
                        selected_by_symbol[symbol] = selected_by_symbol.get(symbol, 0) + 1
                        cooldown_until[symbol] = cooldown_release
                        selected_rows.append(
                            {
                                "run": label,
                                "date": trading_day,
                                "entry_time": entry,
                                "exit_time": exit_time(entry, cycle_minutes),
                                "rank": int(row["rank"]),
                                "symbol": symbol,
                                "weight": float(row["weight"]),
                                "prediction": float(row[score_col]),
                                "prediction_label": (
                                    float(row["_prediction_label"])
                                    if "_prediction_label" in row
                                    and pd.notna(row["_prediction_label"])
                                    else float("nan")
                                ),
                                "label": float(row[label_col]),
                                "execution_label": float(row["execution_label"]),
                                "net_label": float(row["net_label"]),
                                "cost_bps": float(row["_cost_bps"]),
                                "depth_fill_ratio": (
                                    float(row["_depth_fill_ratio"])
                                    if "_depth_fill_ratio" in row
                                    and pd.notna(row["_depth_fill_ratio"])
                                    else float("nan")
                                ),
                                "ask_depth_notional": (
                                    float(row["_ask_depth_notional"])
                                    if "_ask_depth_notional" in row
                                    and pd.notna(row["_ask_depth_notional"])
                                    else float("nan")
                                ),
                                "ask_depth_usable_notional": (
                                    float(row["_ask_depth_usable_notional"])
                                    if "_ask_depth_usable_notional" in row
                                    and pd.notna(row["_ask_depth_usable_notional"])
                                    else float("nan")
                                ),
                                "ask_depth_target_notional": (
                                    float(row["_ask_depth_target_notional"])
                                    if "_ask_depth_target_notional" in row
                                    and pd.notna(row["_ask_depth_target_notional"])
                                    else float("nan")
                                ),
                                "sweep_buy_price": (
                                    float(row["_sweep_buy_price"])
                                    if "_sweep_buy_price" in row
                                    and pd.notna(row["_sweep_buy_price"])
                                    else float("nan")
                                ),
                                "depth_price_impact_bps": (
                                    float(row["_depth_price_impact_bps"])
                                    if "_depth_price_impact_bps" in row
                                    and pd.notna(row["_depth_price_impact_bps"])
                                    else float("nan")
                                ),
                                "capacity_notional": (
                                    float(row["_capacity_notional"])
                                    if "_capacity_notional" in row
                                    and pd.notna(row["_capacity_notional"])
                                    else float("nan")
                                ),
                                "capacity_target_notional": (
                                    float(row["_capacity_target_notional"])
                                    if "_capacity_target_notional" in row
                                    and pd.notna(row["_capacity_target_notional"])
                                    else float("nan")
                                ),
                            }
                        )

                capital_start = capital
                capital *= 1.0 + cycle_return
                cycle_rows.append(
                    {
                        "run": label,
                        "date": trading_day,
                        "entry_time": entry,
                        "exit_time": exit_time(entry, cycle_minutes),
                        "candidate_count": int(len(raw_group)),
                        "eligible_count": int(len(group)),
                        "selected_count": selected_count,
                        "slot_weight": per_symbol_weight,
                        "cash_weight": cash_weight,
                        "cycle_return": cycle_return,
                        "cycle_return_bps": cycle_return * 10_000.0,
                        "median_return": median_return,
                        "win_rate": win_rate,
                        "capital_start": capital_start,
                        "capital_end": capital,
                        "cumulative_return": capital - 1.0,
                    }
                )

    return pd.DataFrame(cycle_rows), pd.DataFrame(selected_rows)


def build_summary(cycles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        cycles.sort_values(["run", "date", "entry_time"])
        .groupby(["run", "date"], as_index=False)
        .agg(
            cycles=("cycle_return", "size"),
            final_return=("cumulative_return", "last"),
            mean_cycle_return=("cycle_return", "mean"),
            positive_cycles=("cycle_return", lambda x: int((x > 0).sum())),
            min_cycle_return=("cycle_return", "min"),
            max_cycle_return=("cycle_return", "max"),
            selected_count_mean=("selected_count", "mean"),
        )
    )
    rows = []
    for run, run_daily in daily.groupby("run", sort=False):
        run_cycles = cycles.loc[cycles["run"] == run]
        final_returns = run_daily["final_return"]
        compounded_month = float((1.0 + final_returns).prod() - 1.0)
        rows.append(
            {
                "run": run,
                "dates": int(run_daily["date"].nunique()),
                "cycles": int(run_cycles.shape[0]),
                "selected_trades": int(run_cycles["selected_count"].sum()),
                "mean_cycle_return": float(run_cycles["cycle_return"].mean()),
                "mean_cycle_return_bps": float(run_cycles["cycle_return"].mean() * 10_000.0),
                "cycle_win_rate": float((run_cycles["cycle_return"] > 0).mean()),
                "mean_day_final_return": float(final_returns.mean()),
                "mean_day_final_return_bps": float(final_returns.mean() * 10_000.0),
                "median_day_final_return": float(final_returns.median()),
                "positive_day_rate": float((final_returns > 0).mean()),
                "compounded_month_return": compounded_month,
                "best_day": str(run_daily.loc[final_returns.idxmax(), "date"]),
                "best_day_return": float(final_returns.max()),
                "worst_day": str(run_daily.loc[final_returns.idxmin(), "date"]),
                "worst_day_return": float(final_returns.min()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_day_final_return", ascending=False)
    return daily, summary


def plot_daily_curves(
    cycles: pd.DataFrame,
    *,
    output_dir: Path,
    entry_times: list[str],
    cycle_minutes: int,
) -> list[Path]:
    daily_dir = output_dir / "daily_curves"
    daily_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    runs = list(dict.fromkeys(cycles["run"].astype(str)))
    entry_start = entry_times[0]
    x_labels = [entry_start] + [exit_time(entry, cycle_minutes) for entry in entry_times]
    x_positions = list(range(len(x_labels)))

    for trading_day, day_frame in cycles.groupby("date", sort=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        for run in runs:
            run_day = day_frame.loc[day_frame["run"] == run].sort_values("entry_time")
            y_values = [0.0] + run_day["cumulative_return"].astype(float).tolist()
            if len(y_values) != len(x_positions):
                continue
            ax.plot(x_positions, y_values, marker="o", linewidth=2.0, label=run)
        ax.axhline(0.0, color="#666666", linewidth=0.9)
        ax.set_title(f"Opening Top20 Intraday Cumulative Return - {trading_day}")
        ax.set_xlabel("Opening window")
        ax.set_ylabel("Cumulative return")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label[:5] for label in x_labels])
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        output_path = daily_dir / f"{trading_day}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        paths.append(output_path)
    return paths


def plot_contact_sheet(cycles: pd.DataFrame, output_path: Path) -> None:
    dates = sorted(cycles["date"].unique())
    runs = list(dict.fromkeys(cycles["run"].astype(str)))
    columns = 4
    rows = (len(dates) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.2, rows * 3.0), sharex=True)
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, trading_day in zip(axes_list, dates, strict=False):
        day_frame = cycles.loc[cycles["date"] == trading_day]
        for run in runs:
            run_day = day_frame.loc[day_frame["run"] == run].sort_values("entry_time")
            y_values = [0.0] + run_day["cumulative_return"].astype(float).tolist()
            ax.plot(range(len(y_values)), y_values, linewidth=1.3, label=run)
        ax.axhline(0.0, color="#777777", linewidth=0.7)
        ax.set_title(str(trading_day), fontsize=9)
        ax.grid(True, alpha=0.2)
    for ax in axes_list[len(dates) :]:
        ax.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(runs)))
    fig.suptitle("Opening Top20 Intraday Curves", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = args.run or [(label, Path(path)) for label, path in DEFAULT_RUNS]
    entry_times = [normalize_time(value) for value in (args.entry_times or DEFAULT_ENTRY_TIMES)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_frame = load_replay_context(
        args.context_input,
        kind=args.context_kind,
        entry_times=entry_times,
        entry_tick_delay=args.context_entry_tick_delay,
        entry_max_gap_seconds=args.context_entry_max_gap_seconds,
        decision_max_lag_seconds=args.context_decision_max_lag_seconds,
    )

    cycles, selected = run_backtest(
        runs,
        entry_times=entry_times,
        context_frame=context_frame,
        context_label_mode=args.context_label_mode,
        cycle_minutes=args.cycle_minutes,
        top_n=args.top_n,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        tradable_statuses=parse_status_values(args.tradable_statuses),
        require_entry_status=args.require_entry_status,
        max_decision_lag_seconds=args.max_decision_lag_seconds,
        max_entry_tick_gap_seconds=args.max_entry_tick_gap_seconds,
        max_spread_bps=args.max_spread_bps,
        min_limit_up_room_bps=args.min_limit_up_room_bps,
        min_ask_volume_1=args.min_ask_volume_1,
        min_bid_volume_1=args.min_bid_volume_1,
        capacity_notional_col=args.capacity_notional_col,
        capacity_volume_col=args.capacity_volume_col,
        capacity_price_col=args.capacity_price_col,
        min_capacity_notional=args.min_capacity_notional,
        max_participation_rate=args.max_participation_rate,
        capital_per_cycle=args.capital_per_cycle,
        ask_depth_levels=args.ask_depth_levels,
        ask_depth_participation_rate=args.ask_depth_participation_rate,
        ask_depth_fill_mode=args.ask_depth_fill_mode,
        allow_decision_depth_fallback=args.allow_decision_depth_fallback,
        max_symbol_trades_per_day=args.max_symbol_trades_per_day,
        symbol_cooldown_minutes=args.symbol_cooldown_minutes,
        max_symbol_weight=args.max_symbol_weight,
        missing_policy=args.missing_constraint,
        score_col=args.score_col,
        label_col=args.label_col,
    )
    daily, summary = build_summary(cycles)

    cycles_path = output_dir / "intraday_cycles.csv"
    selected_path = output_dir / "intraday_selected_trades.csv"
    daily_path = output_dir / "intraday_daily_summary.csv"
    summary_path = output_dir / "intraday_summary.csv"
    trace_path = output_dir / "intraday_trace.json"
    contact_sheet_path = output_dir / "daily_curves_contact_sheet.png"

    cycles.to_csv(cycles_path, index=False)
    selected.to_csv(selected_path, index=False)
    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    daily_plot_paths = plot_daily_curves(
        cycles,
        output_dir=output_dir,
        entry_times=entry_times,
        cycle_minutes=args.cycle_minutes,
    )
    plot_contact_sheet(cycles, contact_sheet_path)

    trace = {
        "backtested_at_utc": datetime.now(UTC).isoformat(),
        "runs": [{"label": label, "path": str(path)} for label, path in runs],
        "top_n": args.top_n,
        "entry_times": entry_times,
        "cycle_minutes": args.cycle_minutes,
        "context_input": args.context_input,
        "context_kind": args.context_kind,
        "context_entry_tick_delay": args.context_entry_tick_delay,
        "context_entry_max_gap_seconds": args.context_entry_max_gap_seconds,
        "context_decision_max_lag_seconds": args.context_decision_max_lag_seconds,
        "context_label_mode": args.context_label_mode,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "tradable_statuses": sorted(parse_status_values(args.tradable_statuses)),
        "require_entry_status": args.require_entry_status,
        "max_decision_lag_seconds": args.max_decision_lag_seconds,
        "max_entry_tick_gap_seconds": args.max_entry_tick_gap_seconds,
        "max_spread_bps": args.max_spread_bps,
        "min_limit_up_room_bps": args.min_limit_up_room_bps,
        "min_ask_volume_1": args.min_ask_volume_1,
        "min_bid_volume_1": args.min_bid_volume_1,
        "capacity_notional_col": args.capacity_notional_col,
        "capacity_volume_col": args.capacity_volume_col,
        "capacity_price_col": args.capacity_price_col,
        "min_capacity_notional": args.min_capacity_notional,
        "max_participation_rate": args.max_participation_rate,
        "capital_per_cycle": args.capital_per_cycle,
        "ask_depth_levels": args.ask_depth_levels,
        "ask_depth_participation_rate": args.ask_depth_participation_rate,
        "ask_depth_fill_mode": args.ask_depth_fill_mode,
        "allow_decision_depth_fallback": args.allow_decision_depth_fallback,
        "max_symbol_trades_per_day": args.max_symbol_trades_per_day,
        "symbol_cooldown_minutes": args.symbol_cooldown_minutes,
        "max_symbol_weight": args.max_symbol_weight,
        "missing_constraint": args.missing_constraint,
        "score_col": args.score_col,
        "label_col": args.label_col,
        "output_dir": str(output_dir),
        "daily_plot_count": len(daily_plot_paths),
        "contact_sheet": str(contact_sheet_path),
    }
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    print("opening_intraday_backtest_complete:")
    print(f"  output_dir: {output_dir}")
    print(f"  cycles: {cycles_path}")
    print(f"  selected_trades: {selected_path}")
    print(f"  daily_summary: {daily_path}")
    print(f"  summary: {summary_path}")
    print(f"  trace: {trace_path}")
    print(f"  daily_plots: {len(daily_plot_paths)} in {output_dir / 'daily_curves'}")
    print(f"  contact_sheet: {contact_sheet_path}")
    print()
    print(summary.to_string(index=False, float_format="{:.6f}".format))


if __name__ == "__main__":
    main()
