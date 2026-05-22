from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import _bootstrap  # noqa: F401
from run_opening_intraday_backtest import (
    DEFAULT_ENTRY_TIMES,
    build_summary,
    load_replay_context,
    normalize_time,
    parse_status_values,
    plot_contact_sheet,
    plot_daily_curves,
    run_backtest,
)


DEFAULT_DELAYS = ("delay0", "delay1", "delay2")
RUN_ID_TEMPLATE = "lgbm_opening_1y_next_month{strong_suffix}_{delay}"
DEFAULT_SCENARIOS = (
    "proxy_top20",
    "cost_10bps",
    "tradable_cost",
    "tradable_cost_5s",
    "liquidity_cost",
    "capacity_10m_5pct",
    "strict_10m_2pct",
)
CORE_PREDICTION_COLUMNS = ("date", "symbol", "prediction", "label")
TIME_PREDICTION_COLUMNS = ("decision_target_timestamp", "timestamp")


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    description: str
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    tradable_statuses: tuple[str, ...] = ()
    require_entry_status: bool = False
    max_decision_lag_seconds: float | None = None
    max_entry_tick_gap_seconds: float | None = None
    max_spread_bps: float | None = None
    min_limit_up_room_bps: float | None = None
    min_ask_volume_1: float | None = None
    min_bid_volume_1: float | None = None
    min_capacity_notional: float = 0.0
    max_participation_rate: float = 0.0
    ask_depth_levels: int = 0
    ask_depth_participation_rate: float = 1.0
    ask_depth_fill_mode: str = "filter"
    max_symbol_trades_per_day: int = 1
    symbol_cooldown_minutes: int = 0
    max_symbol_weight: float = 0.0


SCENARIOS: dict[str, ReplayScenario] = {
    "proxy_top20": ReplayScenario(
        name="proxy_top20",
        description="No extra cost; Top20 replay with at most one selection per symbol per day.",
    ),
    "cost_10bps": ReplayScenario(
        name="cost_10bps",
        description="Adds 5 bps fee and 5 bps slippage.",
        fee_bps=5.0,
        slippage_bps=5.0,
    ),
    "tradable_cost": ReplayScenario(
        name="tradable_cost",
        description=(
            "Cost plus continuous-trading status, decision lag <=5s, "
            "and entry tick-path gap <=10s."
        ),
        fee_bps=5.0,
        slippage_bps=5.0,
        tradable_statuses=("T0", "20", "TRADE"),
        require_entry_status=True,
        max_decision_lag_seconds=5.0,
        max_entry_tick_gap_seconds=10.0,
    ),
    "tradable_cost_5s": ReplayScenario(
        name="tradable_cost_5s",
        description=(
            "Stricter freshness stress: cost plus continuous-trading status, "
            "decision lag <=5s, and entry tick-path gap <=5s."
        ),
        fee_bps=5.0,
        slippage_bps=5.0,
        tradable_statuses=("T0", "20", "TRADE"),
        require_entry_status=True,
        max_decision_lag_seconds=5.0,
        max_entry_tick_gap_seconds=5.0,
    ),
    "liquidity_cost": ReplayScenario(
        name="liquidity_cost",
        description=(
            "Tradable-cost scenario plus spread and positive decision-time "
            "top-of-book depth checks."
        ),
        fee_bps=5.0,
        slippage_bps=5.0,
        tradable_statuses=("T0", "20", "TRADE"),
        require_entry_status=True,
        max_decision_lag_seconds=5.0,
        max_entry_tick_gap_seconds=10.0,
        max_spread_bps=100.0,
        min_ask_volume_1=1.0,
        min_bid_volume_1=1.0,
    ),
    "capacity_10m_5pct": ReplayScenario(
        name="capacity_10m_5pct",
        description=(
            "Liquidity-cost scenario plus 10mm CNY/cycle, 5% turnover "
            "participation, and ten-level entry ask sweep."
        ),
        fee_bps=5.0,
        slippage_bps=5.0,
        tradable_statuses=("T0", "20", "TRADE"),
        require_entry_status=True,
        max_decision_lag_seconds=5.0,
        max_entry_tick_gap_seconds=10.0,
        max_spread_bps=100.0,
        min_ask_volume_1=1.0,
        min_bid_volume_1=1.0,
        min_capacity_notional=100_000.0,
        max_participation_rate=0.05,
        ask_depth_levels=10,
        ask_depth_participation_rate=1.0,
        ask_depth_fill_mode="sweep",
    ),
    "strict_10m_2pct": ReplayScenario(
        name="strict_10m_2pct",
        description=(
            "Stricter freshness, spread, participation, and displayed entry "
            "ask-depth stress for a 10mm CNY/cycle book."
        ),
        fee_bps=5.0,
        slippage_bps=10.0,
        tradable_statuses=("T0", "20", "TRADE"),
        require_entry_status=True,
        max_decision_lag_seconds=5.0,
        max_entry_tick_gap_seconds=5.0,
        max_spread_bps=50.0,
        min_ask_volume_1=1.0,
        min_bid_volume_1=1.0,
        min_capacity_notional=200_000.0,
        max_participation_rate=0.02,
        ask_depth_levels=10,
        ask_depth_participation_rate=0.5,
        ask_depth_fill_mode="sweep",
    ),
    "limit_up_room_10s": ReplayScenario(
        name="limit_up_room_10s",
        description=(
            "Optional limit-up-room stress layered on liquidity_cost. Requires "
            "ask1_to_limit_up_bps in predictions or context."
        ),
        fee_bps=5.0,
        slippage_bps=5.0,
        tradable_statuses=("T0", "20", "TRADE"),
        require_entry_status=True,
        max_decision_lag_seconds=5.0,
        max_entry_tick_gap_seconds=10.0,
        max_spread_bps=100.0,
        min_limit_up_room_bps=5.0,
        min_ask_volume_1=1.0,
        min_bid_volume_1=1.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standard constrained replay grid for LightGBM delay0/1/2 "
            "prediction files after they are fetched from the k8s PVC."
        )
    )
    parser.add_argument(
        "--prediction-root",
        default="output/predictions",
        help="Directory containing <run_id>/predictions_all.parquet.",
    )
    parser.add_argument(
        "--output-root",
        default="output/reports/opening_intraday_lgbm_delay_replays",
        help="Root directory for per-scenario replay outputs.",
    )
    parser.add_argument(
        "--delay",
        action="append",
        choices=DEFAULT_DELAYS,
        help="Delay branch to replay. Defaults to delay0, delay1, and delay2.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(SCENARIOS),
        help=(
            "Scenario to run. Defaults to the standard scenarios; optional "
            "scenarios such as limit_up_room_10s must be selected explicitly."
        ),
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--entry-time",
        action="append",
        dest="entry_times",
        help="Entry decision time. Defaults to 09:30/32/34/36/38.",
    )
    parser.add_argument("--cycle-minutes", type=int, default=2)
    parser.add_argument(
        "--capital-per-cycle",
        type=float,
        default=10_000_000.0,
        help="CNY capital used by participation-rate scenarios.",
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
        "--capacity-notional-col",
        default="turnover_diff_30t",
        help="Visible notional proxy column used by capacity scenarios.",
    )
    parser.add_argument(
        "--capacity-volume-col",
        default="",
        help="Fallback visible volume capacity column.",
    )
    parser.add_argument("--capacity-price-col", default="ask_price_1")
    parser.add_argument(
        "--allow-decision-depth-fallback",
        action="store_true",
        help=(
            "Use decision-time ask depth when entry_ask_* columns are absent. "
            "Prefer leaving this off for delay experiments."
        ),
    )
    parser.add_argument(
        "--missing-constraint",
        choices=["error", "warn", "ignore"],
        default="error",
        help=(
            "Use error by default so scenario names cannot silently overstate "
            "which execution constraints were actually applied."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing prediction files instead of failing.",
    )
    parser.add_argument(
        "--check-interface-only",
        action="store_true",
        help=(
            "Validate prediction/context columns and delay metadata for the "
            "selected replay grid, then exit without running replay."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV/JSON only, without daily curve PNGs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs and scenarios without reading prediction files.",
    )
    return parser.parse_args()


def run_id_for(delay: str, *, strong: bool) -> str:
    return RUN_ID_TEMPLATE.format(
        delay=delay,
        strong_suffix="_strong" if strong else "",
    )


def prediction_runs(prediction_root: Path, delay: str) -> list[tuple[str, Path]]:
    runs = []
    for label, strong in ((f"lgbm_{delay}", False), (f"lgbm_strong_{delay}", True)):
        run_id = run_id_for(delay, strong=strong)
        runs.append((label, prediction_root / run_id / "predictions_all.parquet"))
    return runs


def delay_entry_ticks(delay: str) -> int:
    token = str(delay).strip().lower()
    if not token.startswith("delay"):
        raise SystemExit(f"cannot infer entry tick delay from {delay!r}")
    return int(token.removeprefix("delay"))


def validate_context_delay(
    context: pd.DataFrame,
    *,
    delay: str,
) -> None:
    if context.empty:
        return
    expected = float(delay_entry_ticks(delay))
    column = "entry_delay_ticks"
    if column not in context.columns:
        raise SystemExit(
            f"{delay}: context input is missing {column!r}; cannot verify that "
            "the replay context matches this delay branch. Use raw tick context "
            "or a labeled context/cache materialized for the same delay."
        )
    values = pd.to_numeric(context[column], errors="coerce").dropna()
    if values.empty:
        raise SystemExit(
            f"{delay}: context input has no non-null {column!r}; cannot verify "
            "the context delay branch."
        )
    bad = values.ne(expected)
    if bool(bad.any()):
        observed = sorted(float(value) for value in values.drop_duplicates().head(8))
        raise SystemExit(
            f"{delay}: context delay mismatch; expected {column}={expected:g}, "
            f"observed sample={observed}. Use a per-delay labeled context or raw "
            "tick context so replay can derive labels for each delay branch."
        )


def replay_required_columns(
    scenarios: list[ReplayScenario],
    *,
    capacity_notional_col: str,
    capacity_volume_col: str,
    capacity_price_col: str,
    allow_decision_depth_fallback: bool,
) -> set[str]:
    required: set[str] = set()
    for scenario in scenarios:
        if scenario.tradable_statuses:
            required.add("status")
            if scenario.require_entry_status:
                required.add("entry_status")
        if scenario.max_decision_lag_seconds is not None:
            required.add("decision_lag_seconds")
        if scenario.max_entry_tick_gap_seconds is not None:
            required.add("entry_max_tick_gap_seconds")
        if scenario.max_spread_bps is not None:
            required.add("spread_bps")
        if scenario.min_limit_up_room_bps is not None:
            required.add("ask1_to_limit_up_bps")
        if scenario.min_ask_volume_1 is not None:
            required.add("ask_volume_1")
        if scenario.min_bid_volume_1 is not None:
            required.add("bid_volume_1")
        if scenario.min_capacity_notional > 0 or scenario.max_participation_rate > 0:
            if capacity_notional_col:
                required.add(capacity_notional_col)
            elif capacity_volume_col:
                required.update({capacity_volume_col, capacity_price_col})
        if scenario.ask_depth_levels > 0 and not allow_decision_depth_fallback:
            for level in range(1, int(scenario.ask_depth_levels) + 1):
                required.update(
                    {
                        f"entry_ask_price_{level}",
                        f"entry_ask_volume_{level}",
                    }
                )
    return required


def _prediction_columns(path: Path) -> list[str]:
    try:
        return pq.ParquetFile(path).schema_arrow.names
    except Exception as exc:
        raise SystemExit(f"{path}: cannot read prediction parquet schema: {exc}") from exc


def _validate_prediction_delay(path: Path, *, delay: str) -> None:
    expected = float(delay_entry_ticks(delay))
    column = "entry_delay_ticks"
    try:
        values = pd.read_parquet(path, columns=[column])[column]
    except Exception as exc:
        raise SystemExit(f"{delay}: cannot read {column!r} from {path}: {exc}") from exc
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        raise SystemExit(f"{delay}: {path} has no non-null {column!r}")
    bad = values.ne(expected)
    if bool(bad.any()):
        observed = sorted(float(value) for value in values.drop_duplicates().head(8))
        raise SystemExit(
            f"{delay}: prediction delay mismatch in {path}; expected "
            f"{column}={expected:g}, observed sample={observed}"
        )


def _validate_ask_depth_alternatives(
    *,
    available: set[str],
    scenarios: list[ReplayScenario],
    context_columns: set[str],
    path: Path,
) -> None:
    levels = max((int(scenario.ask_depth_levels) for scenario in scenarios), default=0)
    if levels <= 0:
        return
    missing = []
    for level in range(1, levels + 1):
        entry_pair = {f"entry_ask_price_{level}", f"entry_ask_volume_{level}"}
        decision_pair = {f"ask_price_{level}", f"ask_volume_{level}"}
        if entry_pair.issubset(available) or decision_pair.issubset(available):
            continue
        missing.append(
            f"level{level}: {'/'.join(sorted(entry_pair))} or "
            f"{'/'.join(sorted(decision_pair))}"
        )
    if missing:
        source = "prediction/context" if context_columns else "prediction"
        raise SystemExit(
            f"{path}: {source} missing ask-depth interface columns for "
            f"--allow-decision-depth-fallback: {missing[:5]}"
        )


def validate_prediction_interface(
    path: Path,
    *,
    delay: str,
    scenarios: list[ReplayScenario],
    context_columns: set[str] | None = None,
    capacity_notional_col: str = "turnover_diff_30t",
    capacity_volume_col: str = "",
    capacity_price_col: str = "ask_price_1",
    allow_decision_depth_fallback: bool = False,
) -> dict[str, object]:
    prediction_columns = set(_prediction_columns(path))
    context_columns = set(context_columns or set())
    missing_core = sorted(set(CORE_PREDICTION_COLUMNS) - prediction_columns)
    if missing_core:
        raise SystemExit(f"{path}: missing core prediction columns: {missing_core}")
    if not prediction_columns.intersection(TIME_PREDICTION_COLUMNS):
        raise SystemExit(
            f"{path}: missing a decision timestamp column; expected one of "
            f"{list(TIME_PREDICTION_COLUMNS)}"
        )

    available = prediction_columns | context_columns
    required = replay_required_columns(
        scenarios,
        capacity_notional_col=capacity_notional_col,
        capacity_volume_col=capacity_volume_col,
        capacity_price_col=capacity_price_col,
        allow_decision_depth_fallback=allow_decision_depth_fallback,
    )
    missing = sorted(required - available)
    if missing:
        source = "prediction/context" if context_columns else "prediction"
        raise SystemExit(
            f"{delay}: {path} missing replay interface columns in {source}: "
            f"{missing}. Fetch CPU LightGBM predictions generated from the "
            "delay labeled cache, or provide a matching --context-input."
        )

    if allow_decision_depth_fallback:
        _validate_ask_depth_alternatives(
            available=available,
            scenarios=scenarios,
            context_columns=context_columns,
            path=path,
        )

    if "entry_delay_ticks" in prediction_columns:
        _validate_prediction_delay(path, delay=delay)
        delay_source = "prediction"
    elif "entry_delay_ticks" in context_columns:
        delay_source = "context"
    else:
        raise SystemExit(
            f"{delay}: {path} missing 'entry_delay_ticks'; cannot verify that "
            "this prediction file matches the replay delay branch."
        )

    return {
        "path": str(path),
        "prediction_columns": len(prediction_columns),
        "context_columns": len(context_columns),
        "required_columns": sorted(required),
        "delay_source": delay_source,
    }


def missing_prediction_paths(runs: list[tuple[str, Path]]) -> list[Path]:
    return [path for _, path in runs if not path.exists()]


def write_trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def replay_scenario(
    *,
    delay: str,
    scenario: ReplayScenario,
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
        capital_per_cycle=args.capital_per_cycle,
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
            "context_input": args.context_input,
            "context_kind": args.context_kind,
            "context_entry_tick_delay": delay_entry_ticks(delay)
            if args.context_input
            else None,
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


def main() -> None:
    args = parse_args()
    prediction_root = Path(args.prediction_root)
    output_root = Path(args.output_root)
    delays = args.delay or list(DEFAULT_DELAYS)
    scenario_names = args.scenario or list(DEFAULT_SCENARIOS)
    entry_times = [normalize_time(value) for value in (args.entry_times or DEFAULT_ENTRY_TIMES)]

    plan = []
    for delay in delays:
        runs = prediction_runs(prediction_root, delay)
        for scenario_name in scenario_names:
            plan.append(
                {
                    "delay": delay,
                    "scenario": scenario_name,
                    "runs": [{"label": label, "path": str(path)} for label, path in runs],
                    "output_dir": str(output_root / delay / scenario_name),
                }
            )

    if args.dry_run:
        print(json.dumps({"replay_plan": plan}, indent=2))
        return

    context_by_delay: dict[str, pd.DataFrame] = {}
    if args.context_input:
        for delay in delays:
            context = load_replay_context(
                args.context_input,
                kind=args.context_kind,
                entry_times=entry_times,
                entry_tick_delay=delay_entry_ticks(delay),
                entry_max_gap_seconds=args.context_entry_max_gap_seconds,
                decision_max_lag_seconds=args.context_decision_max_lag_seconds,
            )
            validate_context_delay(context, delay=delay)
            context_by_delay[delay] = context

    selected_scenarios = [SCENARIOS[name] for name in scenario_names]
    summaries = []
    skipped = []
    runs_by_delay: dict[str, list[tuple[str, Path]]] = {}
    interface_checks = []
    for delay in delays:
        runs = prediction_runs(prediction_root, delay)
        missing = missing_prediction_paths(runs)
        if missing:
            message = f"{delay}: missing prediction files: {', '.join(str(path) for path in missing)}"
            if args.allow_missing:
                print(f"skip_warning: {message}")
                skipped.append({"delay": delay, "missing": [str(path) for path in missing]})
                continue
            raise SystemExit(message)

        context_columns = set(context_by_delay.get(delay, pd.DataFrame()).columns)
        for label, path in runs:
            check = validate_prediction_interface(
                path,
                delay=delay,
                scenarios=selected_scenarios,
                context_columns=context_columns,
                capacity_notional_col=args.capacity_notional_col,
                capacity_volume_col=args.capacity_volume_col,
                capacity_price_col=args.capacity_price_col,
                allow_decision_depth_fallback=args.allow_decision_depth_fallback,
            )
            interface_checks.append({"delay": delay, "run": label, **check})
        runs_by_delay[delay] = runs

    if args.check_interface_only:
        print(
            json.dumps(
                {
                    "interface_check": interface_checks,
                    "skipped": skipped,
                    "scenarios": scenario_names,
                },
                indent=2,
            )
        )
        return

    for delay in delays:
        runs = runs_by_delay.get(delay)
        if not runs:
            continue
        for scenario_name in scenario_names:
            scenario = SCENARIOS[scenario_name]
            output_dir = output_root / delay / scenario.name
            print(f"running_replay: delay={delay} scenario={scenario.name}")
            summaries.append(
                replay_scenario(
                    delay=delay,
                    scenario=scenario,
                    runs=runs,
                    output_dir=output_dir,
                    args=args,
                    entry_times=entry_times,
                    context_frame=context_by_delay.get(delay),
                )
            )

    output_root.mkdir(parents=True, exist_ok=True)
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(output_root / "scenario_summary.csv", index=False)
        print("\nreplay_scenario_summary:")
        print(combined.to_string(index=False, float_format="{:.6f}".format))

    write_trace(
        output_root / "scenario_trace.json",
        {
            "prediction_root": str(prediction_root),
            "output_root": str(output_root),
            "delays": list(delays),
            "default_scenarios": list(DEFAULT_SCENARIOS),
            "scenarios": [asdict(SCENARIOS[name]) for name in scenario_names],
            "entry_times": entry_times,
            "top_n": args.top_n,
            "cycle_minutes": args.cycle_minutes,
            "capital_per_cycle": args.capital_per_cycle,
            "context_input": args.context_input,
            "context_kind": args.context_kind,
            "context_entry_max_gap_seconds": args.context_entry_max_gap_seconds,
            "context_decision_max_lag_seconds": args.context_decision_max_lag_seconds,
            "context_label_mode": args.context_label_mode,
            "allow_decision_depth_fallback": args.allow_decision_depth_fallback,
            "missing_constraint": args.missing_constraint,
            "interface_checks": interface_checks,
            "skipped": skipped,
        },
    )
    print(f"\nwrote replay grid: {output_root}")


if __name__ == "__main__":
    main()
