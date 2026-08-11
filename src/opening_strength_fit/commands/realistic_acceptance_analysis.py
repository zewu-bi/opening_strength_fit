from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from opening_strength_fit.analysis import write_json
from opening_strength_fit.artifact_catalog import record_requested_artifacts
from opening_strength_fit.capacity_acceptance import load_label_frame
from opening_strength_fit.commands.arguments import CommandArguments
from opening_strength_fit.config import load_toml, run_id
from opening_strength_fit.realistic_acceptance import (
    DEFAULT_REALISTIC_LABEL_COL,
    REALISTIC_DAILY_SUMMARY,
    REALISTIC_SELECTED,
    REALISTIC_SUMMARY,
    REALISTIC_TRACE,
    RealisticExecutionConstraints,
    apply_realistic_execution_constraints,
    constraints_trace,
    load_realistic_execution_context,
    load_realistic_selected,
    merge_realistic_execution_context,
    realistic_context_columns,
    summarize_realistic_acceptance,
    summarize_realistic_acceptance_overall,
)

ARTIFACTS = (
    REALISTIC_SELECTED,
    REALISTIC_DAILY_SUMMARY,
    REALISTIC_SUMMARY,
    REALISTIC_TRACE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay capacity-selected child orders with practical execution constraints "
            "and compute capacity-weighted next-close acceptance returns."
        )
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--selected-input", action="append")
    parser.add_argument(
        "--execution-input",
        action="append",
        help=(
            "Optional keyed context with execution fields such as status, ask_price_1, "
            "spread_bps, ask1_to_limit_up_bps, ask_depth_notional, or industry."
        ),
    )
    parser.add_argument("--label-input", action="append")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--label-col", default="")
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--capacity-total-notional", type=float, default=None)
    parser.add_argument("--max-daily-symbol-weight", type=float, default=None)
    parser.add_argument("--max-daily-symbol-participation-rate", type=float, default=None)
    parser.add_argument(
        "--daily-capacity-method",
        default="",
        help=(
            "Deprecated no-op kept for older command lines. Per-decision capacity is enforced "
            "by the selected input and is not collapsed into a daily turnover budget."
        ),
    )
    parser.add_argument("--execution-fill-rate", type=float, default=None)
    parser.add_argument("--min-child-notional", type=float, default=None)
    parser.add_argument("--max-symbol-decision-count", type=int, default=None)
    parser.add_argument("--round-lot-shares", type=int, default=None)
    parser.add_argument("--price-col", default="")
    parser.add_argument("--status-col", default="")
    parser.add_argument("--tradable-status", action="append")
    parser.add_argument("--spread-bps-col", default="")
    parser.add_argument("--max-spread-bps", type=float, default=None)
    parser.add_argument("--limit-up-room-bps-col", default="")
    parser.add_argument("--min-limit-up-room-bps", type=float, default=None)
    parser.add_argument("--ask-depth-notional-col", default="")
    parser.add_argument("--max-ask-depth-participation-rate", type=float, default=None)
    parser.add_argument("--industry-col", default="")
    parser.add_argument("--max-daily-industry-weight", type=float, default=None)
    parser.add_argument("--records-dir", default="")
    parser.add_argument("--record-prefix", default="")
    return parser.parse_args()


def _constraints(arguments: CommandArguments) -> RealisticExecutionConstraints:
    defaults = RealisticExecutionConstraints()
    return RealisticExecutionConstraints(
        capacity_total_notional=arguments.float(
            "capacity_total_notional",
            defaults.capacity_total_notional,
        ),
        fee_bps=arguments.float("fee_bps", defaults.fee_bps),
        max_daily_symbol_weight=arguments.float(
            "max_daily_symbol_weight",
            defaults.max_daily_symbol_weight,
        ),
        max_daily_symbol_participation_rate=arguments.float(
            "max_daily_symbol_participation_rate",
            defaults.max_daily_symbol_participation_rate,
        ),
        daily_capacity_method=arguments.string(
            "daily_capacity_method",
            defaults.daily_capacity_method,
        ),
        execution_fill_rate=arguments.float(
            "execution_fill_rate",
            defaults.execution_fill_rate,
        ),
        min_child_notional=arguments.float(
            "min_child_notional",
            defaults.min_child_notional,
        ),
        max_symbol_decision_count=arguments.integer(
            "max_symbol_decision_count",
            defaults.max_symbol_decision_count,
        ),
        round_lot_shares=arguments.integer(
            "round_lot_shares",
            defaults.round_lot_shares,
        ),
        price_col=arguments.string("price_col", defaults.price_col),
        status_col=arguments.string("status_col", defaults.status_col),
        tradable_statuses=arguments.aliased_tuple(
            "tradable_status",
            "tradable_statuses",
        ),
        spread_bps_col=arguments.string("spread_bps_col", defaults.spread_bps_col),
        max_spread_bps=arguments.float("max_spread_bps", defaults.max_spread_bps),
        limit_up_room_bps_col=arguments.string(
            "limit_up_room_bps_col",
            defaults.limit_up_room_bps_col,
        ),
        min_limit_up_room_bps=arguments.float(
            "min_limit_up_room_bps",
            defaults.min_limit_up_room_bps,
        ),
        ask_depth_notional_col=arguments.string(
            "ask_depth_notional_col",
            defaults.ask_depth_notional_col,
        ),
        max_ask_depth_participation_rate=arguments.float(
            "max_ask_depth_participation_rate",
            defaults.max_ask_depth_participation_rate,
        ),
        industry_col=arguments.string("industry_col", defaults.industry_col),
        max_daily_industry_weight=arguments.float(
            "max_daily_industry_weight",
            defaults.max_daily_industry_weight,
        ),
    )


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    arguments = CommandArguments(args, config, "realistic_acceptance")
    run_name = args.run_id or (
        run_id(config, args.config) if args.config else "realistic_acceptance"
    )
    selected_inputs = arguments.tuple("selected_input")
    execution_inputs = arguments.tuple("execution_input")
    label_inputs = arguments.tuple("label_input")
    if not selected_inputs:
        raise SystemExit("pass --selected-input or set [realistic_acceptance].selected_input")
    if not label_inputs:
        raise SystemExit("pass --label-input or set [realistic_acceptance].label_input")

    output_dir_value = CommandArguments(args, config, "output").string(
        "output_dir",
        config_name="local_dir",
    )
    if not output_dir_value:
        raise SystemExit("pass --output-dir or set [output].local_dir")
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_col = arguments.string("label_col", DEFAULT_REALISTIC_LABEL_COL)
    constraints = _constraints(arguments)

    context_columns = realistic_context_columns(constraints)
    selected = load_realistic_selected(selected_inputs, extra_columns=context_columns)
    execution_context = load_realistic_execution_context(
        execution_inputs,
        columns=context_columns,
    )
    selected = merge_realistic_execution_context(selected, execution_context)
    constrained, group_targets = apply_realistic_execution_constraints(selected, constraints)
    labels = load_label_frame(
        label_inputs,
        label_col=label_col,
        dates=set(group_targets["date"].astype(str)),
    )
    daily = summarize_realistic_acceptance(
        constrained,
        group_targets,
        labels,
        constraints=constraints,
        label_col=label_col,
    )
    summary = summarize_realistic_acceptance_overall(daily)

    selected_path = output_dir / REALISTIC_SELECTED
    daily_path = output_dir / REALISTIC_DAILY_SUMMARY
    summary_path = output_dir / REALISTIC_SUMMARY
    trace_path = output_dir / REALISTIC_TRACE
    constrained.to_csv(selected_path, index=False, float_format="%.6f")
    daily.to_csv(daily_path, index=False, float_format="%.6f")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    trace = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_name,
        "variant": arguments.string("variant", run_name),
        "selected_inputs": list(selected_inputs),
        "execution_inputs": list(execution_inputs),
        "execution_context_columns": list(context_columns),
        "execution_context_rows": int(len(execution_context)),
        "label_inputs": list(label_inputs),
        "label_col": label_col,
        "constraints": constraints_trace(constraints),
        "input_selected_rows": int(len(selected)),
        "constrained_selected_rows": int(len(constrained)),
        "group_targets": int(len(group_targets)),
        "label_rows": int(len(labels)),
        "daily_rows": int(len(daily)),
        "selected": str(selected_path),
        "daily_summary": str(daily_path),
        "summary": str(summary_path),
        "record_paths": [],
        "modeling_note": (
            "This is a selected-order replay. Per-decision capacity limits are assumed to "
            "have been enforced while building the selected input. The replay applies "
            "post-selection constraints such as daily symbol weight, execution fill rate, "
            "minimum child notional, and lot rounding; it does not collapse per-decision "
            "turnover into a same-day turnover budget."
        ),
    }
    write_json(trace_path, trace, ensure_ascii=True)

    records_dir = arguments.string("records_dir")
    record_prefix = arguments.string("record_prefix") or run_name
    record_paths = record_requested_artifacts(
        output_dir=output_dir,
        records_dir=records_dir,
        record_prefix=record_prefix,
        names=ARTIFACTS,
    )
    if records_dir:
        trace["record_paths"] = [str(path) for path in record_paths]
        write_json(trace_path, trace, ensure_ascii=True)
        destination = Path(records_dir) / "backtests" / record_prefix / trace_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace_path, destination)

    print("realistic_acceptance_summary:")
    print(summary.to_string(index=False) if not summary.empty else "empty")
    if record_paths:
        print("\nrecorded_realistic_acceptance_outputs:")
        for path in record_paths:
            print(f"  {path}")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
