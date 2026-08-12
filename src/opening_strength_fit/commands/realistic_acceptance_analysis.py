from __future__ import annotations

import argparse
from datetime import UTC, datetime

from opening_strength_fit.artifact_catalog import (
    print_recorded_artifacts,
    record_requested_artifacts,
)
from opening_strength_fit.capacity_acceptance import load_label_frame
from opening_strength_fit.commands import arguments as cmd
from opening_strength_fit.config import config_str, prepare_output_dir
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
    parser = cmd.command_parser(
        description=(
            "Replay capacity-selected child orders with practical execution constraints "
            "and compute capacity-weighted next-close acceptance returns."
        )
    )
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
    cmd.add_arguments(parser, "output-dir run-id variant", default="")
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
    cmd.add_arguments(parser, "execution-fill-rate min-child-notional", type=float, default=None)
    cmd.add_arguments(parser, "max-symbol-decision-count round-lot-shares", type=int, default=None)
    cmd.add_arguments(parser, "price-col status-col", default="")
    parser.add_argument("--tradable-status", action="append")
    parser.add_argument("--spread-bps-col", default="")
    parser.add_argument("--max-spread-bps", type=float, default=None)
    parser.add_argument("--limit-up-room-bps-col", default="")
    parser.add_argument("--min-limit-up-room-bps", type=float, default=None)
    parser.add_argument("--ask-depth-notional-col", default="")
    parser.add_argument("--max-ask-depth-participation-rate", type=float, default=None)
    parser.add_argument("--industry-col", default="")
    parser.add_argument("--max-daily-industry-weight", type=float, default=None)
    cmd.add_arguments(parser, "records-dir record-prefix", default="")
    return parser.parse_args()


def _constraints(arguments: cmd.CommandArguments) -> RealisticExecutionConstraints:
    return arguments.resolve_dataclass(
        RealisticExecutionConstraints(),
        tuple_aliases={"tradable_statuses": "tradable_status"},
    )


def main() -> None:
    args = parse_args()
    config, arguments, run_name = cmd.command_context(args, "realistic_acceptance")
    selected_inputs = arguments.tuple("selected_input")
    execution_inputs = arguments.tuple("execution_input")
    label_inputs = arguments.tuple("label_input")
    if not selected_inputs:
        raise SystemExit("pass --selected-input or set [realistic_acceptance].selected_input")
    if not label_inputs:
        raise SystemExit("pass --label-input or set [realistic_acceptance].label_input")

    if not (args.output_dir or config_str(config, "output", "local_dir", "")):
        raise SystemExit("pass --output-dir or set [output].local_dir")
    output_dir = prepare_output_dir(config, args.output_dir, run_name)
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
    records_dir = arguments.string("records_dir")
    record_prefix = arguments.string("record_prefix") or run_name
    record_paths = record_requested_artifacts(
        output_dir=output_dir,
        records_dir=records_dir,
        record_prefix=record_prefix,
        names=ARTIFACTS,
        trace=(trace_path, trace),
    )

    print("realistic_acceptance_summary:")
    print(summary.to_string(index=False) if not summary.empty else "empty")
    print_recorded_artifacts(record_paths, "realistic_acceptance")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
