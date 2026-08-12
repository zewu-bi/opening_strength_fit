from __future__ import annotations

import argparse
from datetime import UTC, datetime

from opening_strength_fit.artifact_catalog import (
    CAPACITY_ACCEPTANCE_ARTIFACTS,
    print_recorded_artifacts,
    record_requested_artifacts,
)
from opening_strength_fit.capacity_acceptance import (
    DEFAULT_CAPACITY_LABEL_COL,
    DEFAULT_CAPACITY_TOTAL_NOTIONAL,
    load_capacity_selected,
    load_label_frame,
    summarize_capacity_acceptance,
    summarize_capacity_acceptance_overall,
)
from opening_strength_fit.commands import arguments as cmd
from opening_strength_fit.config import config_str, prepare_output_dir


def parse_args() -> argparse.Namespace:
    parser = cmd.command_parser(
        description=(
            "Compute capacity-weighted next-close acceptance returns from a capacity "
            "audit selected allocation file."
        )
    )
    parser.add_argument(
        "--selected-input",
        action="append",
        help="capacity_audit_selected.csv, parquet/csv file, or directory. May be repeated.",
    )
    parser.add_argument(
        "--label-input",
        action="append",
        help="Next-close label parquet/csv file or directory. May be repeated.",
    )
    cmd.add_arguments(parser, "output-dir run-id variant", default="")
    parser.add_argument("--label-col", default="")
    cmd.add_arguments(parser, "fee-bps capacity-total-notional", type=float, default=None)
    cmd.add_arguments(parser, "records-dir record-prefix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, arguments, run_name = cmd.command_context(args, "capacity_acceptance")
    selected_inputs = arguments.tuple("selected_input")
    label_inputs = arguments.tuple("label_input")
    if not selected_inputs:
        raise SystemExit("pass --selected-input or set [capacity_acceptance].selected_input")
    if not label_inputs:
        raise SystemExit("pass --label-input or set [capacity_acceptance].label_input")

    if not (args.output_dir or config_str(config, "output", "local_dir", "")):
        raise SystemExit("pass --output-dir or set [output].local_dir")
    output_dir = prepare_output_dir(config, args.output_dir, run_name)
    label_col = arguments.string("label_col", DEFAULT_CAPACITY_LABEL_COL)
    fee_bps = arguments.float("fee_bps", 0.0)
    capacity_total_notional = arguments.float(
        "capacity_total_notional",
        DEFAULT_CAPACITY_TOTAL_NOTIONAL,
    )

    selected = load_capacity_selected(selected_inputs)
    labels = load_label_frame(
        label_inputs,
        label_col=label_col,
        dates=set(selected["date"].astype(str)),
    )
    daily = summarize_capacity_acceptance(
        selected,
        labels,
        capacity_total_notional=capacity_total_notional,
        fee_bps=fee_bps,
        label_col=label_col,
    )
    summary = summarize_capacity_acceptance_overall(daily)

    daily_path = output_dir / "capacity_acceptance_daily_summary.csv"
    summary_path = output_dir / "capacity_acceptance_summary.csv"
    trace_path = output_dir / "capacity_acceptance_trace.json"
    daily.to_csv(daily_path, index=False, float_format="%.6f")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    trace = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_name,
        "variant": arguments.string("variant", run_name),
        "selected_inputs": list(selected_inputs),
        "label_inputs": list(label_inputs),
        "label_col": label_col,
        "fee_bps": fee_bps,
        "capacity_total_notional": capacity_total_notional,
        "selected_rows": int(len(selected)),
        "label_rows": int(len(labels)),
        "daily_rows": int(len(daily)),
        "summary": str(summary_path),
        "daily_summary": str(daily_path),
        "record_paths": [],
    }
    records_dir = arguments.string("records_dir")
    record_prefix = arguments.string("record_prefix") or run_name
    record_paths = record_requested_artifacts(
        output_dir=output_dir,
        records_dir=records_dir,
        record_prefix=record_prefix,
        names=CAPACITY_ACCEPTANCE_ARTIFACTS,
        trace=(trace_path, trace),
    )

    print("capacity_acceptance_summary:")
    print(summary.to_string(index=False) if not summary.empty else "empty")
    print_recorded_artifacts(record_paths, "capacity_acceptance")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
