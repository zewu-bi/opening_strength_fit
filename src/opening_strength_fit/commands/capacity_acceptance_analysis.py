from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from opening_strength_fit.analysis import write_json
from opening_strength_fit.artifact_catalog import record_requested_artifacts
from opening_strength_fit.capacity_acceptance import (
    DEFAULT_CAPACITY_LABEL_COL,
    DEFAULT_CAPACITY_TOTAL_NOTIONAL,
    load_capacity_selected,
    load_label_frame,
    summarize_capacity_acceptance,
    summarize_capacity_acceptance_overall,
)
from opening_strength_fit.commands.arguments import CommandArguments
from opening_strength_fit.config import load_toml, run_id

ARTIFACTS = (
    "capacity_acceptance_daily_summary.csv",
    "capacity_acceptance_summary.csv",
    "capacity_acceptance_trace.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute capacity-weighted next-close acceptance returns from a capacity "
            "audit selected allocation file."
        )
    )
    parser.add_argument("--config", default="")
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
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--label-col", default="")
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--capacity-total-notional", type=float, default=None)
    parser.add_argument("--records-dir", default="")
    parser.add_argument("--record-prefix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    arguments = CommandArguments(args, config, "capacity_acceptance")
    run_name = args.run_id or (
        run_id(config, args.config) if args.config else "capacity_acceptance"
    )
    selected_inputs = arguments.tuple("selected_input")
    label_inputs = arguments.tuple("label_input")
    if not selected_inputs:
        raise SystemExit("pass --selected-input or set [capacity_acceptance].selected_input")
    if not label_inputs:
        raise SystemExit("pass --label-input or set [capacity_acceptance].label_input")

    output_dir_value = CommandArguments(args, config, "output").string(
        "output_dir",
        config_name="local_dir",
    )
    if not output_dir_value:
        raise SystemExit("pass --output-dir or set [output].local_dir")
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    print("capacity_acceptance_summary:")
    print(summary.to_string(index=False) if not summary.empty else "empty")
    if record_paths:
        print("\nrecorded_capacity_acceptance_outputs:")
        for path in record_paths:
            print(f"  {path}")
    print(f"\nwrote outputs: {output_dir}")


if __name__ == "__main__":
    main()
