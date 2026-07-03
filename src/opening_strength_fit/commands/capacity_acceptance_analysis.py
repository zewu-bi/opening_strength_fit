from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from opening_strength_fit.analysis import write_json
from opening_strength_fit.capacity_acceptance import (
    DEFAULT_CAPACITY_LABEL_COL,
    DEFAULT_CAPACITY_TOTAL_NOTIONAL,
    load_capacity_selected,
    load_label_frame,
    summarize_capacity_acceptance,
    summarize_capacity_acceptance_overall,
)
from opening_strength_fit.config import (
    config_float,
    config_list,
    config_str,
    load_toml,
    run_id,
)

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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--label-col", default="")
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--capacity-total-notional", type=float, default=None)
    parser.add_argument("--records-dir", default="")
    parser.add_argument("--record-prefix", default="")
    return parser.parse_args()


def _arg_list(args: argparse.Namespace, config: dict, name: str) -> tuple[str, ...]:
    values = getattr(args, name, None)
    if values:
        return tuple(values)
    return tuple(config_list(config, "capacity_acceptance", name, ()))


def _arg_str(args: argparse.Namespace, config: dict, name: str, default: str = "") -> str:
    value = getattr(args, name, "")
    if value not in (None, ""):
        return str(value)
    return config_str(config, "capacity_acceptance", name, default)


def _arg_float(args: argparse.Namespace, config: dict, name: str, default: float) -> float:
    value = getattr(args, name, None)
    if value is not None:
        return float(value)
    return config_float(config, "capacity_acceptance", name, default)


def record_capacity_acceptance_outputs(
    *,
    output_dir: Path,
    records_dir: Path,
    record_prefix: str,
) -> list[Path]:
    destination_dir = records_dir / "backtests" / record_prefix
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ARTIFACTS:
        source = output_dir / name
        if not source.exists():
            continue
        destination = destination_dir / name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = args.run_id or (
        run_id(config, args.config) if args.config else "capacity_acceptance"
    )
    selected_inputs = _arg_list(args, config, "selected_input")
    label_inputs = _arg_list(args, config, "label_input")
    if not selected_inputs:
        raise SystemExit("pass --selected-input or set [capacity_acceptance].selected_input")
    if not label_inputs:
        raise SystemExit("pass --label-input or set [capacity_acceptance].label_input")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_col = _arg_str(args, config, "label_col", DEFAULT_CAPACITY_LABEL_COL)
    fee_bps = _arg_float(args, config, "fee_bps", 0.0)
    capacity_total_notional = _arg_float(
        args,
        config,
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
        "variant": args.variant or _arg_str(args, config, "variant", run_name),
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

    records_dir = args.records_dir or config_str(config, "capacity_acceptance", "records_dir", "")
    record_paths: list[Path] = []
    if records_dir:
        record_prefix = (
            args.record_prefix
            or config_str(config, "capacity_acceptance", "record_prefix", "")
            or run_name
        )
        record_paths = record_capacity_acceptance_outputs(
            output_dir=output_dir,
            records_dir=Path(records_dir),
            record_prefix=record_prefix,
        )
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
