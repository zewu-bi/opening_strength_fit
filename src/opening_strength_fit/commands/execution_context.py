from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from opening_strength_fit.analysis import write_json
from opening_strength_fit.config import config_str, load_toml, run_id
from opening_strength_fit.execution_diagnostics import (
    diagnostic_case_from_values,
    diagnostic_cases_from_config,
    run_execution_context_case,
    write_success_marker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract prediction-time execution context for capacity-selected rows."
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--selected-path", default="")
    parser.add_argument("--prediction-root", default="")
    parser.add_argument("--case-output-dir", default="")
    return parser.parse_args()


def _output_root(args: argparse.Namespace, config: dict, run_name: str) -> Path:
    return Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = args.run_id or (run_id(config, args.config) if args.config else "execution_context")
    output_root = _output_root(args, config, run_name)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = diagnostic_cases_from_config(
        config, section_name="execution_context", output_root=output_root
    ) or diagnostic_case_from_values(
        selected_path=args.selected_path,
        prediction_root=args.prediction_root,
        name=args.name,
        case_output_dir=args.case_output_dir,
        output_root=output_root,
    )
    if not cases:
        raise SystemExit(
            "supply [[execution_context.inputs]] in --config or pass "
            "--selected-path/--prediction-root"
        )

    case_records = []
    for case in cases:
        output_path = run_execution_context_case(case)
        case_records.append(
            {
                "name": case.name,
                "selected_path": str(case.selected_path),
                "prediction_root": str(case.prediction_root),
                "output_dir": str(case.output_dir),
                "output_path": str(output_path),
            }
        )

    write_json(
        output_root / "execution_context_trace.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": run_name,
            "cases": case_records,
        },
        ensure_ascii=True,
    )
    write_success_marker(output_root, run_name=run_name, cases=case_records)

    print("\nexecution_context_outputs:")
    for record in case_records:
        print(f"  {record['name']}: {record['output_path']}")
    print(f"\nwrote run trace: {output_root}")


if __name__ == "__main__":
    main()
