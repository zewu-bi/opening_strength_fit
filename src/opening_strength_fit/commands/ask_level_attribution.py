from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from opening_strength_fit.analysis import write_json
from opening_strength_fit.config import config_int, config_str, load_toml, run_id
from opening_strength_fit.execution_diagnostics import (
    DEFAULT_ASK_LEVELS,
    diagnostic_case_from_values,
    diagnostic_cases_from_config,
    run_ask_level_attribution_case,
    write_success_marker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute capacity-selected notional to visible ask book levels."
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--selected-path", default="")
    parser.add_argument("--prediction-root", default="")
    parser.add_argument("--case-output-dir", default="")
    parser.add_argument("--levels", type=int, default=None)
    return parser.parse_args()


def _output_root(args: argparse.Namespace, config: dict, run_name: str) -> Path:
    return Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = args.run_id or (
        run_id(config, args.config) if args.config else "ask_level_attribution"
    )
    output_root = _output_root(args, config, run_name)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = diagnostic_cases_from_config(
        config, section_name="ask_level_attribution", output_root=output_root
    ) or diagnostic_case_from_values(
        selected_path=args.selected_path,
        prediction_root=args.prediction_root,
        name=args.name,
        case_output_dir=args.case_output_dir,
        output_root=output_root,
    )
    if not cases:
        raise SystemExit(
            "supply [[ask_level_attribution.inputs]] in --config or pass "
            "--selected-path/--prediction-root"
        )

    max_level = args.levels or config_int(
        config,
        "ask_level_attribution",
        "levels",
        max(DEFAULT_ASK_LEVELS),
    )
    levels = tuple(range(1, int(max_level) + 1))
    case_records = []
    for case in cases:
        output_path = run_ask_level_attribution_case(case, levels=levels)
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
        output_root / "ask_level_attribution_trace.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": run_name,
            "levels": list(levels),
            "cases": case_records,
        },
        ensure_ascii=True,
    )
    write_success_marker(output_root, run_name=run_name, cases=case_records)

    print("\nask_level_attribution_outputs:")
    for record in case_records:
        print(f"  {record['name']}: {record['output_path']}")
    print(f"\nwrote run trace: {output_root}")


if __name__ == "__main__":
    main()
