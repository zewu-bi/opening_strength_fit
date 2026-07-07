from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from opening_strength_fit.analysis import write_json
from opening_strength_fit.config import config_int, config_str, load_toml, run_id
from opening_strength_fit.execution_diagnostics import (
    DEFAULT_ASK_LEVELS,
    DiagnosticCase,
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


def _config_cases(config: dict, *, output_root: Path) -> list[DiagnosticCase]:
    section = config.get("ask_level_attribution", {})
    inputs = section.get("inputs", []) if isinstance(section, dict) else []
    if not isinstance(inputs, list):
        raise SystemExit("[ask_level_attribution].inputs must be an array of tables")
    cases = []
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise SystemExit("each [[ask_level_attribution.inputs]] entry must be a table")
        name = str(item.get("name", f"case_{index}")).strip()
        selected_path = str(item.get("selected_path", "")).strip()
        prediction_root = str(item.get("prediction_root", "")).strip()
        if not selected_path or not prediction_root:
            raise SystemExit(
                "each [[ask_level_attribution.inputs]] entry requires selected_path "
                "and prediction_root"
            )
        case_output_dir = str(item.get("output_dir", "")).strip()
        cases.append(
            DiagnosticCase(
                name=name,
                selected_path=Path(selected_path),
                prediction_root=Path(prediction_root),
                output_dir=Path(case_output_dir) if case_output_dir else output_root / name,
            )
        )
    return cases


def _arg_case(args: argparse.Namespace, *, output_root: Path) -> list[DiagnosticCase]:
    if not (args.selected_path or args.prediction_root):
        return []
    if not args.selected_path or not args.prediction_root:
        raise SystemExit("--selected-path and --prediction-root must be supplied together")
    name = args.name or "case"
    return [
        DiagnosticCase(
            name=name,
            selected_path=Path(args.selected_path),
            prediction_root=Path(args.prediction_root),
            output_dir=Path(args.case_output_dir) if args.case_output_dir else output_root,
        )
    ]


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = args.run_id or (
        run_id(config, args.config) if args.config else "ask_level_attribution"
    )
    output_root = _output_root(args, config, run_name)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = _config_cases(config, output_root=output_root) or _arg_case(
        args,
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
