from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from opening_strength_fit.analysis import write_json
from opening_strength_fit.commands.arguments import command_config
from opening_strength_fit.config import config_int, config_str, load_toml
from opening_strength_fit.execution_diagnostics import (
    DEFAULT_ASK_LEVELS,
    DiagnosticCase,
    diagnostic_case_from_values,
    diagnostic_cases_from_config,
    run_ask_level_attribution_case,
    run_execution_context_case,
    write_success_marker,
)


def _parser(description: str, *, include_levels: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    for argument in (
        "config",
        "output-dir",
        "run-id",
        "name",
        "selected-path",
        "prediction-root",
        "case-output-dir",
    ):
        parser.add_argument(f"--{argument}", default="")
    if include_levels:
        parser.add_argument("--levels", type=int, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return _parser(
        "Extract prediction-time execution context for capacity-selected rows."
    ).parse_args()


def parse_ask_level_args() -> argparse.Namespace:
    return _parser(
        "Attribute capacity-selected notional to visible ask book levels.",
        include_levels=True,
    ).parse_args()


def _output_root(args: argparse.Namespace, config: dict, run_name: str) -> Path:
    return Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )


def _run_diagnostic_command(
    args: argparse.Namespace,
    *,
    section: str,
    default_run_name: str,
    case_runner: Callable[[DiagnosticCase], Path],
    trace_fields: dict[str, object] | None = None,
) -> None:
    config, run_name = command_config(args, default_run_name)
    output_root = _output_root(args, config, run_name)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = diagnostic_cases_from_config(
        config, section_name=section, output_root=output_root
    ) or diagnostic_case_from_values(
        selected_path=args.selected_path,
        prediction_root=args.prediction_root,
        name=args.name,
        case_output_dir=args.case_output_dir,
        output_root=output_root,
    )
    if not cases:
        raise SystemExit(
            f"supply [[{section}.inputs]] in --config or pass --selected-path/--prediction-root"
        )

    records = []
    for case in cases:
        output_path = case_runner(case)
        records.append(
            {
                "name": case.name,
                "selected_path": str(case.selected_path),
                "prediction_root": str(case.prediction_root),
                "output_dir": str(case.output_dir),
                "output_path": str(output_path),
            }
        )
    write_json(
        output_root / f"{section}_trace.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": run_name,
            **(trace_fields or {}),
            "cases": records,
        },
        ensure_ascii=True,
    )
    write_success_marker(output_root, run_name=run_name, cases=records)
    print(f"\n{section}_outputs:")
    for record in records:
        print(f"  {record['name']}: {record['output_path']}")
    print(f"\nwrote run trace: {output_root}")


def main() -> None:
    _run_diagnostic_command(
        parse_args(),
        section="execution_context",
        default_run_name="execution_context",
        case_runner=run_execution_context_case,
    )


def ask_level_main() -> None:
    args = parse_ask_level_args()
    config = load_toml(args.config) if args.config else {}
    max_level = args.levels or config_int(
        config, "ask_level_attribution", "levels", max(DEFAULT_ASK_LEVELS)
    )
    levels = tuple(range(1, int(max_level) + 1))
    _run_diagnostic_command(
        args,
        section="ask_level_attribution",
        default_run_name="ask_level_attribution",
        case_runner=lambda case: run_ask_level_attribution_case(case, levels=levels),
        trace_fields={"levels": list(levels)},
    )


if __name__ == "__main__":
    main()
