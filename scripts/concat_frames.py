from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.io import read_frame, write_frame
from opening_strength_fit.reports import dataset_summary, print_mapping


def _expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(Path().glob(value)) if any(c in value for c in "*?[") else []
        paths.extend(matches or [Path(value)])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate parquet/csv frames.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sort-columns",
        default="date,symbol,timestamp",
        help="Comma-separated sort columns; missing columns are ignored.",
    )
    args = parser.parse_args()

    paths = _expand_inputs(args.input)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"input path does not exist: {missing[0]}")

    frames = [read_frame(path) for path in paths]
    if not frames:
        raise SystemExit("no input frames supplied")

    out = pd.concat(frames, ignore_index=True)
    sort_columns = [
        column.strip()
        for column in args.sort_columns.split(",")
        if column.strip() and column.strip() in out.columns
    ]
    if sort_columns:
        out = out.sort_values(sort_columns).reset_index(drop=True)

    write_frame(out, args.output)
    print_mapping("concatenated", dataset_summary(out))
    print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
