from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    DEFAULT_TICK_END_OFFSET_US,
    DEFAULT_TICK_START_OFFSET_US,
    get_tick_client,
    normalize_clickhouse_ticks,
    query_tick_window,
)
from opening_strength_fit.config import load_toml
from opening_strength_fit.dataset import load_ticks
from opening_strength_fit.io import write_frame
from opening_strength_fit.model import feature_columns
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.schema import available_depth_levels
from opening_strength_fit.training import build_labeled_frame_from_config


DEFAULT_DATES = ("2021-09-22", "2021-09-23")


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        parts = shlex.split(line, comments=True)
        if not parts or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        os.environ.setdefault(key, value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _sort_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ("date", "symbol", "timestamp")
        if column in ticks.columns
    ]
    if sort_columns:
        return ticks.sort_values(sort_columns).reset_index(drop=True)
    return ticks.reset_index(drop=True)


def load_source_ticks(args: argparse.Namespace) -> pd.DataFrame:
    if args.input:
        return load_ticks(args.input)

    if not args.no_env_file:
        load_env_file(args.env_file)
        args.user = args.user or os.getenv("CLICKHOUSE_USER")
        args.password = args.password or os.getenv("CLICKHOUSE_PASSWORD")
        args.host = os.getenv("CLICKHOUSE_HOST", args.host)
        args.port = _env_int("CLICKHOUSE_PORT", args.port)
        args.table = os.getenv("CLICKHOUSE_TICK_TABLE", args.table)

    if not args.user or not args.password:
        raise SystemExit(
            "missing ClickHouse credentials: set CLICKHOUSE_USER/CLICKHOUSE_PASSWORD, "
            "source .env, or pass --user/--password"
        )

    client = get_tick_client(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
    )
    frames = []
    for symbol in args.symbol:
        for trading_day in args.date:
            ticks = query_tick_window(
                client,
                symbol=symbol,
                trading_day=trading_day,
                table=args.table,
                start_offset_us=args.start_offset_us,
                end_offset_us=args.end_offset_us,
            )
            ticks = normalize_clickhouse_ticks(ticks)
            frames.append(ticks)
            print_mapping(
                f"source_window[{trading_day},{symbol}]",
                dataset_summary(ticks),
            )

    if not frames:
        raise SystemExit("no source windows requested")
    return _sort_ticks(pd.concat(frames, ignore_index=True))


def print_dataset_checks(
    *,
    ticks: pd.DataFrame,
    labeled: pd.DataFrame,
) -> None:
    tick_summary = dataset_summary(ticks)
    labeled_summary = dataset_summary(labeled)
    features = feature_columns(labeled)
    ask1_positive = (
        int((ticks["ask_price_1"] > 0).sum()) if "ask_price_1" in ticks.columns else 0
    )
    bid1_positive = (
        int((ticks["bid_price_1"] > 0).sum()) if "bid_price_1" in ticks.columns else 0
    )
    valid_labels = int(labeled.get("valid_label", labeled["label"].notna()).sum())
    label_rows = len(labeled)
    label_nan_rate = (
        float(labeled["label"].isna().mean()) if label_rows and "label" in labeled else 1.0
    )
    feature_nan_count = (
        int(
            labeled[features]
            .replace([np.inf, -np.inf], np.nan)
            .isna()
            .sum()
            .sum()
        )
        if features
        else 0
    )

    print_mapping(
        "tick_dataset_check",
        {
            "rows": f"{len(ticks):,}",
            "columns": len(ticks.columns),
            "date_range": f"{tick_summary.get('date_min')} -> {tick_summary.get('date_max')}",
            "dates": tick_summary.get("n_dates"),
            "symbols": tick_summary.get("n_symbols"),
            "time_range": f"{tick_summary.get('time_min')} -> {tick_summary.get('time_max')}",
            "depth_levels": available_depth_levels(ticks),
            "ask1_positive_rows": f"{ask1_positive:,}/{len(ticks):,}",
            "bid1_positive_rows": f"{bid1_positive:,}/{len(ticks):,}",
        },
    )
    print_mapping(
        "labeled_dataset_check",
        {
            "rows": f"{label_rows:,}",
            "features": len(features),
            "date_range": f"{labeled_summary.get('date_min')} -> {labeled_summary.get('date_max')}",
            "dates": labeled_summary.get("n_dates"),
            "symbols": labeled_summary.get("n_symbols"),
            "time_range": f"{labeled_summary.get('time_min')} -> {labeled_summary.get('time_max')}",
            "label_coverage": (
                f"{valid_labels:,}/{label_rows:,} "
                f"(nan_rate={label_nan_rate:.2%})"
            ),
            "feature_nan_values": f"{feature_nan_count:,}",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build or load a small opening tick dataset, check feature/label coverage, "
            "and optionally write the raw tick sample for local smoke training."
        )
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional existing tick parquet/csv. If omitted, fetch ClickHouse windows.",
    )
    parser.add_argument("--symbol", nargs="+", default=["000925.SZ"])
    parser.add_argument("--date", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument(
        "--config",
        default="experiments/runs/ridge_opening_full.toml",
        help="Run config used for label/sample settings during inspection.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional parquet/csv path to write the raw tick sample.",
    )
    parser.add_argument(
        "--labeled-output",
        default="",
        help="Optional parquet/csv path to write the labeled debug table.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Print this many raw tick rows. Use 0 to skip.",
    )
    parser.add_argument(
        "--label-preview-rows",
        type=int,
        default=5,
        help="Print this many labeled rows. Use 0 to skip.",
    )
    parser.add_argument(
        "--nan-preview-rows",
        type=int,
        default=5,
        help="Print this many invalid label rows. Use 0 to skip.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Backward-compatible alias for --preview-rows.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--no-env-file", action="store_true")
    parser.add_argument(
        "--host",
        default=os.getenv("CLICKHOUSE_HOST", DEFAULT_CLICKHOUSE_TICK_HOST),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_TICK_PORT),
    )
    parser.add_argument("--user", default=os.getenv("CLICKHOUSE_USER"))
    parser.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD"))
    parser.add_argument(
        "--table",
        default=os.getenv("CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE),
    )
    parser.add_argument("--start-offset-us", type=int, default=DEFAULT_TICK_START_OFFSET_US)
    parser.add_argument("--end-offset-us", type=int, default=DEFAULT_TICK_END_OFFSET_US)
    args = parser.parse_args()
    if args.head is not None:
        args.preview_rows = args.head

    config = load_toml(args.config)
    ticks = load_source_ticks(args)
    labeled = build_labeled_frame_from_config(ticks, config)
    print_dataset_checks(ticks=ticks, labeled=labeled)

    if args.preview_rows > 0:
        print("\ntick_preview:")
        print(ticks.head(args.preview_rows).to_string(index=False))

    if args.label_preview_rows > 0:
        print("\nlabeled_preview:")
        print(labeled.head(args.label_preview_rows).to_string(index=False))

    if args.nan_preview_rows > 0 and "valid_label" in labeled.columns:
        invalid = labeled.loc[~labeled["valid_label"]]
        if not invalid.empty:
            print("\ninvalid_label_preview:")
            cols = [
                column
                for column in ("date", "symbol", "timestamp", "time", "label", "valid_label")
                if column in invalid.columns
            ]
            print(invalid.loc[:, cols].head(args.nan_preview_rows).to_string(index=False))

    if args.output:
        write_frame(ticks, args.output)
        print(f"\nwrote raw ticks: {args.output}")
    if args.labeled_output:
        write_frame(labeled, args.labeled_output)
        print(f"wrote labeled debug table: {args.labeled_output}")


if __name__ == "__main__":
    main()
