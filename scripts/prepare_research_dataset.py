from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    query_tick_day_window,
)
from opening_strength_fit.config import load_toml
from opening_strength_fit.io import read_frame, write_frame
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns
from opening_strength_fit.training import build_labeled_frame_from_config
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX, load_symbol_list


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _date_list(args: argparse.Namespace) -> list[str]:
    if args.date:
        return [str(pd.Timestamp(date).date()) for date in args.date]
    if not args.start_date or not args.end_date:
        raise SystemExit("pass --date or both --start-date and --end-date")
    return [
        str(date.date())
        for date in pd.date_range(args.start_date, args.end_date, freq="D")
    ]


def _read_input_day(path: str, trading_day: str) -> pd.DataFrame:
    root = Path(path)
    if root.is_dir():
        exact_dirs = sorted(root.glob(f"**/date={trading_day}"))
        if exact_dirs:
            return read_frame(exact_dirs[0])
        frame = read_frame(root)
    elif root.suffix.lower() == ".parquet":
        for filter_col in ("date", "TradingDay"):
            try:
                frame = pd.read_parquet(root, filters=[(filter_col, "==", trading_day)])
                if len(frame):
                    break
            except Exception:
                frame = pd.DataFrame()
        if frame.empty:
            frame = pd.read_parquet(root)
    else:
        frame = read_frame(root)

    frame = ensure_timestamp_columns(standardize_columns(frame))
    return frame.loc[frame["date"].astype(str) == trading_day].copy()


def _with_config_overrides(args: argparse.Namespace, config: dict) -> dict:
    out = {section: dict(values) for section, values in config.items()}
    out.setdefault("universe", {})
    out.setdefault("sample", {})
    if args.no_universe_filter:
        out["universe"]["enabled"] = False
    else:
        out["universe"]["enabled"] = True
        out["universe"]["symbol_regex"] = args.universe_regex
        if args.symbols_file:
            out["universe"]["symbols_file"] = args.symbols_file
    if args.sample_mode:
        out["sample"]["mode"] = args.sample_mode
    if args.decision_times:
        out["sample"]["decision_times"] = args.decision_times
    if args.decision_max_lag_seconds is not None:
        out["sample"]["decision_max_lag_seconds"] = args.decision_max_lag_seconds
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a filtered, labeled opening research dataset partitioned by date."
        )
    )
    parser.add_argument("--config", default="experiments/runs/gbm_opening_1y_next_month.toml")
    parser.add_argument("--input", default="", help="Optional raw tick parquet/csv root.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--date", nargs="*", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--sample-mode", choices=["all_ticks", "decision_points"], default=None)
    parser.add_argument("--decision-times", nargs="*", default=None)
    parser.add_argument("--decision-max-lag-seconds", type=int, default=None)
    parser.add_argument("--universe-regex", default=DEFAULT_A_SHARE_SYMBOL_REGEX)
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--no-universe-filter", action="store_true")
    parser.add_argument("--partition-filename", default="part.parquet")
    parser.add_argument("--overwrite", action="store_true")
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

    config = _with_config_overrides(args, load_toml(args.config) if args.config else {})
    dates = _date_list(args)
    output_root = Path(args.output_root)

    client = None
    if not args.input:
        if not args.user or not args.password:
            raise SystemExit(
                "missing ClickHouse credentials: pass --input or set/pass "
                "CLICKHOUSE_USER and CLICKHOUSE_PASSWORD"
            )
        client = get_tick_client(
            host=args.host,
            port=args.port,
            username=args.user,
            password=args.password,
        )

    summaries = []
    symbols = sorted(load_symbol_list(args.symbols_file)) if args.symbols_file else None
    for trading_day in dates:
        partition = output_root / f"year={trading_day[:4]}" / f"date={trading_day}"
        output_path = partition / args.partition_filename
        if output_path.exists() and not args.overwrite:
            print(f"skip existing: {output_path}")
            continue

        if args.input:
            ticks = _read_input_day(args.input, trading_day)
        else:
            ticks = query_tick_day_window(
                client,
                trading_day=trading_day,
                table=args.table,
                start_offset_us=args.start_offset_us,
                end_offset_us=args.end_offset_us,
                symbol_regex=None if args.no_universe_filter else args.universe_regex,
                symbols=symbols,
            )
            if ticks.empty:
                print(f"skip empty source day: {trading_day}")
                continue
            ticks = normalize_clickhouse_ticks(ticks)

        if ticks.empty:
            print(f"skip empty source day: {trading_day}")
            continue

        labeled = build_labeled_frame_from_config(ticks, config)
        if labeled.empty:
            print(f"skip empty labeled day: {trading_day}")
            continue

        write_frame(labeled, output_path)
        summary = {"date": trading_day, "output": str(output_path), **dataset_summary(labeled)}
        summaries.append(summary)
        print_mapping(f"prepared[{trading_day}]", summary)

    if summaries:
        manifest = pd.DataFrame(summaries)
        manifest.to_csv(output_root / "manifest.csv", index=False)
        print(f"\nwrote manifest: {output_root / 'manifest.csv'}")
    else:
        print("\nno partitions written")


if __name__ == "__main__":
    main()
