from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opening_strength_fit import next_close_labels as _next_close_labels
from opening_strength_fit.analysis import (
    NEXT_CLOSE_LABEL_COL,
)
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    clickhouse_connection,
)
from opening_strength_fit.commands.arguments import (
    add_arguments,
    command_context,
    required_io_paths,
)
from opening_strength_fit.config import (
    config_clock_list,
    prepare_output_dir,
)
from opening_strength_fit.horizon_clickhouse_labels import (
    DEFAULT_CLOSE_LOOKBACK_SECONDS,
    DEFAULT_CLOSE_OFFSET_US,
)
from opening_strength_fit.io import frame_columns, read_frame, write_frame, write_json
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.schema import (
    DECISION_KEY_COLUMNS,
    ensure_timestamp_columns,
    normalize_decision_keys,
    standardize_columns,
)

DEFAULT_DECISION_TIMES = tuple(f"09:{minute:02d}:00" for minute in range(31, 41))
KEY_COLUMNS = DECISION_KEY_COLUMNS
compute_clickhouse_close_labels = _next_close_labels.compute_clickhouse_close_labels
_DEFAULT_LABEL_BUILDER = compute_clickhouse_close_labels


def fetch_next_close_labels(base: pd.DataFrame, **kwargs) -> pd.DataFrame:
    label_builder = compute_clickhouse_close_labels
    if label_builder is _DEFAULT_LABEL_BUILDER:
        label_builder = _next_close_labels.compute_clickhouse_close_labels
    return _next_close_labels.fetch_next_close_labels(
        base,
        compute_labels=label_builder,
        **kwargs,
    )


def _read_base_frame(
    path: Path,
    *,
    buy_price_col: str,
    decision_times: tuple[str, ...],
) -> pd.DataFrame:
    available = frame_columns(path)
    time_col = (
        "decision_target_timestamp"
        if not available or "decision_target_timestamp" in available
        else "timestamp"
    )
    required = ["date", "symbol", time_col, buy_price_col]
    if available:
        missing = sorted(set(required) - available)
        if missing:
            raise SystemExit(f"next-close label input missing columns: {missing}")
    frame = read_frame(path, columns=required)
    out = standardize_columns(frame).copy()
    if time_col == "timestamp":
        out = ensure_timestamp_columns(out)
        out["decision_target_timestamp"] = pd.to_datetime(out[time_col], errors="coerce")
    if buy_price_col != "buy_price":
        out = out.rename(columns={buy_price_col: "buy_price"})
    out = normalize_decision_keys(out)
    if decision_times:
        clock = out["decision_target_timestamp"].dt.strftime("%H:%M:%S")
        out = out.loc[clock.isin(set(decision_times))].copy()
    return (
        out[[*KEY_COLUMNS, "buy_price"]]
        .dropna(
            subset=["date", "symbol", "decision_target_timestamp", "buy_price"],
        )
        .drop_duplicates(list(KEY_COLUMNS))
    )


def build_next_close_label_cache(
    input_path: Path,
    output_path: Path,
    *,
    buy_price_col: str = "buy_price",
    decision_times: tuple[str, ...] = DEFAULT_DECISION_TIMES,
    host: str = DEFAULT_CLICKHOUSE_TICK_HOST,
    port: int = DEFAULT_CLICKHOUSE_TICK_PORT,
    username: str = "",
    password: str = "",
    table: str = DEFAULT_CLICKHOUSE_TICK_TABLE,
    close_offset_us: int = DEFAULT_CLOSE_OFFSET_US,
    close_lookback_seconds: int = DEFAULT_CLOSE_LOOKBACK_SECONDS,
    calendar_days_after: int = 14,
    fee_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _read_base_frame(
        input_path,
        buy_price_col=buy_price_col,
        decision_times=decision_times,
    )
    labels = fetch_next_close_labels(
        base,
        host=host,
        port=port,
        username=username,
        password=password,
        table=table,
        close_offset_us=close_offset_us,
        close_lookback_seconds=close_lookback_seconds,
        calendar_days_after=calendar_days_after,
        fee_bps=fee_bps,
    )
    write_frame(labels, output_path)
    return base, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and cache ClickHouse next-close labels for labeled decision rows."
    )
    add_arguments(parser, "config input output output-dir buy-price-col", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config, arguments, run_name = command_context(
        args, "next_close_labels", default_run_name="build_next_close_labels"
    )
    input_path, output_path = required_io_paths(args, config, "next_close_labels")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output already exists, pass --overwrite: {output_path}")

    output_dir = prepare_output_dir(config, args.output_dir, run_name)

    settings = {
        "buy_price_col": arguments.string("buy_price_col", "buy_price"),
        "decision_times": tuple(
            config_clock_list(config, "next_close_labels", "decision_times", DEFAULT_DECISION_TIMES)
        ),
        **clickhouse_connection(config),
        "close_offset_us": arguments.integer("close_offset_us", DEFAULT_CLOSE_OFFSET_US),
        "close_lookback_seconds": arguments.integer(
            "close_lookback_seconds", DEFAULT_CLOSE_LOOKBACK_SECONDS
        ),
        "calendar_days_after": arguments.integer("calendar_days_after", 14),
        "fee_bps": arguments.float("fee_bps", 0.0),
    }

    print_mapping(
        "next_close_labels",
        {
            "run_id": run_name,
            "input": str(input_path),
            "output": str(output_path),
            "buy_price_col": settings["buy_price_col"],
            "decision_times": ",".join(settings["decision_times"]),
            "clickhouse_table": settings["table"],
        },
    )

    base, labels = build_next_close_label_cache(input_path, output_path, **settings)
    summary = {
        "run_id": run_name,
        "input": str(input_path),
        "output": str(output_path),
        "base_rows": int(len(base)),
        "label_rows": int(len(labels)),
        "label_non_null": int(labels[NEXT_CLOSE_LABEL_COL].notna().sum()),
        "date_min": str(labels["date"].min()) if len(labels) else "",
        "date_max": str(labels["date"].max()) if len(labels) else "",
        "decision_times": list(settings["decision_times"]),
        "buy_price_col": settings["buy_price_col"],
    }
    write_json(output_dir / "next_close_label_cache_trace.json", summary)
    write_json(output_dir / "next_close_label_cache_summary.json", summary)
    print_mapping("next_close_label_cache_summary", summary)
    print(f"\nwrote: {output_path}")
    print(f"trace: {output_dir / 'next_close_label_cache_trace.json'}")


if __name__ == "__main__":
    main()
