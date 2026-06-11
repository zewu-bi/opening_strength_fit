from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import (
    NEXT_CLOSE_LABEL_COL,
    normalize_next_close_labels,
    write_json,
)
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
)
from opening_strength_fit.config import (
    config_clock_list,
    config_float,
    config_int,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.horizon_clickhouse_labels import (
    DEFAULT_CLOSE_LOOKBACK_SECONDS,
    DEFAULT_CLOSE_OFFSET_US,
    compute_clickhouse_close_labels,
)
from opening_strength_fit.horizons import HorizonSpec
from opening_strength_fit.io import frame_columns, read_frame, write_frame
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns

DEFAULT_DECISION_TIMES = tuple(f"09:{minute:02d}:00" for minute in range(31, 41))
KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")


def _arg_or_config(args, config: dict, name: str, default: str = "") -> str:
    value = getattr(args, name, "")
    if value not in (None, ""):
        return str(value)
    return config_str(config, "next_close_labels", name, default)


def _available_columns(path: Path) -> set[str]:
    try:
        return frame_columns(path)
    except SystemExit:
        return set()


def _read_base_frame(
    path: Path,
    *,
    buy_price_col: str,
    decision_times: tuple[str, ...],
) -> pd.DataFrame:
    available = _available_columns(path)
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
    out["date"] = out["date"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"],
        errors="coerce",
    )
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


def fetch_next_close_labels(
    base: pd.DataFrame,
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    close_offset_us: int = DEFAULT_CLOSE_OFFSET_US,
    close_lookback_seconds: int = DEFAULT_CLOSE_LOOKBACK_SECONDS,
    calendar_days_after: int = 14,
    fee_bps: float = 0.0,
) -> pd.DataFrame:
    labels = compute_clickhouse_close_labels(
        base[[*KEY_COLUMNS, "buy_price"]].copy(),
        [HorizonSpec(name="next_close", label="next close", seconds=None)],
        host=host or DEFAULT_CLICKHOUSE_TICK_HOST,
        port=int(port),
        username=username,
        password=password,
        table=table,
        close_offset_us=int(close_offset_us),
        close_lookback_seconds=int(close_lookback_seconds),
        calendar_days_after=int(calendar_days_after),
        fee_bps=float(fee_bps),
    )
    return normalize_next_close_labels(labels, key_columns=KEY_COLUMNS)


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
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--buy-price-col", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if args.config else "build_next_close_labels"
    input_raw = args.input or config_str(config, "next_close_labels", "input_path", "")
    output_raw = args.output or config_str(config, "next_close_labels", "output_path", "")
    if not input_raw:
        raise SystemExit("missing input path: pass --input or [next_close_labels].input_path")
    if not output_raw:
        raise SystemExit("missing output path: pass --output or [next_close_labels].output_path")

    input_path = Path(input_raw)
    output_path = Path(output_raw)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output already exists, pass --overwrite: {output_path}")

    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    decision_times = tuple(
        config_clock_list(
            config,
            "next_close_labels",
            "decision_times",
            DEFAULT_DECISION_TIMES,
        )
    )
    buy_price_col = args.buy_price_col or config_str(
        config,
        "next_close_labels",
        "buy_price_col",
        "buy_price",
    )
    host = (
        config_str(config, "clickhouse", "host", "")
        or os.environ.get("CLICKHOUSE_HOST", "")
        or DEFAULT_CLICKHOUSE_TICK_HOST
    )
    port = int(
        config_str(config, "clickhouse", "port", "")
        or os.environ.get("CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_TICK_PORT)
    )
    username = os.environ.get("CLICKHOUSE_USER", "")
    password = os.environ.get("CLICKHOUSE_PASSWORD", "")
    table = (
        config_str(config, "clickhouse", "table", "")
        or os.environ.get("CLICKHOUSE_TICK_TABLE", "")
        or DEFAULT_CLICKHOUSE_TICK_TABLE
    )
    close_offset_us = config_int(
        config,
        "next_close_labels",
        "close_offset_us",
        DEFAULT_CLOSE_OFFSET_US,
    )
    close_lookback_seconds = config_int(
        config,
        "next_close_labels",
        "close_lookback_seconds",
        DEFAULT_CLOSE_LOOKBACK_SECONDS,
    )
    calendar_days_after = config_int(
        config,
        "next_close_labels",
        "calendar_days_after",
        14,
    )
    fee_bps = config_float(config, "next_close_labels", "fee_bps", 0.0)

    print_mapping(
        "next_close_labels",
        {
            "run_id": run_name,
            "input": str(input_path),
            "output": str(output_path),
            "buy_price_col": buy_price_col,
            "decision_times": ",".join(decision_times),
            "clickhouse_table": table,
        },
    )

    base, labels = build_next_close_label_cache(
        input_path,
        output_path,
        buy_price_col=buy_price_col,
        decision_times=decision_times,
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
    summary = {
        "run_id": run_name,
        "input": str(input_path),
        "output": str(output_path),
        "base_rows": int(len(base)),
        "label_rows": int(len(labels)),
        "label_non_null": int(labels[NEXT_CLOSE_LABEL_COL].notna().sum()),
        "date_min": str(labels["date"].min()) if len(labels) else "",
        "date_max": str(labels["date"].max()) if len(labels) else "",
        "decision_times": list(decision_times),
        "buy_price_col": buy_price_col,
    }
    write_json(output_dir / "next_close_label_cache_trace.json", summary)
    (output_dir / "next_close_label_cache_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print_mapping("next_close_label_cache_summary", summary)
    print(f"\nwrote: {output_path}")
    print(f"trace: {output_dir / 'next_close_label_cache_trace.json'}")


if __name__ == "__main__":
    main()
