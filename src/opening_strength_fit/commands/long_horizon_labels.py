from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from opening_strength_fit.commands.training_dataset_build import (
    KEY_COLUMNS,
    RAW_LABEL_TICK_COLUMNS,
    _build_label_base,
    _normalize_keys,
    _validate_output_keys,
    compute_short_label_set,
)
from opening_strength_fit.config import (
    config_float,
    config_int,
    config_list,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import read_frame, write_frame_atomic, write_json
from opening_strength_fit.training_dataset_features import (
    decode_clickhouse_text,
    normalize_clickhouse_date,
)

STATE_TICK_COLUMNS = ("Symbol", "ExchTimeOffsetUs", "Volume", "Turnover")
LABEL_COLUMNS = ("label_hold_10m", "label_hold_1h", "label_same_day_close")
VALID_COLUMNS = ("valid_hold_10m", "valid_hold_1h", "valid_same_day_close")


def _years(config: dict) -> tuple[int, ...]:
    years = tuple(int(value) for value in config_list(config, "dataset", "years", []))
    if not years or len(set(years)) != len(years):
        raise SystemExit("[dataset].years must contain unique years")
    return years


def _complete_year(root: Path, year: int) -> Path:
    year_root = root / f"year={year}"
    if not (year_root / "_SUCCESS").exists():
        raise SystemExit(f"source year is incomplete: {year_root}")
    return year_root


def _tick_path(year_root: Path, trading_day: str) -> Path:
    path = year_root / "ticks" / f"date={trading_day}.parquet"
    if not path.exists():
        raise SystemExit(f"missing raw tick day: {path}")
    return path


def _trading_days(year_root: Path, start_date: str, end_date: str) -> list[str]:
    days = []
    for path in sorted((year_root / "ticks").glob("date=*.parquet")):
        day = path.stem.removeprefix("date=")
        if start_date <= day <= end_date:
            days.append(day)
    return days


def _state_ticks(base_ticks: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    parts = [base_ticks.loc[:, list(STATE_TICK_COLUMNS)].copy()]
    parts.extend(read_frame(path, columns=list(STATE_TICK_COLUMNS)) for path in paths)
    state = pd.concat(parts, ignore_index=True)
    state["Symbol"] = decode_clickhouse_text(state["Symbol"])
    state["ExchTimeOffsetUs"] = pd.to_numeric(state["ExchTimeOffsetUs"], errors="coerce")
    state = state.dropna(subset=["Symbol", "ExchTimeOffsetUs"])
    return (
        state.sort_values(["Symbol", "ExchTimeOffsetUs"], kind="mergesort")
        .drop_duplicates(["Symbol", "ExchTimeOffsetUs"], keep="last")
        .reset_index(drop=True)
    )


def _valid_entry(base: pd.DataFrame, tradable_statuses: tuple[str, ...]) -> pd.Series:
    allowed = {value.upper() for value in tradable_statuses}
    valid = pd.to_numeric(base["buy_price"], errors="coerce").gt(0)
    for column in ("status", "entry_status"):
        if allowed:
            valid &= base[column].astype(str).str.upper().isin(allowed)
    valid &= base["entry_after_cross_section_ready"].fillna(False).astype(bool)
    return valid


def same_day_close_label(
    base: pd.DataFrame,
    close_reference: pd.DataFrame,
    *,
    tradable_statuses: tuple[str, ...],
    fee_bps: float,
) -> pd.DataFrame:
    close = close_reference.rename(
        columns={"TradingDay": "date", "Symbol": "symbol", "ClosePrice": "_close"}
    ).copy()
    close["date"] = normalize_clickhouse_date(close["date"])
    close["symbol"] = decode_clickhouse_text(close["symbol"])
    close = close.drop_duplicates(["date", "symbol"], keep="last")
    out = base[[*KEY_COLUMNS, "buy_price"]].merge(
        close[["date", "symbol", "_close"]],
        on=["date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    buy = pd.to_numeric(out["buy_price"], errors="coerce")
    sell = pd.to_numeric(out["_close"], errors="coerce")
    valid = _valid_entry(base, tradable_statuses) & buy.gt(0) & sell.gt(0)
    label = (sell / buy - 1.0 - float(fee_bps) / 10_000.0).where(valid)
    result = out[list(KEY_COLUMNS)].copy()
    result["label_same_day_close"] = label
    result["valid_same_day_close"] = label.notna()
    return result


def _reuse_next_close(base: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    source = _normalize_keys(read_frame(source_path, columns=[*KEY_COLUMNS, "label_next_close"]))
    duplicate = source.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        raise SystemExit(
            f"next-close source has {int(duplicate.sum())} duplicate keys: {source_path}"
        )
    merged = base[list(KEY_COLUMNS)].merge(
        source,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_keys = merged["_merge"].ne("both")
    if missing_keys.any():
        raise SystemExit(
            f"next-close source misses {int(missing_keys.sum())} base keys: {source_path}"
        )
    merged = merged.drop(columns="_merge")
    merged["valid_next_close"] = merged["label_next_close"].notna()
    return merged


def build_label_year(
    config: dict,
    config_path: Path,
    *,
    year: int,
    start_date: str,
    end_date: str,
    overwrite: bool,
) -> dict[str, object]:
    base_root_value = config_str(config, "dataset", "raw_source_root", "").strip()
    output_root_value = config_str(config, "dataset", "label_output_root", "").strip()
    next_close_root_value = config_str(config, "dataset", "next_close_label_root", "").strip()
    if not base_root_value or not output_root_value or not next_close_root_value:
        raise SystemExit(
            "[dataset] requires raw_source_root, label_output_root, and next_close_label_root"
        )
    base_root = Path(base_root_value)
    output_root = Path(output_root_value)
    next_close_root = Path(next_close_root_value)
    base_year_root = _complete_year(base_root, year)
    state_roots = [Path(value) for value in config_list(config, "dataset", "state_raw_roots", [])]
    state_year_roots = [_complete_year(root, year) for root in state_roots]
    next_close_year_root = _complete_year(next_close_root, year)
    next_close_path = next_close_year_root / "labels.parquet"
    if not next_close_path.exists():
        raise SystemExit(f"missing next-close source: {next_close_path}")

    clocks = tuple(config_list(config, "dataset", "decision_times", []))
    horizons = tuple(int(value) for value in config_list(config, "dataset", "horizons_seconds", []))
    if horizons != (600, 3600):
        raise SystemExit("long label horizons must be [600, 3600]")
    tradable = tuple(config_list(config, "dataset", "tradable_statuses", ["T0", "20", "TRADE"]))
    fee_bps = config_float(config, "dataset", "fee_bps", 0.0)
    days = _trading_days(base_year_root, start_date, end_date)
    if not days:
        raise SystemExit(f"no raw tick days in {start_date}..{end_date}")
    close_reference = read_frame(
        base_year_root / "close_reference.parquet",
        columns=["TradingDay", "Symbol", "ClosePrice"],
    )

    parts = []
    for index, trading_day in enumerate(days, start=1):
        base_path = _tick_path(base_year_root, trading_day)
        base_ticks = read_frame(base_path, columns=list(RAW_LABEL_TICK_COLUMNS))
        base = _build_label_base(
            base_ticks,
            trading_day=trading_day,
            decision_times=clocks,
            feature_tick_start_offset_us=config_int(
                config, "dataset", "feature_tick_start_offset_us", 0
            ),
            entry_delay_seconds=config_int(config, "dataset", "entry_delay_seconds", 6),
        )
        state_paths = [_tick_path(root, trading_day) for root in state_year_roots]
        state = _state_ticks(base_ticks, state_paths)
        timed = compute_short_label_set(
            base,
            state,
            horizons=horizons,
            sell_window_seconds=config_int(config, "dataset", "sell_window_seconds", 60),
            volume_unit_multiplier=config_float(config, "dataset", "volume_unit_multiplier", 1.0),
            fee_bps=fee_bps,
            tradable_statuses=tradable,
        ).rename(
            columns={
                "label_short_10m": "label_hold_10m",
                "valid_short_10m": "valid_hold_10m",
                "label_short_60m": "label_hold_1h",
                "valid_short_60m": "valid_hold_1h",
            }
        )
        same_close = same_day_close_label(
            base,
            close_reference,
            tradable_statuses=tradable,
            fee_bps=fee_bps,
        )
        part = timed.merge(same_close, on=list(KEY_COLUMNS), validate="one_to_one")
        parts.append(part)
        print(
            f"long labels year={year} day={index}/{len(days)} date={trading_day} rows={len(part)}",
            flush=True,
        )

    output = _normalize_keys(pd.concat(parts, ignore_index=True))
    _validate_output_keys(output, clocks)
    reused = _reuse_next_close(output, next_close_path)
    output = output.merge(reused, on=list(KEY_COLUMNS), validate="one_to_one")
    ordered_columns = [
        *KEY_COLUMNS,
        *LABEL_COLUMNS,
        "label_next_close",
        *VALID_COLUMNS,
        "valid_next_close",
    ]
    output = output[ordered_columns]
    label_columns = [*LABEL_COLUMNS, "label_next_close"]
    output[label_columns] = output[label_columns].astype("float32")

    year_root = output_root / f"year={year}"
    output_path = year_root / "labels.parquet"
    success_path = year_root / "_SUCCESS"
    if output_path.exists() and not overwrite:
        raise SystemExit(f"label output exists, pass --overwrite: {output_path}")
    if overwrite and success_path.exists():
        success_path.unlink()
    write_frame_atomic(output, output_path)
    parquet = pq.ParquetFile(output_path)
    manifest = {
        "schema_version": "opening_long_horizon_labels_v1",
        "run_id": run_id(config, config_path),
        "kind": "long_horizon_labels",
        "year": int(year),
        "date_start": start_date,
        "date_end": end_date,
        "rows": len(output),
        "columns": list(output.columns),
        "key_columns": list(KEY_COLUMNS),
        "label_columns": label_columns,
        "valid_columns": [*VALID_COLUMNS, "valid_next_close"],
        "decision_times": list(clocks),
        "source_raw_root": str(base_root),
        "state_raw_roots": [str(root) for root in state_roots],
        "next_close_source": str(next_close_path),
        "definitions": {
            "entry": "decision_target_timestamp+6s clock state from base raw PVC",
            "hold_10m": "entry+600s then 60s cumulative turnover/volume VWAP",
            "hold_1h": "entry+3600s then 60s cumulative turnover/volume VWAP",
            "same_day_close": "same trading day close reference / buy_price - 1",
            "next_close": "reused without recomputation from existing authoritative label root",
        },
        "non_null_rows": {column: int(output[column].notna().sum()) for column in label_columns},
        "file": {"path": str(output_path), "bytes": output_path.stat().st_size},
        "parquet_rows": int(parquet.metadata.num_rows),
        "contains_features": False,
    }
    write_json(year_root / "manifest.json", manifest, sort_keys=True, atomic=True)
    success_path.touch()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build 10m VWAP, 1h VWAP, and same-day-close labels from PVC sources."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_toml(config_path)
    years = _years(config)
    if int(args.year) not in years:
        raise SystemExit(f"year {args.year} is not configured")
    start_date = args.start_date or f"{args.year}-01-01"
    end_date = args.end_date or f"{args.year}-12-31"
    build_label_year(
        config,
        config_path,
        year=int(args.year),
        start_date=start_date,
        end_date=end_date,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
