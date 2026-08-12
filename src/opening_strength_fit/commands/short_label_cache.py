from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.cache_manifest import cache_manifest_path
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    clickhouse_connection,
    deduplicate_tick_timestamps,
    get_tick_client,
    normalize_clickhouse_ticks,
    query_tick_trade_state_day_window,
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
from opening_strength_fit.horizon_clickhouse_labels import target_offset_us
from opening_strength_fit.io import (
    frame_columns,
    read_frame,
    write_frame_atomic,
    write_json,
)
from opening_strength_fit.labels import _clock_state_values, safe_price_return
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.schema import DECISION_KEY_COLUMNS, normalize_decision_keys

KEY_COLUMNS = DECISION_KEY_COLUMNS
DEFAULT_DECISION_TIMES = tuple(f"09:{minute:02d}:00" for minute in range(31, 41))
DEFAULT_TRADABLE_STATUSES = ("T0", "20", "TRADE")
OPTIONAL_ENTRY_COLUMNS = (
    "entry_source_timestamp",
    "entry_state_age_seconds",
    "status",
    "entry_status",
)


def validate_source_cache(
    input_path: Path,
    *,
    require_manifest: bool,
    expected_schema_version: str,
) -> dict[str, object]:
    if not input_path.exists():
        raise SystemExit(f"short-label source cache does not exist: {input_path}")
    manifest_path = cache_manifest_path(input_path)
    if not manifest_path.exists():
        if require_manifest:
            raise SystemExit(f"short-label source manifest does not exist: {manifest_path}")
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"cannot read short-label source manifest {manifest_path}: {error}"
        ) from error
    actual_schema = str(manifest.get("cache_schema_version", ""))
    if expected_schema_version and actual_schema != expected_schema_version:
        raise SystemExit(
            "short-label source schema mismatch: "
            f"expected {expected_schema_version!r}, got {actual_schema!r}"
        )
    cache_file = manifest.get("cache_file", {})
    if isinstance(cache_file, dict) and cache_file.get("bytes") is not None:
        expected_bytes = int(cache_file["bytes"])
        actual_bytes = int(input_path.stat().st_size)
        if actual_bytes != expected_bytes:
            raise SystemExit(
                f"short-label source size mismatch: {actual_bytes} != {expected_bytes}"
            )
    return manifest


def read_short_label_base(
    input_path: Path,
    *,
    decision_times: tuple[str, ...],
    expected_entry_delay_seconds: int,
    tradable_statuses: tuple[str, ...],
) -> pd.DataFrame:
    available = frame_columns(input_path)
    required = {*KEY_COLUMNS, "entry_timestamp", "buy_price"}
    if tradable_statuses:
        required.update({"status", "entry_status"})
    missing = sorted(required - available)
    if missing:
        raise SystemExit(f"short-label source cache missing columns: {missing}")
    columns = [
        *KEY_COLUMNS,
        "entry_timestamp",
        "buy_price",
        *(column for column in OPTIONAL_ENTRY_COLUMNS if column in available),
    ]
    base = normalize_decision_keys(read_frame(input_path, columns=columns), drop_missing=False)
    base["entry_timestamp"] = pd.to_datetime(base["entry_timestamp"], errors="coerce")
    base["buy_price"] = pd.to_numeric(base["buy_price"], errors="coerce")
    if "entry_source_timestamp" in base:
        base["entry_source_timestamp"] = pd.to_datetime(
            base["entry_source_timestamp"], errors="coerce"
        )
    if "entry_state_age_seconds" in base:
        base["entry_state_age_seconds"] = pd.to_numeric(
            base["entry_state_age_seconds"], errors="coerce"
        )
    if decision_times:
        clock = base["decision_target_timestamp"].dt.strftime("%H:%M:%S")
        base = base.loc[clock.isin(set(decision_times))].copy()
    if (missing_keys := base[list(KEY_COLUMNS)].isna().any(axis=1)).any():
        raise SystemExit(f"short-label source has {int(missing_keys.sum())} rows with missing keys")
    if (duplicate_keys := base.duplicated(list(KEY_COLUMNS), keep=False)).any():
        raise SystemExit(
            f"short-label source keys are not unique: {int(duplicate_keys.sum())} duplicate rows"
        )
    observed_times = set(base["decision_target_timestamp"].dt.strftime("%H:%M:%S"))
    if missing_times := sorted(set(decision_times) - observed_times):
        raise SystemExit(f"short-label source missing decision clocks: {missing_times}")
    has_entry = base["entry_timestamp"].notna()
    delay = (
        base.loc[has_entry, "entry_timestamp"] - base.loc[has_entry, "decision_target_timestamp"]
    ) / pd.Timedelta(seconds=1)
    wrong_delay = ~np.isclose(
        delay.to_numpy(dtype="float64"),
        float(expected_entry_delay_seconds),
        rtol=0.0,
        atol=1e-9,
    )
    if wrong_delay.any():
        examples = sorted(set(delay.loc[wrong_delay].astype(float).tolist()))[:5]
        raise SystemExit(
            "short-label source entry delay mismatch: "
            f"expected {expected_entry_delay_seconds}s, examples={examples}"
        )
    return base.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def compute_short_vwap_labels(
    base: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    hold_seconds: int,
    sell_window_seconds: int,
    volume_unit_multiplier: float,
    fee_bps: float,
    tradable_statuses: tuple[str, ...],
) -> pd.DataFrame:
    samples = base.copy()
    samples["timestamp"] = samples["decision_target_timestamp"]
    state = ticks.copy()
    if not state.empty:
        state = normalize_clickhouse_ticks(state)
        state = deduplicate_tick_timestamps(state, mode="latest_local_timestamp")
    else:
        state = pd.DataFrame(columns=["date", "symbol", "timestamp", "volume", "turnover"])

    aligned_states = {
        suffix: _clock_state_values(
            samples,
            seconds=int(hold_seconds) + offset,
            value_columns=("volume", "turnover"),
            suffix=suffix,
            target_timestamp_col="entry_timestamp",
            state_frame=state,
        )
        for suffix, offset in (("sell_start", 0), ("sell_end", int(sell_window_seconds)))
    }

    out_columns = [
        *KEY_COLUMNS,
        "entry_timestamp",
        "buy_price",
        *(column for column in OPTIONAL_ENTRY_COLUMNS if column in samples),
    ]
    out = samples[out_columns].copy()
    for suffix, aligned in aligned_states.items():
        out[f"{suffix}_target_timestamp"] = pd.to_datetime(
            aligned[f"target_timestamp_{suffix}"], errors="coerce"
        )
        out[f"{suffix}_source_timestamp"] = pd.to_datetime(
            aligned[f"timestamp_{suffix}"], errors="coerce"
        )
        out[f"{suffix}_state_age_seconds"] = pd.to_numeric(
            aligned[f"{suffix}_state_age_seconds"], errors="coerce"
        )
        out[f"{suffix}_volume"] = pd.to_numeric(aligned[f"volume_{suffix}"], errors="coerce")
        out[f"{suffix}_turnover"] = pd.to_numeric(aligned[f"turnover_{suffix}"], errors="coerce")

    out["sell_volume"] = out["sell_end_volume"] - out["sell_start_volume"]
    out["sell_turnover"] = out["sell_end_turnover"] - out["sell_start_turnover"]
    denominator = out["sell_volume"] * float(volume_unit_multiplier)
    out["sell_vwap"] = np.where(
        denominator > 0,
        out["sell_turnover"] / denominator,
        np.nan,
    )
    out["gross_label"] = safe_price_return(out["sell_vwap"], out["buy_price"])
    out["label"] = safe_price_return(
        out["sell_vwap"],
        out["buy_price"],
        fee_bps=float(fee_bps),
    )
    out["valid_label"] = (
        out["label"].notna()
        & np.isfinite(out["label"])
        & out["sell_volume"].gt(0)
        & out["sell_turnover"].gt(0)
        & out["buy_price"].gt(0)
        & out["entry_timestamp"].notna()
        & out["sell_start_source_timestamp"].notna()
        & out["sell_end_source_timestamp"].notna()
    )
    if tradable_statuses:
        allowed = {str(status).upper() for status in tradable_statuses}
        out["valid_label"] &= out["status"].astype(str).str.upper().isin(allowed)
        out["valid_label"] &= out["entry_status"].astype(str).str.upper().isin(allowed)
    out["hold_seconds"] = int(hold_seconds)
    out["sell_window_seconds"] = int(sell_window_seconds)
    out["fee_bps"] = float(fee_bps)
    return out


def fetch_short_vwap_labels(
    base: pd.DataFrame,
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    query_start_offset_us: int,
    hold_seconds: int,
    sell_window_seconds: int,
    volume_unit_multiplier: float,
    fee_bps: float,
    tradable_statuses: tuple[str, ...],
) -> pd.DataFrame:
    if not username or not password:
        raise SystemExit(
            "ClickHouse short labels need credentials. Set CLICKHOUSE_USER and CLICKHOUSE_PASSWORD."
        )
    client = get_tick_client(
        host=host or DEFAULT_CLICKHOUSE_TICK_HOST,
        port=int(port),
        username=username,
        password=password,
    )
    compute_labels = partial(
        compute_short_vwap_labels,
        hold_seconds=hold_seconds,
        sell_window_seconds=sell_window_seconds,
        volume_unit_multiplier=volume_unit_multiplier,
        fee_bps=fee_bps,
        tradable_statuses=tradable_statuses,
    )
    parts: list[pd.DataFrame] = []
    for trading_day, day_base in base.groupby("date", sort=True, observed=True):
        targets = day_base["entry_timestamp"] + pd.to_timedelta(
            int(hold_seconds) + int(sell_window_seconds), unit="s"
        )
        valid_targets = targets.dropna()
        if valid_targets.empty:
            end_offset_us = int(query_start_offset_us)
        else:
            end_offset_us = int(target_offset_us(valid_targets).max())
        symbols = sorted(day_base["symbol"].dropna().astype(str).unique())
        raw_ticks = query_tick_trade_state_day_window(
            client,
            trading_day=str(trading_day),
            symbols=symbols,
            table=table,
            start_offset_us=int(query_start_offset_us),
            end_offset_us=end_offset_us,
        )
        labels = compute_labels(day_base, raw_ticks)
        parts.append(labels)
        print_mapping(
            f"short_labels[{trading_day}]",
            {
                "base_rows": int(len(day_base)),
                "tick_rows": int(len(raw_ticks)),
                "valid_labels": int(labels["valid_label"].sum()),
            },
        )
    if not parts:
        return compute_labels(base, pd.DataFrame())
    labels = pd.concat(parts, ignore_index=True)
    if len(labels) != len(base):
        raise SystemExit(f"short-label row count changed: {len(base)} -> {len(labels)}")
    if labels.duplicated(list(KEY_COLUMNS)).any():
        raise SystemExit("short-label output keys are not unique")
    return labels.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def write_short_label_manifest(
    output_path: Path,
    *,
    labels: pd.DataFrame,
    run_name: str,
    config_path: str,
    source_path: Path,
    source_manifest: dict[str, object],
    schema_version: str,
    hold_seconds: int,
    sell_window_seconds: int,
    expected_entry_delay_seconds: int,
    query_start_offset_us: int,
    fee_bps: float,
) -> dict[str, object]:
    manifest = {
        "manifest_version": 1,
        "run_id": run_name,
        "schema_version": schema_version,
        "config_path": config_path,
        "cache_path": str(output_path),
        "cache_file": {"bytes": int(output_path.stat().st_size)},
        "source_cache_path": str(source_path),
        "source_cache_schema_version": str(source_manifest.get("cache_schema_version", "")),
        "key_columns": list(KEY_COLUMNS),
        "rows": int(len(labels)),
        "valid_labels": int(labels["valid_label"].sum()),
        "non_null_labels": int(labels["label"].notna().sum()),
        "decision_times": sorted(
            labels["decision_target_timestamp"].dt.strftime("%H:%M:%S").unique().tolist()
        ),
        "label_definition": {
            "entry": f"decision_target_timestamp+{expected_entry_delay_seconds}s",
            "hold_seconds": int(hold_seconds),
            "sell_window_seconds": int(sell_window_seconds),
            "sell_alignment": "clock_state",
            "sell_price": "cumulative_turnover_delta/cumulative_volume_delta",
            "query_start_offset_us": int(query_start_offset_us),
            "fee_bps": float(fee_bps),
        },
        "columns": [
            {"name": str(col), "dtype": str(dtype)} for col, dtype in labels.dtypes.items()
        ],
    }
    write_json(cache_manifest_path(output_path), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build standalone ClickHouse short-return VWAP labels from v6 decision rows."
    )
    add_arguments(parser, "config input output output-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config, arguments, run_name = command_context(
        args, "short_labels", default_run_name="build_short_labels"
    )
    input_path, output_path = required_io_paths(args, config, "short_labels")
    manifest_path = cache_manifest_path(output_path)
    if (output_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise SystemExit(f"output or manifest already exists, pass --overwrite: {output_path}")

    output_dir = prepare_output_dir(config, args.output_dir, run_name)
    decision_times = tuple(
        config_clock_list(
            config,
            "short_labels",
            "decision_times",
            DEFAULT_DECISION_TIMES,
        )
    )
    label_settings = {
        "hold_seconds": arguments.integer("hold_seconds", 180),
        "sell_window_seconds": arguments.integer("sell_window_seconds", 60),
        "query_start_offset_us": arguments.integer("query_start_offset_us", 33_300_000_000),
        "volume_unit_multiplier": arguments.float("volume_unit_multiplier", 1.0),
        "fee_bps": arguments.float("fee_bps", 0.0),
        "tradable_statuses": arguments.tuple("tradable_statuses", DEFAULT_TRADABLE_STATUSES),
    }
    expected_entry_delay_seconds = arguments.integer("expected_entry_delay_seconds", 6)
    require_source_manifest = arguments.string("require_source_manifest", "true").lower() not in {
        "false",
        "0",
        "no",
        "off",
    }
    expected_source_schema = arguments.string("expected_source_schema_version")
    schema_version = arguments.string("schema_version", "short_label_h180_vwap60_v1")
    if label_settings["hold_seconds"] != 180 or label_settings["sell_window_seconds"] != 60:
        raise SystemExit(
            "this run is locked to hold_seconds=180 and sell_window_seconds=60; "
            f"got {label_settings['hold_seconds']}/{label_settings['sell_window_seconds']}"
        )

    source_manifest = validate_source_cache(
        input_path,
        require_manifest=require_source_manifest,
        expected_schema_version=expected_source_schema,
    )
    base = read_short_label_base(
        input_path,
        decision_times=decision_times,
        expected_entry_delay_seconds=expected_entry_delay_seconds,
        tradable_statuses=label_settings["tradable_statuses"],
    )
    print_mapping(
        "short_labels",
        {
            "run_id": run_name,
            "input": str(input_path),
            "output": str(output_path),
            "base_rows": int(len(base)),
            "decision_times": ",".join(decision_times),
            "entry_delay_seconds": expected_entry_delay_seconds,
            "hold_seconds": label_settings["hold_seconds"],
            "sell_window_seconds": label_settings["sell_window_seconds"],
            "sell_alignment": "clock_state",
            "tick_timestamp_deduplication": "latest_local_timestamp",
        },
    )
    labels = fetch_short_vwap_labels(
        base,
        **clickhouse_connection(config),
        **label_settings,
    )
    write_frame_atomic(labels, output_path)
    manifest = write_short_label_manifest(
        output_path,
        labels=labels,
        run_name=run_name,
        config_path=args.config,
        source_path=input_path,
        source_manifest=source_manifest,
        schema_version=schema_version,
        hold_seconds=label_settings["hold_seconds"],
        sell_window_seconds=label_settings["sell_window_seconds"],
        expected_entry_delay_seconds=expected_entry_delay_seconds,
        query_start_offset_us=label_settings["query_start_offset_us"],
        fee_bps=label_settings["fee_bps"],
    )
    write_json(output_dir / "short_label_cache_trace.json", manifest)
    write_json(
        output_dir / "short_label_cache_summary.json",
        {
            "run_id": run_name,
            "rows": int(len(labels)),
            "valid_labels": int(labels["valid_label"].sum()),
            "valid_label_ratio": float(labels["valid_label"].mean()) if len(labels) else 0.0,
            "hold_seconds": label_settings["hold_seconds"],
            "sell_window_seconds": label_settings["sell_window_seconds"],
        },
    )
    (output_dir / "_SUCCESS").touch()
    print(f"\nwrote: {output_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
