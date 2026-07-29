from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from opening_strength_fit.clickhouse_daily_reference import (
    DEFAULT_DAILY_MARKET_REFERENCE_TABLE,
    attach_daily_market_reference,
    query_lagged_daily_market_reference,
)
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    deduplicate_tick_timestamps,
    get_tick_client,
    normalize_clickhouse_ticks,
    query_tick_day_window,
)
from opening_strength_fit.config import (
    config_bool,
    config_clock_list,
    config_float,
    config_int,
    config_list,
    config_str,
    config_value,
)
from opening_strength_fit.full_day_labels import (
    build_full_day_narrow_labels,
    build_full_day_temporal_labels,
)
from opening_strength_fit.horizon_clickhouse_labels import (
    DEFAULT_CLOSE_LOOKBACK_SECONDS,
    DEFAULT_CLOSE_OFFSET_US,
    compute_clickhouse_close_labels,
    query_trading_dates,
)
from opening_strength_fit.horizons import horizon_specs, label_column_name
from opening_strength_fit.io import read_frame, write_frame_atomic, write_json
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns
from opening_strength_fit.universe import (
    DEFAULT_A_SHARE_SYMBOL_REGEX,
    filter_symbol_universe,
    load_symbol_list,
)

FULL_DAY_START_OFFSET_US = 33_300_000_000
FULL_DAY_END_OFFSET_US = 54_000_000_000
DEFAULT_DECISION_RANGES = (
    "09:31:00..11:29:00",
    "13:01:00..14:59:00",
)
LABEL_ONLY_CLICKHOUSE_COLUMNS = (
    "TradingDay",
    "Symbol",
    "ExchTimeOffsetUs",
    "LocalTimeStamp",
    "TradeNum",
    "Volume",
    "Turnover",
    "AskPrice1",
    "Status",
)
LABEL_ONLY_BASE_COLUMNS = (
    "date",
    "symbol",
    "timestamp",
    "ask_price_1",
    "volume",
    "turnover",
    "status",
)


def _setting(args, config: dict, arg_name: str, key: str, env_name: str, default):
    arg_value = getattr(args, arg_name, None)
    if arg_value not in (None, ""):
        return arg_value
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return env_value
    return config_value(config, "clickhouse", key, default)


def _date_bounds(config: dict) -> tuple[str, str]:
    start = config_value(
        config,
        "data",
        "start_date",
        config_value(config, "clickhouse", "start_date", None),
    )
    end = config_value(
        config,
        "data",
        "end_date",
        config_value(config, "clickhouse", "end_date", None),
    )
    if not start or not end:
        raise SystemExit("full-day ClickHouse cache needs [data].start_date and end_date")
    return str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())


def _labels_only(config: dict) -> bool:
    mode = config_str(config, "full_day_labels", "output_mode", "audit").strip().lower()
    if mode not in {"audit", "labels_only"}:
        raise SystemExit("[full_day_labels].output_mode must be 'audit' or 'labels_only'")
    return mode == "labels_only"


def _project_output(
    frame: pd.DataFrame,
    *,
    horizons: list[str],
    labels_only: bool,
) -> pd.DataFrame:
    if not labels_only:
        return frame
    output_columns = [
        "date",
        "symbol",
        "decision_target_timestamp",
        *(label_column_name(horizon) for horizon in horizons),
    ]
    missing = [column for column in output_columns if column not in frame]
    if missing:
        raise RuntimeError(f"labels-only full-day output missing columns: {missing}")
    out = frame.loc[:, output_columns].copy()
    for horizon in horizons:
        label_col = label_column_name(horizon)
        valid_col = f"valid_{label_col}"
        if valid_col in frame:
            out.loc[~frame[valid_col].fillna(False).astype(bool), label_col] = float("nan")
    return out


def _daily_summary(frame: pd.DataFrame, horizons: list[str]) -> dict[str, object]:
    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame else 0,
    }
    if "decision_target_timestamp" in frame and not frame.empty:
        decisions = pd.to_datetime(frame["decision_target_timestamp"], errors="coerce")
        summary["decision_time_min"] = decisions.dt.strftime("%H:%M:%S").min()
        summary["decision_time_max"] = decisions.dt.strftime("%H:%M:%S").max()
        summary["decision_points"] = int(decisions.nunique())
    valid_counts: dict[str, int] = {}
    for horizon in horizons:
        valid_col = f"valid_{label_column_name(horizon)}"
        if valid_col in frame:
            valid_counts[horizon] = int(frame[valid_col].fillna(False).sum())
    audit = frame.attrs.get("full_day_audit", {})
    summary["valid_rows"] = valid_counts or dict(audit.get("valid_rows", {}))

    comparisons = [("entry_source_timestamp", "entry_timestamp")]
    for horizon in horizons:
        comparisons.extend(
            [
                (
                    f"sell_start_source_timestamp_{horizon}",
                    f"sell_start_target_timestamp_{horizon}",
                ),
                (
                    f"sell_end_source_timestamp_{horizon}",
                    f"sell_end_target_timestamp_{horizon}",
                ),
            ]
        )
    violations = 0
    compared = 0
    for source_col, target_col in comparisons:
        if source_col not in frame or target_col not in frame:
            continue
        source = pd.to_datetime(frame[source_col], errors="coerce")
        target = pd.to_datetime(frame[target_col], errors="coerce")
        comparable = source.notna() & target.notna()
        compared += int(comparable.sum())
        violations += int((comparable & source.gt(target)).sum())
    summary["causal_timestamp_comparisons"] = compared or int(
        audit.get("causal_timestamp_comparisons", 0)
    )
    summary["causal_timestamp_violations"] = violations or int(
        audit.get("causal_timestamp_violations", 0)
    )
    return summary


def _attach_close_labels(
    frame: pd.DataFrame,
    close_horizons: list[str],
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    config: dict,
) -> pd.DataFrame:
    if not close_horizons or frame.empty:
        return frame
    labels = compute_clickhouse_close_labels(
        frame,
        horizon_specs(close_horizons),
        host=host,
        port=port,
        username=username,
        password=password,
        table=table,
        close_offset_us=config_int(
            config, "full_day_labels", "close_offset_us", DEFAULT_CLOSE_OFFSET_US
        ),
        close_lookback_seconds=config_int(
            config,
            "full_day_labels",
            "close_lookback_seconds",
            DEFAULT_CLOSE_LOOKBACK_SECONDS,
        ),
        calendar_days_after=config_int(
            config, "full_day_labels", "next_close_calendar_days_after", 14
        ),
        fee_bps=config_float(config, "labels", "fee_bps", 0.0),
    )
    keys = ["date", "symbol", "decision_target_timestamp"]
    label_cols = [label_column_name(name) for name in close_horizons]
    available = [column for column in label_cols if column in labels]
    out = frame.merge(labels[[*keys, *available]], on=keys, how="left", validate="one_to_one")
    base_valid = out["valid_entry"].fillna(False)
    for name in close_horizons:
        label_col = label_column_name(name)
        if label_col not in out:
            out[label_col] = float("nan")
        out[f"valid_{label_col}"] = out[label_col].notna() & base_valid
    return out


def _prepare_ticks(ticks: pd.DataFrame, config: dict) -> pd.DataFrame:
    if _labels_only(config):
        out = ensure_timestamp_columns(standardize_columns(ticks))
    else:
        out = normalize_clickhouse_ticks(ticks)
    deduplication = config_str(config, "data", "tick_timestamp_deduplication", "").strip()
    if deduplication:
        out = deduplicate_tick_timestamps(out, mode=deduplication)
        print_mapping("tick_timestamp_deduplication", out.attrs["tick_timestamp_deduplication"])
    if config_bool(config, "universe", "enabled", True):
        symbols_file = config_str(config, "universe", "symbols_file", "")
        out = filter_symbol_universe(
            out,
            symbol_regex=config_str(
                config, "universe", "symbol_regex", DEFAULT_A_SHARE_SYMBOL_REGEX
            ),
            symbols=load_symbol_list(symbols_file) if symbols_file else None,
        )
    if _labels_only(config):
        required = set(LABEL_ONLY_BASE_COLUMNS) - {"status"}
        missing = sorted(required - set(out.columns))
        if missing:
            raise SystemExit(f"labels-only full-day ticks missing columns: {missing}")
        columns = [column for column in LABEL_ONLY_BASE_COLUMNS if column in out.columns]
        out = out.loc[:, columns].copy()
    return out


def _build_day(ticks: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    horizons = config_list(config, "full_day_labels", "horizons", ["5m", "30m"])
    timed_horizons = [spec.name for spec in horizon_specs(horizons) if spec.seconds is not None]
    close_horizons = [name for name in horizons if name in {"close", "next_close"}]
    max_lag = config_value(config, "sample", "decision_max_lag_seconds", 5)
    builder = (
        build_full_day_narrow_labels if _labels_only(config) else build_full_day_temporal_labels
    )
    builder_kwargs = {
        "ticks": ticks,
        "decision_times": config_clock_list(
            config,
            "full_day_labels",
            "decision_ranges",
            DEFAULT_DECISION_RANGES,
        ),
        "horizons": timed_horizons,
        "sessions": config_list(
            config,
            "full_day_labels",
            "sessions",
            ["09:30:00-11:30:00", "13:00:00-15:00:00"],
        ),
        "decision_max_lag_seconds": None if max_lag in (None, "") else int(max_lag),
        "entry_clock_delay_seconds": config_int(config, "labels", "entry_clock_delay_seconds", 6),
        "sell_window_trading_seconds": config_int(config, "labels", "sell_window_seconds", 60),
        "buy_price_col": config_str(config, "labels", "buy_price_col", "ask_price_1"),
        "volume_col": config_str(config, "labels", "volume_col", "volume"),
        "turnover_col": config_str(config, "labels", "turnover_col", "turnover"),
        "volume_unit_multiplier": config_float(config, "labels", "volume_unit_multiplier", 1.0),
        "fee_bps": config_float(config, "labels", "fee_bps", 0.0),
        "tradable_statuses": config_list(config, "filters", "tradable_statuses", []),
        "require_cross_section_ready_entry": config_bool(
            config, "labels", "require_entry_after_cross_section_ready", True
        ),
    }
    if not _labels_only(config):
        builder_kwargs.update(
            {
                "entry_tick_delay_audit": config_int(config, "labels", "entry_tick_delay", 2),
                "include_preopen": config_bool(config, "features", "include_preopen", True),
                "preopen_price_mode": config_str(
                    config, "features", "preopen_price_mode", "legacy_last_price"
                ),
                "preopen_match_time": config_str(
                    config, "features", "preopen_match_time", "09:25:00"
                ),
                "build_features": True,
            }
        )
    frame = builder(
        **builder_kwargs,
    )
    return frame, close_horizons


def _path_days(args, config: dict) -> Iterator[tuple[str, pd.DataFrame, object | None]]:
    path = args.input or config_str(config, "data", "tick_path", "")
    if not path:
        raise SystemExit("full-day path source needs --input or [data].tick_path")
    ticks = normalize_clickhouse_ticks(read_frame(path))
    for date, day in ticks.groupby("date", sort=True, observed=True):
        yield str(date), day.reset_index(drop=True), None


def run(args, config: dict, *, run_name: str, output_dir: Path) -> None:
    cache_root_raw = config_value(
        config,
        "cache",
        "path",
        config_value(config, "cache", "labeled_path", ""),
    )
    if not cache_root_raw:
        raise SystemExit("full-day cache needs a directory in [cache].path")
    cache_root = Path(str(cache_root_raw))
    if cache_root.suffix:
        raise SystemExit("full-day [cache].path must be a directory, not a file")
    cache_root.mkdir(parents=True, exist_ok=True)
    success_path = cache_root / "_SUCCESS"
    success_path.unlink(missing_ok=True)

    horizons = config_list(
        config, "full_day_labels", "horizons", ["5m", "30m", "close", "next_close"]
    )
    source = "path" if args.input else config_str(config, "data", "source", "clickhouse")
    source = source.strip().lower()
    overwrite = config_bool(config, "cache", "overwrite", False)

    host = str(
        _setting(
            args,
            config,
            "clickhouse_host",
            "host",
            "CLICKHOUSE_HOST",
            DEFAULT_CLICKHOUSE_TICK_HOST,
        )
    )
    port = int(
        _setting(
            args,
            config,
            "clickhouse_port",
            "port",
            "CLICKHOUSE_PORT",
            DEFAULT_CLICKHOUSE_TICK_PORT,
        )
    )
    username = str(_setting(args, config, "clickhouse_user", "user", "CLICKHOUSE_USER", "") or "")
    password = str(
        _setting(args, config, "clickhouse_password", "password", "CLICKHOUSE_PASSWORD", "") or ""
    )
    table = str(
        _setting(
            args,
            config,
            "clickhouse_table",
            "table",
            "CLICKHOUSE_TICK_TABLE",
            DEFAULT_CLICKHOUSE_TICK_TABLE,
        )
    )
    client = None
    if source == "clickhouse":
        if not username or not password:
            raise SystemExit(
                "missing ClickHouse credentials: set CLICKHOUSE_USER and CLICKHOUSE_PASSWORD"
            )
        client = get_tick_client(host=host, port=port, username=username, password=password)
        start_date, end_date = _date_bounds(config)
        dates = query_trading_dates(client, table=table, start_date=start_date, end_date=end_date)
        use_universe = config_bool(config, "universe", "enabled", True)
        symbols_file = config_str(config, "universe", "symbols_file", "")
        symbols = sorted(load_symbol_list(symbols_file)) if use_universe and symbols_file else None
        symbol_regex = (
            config_str(config, "universe", "symbol_regex", DEFAULT_A_SHARE_SYMBOL_REGEX)
            if use_universe
            else None
        )

        def clickhouse_days() -> Iterator[tuple[str, pd.DataFrame, object | None]]:
            assert client is not None
            for date in dates:
                shard = cache_root / f"date={date}" / "labels.parquet"
                summary_path = shard.with_name("summary.json")
                if shard.exists() and summary_path.exists() and not overwrite:
                    yield date, pd.DataFrame(), client
                    continue
                ticks = query_tick_day_window(
                    client,
                    trading_day=date,
                    table=table,
                    start_offset_us=int(
                        args.start_offset_us
                        if args.start_offset_us is not None
                        else config_value(
                            config,
                            "clickhouse",
                            "start_offset_us",
                            FULL_DAY_START_OFFSET_US,
                        )
                    ),
                    end_offset_us=int(
                        args.end_offset_us
                        if args.end_offset_us is not None
                        else config_value(
                            config,
                            "clickhouse",
                            "end_offset_us",
                            FULL_DAY_END_OFFSET_US,
                        )
                    ),
                    symbol_regex=symbol_regex,
                    symbols=symbols,
                    columns=(LABEL_ONLY_CLICKHOUSE_COLUMNS if _labels_only(config) else None),
                    collapse_local_timestamp=_labels_only(config),
                )
                yield date, ticks, client

        day_iterator = clickhouse_days()
    elif source == "path":
        day_iterator = _path_days(args, config)
    else:
        raise SystemExit("full-day cache data.source must be clickhouse or path")

    daily_summaries: list[dict[str, object]] = []
    for date, raw_ticks, day_client in day_iterator:
        shard = cache_root / f"date={date}" / "labels.parquet"
        summary_path = shard.with_name("summary.json")
        if shard.exists() and summary_path.exists() and not overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["action"] = "resumed"
            daily_summaries.append(summary)
            print_mapping(f"full_day_cache[{date}]", summary)
            continue
        if raw_ticks.empty:
            print(f"skip empty full-day tick source: {date}")
            continue
        ticks = _prepare_ticks(raw_ticks, config)
        if ticks.empty:
            print(f"skip empty full-day universe: {date}")
            continue
        if day_client is not None and config_bool(
            config, "daily_market_reference", "enabled", False
        ):
            reference = query_lagged_daily_market_reference(
                day_client,
                trading_day=date,
                symbols=ticks["symbol"].dropna().astype(str).unique().tolist(),
                table=config_str(
                    config,
                    "daily_market_reference",
                    "table",
                    DEFAULT_DAILY_MARKET_REFERENCE_TABLE,
                ),
                lag_sessions=config_int(config, "daily_market_reference", "lag_sessions", 1),
                market_cap_unit_multiplier=config_float(
                    config,
                    "daily_market_reference",
                    "market_cap_unit_multiplier",
                    10_000.0,
                ),
                share_unit_multiplier=config_float(
                    config,
                    "daily_market_reference",
                    "share_unit_multiplier",
                    10_000.0,
                ),
            )
            ticks = attach_daily_market_reference(ticks, reference)
        labeled, close_horizons = _build_day(ticks, config)
        if close_horizons:
            if source != "clickhouse":
                raise SystemExit("close/next_close labels require data.source=clickhouse")
            labeled = _attach_close_labels(
                labeled,
                close_horizons,
                host=host,
                port=port,
                username=username,
                password=password,
                table=table,
                config=config,
            )
        summary = {"date": date, "action": "built", **_daily_summary(labeled, horizons)}
        if summary["causal_timestamp_violations"]:
            raise RuntimeError(f"causal timestamp violation detected on {date}: {summary}")
        output_frame = _project_output(
            labeled,
            horizons=horizons,
            labels_only=_labels_only(config),
        )
        summary["columns"] = int(len(output_frame.columns))
        summary["output_columns"] = list(output_frame.columns)
        write_frame_atomic(output_frame, shard)
        write_json(summary_path, summary, atomic=True)
        daily_summaries.append(summary)
        print_mapping(f"full_day_cache[{date}]", summary)

    if not daily_summaries:
        raise SystemExit("full-day source produced no cache shards")
    manifest = {
        "run_id": run_name,
        "schema_version": config_str(
            config, "cache", "schema_version", "full_day_clock6_temporal_v1"
        ),
        "cache_root": str(cache_root),
        "source": source,
        "horizons": horizons,
        "sessions": config_list(
            config,
            "full_day_labels",
            "sessions",
            ["09:30:00-11:30:00", "13:00:00-15:00:00"],
        ),
        "decision_times": config_clock_list(
            config,
            "full_day_labels",
            "decision_ranges",
            DEFAULT_DECISION_RANGES,
        ),
        "days": daily_summaries,
        "total_rows": sum(int(item.get("rows", 0)) for item in daily_summaries),
        "causal_timestamp_violations": sum(
            int(item.get("causal_timestamp_violations", 0)) for item in daily_summaries
        ),
    }
    manifest_path = cache_root / "full_day_label_cache_manifest.json"
    write_json(manifest_path, manifest, atomic=True)
    write_json(success_path, {"run_id": run_name}, atomic=True)
    trace = {
        "run_id": run_name,
        "manifest_path": str(manifest_path),
        "cache_root": str(cache_root),
        "summary": {
            "days": len(daily_summaries),
            "rows": manifest["total_rows"],
            "causal_timestamp_violations": manifest["causal_timestamp_violations"],
        },
    }
    write_json(output_dir / "full_day_label_cache_trace.json", trace, atomic=True)
    print_mapping("full_day_label_cache", trace["summary"])
    print(f"\nwrote full-day cache: {cache_root}")
    print(f"manifest: {manifest_path}")
