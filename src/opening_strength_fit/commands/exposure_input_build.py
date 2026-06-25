from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, write_json
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    get_tick_client,
    validate_table_name,
)
from opening_strength_fit.config import (
    config_bool,
    config_int,
    config_list,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.exposure_audit import normalize_audit_frame
from opening_strength_fit.io import read_frame, write_frame
from opening_strength_fit.prediction_frames import prediction_files
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

DEFAULT_DAILY_BAR_TABLE = "stock.daily_bar_jy"
DEFAULT_INDUSTRY_TABLE = "stock.industry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a date,symbol keyed exposure input parquet with market-cap and "
            "industry fields for osf-audit-exposure."
        )
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--predictions", action="append")
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--trace-output", default="")
    parser.add_argument("--pool", choices=["universe", "S", "M", "L"], default="")
    parser.add_argument("--pool-date-lag-sessions", type=int, default=None)
    parser.add_argument("--date-chunk-size", type=int, default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--clickhouse-host", default="")
    parser.add_argument("--clickhouse-port", type=int, default=None)
    parser.add_argument("--clickhouse-user", default="")
    parser.add_argument("--clickhouse-password", default="")
    parser.add_argument("--daily-bar-table", default="")
    parser.add_argument("--industry-table", default="")
    parser.add_argument(
        "--include-decision-timestamp",
        action="store_true",
        help="Write one exposure row per full prediction key instead of daily date,symbol keys.",
    )
    return parser.parse_args()


def _arg_list(
    args: argparse.Namespace,
    config: dict,
    name: str,
    default: Iterable[str],
) -> list[str]:
    value = getattr(args, name)
    if value:
        return list(value)
    return config_list(config, "exposure_input", name, tuple(default))


def _arg_str(args: argparse.Namespace, config: dict, name: str, default: str = "") -> str:
    value = getattr(args, name)
    return (
        str(value)
        if value not in (None, "")
        else config_str(config, "exposure_input", name, default)
    )


def _arg_int(args: argparse.Namespace, config: dict, name: str, default: int) -> int:
    value = getattr(args, name)
    return (
        int(value)
        if value is not None
        else config_int(config, "exposure_input", name, default)
    )


def _arg_bool(args_value: bool, config: dict, name: str, default: bool = False) -> bool:
    return bool(args_value) or config_bool(config, "exposure_input", name, default)


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


def _prediction_files(paths: Iterable[str]) -> list[Path]:
    return [file for raw in paths for file in prediction_files(Path(raw))]


def _read_prediction_keys(files: list[Path]) -> pd.DataFrame:
    frames = []
    for file in files:
        frame = read_frame(file, columns=list(KEY_COLUMNS))
        print(f"  {file}: key_rows={len(frame)}")
        frames.append(frame)
    if not frames:
        raise SystemExit("no prediction files supplied")
    return normalize_audit_frame(pd.concat(frames, ignore_index=True))


def _filter_pool(
    keys: pd.DataFrame,
    *,
    pool: str,
    pool_date_lag_sessions: int,
) -> pd.DataFrame:
    if pool == "universe":
        return keys
    pool_path = DEFAULT_STOCK_POOL_PATHS[pool]
    print(f"loading_stock_pool: pool={pool} path={pool_path}")
    stock_pool = load_stock_pool(pool_path)
    mask = stock_pool_membership_mask(
        keys,
        stock_pool,
        date_lag_sessions=pool_date_lag_sessions,
    )
    return keys.loc[mask].copy()


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    size = max(int(size), 1)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _finite_float(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _query_daily_bar(
    client,
    *,
    table: str,
    dates: list[str],
    symbols: list[str],
) -> pd.DataFrame:
    table = validate_table_name(table)
    return client.query_df(
        f"""select
    toString(TradingDay) as date,
    Symbol as symbol,
    TotalMarketValue as market_cap,
    TotalFloatMarketValue as float_market_cap,
    AMT as daily_amount,
    Turnover as daily_turnover_rate,
    FreeTurnover as free_turnover_rate,
    ClosePrice as close_price
from {table}
where TradingDay in {{dates:Array(Date)}}
  and Symbol in {{symbols:Array(String)}}""",
        parameters={"dates": dates, "symbols": symbols},
    )


def _query_industry(
    client,
    *,
    table: str,
    dates: list[str],
    symbols: list[str],
) -> pd.DataFrame:
    table = validate_table_name(table)
    return client.query_df(
        f"""select
    toString(TradingDay) as date,
    Symbol as symbol,
    SWIndustry1 as industry_sw1,
    SWIndustry2 as industry_sw2,
    SWIndustry3 as industry_sw3,
    SWIndustryCode1 as industry_code_sw1,
    SWIndustryCode2 as industry_code_sw2,
    SWIndustryCode3 as industry_code_sw3
from {table}
where toString(TradingDay) in {{dates:Array(String)}}
  and Symbol in {{symbols:Array(String)}}""",
        parameters={"dates": dates, "symbols": symbols},
    )


def _normalize_daily_exposures(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str)
    for column in (
        "market_cap",
        "float_market_cap",
        "daily_amount",
        "daily_turnover_rate",
        "free_turnover_rate",
        "close_price",
    ):
        if column in out.columns:
            out[column] = _finite_float(out[column])
    if "market_cap" in out.columns:
        out["log_market_cap"] = np.log1p(out["market_cap"].where(out["market_cap"].gt(0)))
    if "float_market_cap" in out.columns:
        out["log_float_market_cap"] = np.log1p(
            out["float_market_cap"].where(out["float_market_cap"].gt(0))
        )
    if "industry_sw1" in out.columns:
        industry = out["industry_sw1"].replace("", pd.NA)
        out["industry"] = industry.combine_first(out.get("industry_code_sw1", industry))
    return out.dropna(subset=["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")


def build_exposure_input(
    *,
    keys: pd.DataFrame,
    client,
    daily_bar_table: str,
    industry_table: str,
    date_chunk_size: int,
) -> pd.DataFrame:
    daily_keys = keys[["date", "symbol"]].drop_duplicates().copy()
    dates = sorted(daily_keys["date"].dropna().astype(str).unique())
    symbols = sorted(daily_keys["symbol"].dropna().astype(str).unique())
    if not dates or not symbols:
        raise SystemExit("no date/symbol keys available after pool filtering")

    frames: list[pd.DataFrame] = []
    for date_chunk in _chunks(dates, date_chunk_size):
        print(
            "query_exposures: "
            f"dates={date_chunk[0]}..{date_chunk[-1]} ({len(date_chunk)}) "
            f"symbols={len(symbols)}"
        )
        daily = _query_daily_bar(
            client,
            table=daily_bar_table,
            dates=date_chunk,
            symbols=symbols,
        )
        industry = _query_industry(
            client,
            table=industry_table,
            dates=date_chunk,
            symbols=symbols,
        )
        if daily.empty:
            daily = pd.DataFrame(columns=["date", "symbol"])
        if industry.empty:
            industry = pd.DataFrame(columns=["date", "symbol"])
        chunk = daily.merge(industry, on=["date", "symbol"], how="outer")
        frames.append(chunk)
    exposures = _normalize_daily_exposures(
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    return daily_keys.merge(exposures, on=["date", "symbol"], how="left", validate="one_to_one")


def _clickhouse_setting(arg_value: object, env_name: str, default: object) -> object:
    if arg_value not in (None, ""):
        return arg_value
    return os.environ.get(env_name, default)


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if config or args.config else "exposure_input"
    if args.env_file:
        load_env_file(args.env_file)
    prediction_paths = _arg_list(args, config, "predictions", ())
    if not prediction_paths:
        raise SystemExit("pass --predictions or set [exposure_input].predictions")
    pool = _arg_str(args, config, "pool", "universe") or "universe"
    pool_date_lag_sessions = _arg_int(args, config, "pool_date_lag_sessions", 0)
    date_chunk_size = _arg_int(args, config, "date_chunk_size", 60)
    daily_bar_table = _arg_str(args, config, "daily_bar_table", DEFAULT_DAILY_BAR_TABLE)
    industry_table = _arg_str(args, config, "industry_table", DEFAULT_INDUSTRY_TABLE)
    include_decision_timestamp = _arg_bool(
        args.include_decision_timestamp,
        config,
        "include_decision_timestamp",
        False,
    )

    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/exposures/{run_name}")
    )
    config_output = config_str(config, "exposure_input", "output", "")
    if args.output_dir:
        output = Path(args.output_dir) / "exposure_input.parquet"
    elif args.output:
        output = Path(args.output)
    elif config_output:
        output = Path(config_output)
    else:
        output = output_dir / "exposure_input.parquet"
    config_trace_output = config_str(config, "exposure_input", "trace_output", "")
    if args.output_dir:
        trace_path = Path(args.output_dir) / "exposure_input_trace.json"
    elif args.trace_output:
        trace_path = Path(args.trace_output)
    elif config_trace_output:
        trace_path = Path(config_trace_output)
    elif config_output or args.output:
        trace_path = output.with_suffix(".trace.json")
    else:
        trace_path = output_dir / "exposure_input_trace.json"

    prediction_files_list = _prediction_files(prediction_paths)
    print(f"reading_prediction_keys: files={len(prediction_files_list)}")
    keys = _read_prediction_keys(prediction_files_list)
    keys = _filter_pool(
        keys,
        pool=pool,
        pool_date_lag_sessions=pool_date_lag_sessions,
    )
    full_keys = keys.drop_duplicates(list(KEY_COLUMNS)).copy()
    print(
        "prediction_key_scope: "
        f"rows={len(keys)} unique_full_keys={len(full_keys)} "
        f"daily_keys={len(full_keys[['date', 'symbol']].drop_duplicates())}"
    )

    host = str(
        _clickhouse_setting(args.clickhouse_host, "CLICKHOUSE_HOST", DEFAULT_CLICKHOUSE_TICK_HOST)
    )
    port = int(
        _clickhouse_setting(args.clickhouse_port, "CLICKHOUSE_PORT", DEFAULT_CLICKHOUSE_TICK_PORT)
    )
    username = str(_clickhouse_setting(args.clickhouse_user, "CLICKHOUSE_USER", "") or "")
    password = str(_clickhouse_setting(args.clickhouse_password, "CLICKHOUSE_PASSWORD", "") or "")
    if not username or not password:
        raise SystemExit("ClickHouse credentials are missing; set CLICKHOUSE_USER/PASSWORD")
    client = get_tick_client(host=host, port=port, username=username, password=password)

    exposures = build_exposure_input(
        keys=full_keys,
        client=client,
        daily_bar_table=daily_bar_table,
        industry_table=industry_table,
        date_chunk_size=date_chunk_size,
    )
    if include_decision_timestamp:
        exposures = full_keys.merge(exposures, on=["date", "symbol"], how="left")

    write_frame(exposures, output)
    missing = {
        column: int(exposures[column].isna().sum())
        for column in exposures.columns
        if column not in {*KEY_COLUMNS, "date", "symbol"}
    }
    trace = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prediction_files": [str(path) for path in prediction_files_list],
        "pool": pool,
        "pool_date_lag_sessions": pool_date_lag_sessions,
        "daily_bar_table": daily_bar_table,
        "industry_table": industry_table,
        "include_decision_timestamp": bool(include_decision_timestamp),
        "rows": int(len(exposures)),
        "date_min": str(exposures["date"].min()) if not exposures.empty else "",
        "date_max": str(exposures["date"].max()) if not exposures.empty else "",
        "symbols": int(exposures["symbol"].nunique()) if not exposures.empty else 0,
        "missing_values": missing,
    }
    write_json(trace_path, trace, ensure_ascii=True)
    print(f"wrote exposure input: {output} rows={len(exposures)}")
    print(f"wrote trace: {trace_path}")


if __name__ == "__main__":
    main()
