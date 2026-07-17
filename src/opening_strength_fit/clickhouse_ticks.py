from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns
from opening_strength_fit.universe import DEFAULT_A_SHARE_SYMBOL_REGEX

DEFAULT_CLICKHOUSE_TICK_HOST = "ch.db.prod.highfortfunds.com"
DEFAULT_CLICKHOUSE_TICK_PORT = 8123
DEFAULT_CLICKHOUSE_TICK_TABLE = "stock.tick"
DEFAULT_TICK_START_OFFSET_US = 33_300_000_000
DEFAULT_TICK_END_OFFSET_US = 35_100_000_000

TICK_FIELD_DESC = {
    # 基础信息
    "TradingDay": "交易日",
    "Symbol": "证券代码",
    # 时间
    "ExchTimeOffsetUs": "交易所时间偏移(微秒)，表示从00:00:00开始经过的微秒数",
    # 最新成交
    "HighPrice": "当日最高价",
    "LowPrice": "当日最低价",
    "LastPrice": "最新成交价",
    "TradeNum": "累计成交笔数",
    "Volume": "累计成交量(股)",
    "Turnover": "累计成交额(元)",
    # 状态
    "Status": "交易状态码",
    # 全盘口统计
    "AvgAskPrice": "全市场平均委卖价格",
    "TotalAskVolume": "总委卖量",
    "TotalAskCount": "总委卖笔数",
    "AvgBidPrice": "全市场平均委买价格",
    "TotalBidVolume": "总委买量",
    "TotalBidCount": "总委买笔数",
    # ETF相关
    "IOPV": "ETF参考净值(IOPV)，普通股票通常为0",
    # 本地时间
    "LocalTimeStamp": "本地接收时间戳",
}

for _level in range(1, 11):
    TICK_FIELD_DESC.update(
        {
            f"AskPrice{_level}": f"卖{_level}价",
            f"AskVolume{_level}": f"卖{_level}量",
            f"AskCount{_level}": f"卖{_level}委托笔数",
            f"BidPrice{_level}": f"买{_level}价",
            f"BidVolume{_level}": f"买{_level}量",
            f"BidCount{_level}": f"买{_level}委托笔数",
        }
    )
del _level

STATUS_DESC = {
    # ========= SZ =========
    "T0": "连续竞价交易",
    "C0": "收盘集合竞价",
    "S0": "停牌",
    "O0": "开盘集合竞价",
    "B0": "午间休市",
    "E0": "闭市结束",
    "A0": "盘后交易或特殊交易阶段",
    "H0": "临时停牌",
    "V0": "波动性中断/异常波动状态",
    "10": "开盘集合竞价阶段",
    "20": "连续竞价阶段",
    # ========= SH =========
    "START": "开市准备阶段",
    "OCALL": "开盘集合竞价",
    "TRADE": "连续竞价交易",
    "CLOSE": "收盘阶段",
    "ENDTR": "交易结束",
    "CCALL": "收盘集合竞价",
    "SUSP": "停牌(Suspend)",
    "FIXOC": "固定价格开盘集合竞价",
    "FIXTR": "固定价格交易",
    "FIXCL": "固定价格收盘",
    "DERSP": "衍生品特殊阶段",
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TICK_TIMESTAMP_DEDUPLICATION_MODES = {"latest_local_timestamp"}


def _validate_table_name(table: str) -> str:
    parts = table.split(".")
    if not parts or not all(_IDENTIFIER_RE.fullmatch(part) for part in parts):
        raise ValueError(f"invalid ClickHouse table name: {table}")
    return ".".join(parts)


def validate_table_name(table: str) -> str:
    return _validate_table_name(table)


def get_tick_client(
    *,
    username: str,
    password: str,
    host: str = DEFAULT_CLICKHOUSE_TICK_HOST,
    port: int = DEFAULT_CLICKHOUSE_TICK_PORT,
    **kwargs,
):
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        **kwargs,
    )


def tick_window_sql(table: str = DEFAULT_CLICKHOUSE_TICK_TABLE) -> str:
    table = _validate_table_name(table)
    return f"""select *
from {table}
where (
    Symbol = {{symbol:String}}
    and TradingDay = {{trading_day:String}}
    and ExchTimeOffsetUs >= {{start_offset_us:UInt64}}
    and ExchTimeOffsetUs <= {{end_offset_us:UInt64}}
)
order by ExchTimeOffsetUs"""


def tick_day_window_sql(
    table: str = DEFAULT_CLICKHOUSE_TICK_TABLE,
    *,
    use_symbols: bool = False,
    use_symbol_regex: bool = True,
) -> str:
    table = _validate_table_name(table)
    clauses = [
        "TradingDay = {trading_day:String}",
        "ExchTimeOffsetUs >= {start_offset_us:UInt64}",
        "ExchTimeOffsetUs <= {end_offset_us:UInt64}",
    ]
    if use_symbols:
        clauses.append("Symbol in {symbols:Array(String)}")
    if use_symbol_regex:
        clauses.append("match(Symbol, {symbol_regex:String})")
    where = "\n    and ".join(clauses)
    return f"""select *
from {table}
where (
    {where}
)
order by Symbol, ExchTimeOffsetUs"""


def query_tick_window(
    client,
    *,
    symbol: str,
    trading_day: str,
    table: str = DEFAULT_CLICKHOUSE_TICK_TABLE,
    start_offset_us: int = DEFAULT_TICK_START_OFFSET_US,
    end_offset_us: int = DEFAULT_TICK_END_OFFSET_US,
) -> pd.DataFrame:
    return client.query_df(
        tick_window_sql(table),
        parameters={
            "symbol": symbol,
            "trading_day": trading_day,
            "start_offset_us": int(start_offset_us),
            "end_offset_us": int(end_offset_us),
        },
    )


def query_tick_day_window(
    client,
    *,
    trading_day: str,
    table: str = DEFAULT_CLICKHOUSE_TICK_TABLE,
    start_offset_us: int = DEFAULT_TICK_START_OFFSET_US,
    end_offset_us: int = DEFAULT_TICK_END_OFFSET_US,
    symbol_regex: str | None = DEFAULT_A_SHARE_SYMBOL_REGEX,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    parameters = {
        "trading_day": trading_day,
        "start_offset_us": int(start_offset_us),
        "end_offset_us": int(end_offset_us),
    }
    if symbols:
        parameters["symbols"] = symbols
    if symbol_regex:
        parameters["symbol_regex"] = symbol_regex
    return client.query_df(
        tick_day_window_sql(
            table,
            use_symbols=bool(symbols),
            use_symbol_regex=bool(symbol_regex),
        ),
        parameters=parameters,
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def normalize_clickhouse_ticks(df: pd.DataFrame) -> pd.DataFrame:
    ticks = standardize_columns(df)
    ticks = ensure_timestamp_columns(ticks)
    for column in ticks.columns:
        if not pd.api.types.is_object_dtype(ticks[column]):
            continue
        if ticks[column].map(lambda value: isinstance(value, (dict, list))).any():
            ticks[column] = ticks[column].map(
                lambda value: (
                    json.dumps(
                        value,
                        default=_json_default,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if isinstance(value, (dict, list))
                    else value
                )
            )
    return ticks


def _latest_numeric_leaf(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return float("nan")
        try:
            return _latest_numeric_leaf(json.loads(stripped))
        except (json.JSONDecodeError, TypeError):
            try:
                return float(stripped)
            except ValueError:
                parsed = pd.to_datetime(stripped, errors="coerce")
                return float(parsed.value) if not pd.isna(parsed) else float("nan")
    if isinstance(value, Mapping):
        leaves = [_latest_numeric_leaf(item) for item in value.values()]
        finite = [item for item in leaves if np.isfinite(item)]
        return max(finite) if finite else float("nan")
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        leaves = [_latest_numeric_leaf(item) for item in value]
        finite = [item for item in leaves if np.isfinite(item)]
        return max(finite) if finite else float("nan")
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        parsed = pd.to_datetime(value, errors="coerce")
        return float(parsed.value) if not pd.isna(parsed) else float("nan")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def deduplicate_tick_timestamps(
    df: pd.DataFrame,
    *,
    mode: str = "latest_local_timestamp",
    key_columns: Sequence[str] = ("date", "symbol", "timestamp"),
) -> pd.DataFrame:
    """Keep one deterministic snapshot for each exchange timestamp.

    The retained row prefers the latest local receive timestamp, followed by the
    largest cumulative trade state.  Removing duplicates before feature and label
    construction makes ``entry_tick_delay`` count distinct exchange snapshots.
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in TICK_TIMESTAMP_DEDUPLICATION_MODES:
        supported = ", ".join(sorted(TICK_TIMESTAMP_DEDUPLICATION_MODES))
        raise SystemExit(
            f"unsupported data.tick_timestamp_deduplication={mode!r}; expected one of: {supported}"
        )

    work = ensure_timestamp_columns(standardize_columns(df)).copy()
    missing = [column for column in key_columns if column not in work.columns]
    if missing:
        raise SystemExit(f"tick timestamp deduplication missing columns: {missing}")

    input_rows = len(work)
    duplicate_mask = work.duplicated(list(key_columns), keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_keys = int(work.loc[duplicate_mask, list(key_columns)].drop_duplicates().shape[0])
    if duplicate_rows:
        duplicates = work.loc[duplicate_mask].copy()
        duplicates["_dedup_local_receive"] = (
            duplicates["local_timestamp"].map(_latest_numeric_leaf)
            if "local_timestamp" in duplicates.columns
            else np.nan
        )
        rank_columns: list[str] = []
        for column in ("trade_num", "volume", "turnover"):
            rank_column = f"_dedup_{column}"
            duplicates[rank_column] = (
                pd.to_numeric(duplicates[column], errors="coerce")
                if column in duplicates.columns
                else np.nan
            )
            rank_columns.append(rank_column)
        fingerprint_columns = [
            column for column in duplicates.columns if not column.startswith("_dedup_")
        ]
        duplicates["_dedup_fingerprint"] = pd.util.hash_pandas_object(
            duplicates[fingerprint_columns].astype("string"),
            index=False,
        ).astype("uint64")
        retained_duplicates = (
            duplicates.sort_values(
                [
                    *key_columns,
                    "_dedup_local_receive",
                    *rank_columns,
                    "_dedup_fingerprint",
                ],
                kind="mergesort",
                na_position="first",
            )
            .drop_duplicates(list(key_columns), keep="last")
            .drop(
                columns=[
                    "_dedup_local_receive",
                    *rank_columns,
                    "_dedup_fingerprint",
                ]
            )
        )
        work = pd.concat(
            [work.loc[~duplicate_mask], retained_duplicates],
            ignore_index=True,
        )

    out = work.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
    out.attrs["tick_timestamp_deduplication"] = {
        "mode": normalized_mode,
        "input_rows": int(input_rows),
        "duplicate_rows": duplicate_rows,
        "duplicate_keys": duplicate_keys,
        "rows_removed": int(input_rows - len(out)),
        "output_rows": int(len(out)),
    }
    return out


def field_description_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"field": field, "description": desc} for field, desc in TICK_FIELD_DESC.items()]
    )
