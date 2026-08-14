from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import get_tick_client, validate_table_name
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "symbol", "decision_target_timestamp"]
GROUPS = ["date", "decision_target_timestamp"]
TOP_N = 100
ENTRY_DELAY_SECONDS = 6
TRADABLE_STATUSES = {"T0", "20", "TRADE"}
SYMBOL_PATTERN = re.compile(r"^(?:(?:00|30)\d{4}\.SZ|(?:60|68)\d{4}\.SH)$")
DATE_PATTERN = re.compile(r"^2025-\d{2}-\d{2}$")
TICK_COLUMNS = [
    "Symbol",
    "ExchTimeOffsetUs",
    "Status",
    "AskPrice1",
    "AskVolume1",
    "BidPrice1",
    "BidVolume1",
    "LastPrice",
    "HighPrice",
]
STATE_COLUMNS = TICK_COLUMNS[2:]


def normalize_text_series(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        )
    )


def normalize_date_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        parsed = pd.to_datetime(values, unit="D", origin="unix", errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List every 2025 Pool-L final-limit hit selected by the 09:31-09:40 "
            "ds350 1m model and verify its point-in-time state in PVC and ClickHouse."
        )
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--raw-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--entry-delay-seconds", type=int, default=ENTRY_DELAY_SECONDS)
    parser.add_argument("--clickhouse-table", default="stock.tick")
    parser.add_argument("--daily-table", default="stock.daily_bar_jy")
    parser.add_argument("--clickhouse-pair-chunk-size", type=int, default=250)
    parser.add_argument("--pvc-workers", type=int, default=8)
    return parser.parse_args()


def _text(values: pd.Series) -> pd.Series:
    return normalize_text_series(values).astype("string")


def _number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _normalize_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = normalize_date_series(out["date"])
    out["symbol"] = _text(out["symbol"])
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"], errors="coerce"
    )
    out["prediction"] = _number(out["prediction"])
    if "label" in out:
        out["label"] = _number(out["label"])
    return out.dropna(subset=[*KEYS, "prediction"])


def load_top100(
    model_root: Path,
    *,
    pool: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_paths = sorted(model_root.glob("month_2025-*/predictions.parquet"))
    expected_folds = {"month_2025-01", "month_2025-07"}
    actual_folds = {path.parent.name for path in prediction_paths}
    if actual_folds != expected_folds:
        raise SystemExit(
            f"expected 2025 prediction folds {sorted(expected_folds)}, got {sorted(actual_folds)}"
        )

    parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for path in prediction_paths:
        columns = [*KEYS, "prediction"]
        # Historical predictions normally retain the realized label. It is diagnostic only:
        # selection below intentionally does not require it to be finite.
        try:
            frame = pd.read_parquet(path, columns=[*columns, "label"])
        except (KeyError, ValueError):
            frame = pd.read_parquet(path, columns=columns)
        frame = _normalize_prediction(frame)
        frame = frame.loc[frame["date"].str.startswith("2025-")].copy()
        input_rows = len(frame)
        frame = frame.loc[stock_pool_membership_mask(frame, pool, date_lag_sessions=0)].copy()
        if frame.duplicated(KEYS).any():
            raise AssertionError(f"duplicate prediction keys in {path}")
        fold_rows.append(
            {
                "fold": path.parent.name,
                "path": str(path),
                "all_a_rows": int(input_rows),
                "pool_l_rows": int(len(frame)),
            }
        )
        parts.append(frame)

    predictions = pd.concat(parts, ignore_index=True)
    if predictions.duplicated(KEYS).any():
        raise AssertionError("2025 prediction folds overlap")
    predictions = predictions.sort_values(
        [*GROUPS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    predictions["candidate_rows"] = predictions.groupby(GROUPS, sort=False)["prediction"].transform(
        "size"
    )
    predictions["score_rank"] = predictions.groupby(GROUPS, sort=False).cumcount() + 1
    predictions["score_percentile"] = (
        1.0 - (predictions["score_rank"] - 1) / predictions["candidate_rows"]
    )
    selected = predictions.loc[predictions["score_rank"].le(int(top_n))].copy()
    group_size = selected.groupby(GROUPS, sort=False).size()
    if group_size.empty or int(group_size.min()) != int(top_n):
        raise AssertionError("at least one date-clock group has fewer than Top100 rows")
    return selected, {
        "prediction_folds": fold_rows,
        "candidate_rows": int(len(predictions)),
        "date_clock_groups": int(predictions.groupby(GROUPS, sort=False).ngroups),
        "selected_rows": int(len(selected)),
    }


def load_pvc_daily(raw_source_root: Path) -> pd.DataFrame:
    daily_path = raw_source_root / "year=2025/daily_reference.parquet"
    close_path = raw_source_root / "year=2025/close_reference.parquet"
    daily = pd.read_parquet(
        daily_path,
        columns=[
            "TradingDay",
            "Symbol",
            "ClosePrice",
            "PreClosePrice",
            "TradeStatus",
            "STStatus",
            "UpdownLimitStatus",
        ],
    ).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "ClosePrice": "pvc_daily_close",
            "PreClosePrice": "pvc_prev_close",
            "TradeStatus": "pvc_daily_trade_status",
            "STStatus": "pvc_st_status",
            "UpdownLimitStatus": "pvc_updown_limit_status",
        }
    )
    daily["date"] = normalize_date_series(daily["date"])
    daily["symbol"] = _text(daily["symbol"])
    for column in (
        "pvc_daily_close",
        "pvc_prev_close",
        "pvc_st_status",
        "pvc_updown_limit_status",
    ):
        daily[column] = _number(daily[column])
    daily["pvc_daily_trade_status"] = _text(daily["pvc_daily_trade_status"]).str.upper()
    daily = daily.drop_duplicates(["date", "symbol"], keep="last")

    close = pd.read_parquet(
        close_path,
        columns=["TradingDay", "Symbol", "ClosePrice", "CloseSourceOffsetUs"],
    ).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "ClosePrice": "pvc_tick_close",
            "CloseSourceOffsetUs": "pvc_close_source_offset_us",
        }
    )
    close["date"] = normalize_date_series(close["date"])
    close["symbol"] = _text(close["symbol"])
    close["pvc_tick_close"] = _number(close["pvc_tick_close"])
    close["pvc_close_source_offset_us"] = _number(close["pvc_close_source_offset_us"])
    close = close.drop_duplicates(["date", "symbol"], keep="last")
    return daily.merge(close, on=["date", "symbol"], how="left", validate="one_to_one")


def add_limit_checks(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    symbol = out["symbol"].astype(str)
    st = _number(out[f"{prefix}_st_status"]).fillna(0).ne(0)
    wide = symbol.str.startswith("30") | symbol.str.startswith("68")
    ratio = np.where(st, 0.05, np.where(wide, 0.20, 0.10))
    prev_close = _number(out[f"{prefix}_prev_close"])
    upper = np.floor(prev_close.to_numpy() * (1.0 + ratio) * 100.0 + 0.50000001) / 100.0
    out[f"{prefix}_rule_upper_limit"] = pd.Series(upper, index=out.index)
    out[f"{prefix}_final_limit"] = _number(out[f"{prefix}_updown_limit_status"]).eq(1)
    out[f"{prefix}_close_equals_rule_limit"] = (
        _number(out[f"{prefix}_daily_close"])
        .sub(out[f"{prefix}_rule_upper_limit"])
        .abs()
        .le(0.0051)
    )
    return out


def select_unique_limit_hits(
    selected: pd.DataFrame, pvc_daily: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = selected.merge(pvc_daily, on=["date", "symbol"], how="left", validate="many_to_one")
    frame = add_limit_checks(frame, prefix="pvc")
    event = frame.loc[frame["pvc_final_limit"]].copy()
    event_counts = event.groupby(["date", "symbol"], sort=False).size().rename("hit_clock_count")
    ordered = event.sort_values(
        ["date", "symbol", "prediction", "decision_target_timestamp"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    strongest = ordered.groupby(["date", "symbol"], sort=False).head(1).copy()
    strongest = strongest.merge(
        event_counts.reset_index(), on=["date", "symbol"], how="left", validate="one_to_one"
    )
    strongest["buy_target_timestamp"] = strongest["decision_target_timestamp"] + pd.to_timedelta(
        ENTRY_DELAY_SECONDS, unit="s"
    )
    return strongest, {
        "limit_hit_rows_before_daily_dedup": int(len(event)),
        "unique_limit_hit_symbol_days": int(len(strongest)),
        "multi_clock_limit_hit_symbol_days": int(strongest["hit_clock_count"].gt(1).sum()),
    }


def _offset_us(timestamp: pd.Series) -> np.ndarray:
    return ((timestamp - timestamp.dt.normalize()) / pd.Timedelta(microseconds=1)).to_numpy(
        dtype="int64"
    )


def align_tick_states(
    keys: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    target_column: str,
    prefix: str,
) -> pd.DataFrame:
    output_columns = [
        "date",
        "symbol",
        f"{prefix}_target_timestamp",
        f"{prefix}_source_offset_us",
        f"{prefix}_source_timestamp",
        f"{prefix}_state_age_seconds",
        *(f"{prefix}_{column}" for column in STATE_COLUMNS),
    ]
    rows: list[pd.DataFrame] = []
    ticks = ticks.copy()
    ticks["date"] = normalize_date_series(ticks["date"])
    ticks["symbol"] = _text(ticks["symbol"])
    ticks["ExchTimeOffsetUs"] = _number(ticks["ExchTimeOffsetUs"])
    for column in STATE_COLUMNS:
        if column != "Status":
            ticks[column] = _number(ticks[column])
    ticks["Status"] = _text(ticks["Status"]).str.upper()
    tick_groups = {
        (str(date), str(symbol)): part.sort_values("ExchTimeOffsetUs", kind="mergesort")
        for (date, symbol), part in ticks.groupby(["date", "symbol"], sort=False)
    }

    for (date, symbol), wanted in keys.groupby(["date", "symbol"], sort=False):
        state = tick_groups.get((str(date), str(symbol)))
        if state is None or state.empty:
            continue
        offsets = state["ExchTimeOffsetUs"].to_numpy(dtype="int64")
        targets = _offset_us(wanted[target_column])
        positions = np.searchsorted(offsets, targets, side="right") - 1
        valid = positions >= 0
        if not valid.any():
            continue
        matched = state.iloc[positions[valid]]
        part = wanted.loc[valid, ["date", "symbol", target_column]].copy()
        part = part.rename(columns={target_column: f"{prefix}_target_timestamp"})
        part[f"{prefix}_source_offset_us"] = matched["ExchTimeOffsetUs"].to_numpy()
        part[f"{prefix}_source_timestamp"] = pd.to_datetime(part["date"]) + pd.to_timedelta(
            part[f"{prefix}_source_offset_us"], unit="us"
        )
        part[f"{prefix}_state_age_seconds"] = (
            part[f"{prefix}_target_timestamp"] - part[f"{prefix}_source_timestamp"]
        ) / pd.Timedelta(seconds=1)
        for column in STATE_COLUMNS:
            part[f"{prefix}_{column}"] = matched[column].to_numpy()
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.concat(rows, ignore_index=True)[output_columns]


def load_pvc_states(
    keys: pd.DataFrame,
    *,
    raw_source_root: Path,
    workers: int,
) -> pd.DataFrame:
    items = [(date, part.copy()) for date, part in keys.groupby("date", sort=True)]

    def read_day(date: str, wanted: pd.DataFrame) -> pd.DataFrame:
        path = raw_source_root / f"year=2025/ticks/date={date}.parquet"
        if not path.exists():
            return pd.DataFrame()
        symbols = sorted(set(wanted["symbol"].astype(str)))
        ticks = pd.read_parquet(path, columns=TICK_COLUMNS, filters=[("Symbol", "in", symbols)])
        ticks = ticks.rename(columns={"Symbol": "symbol"})
        ticks["date"] = date
        decision = align_tick_states(
            wanted,
            ticks,
            target_column="decision_target_timestamp",
            prefix="pvc_decision",
        )
        buy = align_tick_states(
            wanted,
            ticks,
            target_column="buy_target_timestamp",
            prefix="pvc_buy",
        )
        return decision.merge(buy, on=["date", "symbol"], how="outer", validate="one_to_one")

    rows: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(read_day, date, part): date for date, part in items}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if not result.empty:
                rows.append(result)
            if index % 25 == 0 or index == len(items):
                print(f"PVC_PROGRESS {index}/{len(items)}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _validate_pairs(keys: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = sorted(set(zip(keys["date"].astype(str), keys["symbol"].astype(str), strict=True)))
    for date, symbol in pairs:
        if DATE_PATTERN.fullmatch(date) is None or SYMBOL_PATTERN.fullmatch(symbol) is None:
            raise ValueError(f"unsafe ClickHouse key: {(date, symbol)!r}")
    return pairs


def _pair_sql(pairs: list[tuple[str, str]]) -> str:
    return ",".join(f"(toDate('{date}'),'{symbol}')" for date, symbol in pairs)


def query_clickhouse_daily(
    client: object,
    keys: pd.DataFrame,
    *,
    table: str,
    chunk_size: int,
) -> pd.DataFrame:
    pairs = _validate_pairs(keys)
    rows: list[pd.DataFrame] = []
    for start in range(0, len(pairs), int(chunk_size)):
        chunk = pairs[start : start + int(chunk_size)]
        query = f"""
select
    toString(TradingDay) as date,
    Symbol as symbol,
    ClosePrice as ch_daily_close,
    PreClosePrice as ch_prev_close,
    TradeStatus as ch_daily_trade_status,
    STStatus as ch_st_status,
    UpdownLimitStatus as ch_updown_limit_status
from {table}
where (TradingDay, Symbol) in ({_pair_sql(chunk)})
"""
        rows.append(client.query_df(query))
        print(
            f"CH_DAILY_PROGRESS {min(start + len(chunk), len(pairs))}/{len(pairs)}",
            flush=True,
        )
    daily = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    daily["date"] = normalize_date_series(daily["date"])
    daily["symbol"] = _text(daily["symbol"])
    for column in (
        "ch_daily_close",
        "ch_prev_close",
        "ch_st_status",
        "ch_updown_limit_status",
    ):
        daily[column] = _number(daily[column])
    daily["ch_daily_trade_status"] = _text(daily["ch_daily_trade_status"]).str.upper()
    return daily.drop_duplicates(["date", "symbol"], keep="last")


def query_clickhouse_states(
    client: object,
    keys: pd.DataFrame,
    *,
    table: str,
    chunk_size: int,
) -> pd.DataFrame:
    pairs = _validate_pairs(keys)
    rows: list[pd.DataFrame] = []
    for start in range(0, len(pairs), int(chunk_size)):
        chunk = pairs[start : start + int(chunk_size)]
        query = f"""
select
    toString(TradingDay) as date,
    Symbol as symbol,
    ExchTimeOffsetUs,
    Status,
    AskPrice1,
    AskVolume1,
    BidPrice1,
    BidVolume1,
    LastPrice,
    HighPrice
from (
    select
        TradingDay,
        Symbol,
        ExchTimeOffsetUs,
        Status,
        AskPrice1,
        AskVolume1,
        BidPrice1,
        BidVolume1,
        LastPrice,
        HighPrice,
        LocalTimeStamp,
        TradeNum,
        Volume,
        Turnover
    from {table}
    where (TradingDay, Symbol) in ({_pair_sql(chunk)})
      and ExchTimeOffsetUs >= 33300000000
      and ExchTimeOffsetUs <= 34860000000
    order by
        TradingDay,
        Symbol,
        ExchTimeOffsetUs,
        arrayMax(mapValues(LocalTimeStamp)) desc,
        TradeNum desc,
        Volume desc,
        Turnover desc,
        LastPrice desc,
        AskPrice1 desc,
        BidPrice1 desc
    limit 1 by TradingDay, Symbol, ExchTimeOffsetUs
)
order by TradingDay, Symbol, ExchTimeOffsetUs
"""
        ticks = client.query_df(query)
        chunk_index = pd.MultiIndex.from_tuples(chunk, names=["date", "symbol"])
        key_index = pd.MultiIndex.from_arrays(
            [keys["date"].astype(str), keys["symbol"].astype(str)],
            names=["date", "symbol"],
        )
        chunk_keys = keys.loc[key_index.isin(chunk_index)].copy()
        decision = align_tick_states(
            chunk_keys,
            ticks,
            target_column="decision_target_timestamp",
            prefix="ch_decision",
        )
        buy = align_tick_states(
            chunk_keys,
            ticks,
            target_column="buy_target_timestamp",
            prefix="ch_buy",
        )
        rows.append(decision.merge(buy, on=["date", "symbol"], how="outer", validate="one_to_one"))
        del ticks
        print(
            f"CH_TICK_PROGRESS {min(start + len(chunk), len(pairs))}/{len(pairs)}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_tradeability(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    status = _text(out[f"{prefix}_Status"]).str.upper()
    ask = _number(out[f"{prefix}_AskPrice1"])
    volume = _number(out[f"{prefix}_AskVolume1"])
    out[f"{prefix}_tradable"] = status.isin(TRADABLE_STATUSES) & ask.gt(0) & volume.gt(0)
    return out


def close_enough(left: pd.Series, right: pd.Series, tolerance: float = 1e-8) -> pd.Series:
    a = _number(left)
    b = _number(right)
    return a.notna() & b.notna() & a.sub(b).abs().le(float(tolerance))


def add_cross_source_checks(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_limit_checks(frame, prefix="ch")
    for source in ("pvc", "ch"):
        out = add_tradeability(out, prefix=f"{source}_decision")
        out = add_tradeability(out, prefix=f"{source}_buy")

    for point in ("decision", "buy"):
        pvc = f"pvc_{point}"
        ch = f"ch_{point}"
        checks = [
            close_enough(out[f"{pvc}_source_offset_us"], out[f"{ch}_source_offset_us"], 0.1),
            _text(out[f"{pvc}_Status"]).str.upper().eq(_text(out[f"{ch}_Status"]).str.upper()),
        ]
        for column in (
            "AskPrice1",
            "AskVolume1",
            "BidPrice1",
            "BidVolume1",
            "LastPrice",
            "HighPrice",
        ):
            checks.append(close_enough(out[f"{pvc}_{column}"], out[f"{ch}_{column}"]))
        out[f"{point}_pvc_ch_match"] = pd.concat(checks, axis=1).all(axis=1)

    out["pvc_tick_close_matches_daily"] = close_enough(
        out["pvc_tick_close"], out["pvc_daily_close"], 0.0051
    )
    out["daily_pvc_ch_match"] = (
        close_enough(out["pvc_daily_close"], out["ch_daily_close"], 0.0051)
        & close_enough(out["pvc_prev_close"], out["ch_prev_close"], 0.0051)
        & _number(out["pvc_updown_limit_status"]).eq(_number(out["ch_updown_limit_status"]))
    )
    out["official_final_limit_verified"] = (
        out["pvc_final_limit"] & out["ch_final_limit"] & out["daily_pvc_ch_match"]
    )
    out["independent_rule_limit_verified"] = (
        out["pvc_close_equals_rule_limit"] & out["ch_close_equals_rule_limit"]
    )
    out["score_high_verified"] = (
        _number(out["prediction"]).notna()
        & _number(out["score_rank"]).between(1, TOP_N)
        & _number(out["candidate_rows"]).ge(TOP_N)
    )
    out["pvc_final_limit_verified"] = (
        out["pvc_final_limit"]
        & out["pvc_close_equals_rule_limit"]
        & out["pvc_tick_close_matches_daily"]
    )
    out["ch_final_limit_verified"] = out["ch_final_limit"] & out["ch_close_equals_rule_limit"]
    core_required = [
        "score_high_verified",
        "pvc_decision_tradable",
        "pvc_buy_tradable",
        "ch_decision_tradable",
        "ch_buy_tradable",
        "decision_pvc_ch_match",
        "buy_pvc_ch_match",
        "daily_pvc_ch_match",
        "official_final_limit_verified",
    ]
    out["core_checks_pass"] = out[core_required].fillna(False).all(axis=1)
    out["strict_auxiliary_checks_pass"] = (
        out["core_checks_pass"]
        & out["pvc_tick_close_matches_daily"]
        & out["independent_rule_limit_verified"]
    )
    out["all_checks_pass"] = out["core_checks_pass"]
    return out


def main() -> None:
    args = parse_args()
    if args.top_n != TOP_N:
        raise SystemExit(f"this mentor audit is fixed to Top{TOP_N}")
    if args.entry_delay_seconds != ENTRY_DELAY_SECONDS:
        raise SystemExit(f"this mentor audit is fixed to entry delay {ENTRY_DELAY_SECONDS}s")
    args.clickhouse_table = validate_table_name(args.clickhouse_table)
    args.daily_table = validate_table_name(args.daily_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    selected, selection_summary = load_top100(
        args.model_root,
        pool=pool,
        top_n=args.top_n,
    )
    pvc_daily = load_pvc_daily(args.raw_source_root)
    hits, hit_summary = select_unique_limit_hits(selected, pvc_daily)
    if hits.empty:
        raise AssertionError("the 2025 1m model produced no final-limit Top100 hits")
    print(
        f"HITS rows={hit_summary['limit_hit_rows_before_daily_dedup']} "
        f"unique={hit_summary['unique_limit_hit_symbol_days']}",
        flush=True,
    )

    pvc_states = load_pvc_states(
        hits[[*KEYS, "buy_target_timestamp"]],
        raw_source_root=args.raw_source_root,
        workers=args.pvc_workers,
    )
    detail = hits.merge(pvc_states, on=["date", "symbol"], how="left", validate="one_to_one")

    client = get_tick_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 8123)),
        username=os.environ["CLICKHOUSE_USER"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
    )
    try:
        ch_daily = query_clickhouse_daily(
            client,
            hits,
            table=args.daily_table,
            chunk_size=args.clickhouse_pair_chunk_size,
        )
        ch_states = query_clickhouse_states(
            client,
            hits,
            table=args.clickhouse_table,
            chunk_size=args.clickhouse_pair_chunk_size,
        )
    finally:
        client.close()
    ch_daily = add_limit_checks(ch_daily, prefix="ch")
    detail = detail.merge(ch_daily, on=["date", "symbol"], how="left", validate="one_to_one").merge(
        ch_states, on=["date", "symbol"], how="left", validate="one_to_one"
    )
    detail = add_cross_source_checks(detail)
    detail = detail.sort_values(
        ["date", "buy_target_timestamp", "symbol"], kind="mergesort"
    ).reset_index(drop=True)

    mentor = pd.DataFrame(
        {
            "symbol": detail["symbol"].astype(str),
            "日期": detail["date"].astype(str),
            "买入时间点": detail["buy_target_timestamp"].dt.strftime("%H:%M:%S"),
            "信号时间点": detail["decision_target_timestamp"].dt.strftime("%H:%M:%S"),
            "模型分数": detail["prediction"],
            "池内排名": detail["score_rank"].astype("Int64"),
            "当日命中次数": detail["hit_clock_count"].astype("Int64"),
            "PVC买入时可交易": detail["pvc_buy_tradable"],
            "ClickHouse买入时可交易": detail["ch_buy_tradable"],
            "PVC与ClickHouse买入盘口一致": detail["buy_pvc_ch_match"],
            "当天收盘涨停": detail["official_final_limit_verified"],
            "核验结论": np.select(
                [
                    detail["strict_auxiliary_checks_pass"],
                    detail["core_checks_pass"],
                ],
                [
                    "全部通过",
                    "核心通过；附加收盘旁证异常，见audit_exceptions",
                ],
                default="存在核心异常，见audit_detail",
            ),
        }
    )
    minimal = mentor[["symbol", "日期", "买入时间点"]].copy()
    exceptions = detail.loc[~detail["strict_auxiliary_checks_pass"]].copy()
    core_exceptions = detail.loc[~detail["core_checks_pass"]].copy()
    audited_flags = [
        "score_high_verified",
        "pvc_decision_tradable",
        "pvc_buy_tradable",
        "ch_decision_tradable",
        "ch_buy_tradable",
        "decision_pvc_ch_match",
        "buy_pvc_ch_match",
        "daily_pvc_ch_match",
        "official_final_limit_verified",
        "pvc_final_limit_verified",
        "ch_final_limit_verified",
        "independent_rule_limit_verified",
        "pvc_tick_close_matches_daily",
    ]
    summary = {
        "status": "ok",
        "scope": (
            "2025 rolling-OOS; ds350 09:31-09:40 1m model; Pool L; causal Top100 "
            "selected on prediction only; final close-limit joined afterward"
        ),
        "deduplication": (
            "one row per date-symbol; retain the largest prediction across clocks; "
            "earliest clock breaks exact score ties"
        ),
        "buy_time": "decision_target_timestamp + 6 seconds",
        "selection": selection_summary,
        "hits": hit_summary,
        "core_verified_rows": int(detail["core_checks_pass"].sum()),
        "strict_auxiliary_verified_rows": int(detail["strict_auxiliary_checks_pass"].sum()),
        "core_exception_rows": int(len(core_exceptions)),
        "auxiliary_warning_rows": int(len(exceptions)),
        "failed_check_counts": {
            column: int((~detail[column].fillna(False)).sum()) for column in audited_flags
        },
        "outputs": {
            "minimal_table": "mentor_symbol_date_buy_time.csv",
            "mentor_table": "mentor_2025_1m_final_limit_hits.csv",
            "audit_detail": "audit_detail.csv",
            "auxiliary_warnings": "audit_exceptions.csv",
            "core_exceptions": "audit_core_exceptions.csv",
        },
    }

    minimal.to_csv(args.output_dir / "mentor_symbol_date_buy_time.csv", index=False)
    mentor.to_csv(args.output_dir / "mentor_2025_1m_final_limit_hits.csv", index=False)
    detail.to_csv(args.output_dir / "audit_detail.csv", index=False)
    exceptions.to_csv(args.output_dir / "audit_exceptions.csv", index=False)
    core_exceptions.to_csv(args.output_dir / "audit_core_exceptions.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print("AUDIT_SUMMARY_BEGIN", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("AUDIT_SUMMARY_END", flush=True)
    print(f"AUDIT_READY {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
