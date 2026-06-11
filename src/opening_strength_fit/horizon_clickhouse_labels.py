from __future__ import annotations

import os

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import get_tick_client, validate_table_name
from opening_strength_fit.horizons import HorizonLike, key_columns_for_merge, label_column_name
from opening_strength_fit.labels import safe_price_return

DEFAULT_CLOSE_OFFSET_US = 54_000_000_000
DEFAULT_CLOSE_LOOKBACK_SECONDS = 1_800


def clickhouse_setting(value, env_name: str, default):
    if value not in (None, ""):
        return value
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return env_value
    return default


def query_trading_dates(
    client,
    *,
    table: str,
    start_date: str,
    end_date: str,
) -> list[str]:
    table = validate_table_name(table)
    sql = f"""select distinct TradingDay as date
from {table}
where TradingDay >= {{start_date:String}}
  and TradingDay <= {{end_date:String}}
order by TradingDay"""
    frame = client.query_df(
        sql,
        parameters={"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return []
    return [str(value) for value in frame["date"].dropna().astype(str)]


def target_offset_us(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, errors="coerce")
    hour = ts.dt.hour.astype("int64")
    minute = ts.dt.minute.astype("int64")
    second = ts.dt.second.astype("int64")
    microsecond = ts.dt.microsecond.astype("int64")
    return ((hour * 3_600 + minute * 60 + second) * 1_000_000 + microsecond).astype("Int64")


def query_intraday_mid_prices(
    client,
    *,
    table: str,
    dates: list[str],
    symbols: list[str],
    target_offsets: list[int],
    max_lag_seconds: int,
) -> pd.DataFrame:
    if not dates or not symbols or not target_offsets:
        return pd.DataFrame(columns=["date", "symbol", "target_offset_us", "exit_mid_price"])
    table = validate_table_name(table)
    max_lag_us = int(max_lag_seconds) * 1_000_000
    min_offset = int(min(target_offsets))
    max_offset = int(max(target_offsets) + max_lag_us)
    sql = f"""select
    TradingDay as date,
    Symbol as symbol,
    target_offset_us,
    argMin((AskPrice1 + BidPrice1) / 2.0, ExchTimeOffsetUs) as exit_mid_price,
    min(ExchTimeOffsetUs) as matched_offset_us
from (
    select
        TradingDay,
        Symbol,
        ExchTimeOffsetUs,
        AskPrice1,
        BidPrice1,
        arrayJoin({{target_offsets:Array(UInt64)}}) as target_offset_us
    from {table}
    where TradingDay in {{dates:Array(String)}}
      and Symbol in {{symbols:Array(String)}}
      and ExchTimeOffsetUs >= {{min_offset_us:UInt64}}
      and ExchTimeOffsetUs <= {{max_offset_us:UInt64}}
      and AskPrice1 > 0
      and BidPrice1 > 0
)
where 1
  and ExchTimeOffsetUs >= target_offset_us
  and ExchTimeOffsetUs <= target_offset_us + {{max_lag_us:UInt64}}
group by TradingDay, Symbol, target_offset_us"""
    frame = client.query_df(
        sql,
        parameters={
            "dates": dates,
            "symbols": symbols,
            "target_offsets": [int(value) for value in target_offsets],
            "min_offset_us": min_offset,
            "max_offset_us": max_offset,
            "max_lag_us": max_lag_us,
        },
    )
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "target_offset_us", "exit_mid_price"])
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["target_offset_us"] = pd.to_numeric(
        frame["target_offset_us"],
        errors="coerce",
    ).astype("Int64")
    frame["exit_mid_price"] = pd.to_numeric(
        frame["exit_mid_price"],
        errors="coerce",
    )
    return frame.dropna(subset=["target_offset_us", "exit_mid_price"])


def compute_clickhouse_intraday_labels(
    predictions: pd.DataFrame,
    horizons: list[HorizonLike],
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    decision_max_lag_seconds: int,
    fee_bps: float,
    target_end_seconds: int | None,
) -> pd.DataFrame:
    timed_horizons = [spec for spec in horizons if spec.seconds is not None]
    key_cols = key_columns_for_merge(predictions)
    output = predictions[key_cols].copy()
    if not timed_horizons:
        return output
    if not username or not password:
        raise SystemExit(
            "ClickHouse intraday labels need credentials. Set CLICKHOUSE_USER and "
            "CLICKHOUSE_PASSWORD or pass --clickhouse-user/--clickhouse-password."
        )
    if "decision_target_timestamp" not in predictions.columns:
        raise SystemExit("ClickHouse intraday labels require decision_target_timestamp")
    if "buy_price" not in predictions.columns:
        raise SystemExit("ClickHouse intraday labels require prediction column: buy_price")

    sample = predictions[[*key_cols, "buy_price"]].copy()
    sample["date"] = sample["date"].astype(str)
    sample["symbol"] = sample["symbol"].astype(str)
    sample["decision_target_timestamp"] = pd.to_datetime(
        sample["decision_target_timestamp"],
        errors="coerce",
    )
    sample["buy_price"] = pd.to_numeric(sample["buy_price"], errors="coerce")
    sample["_row"] = np.arange(len(sample), dtype="int64")
    sample = sample.dropna(subset=["decision_target_timestamp", "buy_price"])
    if sample.empty:
        return output.drop_duplicates(key_cols)

    target_frames = []
    for spec in timed_horizons:
        targets = sample[
            ["_row", "date", "symbol", "decision_target_timestamp", "buy_price"]
        ].copy()
        targets["horizon"] = spec.name
        targets["target_timestamp"] = targets["decision_target_timestamp"] + pd.to_timedelta(
            int(spec.seconds),
            unit="s",
        )
        if target_end_seconds is not None:
            target_seconds = (
                targets["target_timestamp"].dt.hour.astype("int64") * 3_600
                + targets["target_timestamp"].dt.minute.astype("int64") * 60
                + targets["target_timestamp"].dt.second.astype("int64")
            )
            targets = targets.loc[target_seconds <= int(target_end_seconds)].copy()
        targets["target_offset_us"] = target_offset_us(targets["target_timestamp"])
        target_frames.append(targets)
    target_frame = pd.concat(target_frames, ignore_index=True)
    target_frame = target_frame.dropna(subset=["target_offset_us"])
    if target_frame.empty:
        return output.drop_duplicates(key_cols)

    dates = sorted(target_frame["date"].dropna().astype(str).unique())
    symbols = sorted(target_frame["symbol"].dropna().astype(str).unique())
    offsets = sorted(int(value) for value in target_frame["target_offset_us"].dropna().unique())
    client = get_tick_client(
        host=host,
        port=int(port),
        username=username,
        password=password,
    )
    price_frame = query_intraday_mid_prices(
        client,
        table=table,
        dates=dates,
        symbols=symbols,
        target_offsets=offsets,
        max_lag_seconds=decision_max_lag_seconds,
    )
    if price_frame.empty:
        return output.drop_duplicates(key_cols)

    merged = target_frame.merge(
        price_frame[["date", "symbol", "target_offset_us", "exit_mid_price"]],
        on=["date", "symbol", "target_offset_us"],
        how="left",
    )
    merged["label"] = safe_price_return(
        merged["exit_mid_price"],
        merged["buy_price"],
        fee_bps=fee_bps,
    )
    labels_wide = merged.pivot_table(
        index="_row",
        columns="horizon",
        values="label",
        aggfunc="first",
    )
    output["_row"] = np.arange(len(output), dtype="int64")
    for spec in timed_horizons:
        if spec.name in labels_wide.columns:
            output[label_column_name(spec.name)] = output["_row"].map(labels_wide[spec.name])
    return output.drop(columns=["_row"]).drop_duplicates(key_cols)


def query_close_prices(
    client,
    *,
    table: str,
    dates: list[str],
    symbols: list[str],
    close_offset_us: int,
    close_lookback_seconds: int,
) -> pd.DataFrame:
    if not dates or not symbols:
        return pd.DataFrame(columns=["date", "symbol", "close_price"])
    table = validate_table_name(table)
    start_offset = max(0, int(close_offset_us) - int(close_lookback_seconds) * 1_000_000)
    sql = f"""select
    TradingDay as date,
    Symbol as symbol,
    argMax(LastPrice, ExchTimeOffsetUs) as close_price
from {table}
where TradingDay in {{dates:Array(String)}}
  and Symbol in {{symbols:Array(String)}}
  and ExchTimeOffsetUs >= {{start_offset_us:UInt64}}
  and ExchTimeOffsetUs <= {{close_offset_us:UInt64}}
  and LastPrice > 0
group by TradingDay, Symbol"""
    frame = client.query_df(
        sql,
        parameters={
            "dates": dates,
            "symbols": symbols,
            "start_offset_us": int(start_offset),
            "close_offset_us": int(close_offset_us),
        },
    )
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "close_price"])
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["close_price"] = pd.to_numeric(frame["close_price"], errors="coerce")
    return frame.dropna(subset=["close_price"])


def compute_clickhouse_close_labels(
    predictions: pd.DataFrame,
    horizons: list[HorizonLike],
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    table: str,
    close_offset_us: int,
    close_lookback_seconds: int,
    calendar_days_after: int,
    fee_bps: float,
) -> pd.DataFrame:
    close_horizons = {spec.name for spec in horizons if spec.name in {"close", "next_close"}}
    key_cols = key_columns_for_merge(predictions)
    output = predictions[key_cols].copy()
    if not close_horizons:
        return output
    if not username or not password:
        raise SystemExit(
            "ClickHouse close labels need credentials. Set CLICKHOUSE_USER and "
            "CLICKHOUSE_PASSWORD or pass --clickhouse-user/--clickhouse-password."
        )
    if "buy_price" not in predictions.columns:
        raise SystemExit("close labels require prediction column: buy_price")

    sample = predictions[["date", "symbol", "buy_price", *key_cols[2:]]].copy()
    sample["date"] = sample["date"].astype(str)
    sample["symbol"] = sample["symbol"].astype(str)
    sample["buy_price"] = pd.to_numeric(sample["buy_price"], errors="coerce")
    unique_dates = sorted(sample["date"].dropna().unique())
    unique_symbols = sorted(sample["symbol"].dropna().unique())
    if not unique_dates or not unique_symbols:
        return output

    start_date = str(pd.Timestamp(unique_dates[0]).date())
    end_date = str(
        (pd.Timestamp(unique_dates[-1]) + pd.Timedelta(days=int(calendar_days_after))).date()
    )
    client = get_tick_client(
        host=host,
        port=int(port),
        username=username,
        password=password,
    )
    trading_dates = query_trading_dates(
        client,
        table=table,
        start_date=start_date,
        end_date=end_date,
    )
    needed_dates = [date for date in trading_dates if date >= start_date]
    close_prices = query_close_prices(
        client,
        table=table,
        dates=needed_dates,
        symbols=unique_symbols,
        close_offset_us=close_offset_us,
        close_lookback_seconds=close_lookback_seconds,
    )
    if close_prices.empty:
        return output
    next_date = {
        date: trading_dates[index + 1]
        for index, date in enumerate(trading_dates[:-1])
        if date in set(unique_dates)
    }
    close_by_key = close_prices.set_index(["date", "symbol"])["close_price"]
    sample_index = pd.MultiIndex.from_frame(sample[["date", "symbol"]])
    buy_price = sample["buy_price"].to_numpy(dtype="float64")

    def close_label(exit_price_values: np.ndarray) -> np.ndarray:
        exit_price = pd.to_numeric(
            pd.Series(exit_price_values),
            errors="coerce",
        ).to_numpy(dtype="float64")
        return safe_price_return(exit_price, buy_price, fee_bps=fee_bps).to_numpy()

    if "close" in close_horizons:
        close_price = close_by_key.reindex(sample_index).to_numpy()
        output[label_column_name("close")] = close_label(close_price)

    if "next_close" in close_horizons:
        next_keys = pd.MultiIndex.from_arrays(
            [
                sample["date"].map(next_date).astype("object"),
                sample["symbol"].astype(str),
            ],
            names=["date", "symbol"],
        )
        next_close_price = close_by_key.reindex(next_keys).to_numpy()
        output[label_column_name("next_close")] = close_label(next_close_price)

    return output.drop_duplicates(key_cols)
