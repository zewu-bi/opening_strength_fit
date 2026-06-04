from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from opening_strength_fit.io import read_frame
from opening_strength_fit.labels import safe_price_return
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns

HORIZON_LABEL_CANDIDATES = (
    "alpha_return_{horizon}",
    "label_{horizon}",
    "gross_label_{horizon}",
    "return_{horizon}",
    "{horizon}_label",
    "{horizon}_return",
)


class HorizonLike(Protocol):
    name: str
    label: str
    seconds: int | None


def available_columns(path: Path) -> list[str] | None:
    if not path.exists():
        raise SystemExit(f"input path does not exist: {path}")
    target = path
    if path.is_dir():
        parquet_files = sorted(path.rglob("*.parquet"))
        if not parquet_files:
            return None
        target = parquet_files[0]
    if target.suffix.lower() != ".parquet":
        return None
    return list(pq.ParquetFile(target).schema_arrow.names)


def read_selected_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return read_frame(path)
    available = available_columns(path)
    selected = [column for column in columns if available is None or column in available]
    if not selected:
        return read_frame(path)
    return read_frame(path, columns=selected)


def normalize_frame_times(frame: pd.DataFrame) -> pd.DataFrame:
    out = standardize_columns(frame)
    if {"date", "symbol", "timestamp"}.issubset(out.columns):
        out = ensure_timestamp_columns(out)
    for column in (
        "timestamp",
        "decision_target_timestamp",
        "entry_timestamp",
    ):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def load_prediction(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"prediction input for {label} does not exist: {path}")
    frame = normalize_frame_times(read_frame(path))
    required = {"date", "symbol", "prediction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"prediction input for {label} missing columns: {missing}")
    frame["branch"] = label
    return frame


def filter_decision_times(frame: pd.DataFrame, clocks: set[str]) -> pd.DataFrame:
    if not clocks:
        return frame
    if "decision_time" in frame.columns:
        values = frame["decision_time"].astype(str)
        return frame.loc[values.isin(clocks)].copy()
    timestamp_col = (
        "decision_target_timestamp" if "decision_target_timestamp" in frame.columns else "timestamp"
    )
    if timestamp_col not in frame.columns:
        raise SystemExit("--decision-time requires decision_time or timestamp columns")
    values = pd.to_datetime(frame[timestamp_col], errors="coerce").dt.strftime("%H:%M:%S")
    return frame.loc[values.isin(clocks)].copy()


def label_column_name(horizon: str) -> str:
    return f"alpha_return_{horizon}"


def explicit_label_map(values: list[tuple[str, str]] | None) -> dict[str, str]:
    return dict(values or [])


def label_candidates(horizon: str) -> list[str]:
    candidates = [template.format(horizon=horizon) for template in HORIZON_LABEL_CANDIDATES]
    if horizon == "60s":
        candidates.extend(["label", "gross_label"])
    return list(dict.fromkeys(candidates))


def find_existing_label_column(
    frame: pd.DataFrame,
    horizon: str,
    explicit: dict[str, str],
) -> str | None:
    if horizon in explicit:
        column = explicit[horizon]
        if column not in frame.columns:
            raise SystemExit(f"explicit horizon label {horizon}={column} is missing from input")
        return column
    for column in label_candidates(horizon):
        if column in frame.columns:
            return column
    return None


def merge_label_input(
    predictions: pd.DataFrame,
    label_input: Path,
    horizons: list[HorizonLike],
    explicit: dict[str, str],
) -> pd.DataFrame:
    available = available_columns(label_input)
    key_cols = key_columns_for_merge(predictions)
    candidate_cols = []
    for spec in horizons:
        candidate_cols.extend(label_candidates(spec.name))
        if spec.name in explicit:
            candidate_cols.append(explicit[spec.name])
    if available is not None:
        read_cols = [column for column in [*key_cols, *candidate_cols] if column in available]
        missing_keys = sorted(set(key_cols) - set(read_cols))
        if missing_keys:
            raise SystemExit(f"label input {label_input} is missing merge keys: {missing_keys}")
    else:
        read_cols = None
    labels = normalize_frame_times(read_selected_frame(label_input, read_cols))
    label_cols = []
    for spec in horizons:
        column = find_existing_label_column(labels, spec.name, explicit)
        if column is not None:
            label_cols.append(column)
    label_cols = list(dict.fromkeys(label_cols))
    if not label_cols:
        return predictions
    labels = labels[[*key_cols, *label_cols]].drop_duplicates(key_cols)
    rename = {}
    for spec in horizons:
        column = find_existing_label_column(labels, spec.name, explicit)
        if column and column != label_column_name(spec.name):
            rename[column] = label_column_name(spec.name)
    labels = labels.rename(columns=rename)
    merged = predictions.merge(labels, on=key_cols, how="left", suffixes=("", "_labelctx"))
    for spec in horizons:
        column = label_column_name(spec.name)
        labelctx_column = f"{column}_labelctx"
        if labelctx_column in merged.columns:
            merged[column] = pd.to_numeric(
                merged[labelctx_column],
                errors="coerce",
            ).combine_first(pd.to_numeric(merged.get(column), errors="coerce"))
            merged = merged.drop(columns=[labelctx_column])
    return merged


def key_columns_for_merge(frame: pd.DataFrame) -> list[str]:
    for time_col in ("decision_target_timestamp", "timestamp"):
        if time_col in frame.columns:
            return ["date", "symbol", time_col]
    raise SystemExit("inputs need decision_target_timestamp or timestamp merge key")


def build_base_samples(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "symbol", "entry_timestamp", "buy_price"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"tick-input horizon labels require prediction columns: {missing}")
    key_cols = key_columns_for_merge(frame)
    cols = list(dict.fromkeys([*key_cols, "entry_timestamp", "buy_price"]))
    samples = frame[cols].drop_duplicates(key_cols).copy()
    samples["entry_timestamp"] = pd.to_datetime(
        samples["entry_timestamp"],
        errors="coerce",
    )
    samples["buy_price"] = pd.to_numeric(samples["buy_price"], errors="coerce")
    return samples


def load_ticks_for_horizons(
    tick_input: Path,
    *,
    volume_col: str,
    turnover_col: str,
    price_col: str,
) -> pd.DataFrame:
    requested = {
        "date",
        "symbol",
        "timestamp",
        "time",
        "exch_time_offset_us",
        volume_col,
        turnover_col,
        "mid_price",
        "last_price",
        "ask_price_1",
        "bid_price_1",
    }
    if price_col != "auto":
        requested.add(price_col)
    available = available_columns(tick_input)
    columns = None if available is None else [column for column in requested if column in available]
    ticks = normalize_frame_times(read_selected_frame(tick_input, columns))
    missing = [column for column in ("date", "symbol", "timestamp") if column not in ticks]
    if missing:
        raise SystemExit(f"tick input missing required columns: {missing}")
    return ticks.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)


def resolve_price_column(ticks: pd.DataFrame, price_col: str) -> tuple[pd.DataFrame, str]:
    if price_col != "auto":
        if price_col not in ticks.columns:
            raise SystemExit(f"price column {price_col!r} not found in tick input")
        return ticks, price_col
    if "mid_price" in ticks.columns:
        return ticks, "mid_price"
    if {"ask_price_1", "bid_price_1"}.issubset(ticks.columns):
        ticks = ticks.copy()
        ask = pd.to_numeric(ticks["ask_price_1"], errors="coerce")
        bid = pd.to_numeric(ticks["bid_price_1"], errors="coerce")
        ticks["mid_price"] = np.where((ask > 0) & (bid > 0), (ask + bid) / 2.0, np.nan)
        return ticks, "mid_price"
    if "last_price" in ticks.columns:
        return ticks, "last_price"
    raise SystemExit("tick input needs mid_price, last_price, or bid/ask level 1 for price exits")


def future_vwap_labels(
    samples: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    horizon: HorizonLike,
    volume_col: str,
    turnover_col: str,
    volume_unit_multiplier: float,
    sell_window_seconds: int,
    fee_bps: float,
    max_gap_seconds: float | None,
) -> pd.Series:
    if horizon.seconds is None:
        raise ValueError("future_vwap_labels requires a timed horizon")
    for column in (volume_col, turnover_col):
        if column not in ticks.columns:
            raise SystemExit(f"tick input missing required column for VWAP: {column}")

    out = pd.Series(np.nan, index=samples.index, dtype="float64")
    tolerance = pd.Timedelta(seconds=max_gap_seconds) if max_gap_seconds else None
    tick_groups = {
        key_tuple(key): group.sort_values("timestamp")
        for key, group in ticks.groupby(["date", "symbol"], sort=False, observed=True)
    }

    for key, group in samples.groupby(["date", "symbol"], sort=False, observed=True):
        right = tick_groups.get(key_tuple(key))
        if right is None:
            continue
        sample_group = group.dropna(subset=["entry_timestamp", "buy_price"])
        if sample_group.empty:
            continue
        right_frame = (
            right[["timestamp", volume_col, turnover_col]]
            .dropna(subset=["timestamp"])
            .rename(columns={"timestamp": "_future_ts"})
            .sort_values("_future_ts")
        )
        if right_frame.empty:
            continue
        left_start = pd.DataFrame(
            {
                "_row": sample_group.index.to_numpy(),
                "_target_ts": sample_group["entry_timestamp"]
                + pd.to_timedelta(horizon.seconds, unit="s"),
            }
        ).sort_values("_target_ts")
        left_end = pd.DataFrame(
            {
                "_row": sample_group.index.to_numpy(),
                "_target_ts": sample_group["entry_timestamp"]
                + pd.to_timedelta(horizon.seconds + sell_window_seconds, unit="s"),
            }
        ).sort_values("_target_ts")
        start = pd.merge_asof(
            left_start,
            right_frame,
            left_on="_target_ts",
            right_on="_future_ts",
            direction="forward",
            tolerance=tolerance,
        ).set_index("_row")
        end = pd.merge_asof(
            left_end,
            right_frame,
            left_on="_target_ts",
            right_on="_future_ts",
            direction="forward",
            tolerance=tolerance,
        ).set_index("_row")
        common = start.index.intersection(end.index)
        if common.empty:
            continue
        start_volume = pd.to_numeric(start.loc[common, volume_col], errors="coerce")
        end_volume = pd.to_numeric(end.loc[common, volume_col], errors="coerce")
        start_turnover = pd.to_numeric(
            start.loc[common, turnover_col],
            errors="coerce",
        )
        end_turnover = pd.to_numeric(end.loc[common, turnover_col], errors="coerce")
        sell_volume = end_volume - start_volume
        sell_turnover = end_turnover - start_turnover
        denominator = sell_volume * float(volume_unit_multiplier)
        sell_vwap = sell_turnover / denominator.replace(0, np.nan)
        buy_price = pd.to_numeric(samples.loc[common, "buy_price"], errors="coerce")
        label = safe_price_return(sell_vwap, buy_price, fee_bps=fee_bps)
        valid = sell_volume.gt(0) & sell_turnover.gt(0) & label.notna()
        out.loc[common[valid.to_numpy()]] = label.loc[valid].to_numpy()

    return out


def key_tuple(key: object) -> tuple[object, ...]:
    return key if isinstance(key, tuple) else (key,)


def next_date_map(ticks: pd.DataFrame) -> dict[str, str]:
    dates = sorted(ticks["date"].astype(str).dropna().unique())
    return {date: dates[index + 1] for index, date in enumerate(dates[:-1])}


def price_exit_labels(
    samples: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    horizon: HorizonLike,
    price_col: str,
    open_time: str,
    close_time: str,
    fee_bps: float,
    max_gap_seconds: float | None,
) -> pd.Series:
    out = pd.Series(np.nan, index=samples.index, dtype="float64")
    tolerance = pd.Timedelta(seconds=max_gap_seconds) if max_gap_seconds else None

    work = samples.dropna(subset=["buy_price"]).copy()
    if horizon.name == "close":
        work["_target_date"] = work["date"].astype(str)
        work["_target_time"] = close_time
        direction = "backward"
    elif horizon.name in {"next_open", "next_close"}:
        next_dates = next_date_map(ticks)
        work["_target_date"] = work["date"].astype(str).map(next_dates)
        work["_target_time"] = open_time if horizon.name == "next_open" else close_time
        direction = "forward" if horizon.name == "next_open" else "backward"
    else:
        raise ValueError(f"unsupported price horizon: {horizon.name}")

    work = work.dropna(subset=["_target_date"])
    if work.empty:
        return out
    work["_target_ts"] = pd.to_datetime(
        work["_target_date"].astype(str) + " " + work["_target_time"].astype(str),
        errors="coerce",
    )
    tick_groups = {
        key_tuple(key): group.sort_values("timestamp")
        for key, group in ticks.groupby(["date", "symbol"], sort=False, observed=True)
    }

    for key, group in work.groupby(["_target_date", "symbol"], sort=False, observed=True):
        right = tick_groups.get(key_tuple(key))
        if right is None:
            continue
        right_frame = (
            right[["timestamp", price_col]]
            .dropna(subset=["timestamp", price_col])
            .rename(columns={"timestamp": "_future_ts"})
            .sort_values("_future_ts")
        )
        if right_frame.empty:
            continue
        left = pd.DataFrame(
            {
                "_row": group.index.to_numpy(),
                "_target_ts": group["_target_ts"].to_numpy(),
            }
        ).sort_values("_target_ts")
        merged = pd.merge_asof(
            left,
            right_frame,
            left_on="_target_ts",
            right_on="_future_ts",
            direction=direction,
            tolerance=tolerance,
        ).set_index("_row")
        if merged.empty:
            continue
        exit_price = pd.to_numeric(merged[price_col], errors="coerce")
        buy_price = pd.to_numeric(samples.loc[merged.index, "buy_price"], errors="coerce")
        label = safe_price_return(exit_price, buy_price, fee_bps=fee_bps)
        valid = label.notna()
        out.loc[merged.index[valid.to_numpy()]] = label.loc[valid].to_numpy()

    return out


def compute_tick_horizon_labels(
    predictions: pd.DataFrame,
    tick_input: Path,
    horizons: list[HorizonLike],
    *,
    volume_col: str,
    turnover_col: str,
    volume_unit_multiplier: float,
    sell_window_seconds: int,
    fee_bps: float,
    price_col: str,
    open_time: str,
    close_time: str,
    max_future_gap_seconds: float | None,
    max_price_gap_seconds: float | None,
) -> pd.DataFrame:
    samples = build_base_samples(predictions)
    ticks = load_ticks_for_horizons(
        tick_input,
        volume_col=volume_col,
        turnover_col=turnover_col,
        price_col=price_col,
    )
    ticks, resolved_price_col = resolve_price_column(ticks, price_col)
    labels = samples[key_columns_for_merge(samples)].copy()
    for spec in horizons:
        column = label_column_name(spec.name)
        if spec.seconds is not None:
            labels[column] = future_vwap_labels(
                samples,
                ticks,
                horizon=spec,
                volume_col=volume_col,
                turnover_col=turnover_col,
                volume_unit_multiplier=volume_unit_multiplier,
                sell_window_seconds=sell_window_seconds,
                fee_bps=fee_bps,
                max_gap_seconds=max_future_gap_seconds,
            )
        else:
            labels[column] = price_exit_labels(
                samples,
                ticks,
                horizon=spec,
                price_col=resolved_price_col,
                open_time=open_time,
                close_time=close_time,
                fee_bps=fee_bps,
                max_gap_seconds=max_price_gap_seconds,
            )
    return labels


def attach_available_prediction_labels(
    predictions: pd.DataFrame,
    horizons: list[HorizonLike],
    explicit: dict[str, str],
) -> pd.DataFrame:
    out = predictions.copy()
    for spec in horizons:
        target = label_column_name(spec.name)
        if target in out.columns:
            continue
        source = find_existing_label_column(out, spec.name, explicit)
        if source is not None:
            out[target] = pd.to_numeric(out[source], errors="coerce")
    return out


def load_sample_context(
    predictions: pd.DataFrame,
    sample_context: str,
    *,
    exit_price_col: str,
) -> pd.DataFrame:
    key_cols = ["date", "symbol", "decision_target_timestamp"]
    required_columns = [*key_cols, exit_price_col]
    columns = [*key_cols, exit_price_col]
    if exit_price_col == "mid_price":
        columns.extend(["ask_price_1", "bid_price_1"])
    columns = list(dict.fromkeys(columns))
    if sample_context:
        path = Path(sample_context)
        available = available_columns(path)
        read_cols = columns if available is None else [col for col in columns if col in available]
        missing = sorted(set(required_columns) - set(read_cols))
        if missing:
            raise SystemExit(f"sample context is missing required columns: {missing}")
        context = normalize_frame_times(read_selected_frame(path, read_cols))
    else:
        missing = [col for col in required_columns if col not in predictions.columns]
        if missing:
            raise SystemExit(
                f"sampled intraday decay needs --sample-context or prediction columns: {missing}"
            )
        context = predictions[[col for col in columns if col in predictions.columns]].copy()
    context["date"] = context["date"].astype(str)
    context["symbol"] = context["symbol"].astype(str)
    context["decision_target_timestamp"] = pd.to_datetime(
        context["decision_target_timestamp"],
        errors="coerce",
    )
    if exit_price_col == "mid_price" and {"ask_price_1", "bid_price_1"}.issubset(context.columns):
        ask = pd.to_numeric(context["ask_price_1"], errors="coerce")
        bid = pd.to_numeric(context["bid_price_1"], errors="coerce")
        context[exit_price_col] = np.where((ask > 0) & (bid > 0), (ask + bid) / 2.0, np.nan)
    context[exit_price_col] = pd.to_numeric(context[exit_price_col], errors="coerce")
    return (
        context.dropna(subset=["decision_target_timestamp", exit_price_col])
        .sort_values(key_cols)
        .drop_duplicates(key_cols)
        .reset_index(drop=True)
    )


def compute_sampled_intraday_labels(
    predictions: pd.DataFrame,
    context: pd.DataFrame,
    horizons: list[HorizonLike],
    *,
    exit_price_col: str,
    fee_bps: float,
    target_end_seconds: int | None,
) -> pd.DataFrame:
    timed_horizons = [spec for spec in horizons if spec.seconds is not None]
    if not timed_horizons:
        return pd.DataFrame(columns=key_columns_for_merge(predictions))
    key_cols = key_columns_for_merge(predictions)
    if "decision_target_timestamp" not in key_cols:
        return pd.DataFrame(columns=key_cols)
    required = [*key_cols, "buy_price"]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise SystemExit(f"sampled intraday labels require prediction columns: {missing}")

    base = predictions[required].copy()
    base["_row"] = np.arange(len(base), dtype="int64")
    base["date"] = base["date"].astype(str)
    base["symbol"] = base["symbol"].astype(str)
    base["decision_target_timestamp"] = pd.to_datetime(
        base["decision_target_timestamp"],
        errors="coerce",
    )
    base["buy_price"] = pd.to_numeric(base["buy_price"], errors="coerce")

    output = predictions[key_cols].copy()
    output["_row"] = np.arange(len(output), dtype="int64")
    right = context.rename(
        columns={
            "decision_target_timestamp": "_target_ts",
            exit_price_col: "_exit_price",
        }
    )[["date", "symbol", "_target_ts", "_exit_price"]]

    for spec in timed_horizons:
        target_col = label_column_name(spec.name)
        left = base[["_row", "date", "symbol", "decision_target_timestamp", "buy_price"]].copy()
        left["_target_ts"] = left["decision_target_timestamp"] + pd.to_timedelta(
            int(spec.seconds),
            unit="s",
        )
        if target_end_seconds is not None:
            target_seconds = (
                left["_target_ts"].dt.hour.astype("int64") * 3_600
                + left["_target_ts"].dt.minute.astype("int64") * 60
                + left["_target_ts"].dt.second.astype("int64")
            )
            left = left.loc[target_seconds <= int(target_end_seconds)].copy()
        merged = left.merge(
            right,
            on=["date", "symbol", "_target_ts"],
            how="left",
            sort=False,
        )
        exit_price = pd.to_numeric(merged["_exit_price"], errors="coerce")
        buy_price = pd.to_numeric(merged["buy_price"], errors="coerce")
        label = safe_price_return(exit_price, buy_price, fee_bps=fee_bps)
        aligned = pd.Series(label.to_numpy(), index=merged["_row"].to_numpy())
        output[target_col] = output["_row"].map(aligned)

    return output.drop(columns=["_row"]).drop_duplicates(key_cols)
