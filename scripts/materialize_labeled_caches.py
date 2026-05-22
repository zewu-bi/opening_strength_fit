from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import _bootstrap  # noqa: F401
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_PORT,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
    DEFAULT_TICK_END_OFFSET_US,
    DEFAULT_TICK_START_OFFSET_US,
    get_tick_client,
    normalize_clickhouse_ticks,
    query_tick_day_window,
)
from opening_strength_fit.config import config_value, load_toml
from opening_strength_fit.dataset import _add_clock_seconds, _is_decision_point_mode
from opening_strength_fit.features import build_feature_frame
from opening_strength_fit.labels import build_trade_labels
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.sampling import DEFAULT_DECISION_TIMES, sample_labeled_frame
from opening_strength_fit.schema import (
    PRICE_LEVELS,
    ask_price_col,
    ask_volume_col,
    filter_time_range,
)
from opening_strength_fit.training import (
    _CacheLockHeartbeat,
    _acquire_cache_lock,
    _bool_config,
    _cache_timeout_seconds,
    _clear_cache_ready,
    _clickhouse_date_bounds,
    _clickhouse_setting,
    _clock_list_config,
    _float_config,
    _int_config,
    _list_config,
    _mark_cache_ready,
    _release_cache_lock,
    _str_config,
    build_training_parser,
)
from opening_strength_fit.universe import (
    DEFAULT_A_SHARE_SYMBOL_REGEX,
    filter_symbol_universe,
    load_symbol_list,
)


DEFAULT_DELAYS = (0, 1, 2)
DEFAULT_CACHE_DIR = "/mnt/output/opening_strength_fit/cache"
DEFAULT_FILE_TEMPLATE = "opening_1y_next_month_delay{delay}_labeled.parquet"


class StreamingParquetWriter:
    def __init__(self, path: Path, *, compression: str = "snappy") -> None:
        self.path = path
        self.tmp_path = path.with_name(
            f".{path.name}.{os.getpid()}.tmp{''.join(path.suffixes) or '.parquet'}"
        )
        self.compression = compression
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.summary = _StreamingSummary()

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.schema is None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.tmp_path,
                self.schema,
                compression=self.compression,
            )
        else:
            table = pa.Table.from_pandas(
                frame[list(self.schema.names)],
                schema=self.schema,
                preserve_index=False,
            )
        assert self.writer is not None
        self.writer.write_table(table)
        self.summary.update(frame)

    def close(self) -> dict[str, object]:
        if self.writer is None:
            raise SystemExit(f"no labeled rows were written for {self.path}")
        self.writer.close()
        self.writer = None
        os.replace(self.tmp_path, self.path)
        result = self.summary.as_dict()
        result["path"] = str(self.path)
        result["bytes"] = self.path.stat().st_size
        return result

    def cleanup(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        try:
            self.tmp_path.unlink()
        except FileNotFoundError:
            return


class _StreamingSummary:
    def __init__(self) -> None:
        self.rows = 0
        self.columns = 0
        self.dates: set[str] = set()
        self.symbols: set[str] = set()
        self.time_min: str | None = None
        self.time_max: str | None = None
        self.non_null_labels = 0
        self.valid_labels = 0
        self.label_nan_rows = 0

    def update(self, frame: pd.DataFrame) -> None:
        self.rows += len(frame)
        self.columns = max(self.columns, len(frame.columns))
        if "date" in frame.columns:
            self.dates.update(frame["date"].astype(str).dropna().unique())
        if "symbol" in frame.columns:
            self.symbols.update(frame["symbol"].astype(str).dropna().unique())
        if "timestamp" in frame.columns and len(frame):
            clock = frame["timestamp"].dt.strftime("%H:%M:%S")
            current_min = clock.min()
            current_max = clock.max()
            self.time_min = (
                current_min
                if self.time_min is None
                else min(self.time_min, current_min)
            )
            self.time_max = (
                current_max
                if self.time_max is None
                else max(self.time_max, current_max)
            )
        if "label" in frame.columns:
            labels = frame["label"]
            self.non_null_labels += int(labels.notna().sum())
            self.label_nan_rows += int(labels.isna().sum())
        if "valid_label" in frame.columns:
            self.valid_labels += int(frame["valid_label"].sum())

    def as_dict(self) -> dict[str, object]:
        date_min = min(self.dates) if self.dates else None
        date_max = max(self.dates) if self.dates else None
        return {
            "rows": self.rows,
            "columns": self.columns,
            "date_min": date_min,
            "date_max": date_max,
            "n_dates": len(self.dates),
            "n_symbols": len(self.symbols),
            "time_min": self.time_min,
            "time_max": self.time_max,
            "non_null_labels": self.non_null_labels,
            "valid_labels": self.valid_labels,
            "label_nan_rate": (
                float(self.label_nan_rows / self.rows) if self.rows else 1.0
            ),
        }


def _int_list(value: object, default: Iterable[int]) -> list[int]:
    if value in (None, ""):
        return list(default)
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = list(value)  # type: ignore[arg-type]
    return sorted({int(part) for part in parts})


def _feature_frame_from_config(
    ticks: pd.DataFrame,
    config: dict,
    *,
    universe_symbols: set[str] | None,
) -> pd.DataFrame:
    volume_unit_multiplier = _float_config(
        config,
        "labels",
        "volume_unit_multiplier",
        1.0,
    )
    use_universe = _bool_config(config, "universe", "enabled", True)
    if use_universe:
        ticks = filter_symbol_universe(
            ticks,
            symbol_regex=_str_config(
                config,
                "universe",
                "symbol_regex",
                DEFAULT_A_SHARE_SYMBOL_REGEX,
            ),
            symbols=universe_symbols,
        )
    return build_feature_frame(
        ticks,
        include_preopen=_bool_config(config, "features", "include_preopen", True),
        volume_col=_str_config(config, "labels", "volume_col", "volume"),
        turnover_col=_str_config(config, "labels", "turnover_col", "turnover"),
        volume_unit_multiplier=volume_unit_multiplier,
    )


def _key_tuple(key: object) -> tuple[object, ...]:
    return key if isinstance(key, tuple) else (key,)


def _future_values_for_samples(
    samples: pd.DataFrame,
    full: pd.DataFrame,
    *,
    seconds: int,
    value_columns: Sequence[str],
    suffix: str,
    group_columns: Sequence[str] = ("date", "symbol"),
    timestamp_col: str = "timestamp",
    target_timestamp_col: str = "timestamp",
    max_gap_seconds: int | None = None,
) -> pd.DataFrame:
    tolerance = (
        pd.Timedelta(seconds=max_gap_seconds) if max_gap_seconds is not None else None
    )
    out = pd.DataFrame(index=samples.index)
    out[f"timestamp_{suffix}"] = pd.NaT
    for column in value_columns:
        out[f"{column}_{suffix}"] = pd.Series(pd.NA, index=samples.index, dtype="object")

    full_groups = {
        _key_tuple(key): group.sort_values(timestamp_col)
        for key, group in full.groupby(list(group_columns), sort=False, observed=True)
    }
    for key, sample_group in samples.groupby(
        list(group_columns),
        sort=False,
        observed=True,
    ):
        right_group = full_groups.get(_key_tuple(key))
        if right_group is None:
            continue
        left = pd.DataFrame(
            {
                "_row": sample_group.index.to_numpy(),
                "_target_ts": sample_group[target_timestamp_col]
                + pd.to_timedelta(seconds, unit="s"),
            }
        ).dropna(subset=["_target_ts"])
        left["_target_ts"] = pd.to_datetime(left["_target_ts"]).astype("datetime64[ns]")
        right = (
            right_group[[timestamp_col, *value_columns]]
            .dropna(subset=[timestamp_col])
            .rename(columns={timestamp_col: "_future_ts"})
            .sort_values("_future_ts")
        )
        right["_future_ts"] = pd.to_datetime(right["_future_ts"]).astype(
            "datetime64[ns]"
        )
        if left.empty or right.empty:
            continue
        merged = pd.merge_asof(
            left.sort_values("_target_ts"),
            right,
            left_on="_target_ts",
            right_on="_future_ts",
            direction="forward",
            tolerance=tolerance,
        ).set_index("_row")
        out.loc[merged.index, f"timestamp_{suffix}"] = merged["_future_ts"]
        for column in value_columns:
            out.loc[merged.index, f"{column}_{suffix}"] = merged[column]
    return out


def _entry_values_for_samples(
    samples: pd.DataFrame,
    full: pd.DataFrame,
    *,
    offset_ticks: int,
    value_columns: Sequence[str],
    suffix: str = "entry",
    group_columns: Sequence[str] = ("date", "symbol"),
    timestamp_col: str = "timestamp",
    max_gap_seconds: int | None = None,
) -> pd.DataFrame:
    if offset_ticks < 0:
        raise SystemExit("entry_tick_delay must be non-negative")
    if "_source_row" not in samples.columns:
        raise SystemExit("decision samples are missing _source_row")

    timing_columns = (
        f"{suffix}_delay_ticks",
        f"{suffix}_delay_seconds",
        f"{suffix}_max_tick_gap_seconds",
    )
    out = pd.DataFrame(index=samples.index)
    out[f"timestamp_{suffix}"] = pd.NaT
    for column in timing_columns:
        out[column] = pd.Series(np.nan, index=samples.index, dtype="float64")
    for column in value_columns:
        out[f"{column}_{suffix}"] = pd.Series(
            pd.NA,
            index=samples.index,
            dtype="object",
        )

    full_groups = {
        _key_tuple(key): group.sort_values(timestamp_col)
        for key, group in full.groupby(list(group_columns), sort=False, observed=True)
    }
    for key, sample_group in samples.groupby(
        list(group_columns),
        sort=False,
        observed=True,
    ):
        group = full_groups.get(_key_tuple(key))
        if group is None:
            continue
        positions = pd.Series(np.arange(len(group)), index=group.index)
        sample_rows = sample_group["_source_row"].astype(int)
        sample_positions = positions.reindex(sample_rows.to_numpy()).to_numpy()
        valid_position = ~pd.isna(sample_positions)
        sample_positions = np.where(valid_position, sample_positions, -1).astype(int)
        target_positions = sample_positions + int(offset_ticks)
        valid_target = valid_position & (target_positions >= 0) & (
            target_positions < len(group)
        )
        if not valid_target.any():
            continue
        rows = sample_group.index.to_numpy()
        valid_rows = rows[valid_target]
        valid_targets = target_positions[valid_target]
        target = group.iloc[valid_targets]
        sample_timestamps = pd.to_datetime(
            sample_group.loc[valid_rows, timestamp_col],
            errors="coerce",
        ).astype("datetime64[ns]")
        entry_timestamps = pd.to_datetime(
            target[timestamp_col],
            errors="coerce",
        ).astype("datetime64[ns]")
        valid_entry = sample_timestamps.notna().to_numpy() & entry_timestamps.notna().to_numpy()
        out.loc[valid_rows, f"timestamp_{suffix}"] = entry_timestamps.to_numpy()
        out.loc[valid_rows, f"{suffix}_delay_ticks"] = np.where(
            valid_entry,
            float(offset_ticks),
            np.nan,
        )
        out.loc[valid_rows, f"{suffix}_delay_seconds"] = (
            entry_timestamps.to_numpy() - sample_timestamps.to_numpy()
        ) / pd.Timedelta(seconds=1)
        if offset_ticks == 0:
            max_tick_gap = np.where(valid_entry, 0.0, np.nan)
        else:
            group_timestamps = pd.to_datetime(
                group[timestamp_col],
                errors="coerce",
            ).astype("datetime64[ns]")
            step_gap = (
                group_timestamps.diff() / pd.Timedelta(seconds=1)
            ).to_numpy(dtype="float64")
            path_gaps = np.vstack(
                [step_gap[sample_positions[valid_target] + step] for step in range(1, offset_ticks + 1)]
            ).T
            complete_path = ~np.isnan(path_gaps).any(axis=1)
            max_tick_gap = np.where(complete_path, np.nanmax(path_gaps, axis=1), np.nan)
        out.loc[valid_rows, f"{suffix}_max_tick_gap_seconds"] = max_tick_gap
        for column in value_columns:
            out.loc[valid_rows, f"{column}_{suffix}"] = target[column].to_numpy()

    if max_gap_seconds is not None:
        entry_gap = out[f"{suffix}_max_tick_gap_seconds"]
        valid_gap = entry_gap.notna() & entry_gap.ge(0) & entry_gap.le(max_gap_seconds)
        out.loc[~valid_gap, f"timestamp_{suffix}"] = pd.NaT
        for column in timing_columns:
            out.loc[~valid_gap, column] = np.nan
        for column in value_columns:
            out.loc[~valid_gap, f"{column}_{suffix}"] = np.nan

    return out


def _select_decision_points_fast(
    frame: pd.DataFrame,
    *,
    decision_times: Sequence[str],
    max_lag_seconds: int | None,
) -> pd.DataFrame:
    if not decision_times:
        raise SystemExit("decision point sampling needs at least one decision time")
    if frame.empty:
        return frame.copy()

    tolerance = (
        pd.Timedelta(seconds=max_lag_seconds) if max_lag_seconds is not None else None
    )
    selected_parts = []
    work = frame.sort_values(["date", "symbol", "timestamp"]).copy()
    if "_source_row" not in work.columns:
        work["_source_row"] = work.index

    for trading_day, day in work.groupby("date", sort=False, observed=True):
        symbols = day["symbol"].astype(str).drop_duplicates().to_numpy()
        if len(symbols) == 0:
            continue
        times = np.asarray(list(decision_times), dtype=object)
        targets = pd.DataFrame(
            {
                "symbol": np.repeat(symbols, len(times)),
                "decision_time": np.tile(times, len(symbols)),
            }
        )
        targets["_target_ts"] = pd.to_datetime(
            [f"{trading_day} {clock}" for clock in targets["decision_time"]]
        )
        right = (
            day[["symbol", "timestamp", "_source_row"]]
            .assign(symbol=lambda df: df["symbol"].astype(str))
            .sort_values(["timestamp", "symbol"])
        )
        merged = pd.merge_asof(
            targets.sort_values(["_target_ts", "symbol"]),
            right,
            by="symbol",
            left_on="_target_ts",
            right_on="timestamp",
            direction="forward",
            tolerance=tolerance,
        ).dropna(subset=["_source_row"])
        if merged.empty:
            continue
        chosen = work.loc[merged["_source_row"].astype(int).to_numpy()].copy()
        chosen["decision_time"] = merged["decision_time"].to_numpy()
        chosen["decision_target_timestamp"] = merged["_target_ts"].to_numpy()
        chosen["decision_lag_seconds"] = (
            chosen["timestamp"].to_numpy() - chosen["decision_target_timestamp"]
        ) / pd.Timedelta(seconds=1)
        selected_parts.append(chosen)

    if not selected_parts:
        return work.iloc[:0].copy()
    return (
        pd.concat(selected_parts, ignore_index=True)
        .sort_values(["date", "symbol", "timestamp"])
        .reset_index(drop=True)
    )


def _decision_labeled_from_feature_frame(
    features: pd.DataFrame,
    *,
    buy_price_col: str,
    volume_col: str,
    turnover_col: str,
    hold_seconds: int,
    sell_window_seconds: int,
    volume_unit_multiplier: float,
    fee_bps: float,
    entry_tick_delay: int,
    entry_max_gap_seconds: int | None,
    sample_start_time: str,
    sample_end_time: str,
    max_future_gap_seconds: int | None,
    tradable_statuses: Sequence[str] | None,
    decision_times: Sequence[str],
    decision_max_lag_seconds: int | None,
) -> pd.DataFrame:
    full = features.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)
    full["_source_row"] = full.index
    samples = _select_decision_points_fast(
        full,
        decision_times=decision_times,
        max_lag_seconds=decision_max_lag_seconds,
    )
    if samples.empty:
        return samples.drop(columns=["_source_row"], errors="ignore")

    entry_value_columns = [buy_price_col]
    for level in PRICE_LEVELS:
        for column in (ask_price_col(level), ask_volume_col(level)):
            if column in full.columns:
                entry_value_columns.append(column)
    if "status" in full.columns:
        entry_value_columns.append("status")
    entry_value_columns = list(dict.fromkeys(entry_value_columns))
    entry = _entry_values_for_samples(
        samples,
        full,
        offset_ticks=int(entry_tick_delay),
        value_columns=entry_value_columns,
        max_gap_seconds=entry_max_gap_seconds,
    )

    work = samples.copy()
    work["entry_timestamp"] = pd.to_datetime(
        entry["timestamp_entry"],
        errors="coerce",
    ).astype("datetime64[ns]")
    work["entry_delay_ticks"] = pd.to_numeric(
        entry["entry_delay_ticks"],
        errors="coerce",
    ).astype("float64")
    work["entry_delay_seconds"] = pd.to_numeric(
        entry["entry_delay_seconds"],
        errors="coerce",
    ).astype("float64")
    work["entry_max_tick_gap_seconds"] = pd.to_numeric(
        entry["entry_max_tick_gap_seconds"],
        errors="coerce",
    ).astype("float64")
    work["buy_price"] = pd.to_numeric(
        entry[f"{buy_price_col}_entry"],
        errors="coerce",
    ).astype("float64")
    for level in PRICE_LEVELS:
        price_col = ask_price_col(level)
        volume_col_name = ask_volume_col(level)
        if f"{price_col}_entry" in entry.columns:
            work[f"entry_{price_col}"] = pd.to_numeric(
                entry[f"{price_col}_entry"],
                errors="coerce",
            ).astype("float64")
        if f"{volume_col_name}_entry" in entry.columns:
            work[f"entry_{volume_col_name}"] = pd.to_numeric(
                entry[f"{volume_col_name}_entry"],
                errors="coerce",
            ).astype("float64")
    if "status_entry" in entry.columns:
        work["entry_status"] = entry["status_entry"]

    value_columns = [volume_col, turnover_col]
    sell_start = _future_values_for_samples(
        work,
        full,
        seconds=hold_seconds,
        value_columns=value_columns,
        suffix="sell_start",
        target_timestamp_col="entry_timestamp",
        max_gap_seconds=max_future_gap_seconds,
    )
    sell_end = _future_values_for_samples(
        work,
        full,
        seconds=hold_seconds + sell_window_seconds,
        value_columns=value_columns,
        suffix="sell_end",
        target_timestamp_col="entry_timestamp",
        max_gap_seconds=max_future_gap_seconds,
    )
    work = pd.concat([work, sell_start, sell_end], axis=1)

    start_volume = pd.to_numeric(
        work[f"{volume_col}_sell_start"],
        errors="coerce",
    ).astype("float64")
    end_volume = pd.to_numeric(
        work[f"{volume_col}_sell_end"],
        errors="coerce",
    ).astype("float64")
    start_turnover = pd.to_numeric(
        work[f"{turnover_col}_sell_start"],
        errors="coerce",
    ).astype("float64")
    end_turnover = pd.to_numeric(
        work[f"{turnover_col}_sell_end"],
        errors="coerce",
    ).astype("float64")
    work["sell_volume"] = end_volume - start_volume
    work["sell_turnover"] = end_turnover - start_turnover
    denominator = work["sell_volume"] * float(volume_unit_multiplier)
    work["sell_vwap"] = np.where(
        denominator > 0,
        work["sell_turnover"] / denominator,
        np.nan,
    )
    work["gross_label"] = np.where(
        work["buy_price"] > 0,
        work["sell_vwap"] / work["buy_price"] - 1.0,
        np.nan,
    )
    work["label"] = work["gross_label"] - float(fee_bps) / 10_000.0
    work["valid_label"] = (
        work["label"].notna()
        & np.isfinite(work["label"])
        & (work["sell_volume"] > 0)
        & (work["sell_turnover"] > 0)
        & (work["buy_price"] > 0)
        & work["entry_timestamp"].notna()
    )
    if tradable_statuses and "status" in work.columns:
        allowed = {str(status).upper() for status in tradable_statuses}
        work["valid_label"] &= work["status"].astype(str).str.upper().isin(allowed)
        if "entry_status" in work.columns:
            work["valid_label"] &= (
                work["entry_status"].astype(str).str.upper().isin(allowed)
            )

    work = filter_time_range(
        work,
        sample_start_time,
        sample_end_time,
        include_end=True,
    )
    return work.drop(columns=["_source_row"], errors="ignore")


def _labeled_from_feature_frame(
    features: pd.DataFrame,
    config: dict,
    *,
    delay: int,
) -> pd.DataFrame:
    volume_unit_multiplier = _float_config(
        config,
        "labels",
        "volume_unit_multiplier",
        1.0,
    )
    sample_mode = _str_config(config, "sample", "mode", "all_ticks")
    max_decision_lag = config_value(
        config,
        "sample",
        "decision_max_lag_seconds",
        5,
    )
    decision_lag_seconds = (
        None if max_decision_lag in (None, "") else int(max_decision_lag)
    )
    sample_end_time = _str_config(config, "sample", "end_time", "09:40:00")
    label_sample_end_time = (
        _add_clock_seconds(sample_end_time, decision_lag_seconds)
        if _is_decision_point_mode(sample_mode)
        else sample_end_time
    )
    decision_times = _clock_list_config(
        config,
        "sample",
        "decision_times",
        DEFAULT_DECISION_TIMES,
    )
    if _is_decision_point_mode(sample_mode):
        return _decision_labeled_from_feature_frame(
            features,
            buy_price_col=_str_config(config, "labels", "buy_price_col", "ask_price_1"),
            volume_col=_str_config(config, "labels", "volume_col", "volume"),
            turnover_col=_str_config(config, "labels", "turnover_col", "turnover"),
            hold_seconds=_int_config(config, "labels", "hold_seconds", 60),
            sell_window_seconds=_int_config(config, "labels", "sell_window_seconds", 60),
            volume_unit_multiplier=volume_unit_multiplier,
            fee_bps=_float_config(config, "labels", "fee_bps", 0.0),
            entry_tick_delay=int(delay),
            entry_max_gap_seconds=config_value(
                config,
                "labels",
                "entry_max_gap_seconds",
                None,
            ),
            sample_start_time=_str_config(config, "sample", "start_time", "09:30:00"),
            sample_end_time=label_sample_end_time,
            max_future_gap_seconds=config_value(
                config,
                "labels",
                "max_future_gap_seconds",
                None,
            ),
            tradable_statuses=_list_config(config, "filters", "tradable_statuses", []),
            decision_times=decision_times,
            decision_max_lag_seconds=decision_lag_seconds,
        )

    labeled = build_trade_labels(
        features,
        buy_price_col=_str_config(config, "labels", "buy_price_col", "ask_price_1"),
        volume_col=_str_config(config, "labels", "volume_col", "volume"),
        turnover_col=_str_config(config, "labels", "turnover_col", "turnover"),
        hold_seconds=_int_config(config, "labels", "hold_seconds", 60),
        sell_window_seconds=_int_config(config, "labels", "sell_window_seconds", 60),
        volume_unit_multiplier=volume_unit_multiplier,
        fee_bps=_float_config(config, "labels", "fee_bps", 0.0),
        entry_tick_delay=int(delay),
        entry_max_gap_seconds=config_value(
            config,
            "labels",
            "entry_max_gap_seconds",
            None,
        ),
        sample_start_time=_str_config(config, "sample", "start_time", "09:30:00"),
        sample_end_time=label_sample_end_time,
        max_future_gap_seconds=config_value(
            config,
            "labels",
            "max_future_gap_seconds",
            None,
        ),
        tradable_statuses=_list_config(config, "filters", "tradable_statuses", []),
    )
    return sample_labeled_frame(
        labeled,
        mode=sample_mode,
        decision_times=decision_times,
        max_lag_seconds=decision_lag_seconds,
    )


def _cache_path(config: dict, cache_dir: Path, delay: int) -> Path:
    cache_config = config.get("cache", {})
    specific = cache_config.get(f"delay{delay}_labeled_path", "")
    template = str(
        specific
        or cache_config.get("file_template", DEFAULT_FILE_TEMPLATE)
    )
    formatted = template.format(delay=delay, delay_label=f"delay{delay}")
    path = Path(formatted)
    return path if path.is_absolute() else cache_dir / path


def _lock_targets(
    paths: Iterable[Path],
    *,
    timeout_seconds: int,
) -> list[Path]:
    lock_paths = []
    for path in paths:
        lock_path = Path(f"{path}.lock")
        status = _acquire_cache_lock(
            lock_path,
            timeout_seconds,
            cache_path=None,
            cache_read=False,
        )
        if status != "acquired":
            raise SystemExit(f"failed to acquire labeled cache lock: {lock_path}")
        _clear_cache_ready(lock_path)
        lock_paths.append(lock_path)
    return lock_paths


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize(args) -> None:
    if not args.config:
        raise SystemExit("--config is required")
    config = load_toml(args.config)
    cache_dir = Path(
        args.cache_dir
        or config_value(config, "cache", "dir", DEFAULT_CACHE_DIR)
    )
    delays = _int_list(
        args.delay
        if args.delay
        else config_value(config, "cache", "delays", DEFAULT_DELAYS),
        DEFAULT_DELAYS,
    )
    compression = str(config_value(config, "cache", "compression", "snappy"))
    overwrite = bool(args.overwrite or _bool_config(config, "cache", "overwrite", False))
    output_paths = {delay: _cache_path(config, cache_dir, delay) for delay in delays}

    active_paths = {
        delay: path
        for delay, path in output_paths.items()
        if overwrite or not path.exists()
    }
    skipped = {
        delay: str(path)
        for delay, path in output_paths.items()
        if delay not in active_paths
    }
    if skipped:
        print_mapping("labeled_cache_skipped", skipped)
    if not active_paths:
        print("all requested labeled caches already exist")
        return

    start_date, end_date = _clickhouse_date_bounds(args, config)
    dates = [str(date.date()) for date in pd.date_range(start_date, end_date, freq="D")]
    use_universe = _bool_config(config, "universe", "enabled", True)
    symbols_file = _str_config(config, "universe", "symbols_file", "")
    symbols = sorted(load_symbol_list(symbols_file)) if use_universe and symbols_file else None
    symbol_regex = (
        _str_config(config, "universe", "symbol_regex", DEFAULT_A_SHARE_SYMBOL_REGEX)
        if use_universe
        else None
    )
    host = str(
        _clickhouse_setting(
            args,
            config,
            "clickhouse_host",
            "host",
            "CLICKHOUSE_HOST",
            DEFAULT_CLICKHOUSE_TICK_HOST,
        )
    )
    port = int(
        _clickhouse_setting(
            args,
            config,
            "clickhouse_port",
            "port",
            "CLICKHOUSE_PORT",
            DEFAULT_CLICKHOUSE_TICK_PORT,
        )
    )
    user = _clickhouse_setting(
        args,
        config,
        "clickhouse_user",
        "user",
        "CLICKHOUSE_USER",
        None,
    )
    password = _clickhouse_setting(
        args,
        config,
        "clickhouse_password",
        "password",
        "CLICKHOUSE_PASSWORD",
        None,
    )
    table = str(
        _clickhouse_setting(
            args,
            config,
            "clickhouse_table",
            "table",
            "CLICKHOUSE_TICK_TABLE",
            DEFAULT_CLICKHOUSE_TICK_TABLE,
        )
    )
    start_offset_us = int(
        args.start_offset_us
        if args.start_offset_us is not None
        else config_value(
            config,
            "clickhouse",
            "start_offset_us",
            DEFAULT_TICK_START_OFFSET_US,
        )
    )
    end_offset_us = int(
        args.end_offset_us
        if args.end_offset_us is not None
        else config_value(
            config,
            "clickhouse",
            "end_offset_us",
            DEFAULT_TICK_END_OFFSET_US,
        )
    )
    if not user or not password:
        raise SystemExit(
            "missing ClickHouse credentials: set CLICKHOUSE_USER and "
            "CLICKHOUSE_PASSWORD, pass CLI overrides, or configure a K8s secret."
        )

    print_mapping(
        "materialize_labeled_caches",
        {
            "delays": ",".join(str(delay) for delay in active_paths),
            "cache_dir": str(cache_dir),
            "overwrite": overwrite,
            "date_start": start_date,
            "date_end": end_date,
            "calendar_days": len(dates),
            "clickhouse_host": host,
            "clickhouse_table": table,
            "start_offset_us": start_offset_us,
            "end_offset_us": end_offset_us,
            "symbol_regex": symbol_regex or "",
            "symbols": len(symbols) if symbols else 0,
        },
    )

    client = get_tick_client(
        host=host,
        port=port,
        username=str(user),
        password=str(password),
    )
    timeout_seconds = _cache_timeout_seconds(config)
    lock_paths = _lock_targets(active_paths.values(), timeout_seconds=timeout_seconds)
    writers = {
        delay: StreamingParquetWriter(path, compression=compression)
        for delay, path in active_paths.items()
    }
    try:
        heartbeats = [_CacheLockHeartbeat(lock_path) for lock_path in lock_paths]
        for heartbeat in heartbeats:
            heartbeat.__enter__()
        try:
            for trading_day in dates:
                ticks = query_tick_day_window(
                    client,
                    trading_day=trading_day,
                    table=table,
                    start_offset_us=start_offset_us,
                    end_offset_us=end_offset_us,
                    symbol_regex=symbol_regex,
                    symbols=symbols,
                )
                if ticks.empty:
                    print(f"skip empty ClickHouse day: {trading_day}")
                    continue
                ticks = normalize_clickhouse_ticks(ticks)
                print_mapping(
                    f"clickhouse_ticks[{trading_day}]",
                    dataset_summary(ticks),
                )
                features = _feature_frame_from_config(
                    ticks,
                    config,
                    universe_symbols=set(symbols) if use_universe and symbols else None,
                )
                print_mapping(
                    f"features[{trading_day}]",
                    dataset_summary(features),
                )
                for delay, writer in writers.items():
                    labeled = _labeled_from_feature_frame(
                        features,
                        config,
                        delay=delay,
                    )
                    writer.write(labeled)
                    print_mapping(
                        f"labeled_delay{delay}[{trading_day}]",
                        dataset_summary(labeled),
                    )
        finally:
            for heartbeat in reversed(heartbeats):
                heartbeat.__exit__(None, None, None)

        summaries = {}
        for delay, writer in writers.items():
            summary = writer.close()
            summaries[f"delay{delay}"] = summary
            lock_path = Path(f"{writer.path}.lock")
            _mark_cache_ready(writer.path, lock_path)
            print_mapping(f"wrote_labeled_cache_delay{delay}", summary)

        manifest_path = Path(
            args.manifest
            or config_value(
                config,
                "cache",
                "manifest_path",
                cache_dir / "opening_1y_next_month_delay_cache_manifest.json",
            )
        )
        _write_manifest(
            manifest_path,
            {
                "config": str(args.config),
                "delays": delays,
                "active_delays": sorted(active_paths),
                "skipped": skipped,
                "date_start": start_date,
                "date_end": end_date,
                "decision_times": _clock_list_config(
                    config,
                    "sample",
                    "decision_times",
                    [],
                ),
                "summaries": summaries,
            },
        )
        print(f"wrote_manifest: {manifest_path}")
    except Exception:
        for writer in writers.values():
            writer.cleanup()
        raise
    finally:
        for lock_path in lock_paths:
            _release_cache_lock(lock_path)


def main() -> None:
    parser = build_training_parser(
        "Materialize ClickHouse labeled parquet caches for multiple entry delays."
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Directory for labeled cache files. Default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--delay",
        action="append",
        type=int,
        help="Entry tick delay to materialize. Repeatable; defaults to config or 0,1,2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing labeled parquet files.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest JSON path. Defaults to [cache].manifest_path or cache dir.",
    )
    args = parser.parse_args()
    materialize(args)


if __name__ == "__main__":
    main()
