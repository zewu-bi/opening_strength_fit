from __future__ import annotations

import re

import numpy as np
import pandas as pd

from opening_strength_fit.schema import (
    PREOPEN_END,
    PREOPEN_START,
    ask_price_col,
    ask_volume_col,
    available_depth_levels,
    bid_price_col,
    bid_volume_col,
    ensure_timestamp_columns,
    filter_time_range,
    normalize_clock_time,
)


def safe_divide(numerator, denominator):
    denominator = np.asarray(denominator, dtype="float64")
    numerator = np.asarray(numerator, dtype="float64")
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    out = np.full_like(numerator, np.nan, dtype="float64")
    return np.divide(numerator, denominator, out=out, where=denominator != 0)


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _sum_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=df.index)
    values = df[columns].apply(pd.to_numeric, errors="coerce").astype("float64")
    return values.sum(axis=1, min_count=1)


def add_order_book_features(ticks: pd.DataFrame) -> pd.DataFrame:
    out = ticks.copy()
    levels = available_depth_levels(out)

    if "bid_price_1" in out.columns and "ask_price_1" in out.columns:
        bid1 = _numeric_series(out["bid_price_1"])
        ask1 = _numeric_series(out["ask_price_1"])
        out["mid_price"] = (bid1 + ask1) / 2.0
        out["spread_abs"] = ask1 - bid1
        out["spread_bps"] = safe_divide(out["spread_abs"], out["mid_price"]) * 10_000

    bid_volume_columns = [bid_volume_col(level) for level in levels]
    ask_volume_columns = [ask_volume_col(level) for level in levels]
    out["bid_depth_10"] = _sum_columns(out, bid_volume_columns)
    out["ask_depth_10"] = _sum_columns(out, ask_volume_columns)
    out["depth_imbalance_10"] = safe_divide(
        out["bid_depth_10"] - out["ask_depth_10"],
        out["bid_depth_10"] + out["ask_depth_10"],
    )

    if "bid_volume_1" in out.columns and "ask_volume_1" in out.columns:
        bid_volume_1 = _numeric_series(out["bid_volume_1"])
        ask_volume_1 = _numeric_series(out["ask_volume_1"])
        out["depth_imbalance_1"] = safe_divide(
            bid_volume_1 - ask_volume_1,
            bid_volume_1 + ask_volume_1,
        )

    if "limit_up_price" in out.columns and "ask_price_1" in out.columns:
        limit_up = _numeric_series(out["limit_up_price"])
        ask1 = _numeric_series(out["ask_price_1"])
        out["ask1_to_limit_up_bps"] = safe_divide(limit_up - ask1, ask1) * 10_000

    for level in levels:
        bid_price = bid_price_col(level)
        ask_price = ask_price_col(level)
        if level > 1 and bid_price in out.columns and "bid_price_1" in out.columns:
            bid1 = _numeric_series(out["bid_price_1"])
            bid_level = _numeric_series(out[bid_price])
            out[f"bid_gap_{level}_bps"] = safe_divide(bid1 - bid_level, bid1) * 10_000
        if level > 1 and ask_price in out.columns and "ask_price_1" in out.columns:
            ask1 = _numeric_series(out["ask_price_1"])
            ask_level = _numeric_series(out[ask_price])
            out[f"ask_gap_{level}_bps"] = safe_divide(ask_level - ask1, ask1) * 10_000
    return out


def add_trade_features(
    ticks: pd.DataFrame,
    *,
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    volume_unit_multiplier: float = 1.0,
) -> pd.DataFrame:
    out = ticks.sort_values(["date", "symbol", "timestamp"]).copy()
    group = out.groupby(["date", "symbol"], sort=False)

    if volume_col in out.columns:
        out["volume_diff_1t"] = group[volume_col].diff()
    if turnover_col in out.columns:
        out["turnover_diff_1t"] = group[turnover_col].diff()
    if volume_col in out.columns and turnover_col in out.columns:
        denominator = out["volume_diff_1t"] * float(volume_unit_multiplier)
        out["trade_vwap_1t"] = safe_divide(out["turnover_diff_1t"], denominator)
        for window in (3, 10, 30):
            out[f"volume_diff_{window}t"] = group[volume_col].diff(window)
            out[f"turnover_diff_{window}t"] = group[turnover_col].diff(window)
            out[f"trade_vwap_{window}t"] = safe_divide(
                out[f"turnover_diff_{window}t"],
                out[f"volume_diff_{window}t"] * float(volume_unit_multiplier),
            )
    return out


def add_momentum_features(ticks: pd.DataFrame) -> pd.DataFrame:
    out = ticks.sort_values(["date", "symbol", "timestamp"]).copy()
    group = out.groupby(["date", "symbol"], sort=False)
    price_col = None
    for candidate in ("last_price", "mid_price", "ask_price_1"):
        if candidate in out.columns:
            price_col = candidate
            break
    if price_col is None:
        return out

    if "prev_close" in out.columns:
        out["return_vs_prev_close"] = safe_divide(
            out[price_col] - out["prev_close"],
            out["prev_close"],
        )
    if "open_price" in out.columns:
        out["return_vs_open"] = safe_divide(
            out[price_col] - out["open_price"],
            out["open_price"],
        )

    for periods in (1, 3, 10, 30):
        out[f"return_{periods}t"] = group[price_col].pct_change(periods)
    return out


def add_postopen_decision_features(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    out = ensure_timestamp_columns(frame)
    time_col = (
        "decision_target_timestamp" if "decision_target_timestamp" in out.columns else "timestamp"
    )
    out = out.sort_values(["date", "symbol", time_col]).reset_index(drop=True)
    group = out.groupby(["date", "symbol"], sort=False)

    timestamp = pd.to_datetime(out[time_col], errors="coerce")
    open_timestamp = pd.to_datetime(
        out["date"].astype(str) + " 09:30:00",
        errors="coerce",
    )
    out["postopen_minutes_since_0930"] = (timestamp - open_timestamp).dt.total_seconds() / 60.0

    dynamic_columns = [
        "ask_volume_1",
        "bid_volume_1",
        "ask_depth_10",
        "bid_depth_10",
        "depth_imbalance_1",
        "depth_imbalance_10",
        "spread_bps",
        "mid_price",
        "ask_price_1",
        "bid_price_1",
        "volume",
        "turnover",
    ]
    for column in dynamic_columns:
        if column not in out.columns:
            continue
        values = _numeric_series(out[column])
        for window in windows:
            lagged = pd.to_numeric(group[column].shift(window), errors="coerce")
            diff = values - lagged
            out[f"postopen_{column}_diff_{window}m"] = diff
            out[f"postopen_{column}_rel_{window}m"] = safe_divide(
                diff,
                lagged.abs(),
            )

    if "ask_volume_1" in out.columns and "ask_depth_10" in out.columns:
        out["postopen_ask1_depth_share"] = safe_divide(
            out["ask_volume_1"],
            out["ask_depth_10"],
        )
    if "bid_volume_1" in out.columns and "bid_depth_10" in out.columns:
        out["postopen_bid1_depth_share"] = safe_divide(
            out["bid_volume_1"],
            out["bid_depth_10"],
        )
    if "postopen_bid1_depth_share" in out.columns and "postopen_ask1_depth_share" in out.columns:
        out["postopen_top_depth_share_imbalance"] = (
            out["postopen_bid1_depth_share"] - out["postopen_ask1_depth_share"]
        )

    ask_gap_cols = [f"ask_gap_{level}_bps" for level in range(2, 11)]
    bid_gap_cols = [f"bid_gap_{level}_bps" for level in range(2, 11)]
    present_ask_gaps = [column for column in ask_gap_cols if column in out.columns]
    present_bid_gaps = [column for column in bid_gap_cols if column in out.columns]
    if present_ask_gaps:
        out["postopen_ask_gap_mean_2_10"] = out[present_ask_gaps].mean(axis=1)
        out["postopen_ask_gap_std_2_10"] = out[present_ask_gaps].std(axis=1)
    if present_bid_gaps:
        out["postopen_bid_gap_mean_2_10"] = out[present_bid_gaps].mean(axis=1)
        out["postopen_bid_gap_std_2_10"] = out[present_bid_gaps].std(axis=1)
    if "spread_bps" in out.columns and "depth_imbalance_1" in out.columns:
        out["postopen_spread_x_imbalance_1"] = _numeric_series(out["spread_bps"]) * _numeric_series(
            out["depth_imbalance_1"]
        )
    if "volume_diff_1t" in out.columns and "ask_volume_1" in out.columns:
        out["postopen_trade_vs_ask1_queue"] = safe_divide(
            out["volume_diff_1t"],
            out["ask_volume_1"],
        )
    return out


def _column_values(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return _numeric_series(df[column])


def _matching_columns(
    columns: pd.Index,
    *,
    include_columns: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] = (),
    include_patterns: tuple[str, ...] = (),
) -> list[str]:
    explicit = set(include_columns)
    compiled = [re.compile(pattern) for pattern in include_patterns]
    matched: list[str] = []
    for column in columns:
        name = str(column)
        if name in explicit:
            matched.append(name)
            continue
        if include_prefixes and name.startswith(include_prefixes):
            matched.append(name)
            continue
        if compiled and any(pattern.search(name) for pattern in compiled):
            matched.append(name)
    return list(dict.fromkeys(matched))


def add_cross_sectional_relative_features(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] = (),
    include_patterns: tuple[str, ...] = (),
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    modes: tuple[str, ...] = ("zscore", "rank_centered"),
    prefix: str = "xs_rel_",
    rank_method: str = "average",
) -> pd.DataFrame:
    """Add within-cross-section relative versions of configured numeric columns."""

    available_group_cols = tuple(column for column in group_cols if column in frame.columns)
    if not available_group_cols:
        raise ValueError("cross-sectional relative features need at least one group column")

    source_columns = [
        column
        for column in _matching_columns(
            frame.columns,
            include_columns=columns,
            include_prefixes=include_prefixes,
            include_patterns=include_patterns,
        )
        if column not in available_group_cols
    ]
    source_columns = [
        column
        for column in source_columns
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column])
    ]
    if not source_columns:
        return frame.copy()

    normalized_modes = tuple(mode.strip().lower() for mode in modes if mode.strip())
    valid_modes = {"demean", "zscore", "rank_pct", "rank_centered", "rank"}
    unknown_modes = sorted(set(normalized_modes) - valid_modes)
    if unknown_modes:
        raise ValueError(f"unknown cross-sectional relative feature modes: {unknown_modes}")

    out = frame.copy()
    group_keys = [out[column] for column in available_group_cols]
    new_columns: dict[str, pd.Series] = {}
    for column in source_columns:
        values = pd.to_numeric(out[column], errors="coerce").astype("float64")
        grouped = values.groupby(group_keys, sort=False)
        centered = None
        rank_pct = None
        for mode in normalized_modes:
            if mode in {"demean", "zscore"} and centered is None:
                centered = values - grouped.transform("mean")
            if mode == "demean":
                new_columns[f"{prefix}{column}_demean"] = centered
            elif mode == "zscore":
                std = grouped.transform("std")
                new_columns[f"{prefix}{column}_zscore"] = pd.Series(
                    safe_divide(centered, std),
                    index=out.index,
                )
            elif mode in {"rank_pct", "rank_centered", "rank"}:
                if rank_pct is None:
                    rank_pct = grouped.rank(method=rank_method, pct=True)
                if mode == "rank_pct":
                    new_columns[f"{prefix}{column}_rank_pct"] = rank_pct
                else:
                    new_columns[f"{prefix}{column}_rank_centered"] = rank_pct - 0.5

    return pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)


def _price_bucket_labels(edges: tuple[float, ...]) -> list[str]:
    if len(edges) == 2:
        return ["cheap", "mid", "expensive"]
    return [f"bucket{i}" for i in range(len(edges) + 1)]


def add_price_scale_features(
    frame: pd.DataFrame,
    *,
    price_col: str = "ask_price_1",
    tick_size: float = 0.01,
    bucket_edges: tuple[float, ...] = (5.0, 20.0),
    interaction_columns: tuple[str, ...] = (),
    prefix: str = "price_scale_",
) -> pd.DataFrame:
    """Add price-regime and tick/bps scale features.

    These features make the A-share fixed 0.01 tick explicit, so low-price and
    high-price names do not have to share one raw orderbook interpretation.
    """

    if price_col not in frame.columns:
        return frame.copy()

    out = frame.copy()
    price = _numeric_series(out[price_col]).where(lambda s: s > 0.0)
    tick_size = float(tick_size)
    edges = tuple(sorted(float(edge) for edge in bucket_edges))
    labels = _price_bucket_labels(edges)

    bucket_code = pd.Series(np.nan, index=out.index, dtype="float64")
    valid = price.notna()
    if valid.any():
        bucket_code.loc[valid] = np.searchsorted(edges, price.loc[valid], side="right")

    new_columns: dict[str, pd.Series] = {
        f"{prefix}log_price": np.log(price),
        f"{prefix}tick_bps": safe_divide(tick_size, price) * 10_000.0,
        f"{prefix}bucket_code": bucket_code,
    }
    for code, label in enumerate(labels):
        new_columns[f"{prefix}bucket_{label}"] = bucket_code.eq(float(code)).astype("int8")

    if "spread_abs" in out.columns:
        new_columns[f"{prefix}spread_ticks"] = safe_divide(
            _numeric_series(out["spread_abs"]),
            tick_size,
        )
    elif {"ask_price_1", "bid_price_1"}.issubset(out.columns):
        new_columns[f"{prefix}spread_ticks"] = safe_divide(
            _numeric_series(out["ask_price_1"]) - _numeric_series(out["bid_price_1"]),
            tick_size,
        )

    if "ask_price_1" in out.columns:
        ask1 = _numeric_series(out["ask_price_1"])
        for level in range(2, 11):
            column = f"ask_price_{level}"
            if column in out.columns:
                new_columns[f"{prefix}ask_gap_{level}_ticks"] = safe_divide(
                    _numeric_series(out[column]) - ask1,
                    tick_size,
                )
    if "bid_price_1" in out.columns:
        bid1 = _numeric_series(out["bid_price_1"])
        for level in range(2, 11):
            column = f"bid_price_{level}"
            if column in out.columns:
                new_columns[f"{prefix}bid_gap_{level}_ticks"] = safe_divide(
                    bid1 - _numeric_series(out[column]),
                    tick_size,
                )

    for column in (
        "ask_volume_1",
        "bid_volume_1",
        "ask_depth_3",
        "bid_depth_3",
        "ask_depth_10",
        "bid_depth_10",
        "postopen_v2_ask_depth_3",
        "postopen_v2_bid_depth_3",
        "postopen_v2_ask_depth_10",
        "postopen_v2_bid_depth_10",
        "volume_diff_1t",
        "volume_diff_3t",
    ):
        if column in out.columns:
            new_columns[f"{prefix}{column}_notional"] = _numeric_series(out[column]) * price

    interaction_frame = pd.DataFrame(new_columns, index=out.index)
    for column in interaction_columns:
        if column not in out.columns:
            continue
        values = _numeric_series(out[column])
        for label in labels:
            bucket = interaction_frame[f"{prefix}bucket_{label}"].astype("float64")
            interaction_frame[f"{prefix}{column}_x_{label}"] = values * bucket

    return pd.concat([out, interaction_frame], axis=1)


def _sum_present_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [column for column in columns if column in df.columns]
    return _sum_columns(df, present)


def _weighted_mean(
    values: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    weighted = values.astype("float64") * weights.astype("float64")
    return safe_divide(
        weighted.sum(axis=1, min_count=1),
        weights.sum(axis=1, min_count=1),
    )


class _PostOpenV2Builder:
    def __init__(self, frame: pd.DataFrame) -> None:
        out = ensure_timestamp_columns(frame)
        self.time_col = (
            "decision_target_timestamp"
            if "decision_target_timestamp" in out.columns
            else "timestamp"
        )
        self.out = out.sort_values(["date", "symbol", self.time_col]).reset_index(
            drop=True,
        )
        self.group_keys = [self.out["date"], self.out["symbol"]]
        self.new_columns: dict[str, pd.Series] = {}

    def has(self, column: str) -> bool:
        return column in self.out.columns or column in self.new_columns

    def add(self, name: str, values) -> None:
        if isinstance(values, pd.Series):
            self.new_columns[name] = values.reset_index(drop=True)
        else:
            self.new_columns[name] = pd.Series(values, index=self.out.index)

    def series(self, column: str) -> pd.Series:
        if column in self.new_columns:
            return self.new_columns[column]
        return self.out[column]

    def numeric(self, column: str) -> pd.Series:
        return pd.to_numeric(self.series(column), errors="coerce").astype("float64")

    def shifted(self, column: str, window: int) -> pd.Series:
        return self.numeric(column).groupby(self.group_keys, sort=False).shift(window)

    def first(self, column: str) -> pd.Series:
        return self.numeric(column).groupby(self.group_keys, sort=False).transform("first")

    def finish(self) -> pd.DataFrame:
        if not self.new_columns:
            return self.out
        return pd.concat(
            [self.out, pd.DataFrame(self.new_columns, index=self.out.index)],
            axis=1,
        )


def _gap_level(column: str) -> int:
    return int(column.split("_")[2])


def _gap_columns(
    out: pd.DataFrame,
    *,
    side: str,
    present_levels: list[int],
) -> list[str]:
    return [
        f"{side}_gap_{level}_bps"
        for level in present_levels
        if level > 1 and f"{side}_gap_{level}_bps" in out.columns
    ]


def _add_weighted_gap_features(
    builder: _PostOpenV2Builder,
    *,
    side: str,
    depth: int,
    gap_columns: list[str],
) -> None:
    if not gap_columns:
        return
    out = builder.out
    volume_column = ask_volume_col if side == "ask" else bid_volume_col
    pairs = [
        (column, volume_column(_gap_level(column)))
        for column in gap_columns
        if volume_column(_gap_level(column)) in out.columns
    ]
    if pairs:
        gap_values = out[[column for column, _ in pairs]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        gap_weights = out[[column for _, column in pairs]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        builder.add(
            f"postopen_v2_{side}_gap_weighted_{depth}_bps",
            _weighted_mean(gap_values, gap_weights),
        )
    builder.add(
        f"postopen_v2_{side}_gap_max_{depth}_bps",
        out[gap_columns].apply(pd.to_numeric, errors="coerce").max(axis=1),
    )


def _add_depth_state_features(
    builder: _PostOpenV2Builder,
    *,
    depth_levels: tuple[int, ...],
) -> None:
    out = builder.out
    levels = available_depth_levels(out)
    for depth in depth_levels:
        present_levels = [level for level in levels if level <= depth]
        if not present_levels:
            continue
        ask_depth = _sum_present_columns(
            out,
            [ask_volume_col(level) for level in present_levels],
        )
        bid_depth = _sum_present_columns(
            out,
            [bid_volume_col(level) for level in present_levels],
        )
        builder.add(f"postopen_v2_ask_depth_{depth}", ask_depth)
        builder.add(f"postopen_v2_bid_depth_{depth}", bid_depth)
        builder.add(
            f"postopen_v2_depth_imbalance_{depth}",
            safe_divide(bid_depth - ask_depth, bid_depth + ask_depth),
        )
        if "ask_volume_1" in out.columns:
            builder.add(
                f"postopen_v2_ask1_share_depth_{depth}",
                safe_divide(out["ask_volume_1"], ask_depth),
            )
        if "bid_volume_1" in out.columns:
            builder.add(
                f"postopen_v2_bid1_share_depth_{depth}",
                safe_divide(out["bid_volume_1"], bid_depth),
            )
        _add_weighted_gap_features(
            builder,
            side="ask",
            depth=depth,
            gap_columns=_gap_columns(out, side="ask", present_levels=present_levels),
        )
        _add_weighted_gap_features(
            builder,
            side="bid",
            depth=depth,
            gap_columns=_gap_columns(out, side="bid", present_levels=present_levels),
        )


def _add_depth_concentration_features(builder: _PostOpenV2Builder) -> None:
    if builder.has("postopen_v2_ask_depth_3") and builder.has("postopen_v2_ask_depth_10"):
        builder.add(
            "postopen_v2_ask_depth_concentration_3_10",
            safe_divide(
                builder.series("postopen_v2_ask_depth_3"),
                builder.series("postopen_v2_ask_depth_10"),
            ),
        )
    if builder.has("postopen_v2_bid_depth_3") and builder.has("postopen_v2_bid_depth_10"):
        builder.add(
            "postopen_v2_bid_depth_concentration_3_10",
            safe_divide(
                builder.series("postopen_v2_bid_depth_3"),
                builder.series("postopen_v2_bid_depth_10"),
            ),
        )
    if builder.has("postopen_v2_bid_depth_concentration_3_10") and builder.has(
        "postopen_v2_ask_depth_concentration_3_10"
    ):
        builder.add(
            "postopen_v2_depth_concentration_imbalance_3_10",
            builder.series("postopen_v2_bid_depth_concentration_3_10")
            - builder.series("postopen_v2_ask_depth_concentration_3_10"),
        )


def _add_gap_shape_features(
    builder: _PostOpenV2Builder,
    *,
    depth_levels: tuple[int, ...],
) -> None:
    out = builder.out
    for depth in depth_levels:
        ask_gap = f"ask_gap_{depth}_bps"
        bid_gap = f"bid_gap_{depth}_bps"
        if depth > 1 and ask_gap in out.columns:
            builder.add(
                f"postopen_v2_ask_gap_slope_{depth}_bps",
                safe_divide(out[ask_gap], depth - 1),
            )
        if depth > 1 and bid_gap in out.columns:
            builder.add(
                f"postopen_v2_bid_gap_slope_{depth}_bps",
                safe_divide(out[bid_gap], depth - 1),
            )
    if "ask_gap_2_bps" in out.columns and "ask_gap_10_bps" in out.columns:
        builder.add(
            "postopen_v2_ask_gap_curve_2_10_bps",
            _numeric_series(out["ask_gap_10_bps"]) - _numeric_series(out["ask_gap_2_bps"]),
        )
    if "bid_gap_2_bps" in out.columns and "bid_gap_10_bps" in out.columns:
        builder.add(
            "postopen_v2_bid_gap_curve_2_10_bps",
            _numeric_series(out["bid_gap_10_bps"]) - _numeric_series(out["bid_gap_2_bps"]),
        )


def _add_trade_impact_features(builder: _PostOpenV2Builder) -> None:
    out = builder.out
    for ticks in (1, 3, 10, 30):
        volume_col_name = f"volume_diff_{ticks}t"
        turnover_col_name = f"turnover_diff_{ticks}t"
        vwap_col_name = f"trade_vwap_{ticks}t"
        if volume_col_name in out.columns:
            volume = _numeric_series(out[volume_col_name])
            builder.add(
                f"postopen_v2_trade_volume_to_ask1_{ticks}t",
                safe_divide(volume, _column_values(out, "ask_volume_1")),
            )
            builder.add(
                f"postopen_v2_trade_volume_to_bid1_{ticks}t",
                safe_divide(volume, _column_values(out, "bid_volume_1")),
            )
            builder.add(
                f"postopen_v2_trade_volume_to_ask_depth10_{ticks}t",
                safe_divide(volume, _column_values(out, "ask_depth_10")),
            )
            builder.add(
                f"postopen_v2_trade_volume_to_bid_depth10_{ticks}t",
                safe_divide(volume, _column_values(out, "bid_depth_10")),
            )
        if turnover_col_name in out.columns:
            turnover = _numeric_series(out[turnover_col_name])
            builder.add(
                f"postopen_v2_trade_turnover_to_depth_notional_{ticks}t",
                safe_divide(
                    turnover,
                    _column_values(out, "ask_depth_10") * _column_values(out, "ask_price_1"),
                ),
            )
        if vwap_col_name in out.columns:
            vwap = _numeric_series(out[vwap_col_name])
            if "mid_price" in out.columns:
                builder.add(
                    f"postopen_v2_trade_vwap_vs_mid_{ticks}t_bps",
                    safe_divide(vwap - _numeric_series(out["mid_price"]), out["mid_price"])
                    * 10_000,
                )
            if "ask_price_1" in out.columns:
                builder.add(
                    f"postopen_v2_trade_vwap_vs_ask1_{ticks}t_bps",
                    safe_divide(
                        vwap - _numeric_series(out["ask_price_1"]),
                        out["ask_price_1"],
                    )
                    * 10_000,
                )


def _trajectory_columns(builder: _PostOpenV2Builder) -> list[str]:
    candidates = (
        "ask_volume_1",
        "bid_volume_1",
        "ask_depth_10",
        "bid_depth_10",
        "depth_imbalance_1",
        "depth_imbalance_10",
        "spread_bps",
        "mid_price",
        "ask_price_1",
        "bid_price_1",
        "volume",
        "turnover",
        "postopen_v2_ask_depth_3",
        "postopen_v2_bid_depth_3",
        "postopen_v2_depth_imbalance_3",
        "postopen_v2_ask_depth_concentration_3_10",
        "postopen_v2_bid_depth_concentration_3_10",
    )
    return [column for column in candidates if builder.has(column)]


def _add_trajectory_features(
    builder: _PostOpenV2Builder,
    *,
    windows: tuple[int, ...],
) -> None:
    for column in _trajectory_columns(builder):
        values = builder.numeric(column)
        first = builder.first(column)
        builder.add(f"postopen_v2_{column}_from_open_diff", values - first)
        builder.add(
            f"postopen_v2_{column}_from_open_rel",
            safe_divide(values - first, first.abs()),
        )
        for window in windows:
            lagged = builder.shifted(column, window)
            diff = values - lagged
            builder.add(f"postopen_v2_{column}_diff_{window}m", diff)
            builder.add(
                f"postopen_v2_{column}_rel_{window}m",
                safe_divide(diff, lagged.abs()),
            )


def _add_queue_response_features(builder: _PostOpenV2Builder) -> None:
    out = builder.out
    if builder.has("postopen_v2_ask_volume_1_diff_1m") and "volume_diff_1t" in out.columns:
        builder.add(
            "postopen_v2_ask1_queue_replenish_vs_trade_1m",
            safe_divide(
                builder.series("postopen_v2_ask_volume_1_diff_1m"),
                _column_values(out, "volume_diff_1t").abs(),
            ),
        )
    if builder.has("postopen_v2_bid_volume_1_diff_1m") and "volume_diff_1t" in out.columns:
        builder.add(
            "postopen_v2_bid1_queue_replenish_vs_trade_1m",
            safe_divide(
                builder.series("postopen_v2_bid_volume_1_diff_1m"),
                _column_values(out, "volume_diff_1t").abs(),
            ),
        )
    if builder.has("postopen_v2_spread_bps_diff_1m"):
        builder.add(
            "postopen_v2_spread_compression_1m",
            -builder.series("postopen_v2_spread_bps_diff_1m"),
        )

    for window in (1, 2, 3, 5):
        trade_column = f"postopen_v2_volume_diff_{window}m"
        if not builder.has(trade_column):
            continue
        trade_volume = builder.series(trade_column).abs()
        ask_queue_column = f"postopen_v2_ask_volume_1_diff_{window}m"
        bid_queue_column = f"postopen_v2_bid_volume_1_diff_{window}m"
        if builder.has(ask_queue_column):
            builder.add(
                f"postopen_v2_queue_ask1_replenish_vs_trade_{window}m",
                safe_divide(builder.series(ask_queue_column), trade_volume),
            )
        if builder.has(bid_queue_column):
            builder.add(
                f"postopen_v2_queue_bid1_replenish_vs_trade_{window}m",
                safe_divide(builder.series(bid_queue_column), trade_volume),
            )
        if builder.has("postopen_v2_bid_depth_10"):
            bid_depth_diff = builder.numeric("postopen_v2_bid_depth_10") - builder.shifted(
                "postopen_v2_bid_depth_10",
                window,
            )
            builder.add(
                f"postopen_v2_queue_bid_depth10_replenish_vs_trade_{window}m",
                safe_divide(bid_depth_diff, trade_volume),
            )
        spread_column = f"postopen_v2_spread_bps_diff_{window}m"
        if builder.has(spread_column):
            builder.add(
                f"postopen_v2_queue_spread_compression_{window}m",
                -builder.series(spread_column),
            )


def add_postopen_v2_decision_features(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (1, 2, 3, 5),
    depth_levels: tuple[int, ...] = (3, 5, 10),
) -> pd.DataFrame:
    """Add richer post-open state, trajectory, and trade-impact features.

    The input is still a decision-row frame, so every feature uses only the
    current row and earlier decision rows within the same symbol/day.
    """

    builder = _PostOpenV2Builder(frame)
    _add_depth_state_features(builder, depth_levels=depth_levels)
    _add_depth_concentration_features(builder)
    _add_gap_shape_features(builder, depth_levels=depth_levels)
    _add_trade_impact_features(builder)
    _add_trajectory_features(builder, windows=windows)
    _add_queue_response_features(builder)
    return builder.finish()


def _rolling_linear_slope(values: pd.Series, window: int) -> pd.Series:
    if window < 2:
        return pd.Series(np.nan, index=values.index, dtype="float64")

    values = _numeric_series(values)
    rolling = values.rolling(window=window, min_periods=window)
    rolling_sum = rolling.sum()
    valid_window = rolling.count().eq(float(window))

    weighted_sum = pd.Series(0.0, index=values.index, dtype="float64")
    for offset in range(window - 1):
        weight = float(window - 1 - offset)
        weighted_sum = weighted_sum + values.shift(offset) * weight

    x_mean = float(window - 1) / 2.0
    denom = float(window * (window**2 - 1)) / 12.0
    slope = (weighted_sum - x_mean * rolling_sum) / denom
    return slope.where(valid_window)


def _grouped_rolling_linear_slope(
    values: pd.Series,
    group_keys: list[pd.Series],
    window: int,
) -> pd.Series:
    if window < 2:
        return pd.Series(np.nan, index=values.index, dtype="float64")

    values = _numeric_series(values)
    grouped = values.groupby(group_keys, sort=False)
    rolling = grouped.rolling(window=window, min_periods=window)
    group_levels = list(range(len(group_keys)))
    rolling_sum = rolling.sum().reset_index(level=group_levels, drop=True)
    valid_window = rolling.count().reset_index(level=group_levels, drop=True).eq(float(window))

    weighted_sum = pd.Series(0.0, index=values.index, dtype="float64")
    for offset in range(window - 1):
        weight = float(window - 1 - offset)
        shifted = values if offset == 0 else grouped.shift(offset)
        weighted_sum = weighted_sum + shifted * weight

    x_mean = float(window - 1) / 2.0
    denom = float(window * (window**2 - 1)) / 12.0
    slope = (weighted_sum - x_mean * rolling_sum) / denom
    return slope.where(valid_window)


def _normalized_row_clock(frame: pd.DataFrame, time_col: str) -> pd.Series:
    if "decision_time" in frame.columns:
        raw = frame["decision_time"].astype(str)
        extracted = raw.str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False).fillna("")
        return extracted.map(lambda value: normalize_clock_time(value) if value else "")
    return pd.to_datetime(frame[time_col], errors="coerce").dt.strftime("%H:%M:%S").fillna("")


def add_path_shape_confirmation_features(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (2, 3, 5),
    prefix: str = "path_shape_",
) -> pd.DataFrame:
    """Add causal post-open path shape and confirmation features."""

    out = ensure_timestamp_columns(frame)
    time_col = (
        "decision_target_timestamp" if "decision_target_timestamp" in out.columns else "timestamp"
    )
    out = out.sort_values(["date", "symbol", time_col]).reset_index(drop=True)
    group_keys = [out["date"], out["symbol"]]
    group_cols = ["date", "symbol"]
    new_columns: dict[str, pd.Series] = {}

    if "mid_price" in out.columns:
        mid = _numeric_series(out["mid_price"])
        per_symbol = mid.groupby(group_keys, sort=False)
        expanding_high = per_symbol.cummax()
        expanding_low = per_symbol.cummin()
        first_mid = per_symbol.transform("first")
        mid_move_1 = mid - per_symbol.shift(1)
        mid_move_1_bps = pd.Series(
            safe_divide(mid_move_1, per_symbol.shift(1)) * 10_000,
            index=out.index,
        )
        positive_move = mid_move_1_bps.gt(0.0).astype("float64")

        new_columns[f"{prefix}mid_drawdown_from_open_high_bps"] = (
            safe_divide(mid - expanding_high, expanding_high) * 10_000
        )
        new_columns[f"{prefix}mid_recovery_from_open_low_bps"] = (
            safe_divide(mid - expanding_low, expanding_low) * 10_000
        )
        new_columns[f"{prefix}mid_from_first_bps"] = safe_divide(mid - first_mid, first_mid) * 10_000
        new_columns[f"{prefix}mid_new_high_flag"] = mid.ge(expanding_high).astype("int8")
        new_columns[f"{prefix}mid_new_low_flag"] = mid.le(expanding_low).astype("int8")
        new_columns[f"{prefix}return_positive_fraction"] = (
            positive_move.groupby(group_keys, sort=False)
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        for window in windows:
            if window < 2:
                continue
            prior = per_symbol.shift(window - 1)
            move_bps = pd.Series(
                safe_divide(mid - prior, prior) * 10_000,
                index=out.index,
            )
            new_columns[f"{prefix}mid_move_{window}m_bps"] = move_bps
            new_columns[f"{prefix}return_accel_1m_vs_{window}m"] = (
                mid_move_1_bps - safe_divide(move_bps, float(window - 1))
            )
            roll_high = (
                mid.groupby(group_keys, sort=False)
                .rolling(window=window, min_periods=window)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )
            roll_low = (
                mid.groupby(group_keys, sort=False)
                .rolling(window=window, min_periods=window)
                .min()
                .reset_index(level=[0, 1], drop=True)
            )
            new_columns[f"{prefix}mid_position_roll{window}"] = safe_divide(
                mid - roll_low,
                roll_high - roll_low,
            )

        if "spread_bps" in out.columns:
            spread = _numeric_series(out["spread_bps"])
            spread_change = spread - spread.groupby(group_keys, sort=False).shift(1)
            new_columns[f"{prefix}spread_compress_after_upmove"] = (
                -spread_change * positive_move
            )
            new_columns[f"{prefix}spread_widen_after_upmove"] = spread_change * positive_move

        if "depth_imbalance_10" in out.columns:
            imbalance = _numeric_series(out["depth_imbalance_10"])
            for window in windows:
                if window < 2:
                    continue
                slope = _grouped_rolling_linear_slope(imbalance, group_keys, window)
                new_columns[f"{prefix}imbalance_slope_roll{window}"] = slope
                new_columns[f"{prefix}imbalance_slope_confirm_return_roll{window}"] = (
                    slope * move_positive_over_window(out, group_cols, mid, window)
                )

        if {"bid_depth_10", "ask_depth_10"}.issubset(out.columns):
            bid_depth = _numeric_series(out["bid_depth_10"])
            ask_depth = _numeric_series(out["ask_depth_10"])
            bid_change = bid_depth - bid_depth.groupby(group_keys, sort=False).shift(1)
            ask_change = ask_depth - ask_depth.groupby(group_keys, sort=False).shift(1)
            new_columns[f"{prefix}bid_depth_support_after_upmove"] = bid_change * positive_move
            new_columns[f"{prefix}ask_depth_fade_after_upmove"] = (-ask_change) * positive_move
            new_columns[f"{prefix}depth_support_imbalance_after_upmove"] = (
                bid_change - ask_change
            ) * positive_move

    return pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)


def move_positive_over_window(
    frame: pd.DataFrame,
    group_cols: list[str],
    values: pd.Series,
    window: int,
) -> pd.Series:
    group_keys = [frame[column] for column in group_cols]
    prior = values.groupby(group_keys, sort=False).shift(window - 1)
    return (values - prior).gt(0.0).astype("float64")


def add_historical_same_minute_surprise_features(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    windows: tuple[int, ...] = (20, 60),
    min_periods: int = 10,
    modes: tuple[str, ...] = ("zscore", "ratio"),
    prefix: str = "hist_surprise_",
) -> pd.DataFrame:
    """Add symbol x decision-clock historical surprise features using prior dates only."""

    out = ensure_timestamp_columns(frame)
    if "symbol" not in out.columns or "date" not in out.columns:
        raise ValueError("historical surprise features require date and symbol columns")
    time_col = (
        "decision_target_timestamp" if "decision_target_timestamp" in out.columns else "timestamp"
    )
    clock = _normalized_row_clock(out, time_col)
    work = out.assign(_hist_clock=clock, _hist_orig_order=np.arange(len(out)))
    work = work.sort_values(["symbol", "_hist_clock", "date", time_col]).reset_index(drop=True)
    group_keys = [work["symbol"], work["_hist_clock"]]
    modes = tuple(mode.strip().lower() for mode in modes if mode.strip())
    valid_modes = {"zscore", "ratio", "diff"}
    unknown_modes = sorted(set(modes) - valid_modes)
    if unknown_modes:
        raise ValueError(f"unknown historical surprise modes: {unknown_modes}")

    selected = [
        column
        for column in columns
        if column in work.columns and pd.api.types.is_numeric_dtype(work[column])
    ]
    if not selected:
        return out.copy()

    min_periods = max(2, int(min_periods))
    new_sorted: dict[str, pd.Series] = {}
    for column in selected:
        values = pd.to_numeric(work[column], errors="coerce").astype("float64")
        past = values.groupby(group_keys, sort=False).shift(1)
        for window in windows:
            if window < 2:
                continue
            rolling = past.groupby(group_keys, sort=False).rolling(
                window=int(window),
                min_periods=min_periods,
            )
            mean = rolling.mean().reset_index(level=[0, 1], drop=True)
            std = rolling.std().reset_index(level=[0, 1], drop=True)
            if "zscore" in modes:
                new_sorted[f"{prefix}{column}_{window}d_zscore"] = pd.Series(
                    safe_divide(values - mean, std),
                    index=work.index,
                )
            if "ratio" in modes:
                new_sorted[f"{prefix}{column}_{window}d_ratio"] = pd.Series(
                    safe_divide(values, mean),
                    index=work.index,
                )
            if "diff" in modes:
                new_sorted[f"{prefix}{column}_{window}d_diff"] = values - mean

    if not new_sorted:
        return out.copy()
    features = pd.DataFrame(new_sorted, index=work.index)
    features["_hist_orig_order"] = work["_hist_orig_order"].to_numpy()
    features = features.sort_values("_hist_orig_order").drop(columns="_hist_orig_order")
    features.index = out.index
    return pd.concat([out.copy(), features], axis=1)


def build_preopen_features(
    ticks: pd.DataFrame,
    *,
    volume_col: str = "volume",
    turnover_col: str = "turnover",
) -> pd.DataFrame:
    preopen = filter_time_range(
        ticks,
        PREOPEN_START,
        PREOPEN_END,
        include_end=True,
    )
    if preopen.empty:
        return pd.DataFrame(columns=["date", "symbol"])

    price_col = None
    for candidate in ("last_price", "mid_price", "ask_price_1"):
        if candidate in preopen.columns:
            price_col = candidate
            break

    agg_spec: dict[str, tuple[str, str]] = {}
    if volume_col in preopen.columns:
        agg_spec["preopen_volume"] = (volume_col, "max")
    if turnover_col in preopen.columns:
        agg_spec["preopen_turnover"] = (turnover_col, "max")
    if price_col is not None:
        agg_spec["preopen_last_price"] = (price_col, "last")
        agg_spec["preopen_price_min"] = (price_col, "min")
        agg_spec["preopen_price_max"] = (price_col, "max")
    if "depth_imbalance_10" in preopen.columns:
        agg_spec["preopen_depth_imbalance_10"] = ("depth_imbalance_10", "last")

    if not agg_spec:
        return preopen[["date", "symbol"]].drop_duplicates()
    return preopen.groupby(["date", "symbol"], as_index=False).agg(**agg_spec)


def build_feature_frame(
    ticks: pd.DataFrame,
    *,
    include_preopen: bool = True,
    volume_col: str = "volume",
    turnover_col: str = "turnover",
    volume_unit_multiplier: float = 1.0,
) -> pd.DataFrame:
    out = ensure_timestamp_columns(ticks)
    out = out.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)
    out = add_order_book_features(out)
    out = add_trade_features(
        out,
        volume_col=volume_col,
        turnover_col=turnover_col,
        volume_unit_multiplier=volume_unit_multiplier,
    )
    out = add_momentum_features(out)

    if include_preopen:
        preopen = build_preopen_features(
            out,
            volume_col=volume_col,
            turnover_col=turnover_col,
        )
        if not preopen.empty:
            out = out.merge(preopen, on=["date", "symbol"], how="left")
            if "prev_close" in out.columns and "preopen_last_price" in out.columns:
                out["preopen_return_vs_prev_close"] = safe_divide(
                    out["preopen_last_price"] - out["prev_close"],
                    out["prev_close"],
                )
            if {
                "preopen_price_min",
                "preopen_price_max",
                "preopen_last_price",
            }.issubset(out.columns):
                price_range = out["preopen_price_max"] - out["preopen_price_min"]
                reference_price = (
                    out["prev_close"] if "prev_close" in out.columns else out["preopen_last_price"]
                )
                out["auction_price_range_bps"] = safe_divide(price_range, reference_price) * 10_000
                out["auction_last_position_in_range"] = safe_divide(
                    out["preopen_last_price"] - out["preopen_price_min"],
                    price_range,
                )
    return out
