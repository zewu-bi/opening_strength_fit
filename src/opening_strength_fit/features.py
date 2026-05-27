from __future__ import annotations

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
        out["ask1_to_limit_up_bps"] = (
            safe_divide(limit_up - ask1, ask1)
            * 10_000
        )

    for level in levels:
        bid_price = bid_price_col(level)
        ask_price = ask_price_col(level)
        if level > 1 and bid_price in out.columns and "bid_price_1" in out.columns:
            bid1 = _numeric_series(out["bid_price_1"])
            bid_level = _numeric_series(out[bid_price])
            out[f"bid_gap_{level}_bps"] = (
                safe_divide(bid1 - bid_level, bid1)
                * 10_000
            )
        if level > 1 and ask_price in out.columns and "ask_price_1" in out.columns:
            ask1 = _numeric_series(out["ask_price_1"])
            ask_level = _numeric_series(out[ask_price])
            out[f"ask_gap_{level}_bps"] = (
                safe_divide(ask_level - ask1, ask1)
                * 10_000
            )
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
        "decision_target_timestamp"
        if "decision_target_timestamp" in out.columns
        else "timestamp"
    )
    out = out.sort_values(["date", "symbol", time_col]).reset_index(drop=True)
    group = out.groupby(["date", "symbol"], sort=False)

    timestamp = pd.to_datetime(out[time_col], errors="coerce")
    open_timestamp = pd.to_datetime(
        out["date"].astype(str) + " 09:30:00",
        errors="coerce",
    )
    out["postopen_minutes_since_0930"] = (
        (timestamp - open_timestamp).dt.total_seconds() / 60.0
    )

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
    if (
        "postopen_bid1_depth_share" in out.columns
        and "postopen_ask1_depth_share" in out.columns
    ):
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
        out["postopen_spread_x_imbalance_1"] = (
            _numeric_series(out["spread_bps"]) * _numeric_series(out["depth_imbalance_1"])
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
    if (
        builder.has("postopen_v2_bid_depth_concentration_3_10")
        and builder.has("postopen_v2_ask_depth_concentration_3_10")
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
                    _column_values(out, "ask_depth_10")
                    * _column_values(out, "ask_price_1"),
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
    return out
