from __future__ import annotations

import pandas as pd

from opening_strength_fit.feature_utils import (
    _column_values,
    _numeric_series,
    _sum_present_columns,
    _weighted_mean,
    safe_divide,
)
from opening_strength_fit.schema import (
    ask_volume_col,
    available_depth_levels,
    bid_volume_col,
    ensure_timestamp_columns,
)

POSTOPEN_DYNAMIC_COLUMNS = (
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
)


def _lagged_by_minutes(
    frame: pd.DataFrame,
    values: pd.Series,
    timestamp: pd.Series,
    minutes: int,
) -> pd.Series:
    source_index = pd.MultiIndex.from_arrays(
        [frame["date"], frame["symbol"], timestamp],
        names=["date", "symbol", "timestamp"],
    )
    if source_index.has_duplicates:
        raise ValueError("post-open features require unique date/symbol/decision timestamps")
    target_index = pd.MultiIndex.from_arrays(
        [frame["date"], frame["symbol"], timestamp - pd.Timedelta(minutes=minutes)],
        names=source_index.names,
    )
    lookup = pd.Series(values.to_numpy(), index=source_index)
    return pd.Series(lookup.reindex(target_index).to_numpy(), index=frame.index)


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
    timestamp = pd.to_datetime(out[time_col], errors="coerce")
    open_timestamp = pd.to_datetime(
        out["date"].astype(str) + " 09:30:00",
        errors="coerce",
    )
    out["postopen_minutes_since_0930"] = (timestamp - open_timestamp).dt.total_seconds() / 60.0

    for column in POSTOPEN_DYNAMIC_COLUMNS:
        if column not in out.columns:
            continue
        values = _numeric_series(out[column])
        for window in windows:
            lagged = _lagged_by_minutes(out, values, timestamp, window)
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
        self.timestamp = pd.to_datetime(self.out[self.time_col], errors="coerce")
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
        return _lagged_by_minutes(self.out, self.numeric(column), self.timestamp, window)

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
        *POSTOPEN_DYNAMIC_COLUMNS,
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
    reference_name: str,
) -> None:
    reference_name = reference_name.strip().lower()
    if reference_name not in {"open", "start"}:
        raise ValueError("post-open trajectory reference_name must be 'open' or 'start'")
    for column in _trajectory_columns(builder):
        values = builder.numeric(column)
        first = builder.first(column)
        builder.add(f"postopen_v2_{column}_from_{reference_name}_diff", values - first)
        builder.add(
            f"postopen_v2_{column}_from_{reference_name}_rel",
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
    reference_name: str = "open",
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
    _add_trajectory_features(
        builder,
        windows=windows,
        reference_name=reference_name,
    )
    _add_queue_response_features(builder)
    return builder.finish()
