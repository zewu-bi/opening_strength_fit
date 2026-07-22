from __future__ import annotations

from collections import Counter
from importlib import import_module

import pandas as pd

from opening_strength_fit.optimization_direction_data import (
    DEFAULT_DIRECTIONS,
    NEXT_CLOSE_CAPITAL_DIVISOR,
    RETURN_BPS_DENOMINATOR,
    DirectionSpec,
)
from opening_strength_fit.pool_internal_plot_svg import (
    PLOT_COLORS,
)

DIRECTION_COLORS = {
    "baseline": "#1f2937",
    "baseline_pool_l": "#1f2937",
    "baseline_universe": "#7f7f7f",
    "market": "#7f7f7f",
    "background": "#ff7f0e",
}

DISPLAY_LABELS = {
    "market": "market",
    "background": "pool",
    "xs_relative": "xs",
    "hist_surprise": "deviation",
    "hist_path": "deviation+path",
    "hist_path_zscore": "deviation+path zscore",
    "rank_centered": "deviation+path rank",
    "path_shape": "path",
    "scale_norm": "scale",
    "clock_segment": "clock",
}

DEFAULT_PLOT_DIRECTION_KEYS = ("hist_surprise", "path_shape")
MIN_PLOT_DIRECTIONS = 1
MAX_PLOT_DIRECTIONS = 3
BPS_PER_PERCENT = 100.0
AUTO_COLOR_SEQUENCE = (
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#17becf",
    "#bcbd22",
    "#8c564b",
    "#e377c2",
)

CUMULATIVE_PERCENT_DISPLAY_COLUMNS = {
    "next_cumulative_net_return_bps": "next_cumulative_net_return_pct",
    "next_cumulative_vs_baseline_bps": "next_cumulative_vs_baseline_pct",
    "next_cumulative_alpha_bps": "next_cumulative_alpha_pct",
    "next_cumulative_alpha_vs_market_bps": "next_cumulative_alpha_vs_market_pct",
    "next_cumulative_internal_excess_return_bps": ("next_cumulative_internal_excess_return_pct"),
}
CUMULATIVE_NET_RETURN_PANEL_TITLE = "扣除手续费累和收益"
CUMULATIVE_MARKET_ALPHA_PANEL_TITLE = "对比全A股市场平均alpha"
CUMULATIVE_POOL_L_EXCESS_PANEL_TITLE = "相对 pool_L 累和超额"
CUMULATIVE_MODE_TOP100 = "top100"
CUMULATIVE_MODE_CAPACITY = "capacity"
CUMULATIVE_MODE_REALISTIC = "realistic"
CUMULATIVE_MODES = (CUMULATIVE_MODE_TOP100, CUMULATIVE_MODE_CAPACITY, CUMULATIVE_MODE_REALISTIC)
CUMULATIVE_RELATIVE_MODE_MARKET = "market"
CUMULATIVE_RELATIVE_MODE_POOL_L = "pool_l"
CUMULATIVE_RELATIVE_MODES = (
    CUMULATIVE_RELATIVE_MODE_MARKET,
    CUMULATIVE_RELATIVE_MODE_POOL_L,
)


def default_plot_directions(
    directions: tuple[DirectionSpec, ...] = DEFAULT_DIRECTIONS,
) -> tuple[DirectionSpec, ...]:
    selected = tuple(
        direction for direction in directions if direction.key in DEFAULT_PLOT_DIRECTION_KEYS
    )
    missing_keys = sorted(
        set(DEFAULT_PLOT_DIRECTION_KEYS) - {direction.key for direction in selected}
    )
    if missing_keys:
        raise ValueError(f"missing default plot directions: {missing_keys}")
    return selected


def validate_plot_directions(directions: tuple[DirectionSpec, ...]) -> tuple[DirectionSpec, ...]:
    if not MIN_PLOT_DIRECTIONS <= len(directions) <= MAX_PLOT_DIRECTIONS:
        raise ValueError(
            "acceptance plots require 1-3 comparison models besides baseline; "
            f"got {len(directions)}"
        )
    reserved_keys = {"baseline", "baseline_pool_l", "baseline_universe", "market", "background"}
    key_counts = Counter(direction.key for direction in directions)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    if duplicate_keys:
        raise ValueError(f"duplicate direction keys: {duplicate_keys}")
    reserved_direction_keys = sorted(
        direction.key for direction in directions if direction.key in reserved_keys
    )
    if reserved_direction_keys:
        raise ValueError(f"direction keys are reserved: {reserved_direction_keys}")
    return directions


def cumulative_plot_series(
    model_series: tuple[str, ...],
    *,
    relative_mode: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the fixed top-panel series and mode-specific lower-panel series."""

    top_series = ("market", "background", *model_series)
    if relative_mode == CUMULATIVE_RELATIVE_MODE_POOL_L:
        return top_series, model_series
    if relative_mode == CUMULATIVE_RELATIVE_MODE_MARKET:
        return top_series, ("background", *model_series)
    raise ValueError(
        f"unknown cumulative relative mode {relative_mode!r}; expected {CUMULATIVE_RELATIVE_MODES}"
    )


def combine_overlay_acceptance_data(
    short_universe_data: pd.DataFrame,
    next_pool_data: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["test_month", "variant", "pool", "pool_label"]
    return short_universe_data[key_columns + ["short_rank_ic"]].merge(
        next_pool_data[key_columns + ["next_internal_excess_bps"]],
        on=key_columns,
        how="inner",
    )


def combine_net_alpha_cumulative_data(
    realized_cumulative_output: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["pool", "pool_label", "week_start", "variant"]
    value_columns = [
        "decision_groups",
        "clocks",
        "candidate_rows",
        "selected_rows",
        "capacity_decision_groups",
        "capacity_daily_capital_fraction",
        "capacity_total_notional",
        "capacity_decision_notional",
        "capacity_fee_bps_per_trade",
        "capacity_audit_fee_bps_per_trade",
        "capacity_additional_fee_bps_per_trade",
        "capacity_additional_fee_bps",
        "pool_next_mean_bps",
        "pool_turnover",
        "pool_turnover_source",
        "pool_fee_bps",
        "pool_next_net_return_bps",
        "pool_next_capital_net_return_bps",
        "pool_next_cumulative_net_return_bps",
        "pool_next_net_pnl",
        "pool_next_cumulative_net_pnl",
        "selected_next_mean_bps",
        "selected_turnover",
        "selected_fee_bps",
        "next_internal_excess_bps",
        "next_capital_internal_excess_bps",
        "next_cumulative_internal_excess_return_bps",
        "next_internal_excess_pnl",
        "next_cumulative_internal_excess_pnl",
        "fee_bps",
        "next_net_return_bps",
        "next_capital_net_return_bps",
        "next_cumulative_net_return_bps",
        "next_net_pnl",
        "next_cumulative_net_pnl",
    ]
    columns = key_columns + [
        column for column in value_columns if column in realized_cumulative_output
    ]
    combined = realized_cumulative_output[columns].copy()
    combined["next_alpha_bps"] = (
        combined["next_net_return_bps"] - combined["pool_next_net_return_bps"]
    )
    combined["next_cumulative_alpha_bps"] = (
        combined["next_cumulative_net_return_bps"] - combined["pool_next_cumulative_net_return_bps"]
    )
    combined["next_capital_alpha_bps"] = (
        combined["next_capital_net_return_bps"] - combined["pool_next_capital_net_return_bps"]
    )
    if {"next_net_pnl", "pool_next_net_pnl"}.issubset(combined.columns):
        combined["next_alpha_pnl"] = combined["next_net_pnl"] - combined["pool_next_net_pnl"]
    if {"next_cumulative_net_pnl", "pool_next_cumulative_net_pnl"}.issubset(combined.columns):
        combined["next_cumulative_alpha_pnl"] = (
            combined["next_cumulative_net_pnl"] - combined["pool_next_cumulative_net_pnl"]
        )
    return combined


def add_cumulative_percent_display_columns(cumulative_data: pd.DataFrame) -> pd.DataFrame:
    data = cumulative_data.copy()
    for source, target in CUMULATIVE_PERCENT_DISPLAY_COLUMNS.items():
        data[target] = pd.to_numeric(data[source], errors="coerce") / BPS_PER_PERCENT
    return data


def add_background_cumulative_data(
    cumulative_data: pd.DataFrame,
    *,
    baseline_key: str,
) -> pd.DataFrame:
    baseline = cumulative_data.loc[cumulative_data["pool"].astype(str).eq(baseline_key)].copy()
    if baseline.empty:
        raise ValueError(f"cumulative data has no baseline rows for {baseline_key!r}")
    baseline["week_start"] = pd.to_datetime(baseline["week_start"], errors="coerce")
    baseline = baseline.dropna(subset=["week_start"]).sort_values("week_start")
    background = baseline.copy()
    background_label = DISPLAY_LABELS.get("background", "background")
    background["pool"] = "background"
    background["pool_label"] = background_label
    background["variant"] = background_label
    background["selected_next_mean_bps"] = pd.NA
    background["selected_turnover"] = pd.NA
    background["selected_fee_bps"] = pd.NA
    for column in (
        "capacity_fee_bps_per_trade",
        "capacity_audit_fee_bps_per_trade",
        "capacity_additional_fee_bps_per_trade",
        "capacity_additional_fee_bps",
    ):
        if column in background.columns:
            background[column] = pd.NA
    background["next_internal_excess_bps"] = pd.NA
    background["fee_bps"] = background["pool_fee_bps"]
    background["next_net_return_bps"] = background["pool_next_net_return_bps"]
    background["next_capital_net_return_bps"] = background["pool_next_capital_net_return_bps"]
    background["next_cumulative_net_return_bps"] = background["pool_next_cumulative_net_return_bps"]
    if "pool_next_net_pnl" in background.columns:
        background["next_net_pnl"] = background["pool_next_net_pnl"]
    if "pool_next_cumulative_net_pnl" in background.columns:
        background["next_cumulative_net_pnl"] = background["pool_next_cumulative_net_pnl"]
    background["next_alpha_bps"] = pd.NA
    background["next_cumulative_alpha_bps"] = pd.NA
    background["next_capital_internal_excess_bps"] = 0.0
    background["next_cumulative_internal_excess_return_bps"] = 0.0
    if "next_alpha_pnl" in background.columns:
        background["next_alpha_pnl"] = pd.NA
    if "next_cumulative_alpha_pnl" in background.columns:
        background["next_cumulative_alpha_pnl"] = pd.NA
    if "next_internal_excess_pnl" in background.columns:
        background["next_internal_excess_pnl"] = 0.0
    if "next_cumulative_internal_excess_pnl" in background.columns:
        background["next_cumulative_internal_excess_pnl"] = 0.0
    background["next_internal_excess_vs_baseline_bps"] = pd.NA
    background["next_cumulative_internal_excess_vs_baseline_bps"] = pd.NA
    background["next_cumulative_vs_baseline_bps"] = pd.NA
    out = cumulative_data.copy()
    if "next_cumulative_vs_baseline_bps" not in out.columns:
        out["next_cumulative_vs_baseline_bps"] = pd.NA
    if "next_cumulative_internal_excess_vs_baseline_bps" not in out.columns:
        out["next_cumulative_internal_excess_vs_baseline_bps"] = pd.NA
    return pd.concat([out, background], ignore_index=True)


def add_market_cumulative_data(
    cumulative_data: pd.DataFrame,
    *,
    source_key: str = "baseline_universe",
    market_key: str = "market",
) -> pd.DataFrame:
    source = cumulative_data.loc[cumulative_data["pool"].astype(str).eq(source_key)].copy()
    if source.empty:
        raise ValueError(f"cumulative data has no market source rows for {source_key!r}")
    source["week_start"] = pd.to_datetime(source["week_start"], errors="coerce")
    source = source.dropna(subset=["week_start"]).sort_values("week_start")
    if source.empty:
        raise ValueError(f"cumulative data has no dated market source rows for {source_key!r}")

    market = source.copy()
    market_label = DISPLAY_LABELS.get(market_key, market_key)
    raw_market_next_bps = pd.to_numeric(market["pool_next_mean_bps"], errors="coerce")
    if "capacity_daily_capital_fraction" in market.columns:
        capital_fraction = pd.to_numeric(
            market["capacity_daily_capital_fraction"],
            errors="coerce",
        ).fillna(1.0 / NEXT_CLOSE_CAPITAL_DIVISOR)
    else:
        capital_fraction = 1.0 / NEXT_CLOSE_CAPITAL_DIVISOR
    market["pool"] = market_key
    market["pool_label"] = market_label
    market["variant"] = market_label
    market["selected_next_mean_bps"] = pd.NA
    market["selected_turnover"] = pd.NA
    market["selected_fee_bps"] = pd.NA
    market["pool_turnover"] = pd.NA
    market["pool_turnover_source"] = "universe_pool_next_mean_bps"
    market["pool_fee_bps"] = 0.0
    market["fee_bps"] = 0.0
    market["next_internal_excess_bps"] = pd.NA
    market["next_net_return_bps"] = raw_market_next_bps
    market["pool_next_net_return_bps"] = raw_market_next_bps
    market["next_capital_net_return_bps"] = raw_market_next_bps * capital_fraction
    market["pool_next_capital_net_return_bps"] = market["next_capital_net_return_bps"]
    market["next_cumulative_net_return_bps"] = market["next_capital_net_return_bps"].cumsum()
    market["pool_next_cumulative_net_return_bps"] = market["next_cumulative_net_return_bps"]
    if "capacity_total_notional" in market.columns:
        total_notional = pd.to_numeric(market["capacity_total_notional"], errors="coerce")
        market["next_net_pnl"] = (
            market["next_capital_net_return_bps"] / RETURN_BPS_DENOMINATOR * total_notional
        )
        market["pool_next_net_pnl"] = market["next_net_pnl"]
        market["next_cumulative_net_pnl"] = market["next_net_pnl"].fillna(0.0).cumsum()
        market["pool_next_cumulative_net_pnl"] = market["next_cumulative_net_pnl"]
    market["next_alpha_bps"] = pd.NA
    market["next_capital_alpha_bps"] = pd.NA
    market["next_cumulative_alpha_bps"] = pd.NA
    if "next_alpha_pnl" in market.columns:
        market["next_alpha_pnl"] = pd.NA
    if "next_cumulative_alpha_pnl" in market.columns:
        market["next_cumulative_alpha_pnl"] = pd.NA
    market["next_capital_internal_excess_bps"] = pd.NA
    market["next_cumulative_internal_excess_return_bps"] = pd.NA
    for column in (
        "next_vs_baseline_bps",
        "next_cumulative_vs_baseline_bps",
        "next_internal_excess_vs_baseline_bps",
        "next_cumulative_internal_excess_vs_baseline_bps",
        "next_alpha_vs_market_bps",
        "next_capital_alpha_vs_market_bps",
        "next_cumulative_alpha_vs_market_bps",
    ):
        market[column] = pd.NA

    out = cumulative_data.loc[~cumulative_data["pool"].astype(str).eq(source_key)].copy()
    for column in market.columns:
        if column not in out.columns:
            out[column] = pd.NA
    for column in out.columns:
        if column not in market.columns:
            market[column] = pd.NA
    market = market[out.columns]
    market["week_start"] = market["week_start"].dt.strftime("%Y-%m-%d")
    return pd.concat([out, market], ignore_index=True)


def add_cumulative_baseline_relative_data(
    cumulative_data: pd.DataFrame,
    *,
    baseline_key: str,
    comparison_keys: tuple[str, ...],
) -> pd.DataFrame:
    data = cumulative_data.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], errors="coerce")
    data = data.dropna(subset=["week_start"])
    baseline = data.loc[
        data["pool"].astype(str).eq(baseline_key),
        [
            "week_start",
            "next_net_return_bps",
            "next_cumulative_net_return_bps",
            "next_capital_internal_excess_bps",
            "next_cumulative_internal_excess_return_bps",
        ],
    ].rename(
        columns={
            "next_net_return_bps": "baseline_next_net_bps",
            "next_cumulative_net_return_bps": "baseline_next_cumulative_net_bps",
            "next_capital_internal_excess_bps": "baseline_next_capital_internal_excess_bps",
            "next_cumulative_internal_excess_return_bps": (
                "baseline_next_cumulative_internal_excess_return_bps"
            ),
        }
    )
    if baseline.empty:
        raise ValueError(f"cumulative data has no baseline rows for {baseline_key!r}")
    data["next_vs_baseline_bps"] = pd.NA
    data["next_cumulative_vs_baseline_bps"] = pd.NA
    data["next_internal_excess_vs_baseline_bps"] = pd.NA
    data["next_cumulative_internal_excess_vs_baseline_bps"] = pd.NA
    for key in comparison_keys:
        item = data.loc[data["pool"].astype(str).eq(key), ["week_start"]].merge(
            baseline,
            on="week_start",
            how="left",
        )
        index = data.index[data["pool"].astype(str).eq(key)]
        if len(index) != len(item):
            raise ValueError(f"cannot align cumulative baseline rows for {key!r}")
        daily_relative_bps = (
            pd.to_numeric(data.loc[index, "next_net_return_bps"], errors="coerce").to_numpy()
            - pd.to_numeric(item["baseline_next_net_bps"], errors="coerce").to_numpy()
        )
        data.loc[index, "next_vs_baseline_bps"] = daily_relative_bps
        daily_internal_relative_bps = (
            pd.to_numeric(
                data.loc[index, "next_capital_internal_excess_bps"],
                errors="coerce",
            ).to_numpy()
            - pd.to_numeric(
                item["baseline_next_capital_internal_excess_bps"],
                errors="coerce",
            ).to_numpy()
        )
        data.loc[index, "next_internal_excess_vs_baseline_bps"] = daily_internal_relative_bps
        data.loc[index, "next_cumulative_vs_baseline_bps"] = (
            pd.to_numeric(
                data.loc[index, "next_cumulative_net_return_bps"],
                errors="coerce",
            ).to_numpy()
            - pd.to_numeric(item["baseline_next_cumulative_net_bps"], errors="coerce").to_numpy()
        )
        data.loc[index, "next_cumulative_internal_excess_vs_baseline_bps"] = (
            pd.to_numeric(
                data.loc[index, "next_cumulative_internal_excess_return_bps"],
                errors="coerce",
            ).to_numpy()
            - pd.to_numeric(
                item["baseline_next_cumulative_internal_excess_return_bps"],
                errors="coerce",
            ).to_numpy()
        )
    data["week_start"] = data["week_start"].dt.strftime("%Y-%m-%d")
    return data


def add_cumulative_market_relative_data(
    cumulative_data: pd.DataFrame,
    *,
    market_key: str,
    comparison_keys: tuple[str, ...],
) -> pd.DataFrame:
    data = cumulative_data.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], errors="coerce")
    data = data.dropna(subset=["week_start"])
    market = data.loc[
        data["pool"].astype(str).eq(market_key),
        [
            "week_start",
            "next_net_return_bps",
            "next_capital_net_return_bps",
            "next_cumulative_net_return_bps",
        ],
    ].rename(
        columns={
            "next_net_return_bps": "market_next_net_bps",
            "next_capital_net_return_bps": "market_next_capital_net_bps",
            "next_cumulative_net_return_bps": "market_next_cumulative_net_bps",
        }
    )
    if market.empty:
        raise ValueError(f"cumulative data has no market rows for {market_key!r}")

    data["next_alpha_vs_market_bps"] = pd.NA
    data["next_capital_alpha_vs_market_bps"] = pd.NA
    data["next_cumulative_alpha_vs_market_bps"] = pd.NA
    for key in comparison_keys:
        item = data.loc[data["pool"].astype(str).eq(key), ["week_start"]].merge(
            market,
            on="week_start",
            how="left",
        )
        index = data.index[data["pool"].astype(str).eq(key)]
        if len(index) != len(item):
            raise ValueError(f"cannot align cumulative market rows for {key!r}")
        if item["market_next_cumulative_net_bps"].isna().any():
            raise ValueError(f"missing market rows for cumulative alpha key {key!r}")
        data.loc[index, "next_alpha_vs_market_bps"] = (
            pd.to_numeric(data.loc[index, "next_net_return_bps"], errors="coerce").to_numpy()
            - pd.to_numeric(item["market_next_net_bps"], errors="coerce").to_numpy()
        )
        data.loc[index, "next_capital_alpha_vs_market_bps"] = (
            pd.to_numeric(
                data.loc[index, "next_capital_net_return_bps"],
                errors="coerce",
            ).to_numpy()
            - pd.to_numeric(item["market_next_capital_net_bps"], errors="coerce").to_numpy()
        )
        data.loc[index, "next_cumulative_alpha_vs_market_bps"] = (
            pd.to_numeric(
                data.loc[index, "next_cumulative_net_return_bps"],
                errors="coerce",
            ).to_numpy()
            - pd.to_numeric(item["market_next_cumulative_net_bps"], errors="coerce").to_numpy()
        )
    data["week_start"] = data["week_start"].dt.strftime("%Y-%m-%d")
    return data


def _panel_values(data: pd.DataFrame, *, pools: tuple[str, ...], column: str) -> pd.Series:
    return data.loc[data["pool"].astype(str).isin(pools), column]


def apply_display_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "pool" not in out.columns:
        return out
    labels = out["pool"].astype(str).map(DISPLAY_LABELS)
    if "pool_label" in out.columns:
        out["pool_label"] = labels.fillna(out["pool_label"])
    if "variant" in out.columns:
        out["variant"] = labels.fillna(out["variant"])
    return out


def capacity_label(
    *,
    capacity_total_notional: float | None,
    capacity_decision_notional: float | None,
) -> str:
    if not capacity_total_notional:
        return ""
    total_yi = float(capacity_total_notional) / 100_000_000.0
    return f"{total_yi:g}亿容量"


def realistic_label(
    *,
    capacity_total_notional: float | None,
) -> str:
    if not capacity_total_notional:
        return "真实约束"
    total_yi = float(capacity_total_notional) / 100_000_000.0
    return f"{total_yi:g}亿真实约束"


def _attach_capacity_fraction_to_market_source(
    market_source: pd.DataFrame,
    capacity_data: pd.DataFrame,
) -> pd.DataFrame:
    fraction_source = capacity_data.loc[
        capacity_data["pool"].astype(str).eq("baseline_pool_l"),
        [
            "week_start",
            "capacity_daily_capital_fraction",
            "capacity_total_notional",
            "capacity_decision_notional",
        ],
    ].copy()
    if fraction_source.empty:
        raise ValueError("capacity cumulative data has no baseline rows for market scaling")
    out = market_source.drop(
        columns=[
            "capacity_daily_capital_fraction",
            "capacity_total_notional",
            "capacity_decision_notional",
        ],
        errors="ignore",
    ).merge(fraction_source, on="week_start", how="left")
    if out["capacity_daily_capital_fraction"].isna().any():
        raise ValueError("missing capacity fraction rows for market source")
    return out


def _replace_capacity_pool_source(
    capacity_data: pd.DataFrame,
    realized_source: pd.DataFrame,
    *,
    baseline_key: str = "baseline_pool_l",
) -> pd.DataFrame:
    source_columns = [
        "week_start",
        "pool_next_mean_bps",
        "pool_turnover",
        "pool_turnover_source",
        "pool_fee_bps",
        "pool_next_net_return_bps",
    ]
    source = realized_source.loc[
        realized_source["pool"].astype(str).eq(baseline_key),
        source_columns,
    ].copy()
    if source.empty:
        raise ValueError(f"realized source has no rows for {baseline_key!r}")
    source = source.rename(columns={column: f"{column}_source" for column in source_columns[1:]})
    out = capacity_data.drop(
        columns=[
            "pool_next_mean_bps",
            "pool_turnover",
            "pool_turnover_source",
            "pool_fee_bps",
            "pool_next_net_return_bps",
            "pool_next_capital_net_return_bps",
            "pool_next_cumulative_net_return_bps",
            "pool_next_net_pnl",
            "pool_next_cumulative_net_pnl",
            "next_internal_excess_bps",
            "next_capital_internal_excess_bps",
            "next_cumulative_internal_excess_return_bps",
            "next_internal_excess_pnl",
            "next_cumulative_internal_excess_pnl",
        ],
        errors="ignore",
    ).merge(source, on="week_start", how="left")
    missing_source = out["pool_next_net_return_bps_source"].isna()
    if missing_source.any():
        missing_dates = sorted(out.loc[missing_source, "week_start"].astype(str).unique())[:5]
        raise ValueError(f"missing realized pool source rows for dates: {missing_dates}")
    for column in source_columns[1:]:
        out[column] = out.pop(f"{column}_source")
    out["pool_next_capital_net_return_bps"] = pd.to_numeric(
        out["pool_next_net_return_bps"], errors="coerce"
    ) * pd.to_numeric(out["capacity_daily_capital_fraction"], errors="coerce")
    out["next_internal_excess_bps"] = pd.to_numeric(
        out["next_net_return_bps"], errors="coerce"
    ) - pd.to_numeric(out["pool_next_net_return_bps"], errors="coerce")
    out["next_capital_internal_excess_bps"] = pd.to_numeric(
        out["next_internal_excess_bps"], errors="coerce"
    ) * pd.to_numeric(out["capacity_daily_capital_fraction"], errors="coerce")
    total_notional = pd.to_numeric(out["capacity_total_notional"], errors="coerce")
    out["pool_next_net_pnl"] = (
        out["pool_next_capital_net_return_bps"] / RETURN_BPS_DENOMINATOR * total_notional
    )
    out["next_internal_excess_pnl"] = (
        out["next_capital_internal_excess_bps"] / RETURN_BPS_DENOMINATOR * total_notional
    )
    out = out.sort_values(["pool", "week_start"]).copy()
    for _, item in out.groupby("pool", sort=False):
        out.loc[item.index, "pool_next_cumulative_net_return_bps"] = (
            item["pool_next_capital_net_return_bps"].fillna(0.0).cumsum()
        )
        out.loc[item.index, "next_cumulative_internal_excess_return_bps"] = (
            item["next_capital_internal_excess_bps"].fillna(0.0).cumsum()
        )
        out.loc[item.index, "pool_next_cumulative_net_pnl"] = (
            item["pool_next_net_pnl"].fillna(0.0).cumsum()
        )
        out.loc[item.index, "next_cumulative_internal_excess_pnl"] = (
            item["next_internal_excess_pnl"].fillna(0.0).cumsum()
        )
    return out


def ensure_plot_colors(keys: tuple[str, ...]) -> None:
    PLOT_COLORS.update(DIRECTION_COLORS)
    assigned_keys: set[str] = set()
    sequence_index = 0
    for key in keys:
        if key in DIRECTION_COLORS or key in assigned_keys:
            continue
        PLOT_COLORS[key] = AUTO_COLOR_SEQUENCE[sequence_index % len(AUTO_COLOR_SEQUENCE)]
        assigned_keys.add(key)
        sequence_index += 1


def write_optimization_direction_plots(**kwargs) -> dict[str, str]:
    """Forward the historical plots-module API to its workflow owner."""

    workflow = import_module("opening_strength_fit.optimization_acceptance_workflow")
    return workflow.write_optimization_direction_plots(**kwargs)
