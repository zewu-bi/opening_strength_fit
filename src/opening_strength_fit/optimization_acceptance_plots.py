from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import write_json
from opening_strength_fit.optimization_direction_data import (
    CUMULATIVE_DECISION_NORMALIZER,
    DEFAULT_DIRECTIONS,
    DEFAULT_POOL_FEE_MODE,
    DEFAULT_REALIZED_FEE_BPS,
    NEXT_CLOSE_CAPITAL_DIVISOR,
    RETURN_BPS_DENOMINATOR,
    DirectionSpec,
    line_axis,
    line_step,
    load_horizon_plot_data,
    load_realized_cumulative_plot_data,
    source_files,
)
from opening_strength_fit.pool_internal_plot_svg import (
    PLOT_COLORS,
    write_two_panel_bar_svg,
    write_two_panel_line_svg,
)

DIRECTION_COLORS = {
    "baseline": "#1f2937",
    "baseline_pool_l": "#1f2937",
    "baseline_universe": "#7f7f7f",
    "market": "#7f7f7f",
    "background": "#ff7f0e",
}

DISPLAY_LABELS = {
    "baseline": "baseline",
    "baseline_pool_l": "baseline",
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
    "next_cumulative_alpha_vs_market_bps": "next_cumulative_alpha_vs_market_pct",
}
CUMULATIVE_NET_RETURN_PANEL_TITLE = "扣除手续费累和收益"
CUMULATIVE_MARKET_ALPHA_PANEL_TITLE = "对比全A股市场平均alpha"


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
    combined = realized_cumulative_output[
        key_columns
        + [
            "candidate_rows",
            "selected_rows",
            "pool_next_mean_bps",
            "pool_turnover",
            "pool_turnover_source",
            "pool_fee_bps",
            "pool_next_net_return_bps",
            "pool_next_capital_net_return_bps",
            "pool_next_cumulative_net_return_bps",
            "selected_next_mean_bps",
            "selected_turnover",
            "selected_fee_bps",
            "next_internal_excess_bps",
            "next_capital_internal_excess_bps",
            "next_cumulative_internal_excess_return_bps",
            "fee_bps",
            "next_net_return_bps",
            "next_capital_net_return_bps",
            "next_cumulative_net_return_bps",
        ]
    ].copy()
    combined["next_alpha_bps"] = (
        combined["next_net_return_bps"] - combined["pool_next_net_return_bps"]
    )
    combined["next_cumulative_alpha_bps"] = (
        combined["next_cumulative_net_return_bps"] - combined["pool_next_cumulative_net_return_bps"]
    )
    combined["next_capital_alpha_bps"] = (
        combined["next_capital_net_return_bps"] - combined["pool_next_capital_net_return_bps"]
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
    background["next_internal_excess_bps"] = pd.NA
    background["fee_bps"] = background["pool_fee_bps"]
    background["next_net_return_bps"] = background["pool_next_net_return_bps"]
    background["next_capital_net_return_bps"] = background["pool_next_capital_net_return_bps"]
    background["next_cumulative_net_return_bps"] = background["pool_next_cumulative_net_return_bps"]
    background["next_alpha_bps"] = pd.NA
    background["next_cumulative_alpha_bps"] = pd.NA
    background["next_capital_internal_excess_bps"] = 0.0
    background["next_cumulative_internal_excess_return_bps"] = 0.0
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
    market["next_capital_net_return_bps"] = raw_market_next_bps / NEXT_CLOSE_CAPITAL_DIVISOR
    market["pool_next_capital_net_return_bps"] = market["next_capital_net_return_bps"]
    market["next_cumulative_net_return_bps"] = market["next_capital_net_return_bps"].cumsum()
    market["pool_next_cumulative_net_return_bps"] = market["next_cumulative_net_return_bps"]
    market["next_alpha_bps"] = pd.NA
    market["next_capital_alpha_bps"] = pd.NA
    market["next_cumulative_alpha_bps"] = pd.NA
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


def write_optimization_direction_plots(
    *,
    backtests_root: Path,
    output_dir: Path,
    directions: tuple[DirectionSpec, ...] | None = None,
    pool: str = "pool_L",
    include_baseline_pool_cumulative: bool = True,
    include_baseline_universe_cumulative: bool = False,
    baseline_run_id: str = "baseline_2022_2025_cluster",
    realized_fee_bps: float = DEFAULT_REALIZED_FEE_BPS,
    pool_turnover_path: str | Path | None = "auto",
    pool_fee_mode: str = DEFAULT_POOL_FEE_MODE,
    title_prefix: str = "2022-2025",
    top_n: int = 100,
) -> dict[str, str]:
    if directions is None:
        plot_directions = default_plot_directions()
    else:
        plot_directions = validate_plot_directions(tuple(directions))
    if not include_baseline_pool_cumulative:
        raise ValueError("baseline pool cumulative series is required for cumulative plots")

    output_dir.mkdir(parents=True, exist_ok=True)
    series = tuple(direction.key for direction in plot_directions)
    model_cumulative_series = ("baseline_pool_l", *series)
    top_cumulative_series = ("market", "background", *model_cumulative_series)
    alpha_cumulative_series = ("background", *model_cumulative_series)
    acceptance_directions = (
        DirectionSpec(key="baseline", label="baseline", run_id=baseline_run_id),
        *plot_directions,
    )
    acceptance_series = tuple(direction.key for direction in acceptance_directions)
    ensure_plot_colors((*acceptance_series, *top_cumulative_series, *alpha_cumulative_series))

    short_universe_data = apply_display_labels(
        load_horizon_plot_data(
            backtests_root=backtests_root,
            directions=acceptance_directions,
            pool="universe",
            horizon="short",
        )
    )
    next_pool_data = apply_display_labels(
        load_horizon_plot_data(
            backtests_root=backtests_root,
            directions=acceptance_directions,
            pool=pool,
            horizon="next",
        )
    )
    realized_cumulative_data = load_realized_cumulative_plot_data(
        backtests_root=backtests_root,
        directions=plot_directions,
        pool=pool,
        include_baseline_pool=include_baseline_pool_cumulative,
        include_baseline_universe=True,
        baseline_run_id=baseline_run_id,
        fee_bps=realized_fee_bps,
        pool_turnover_path=pool_turnover_path,
        pool_fee_mode=pool_fee_mode,
    )
    realized_cumulative_output = apply_display_labels(
        realized_cumulative_data.drop(
            columns=[
                "pool_short_mean_bps",
                "selected_short_mean_bps",
                "short_internal_excess_bps",
                "short_net_return_bps",
                "short_cumulative_net_return_bps",
            ],
            errors="ignore",
        )
    )
    overlay_acceptance_data = combine_overlay_acceptance_data(
        short_universe_data,
        next_pool_data,
    )
    net_alpha_cumulative_data = combine_net_alpha_cumulative_data(
        realized_cumulative_output,
    )
    net_alpha_cumulative_data = add_cumulative_baseline_relative_data(
        net_alpha_cumulative_data,
        baseline_key="baseline_pool_l",
        comparison_keys=series,
    )
    net_alpha_cumulative_data = add_background_cumulative_data(
        net_alpha_cumulative_data,
        baseline_key="baseline_pool_l",
    )
    net_alpha_cumulative_data = add_market_cumulative_data(net_alpha_cumulative_data)
    net_alpha_cumulative_data = add_cumulative_market_relative_data(
        net_alpha_cumulative_data,
        market_key="market",
        comparison_keys=alpha_cumulative_series,
    )

    overlay_acceptance_csv = output_dir / "optimization_directions_overlay_acceptance_plot_data.csv"
    overlay_acceptance_svg = output_dir / "optimization_directions_overlay_acceptance.svg"
    net_alpha_cumulative_csv = (
        output_dir / "optimization_directions_net_alpha_cumulative_plot_data.csv"
    )
    net_alpha_cumulative_svg = output_dir / "optimization_directions_net_alpha_cumulative.svg"
    trace_path = output_dir / "optimization_directions_trace.json"
    top_n_label = f"Top{top_n}"

    overlay_acceptance_data.to_csv(overlay_acceptance_csv, index=False, float_format="%.6f")
    write_two_panel_bar_svg(
        overlay_acceptance_data,
        title=f"{title_prefix} short rank IC和next pool_L 超额",
        panels=[
            {
                "title": "short universe rank IC",
                "ylabel": "Rank IC",
                "column": "short_rank_ic",
                "default_ylim": (0.12, 0.17),
                "tick_step": 0.01,
                "tick_decimals": 3,
                "label_decimals": 3,
                "adaptive_ylim": True,
                "include_zero": False,
                "target_ticks": 6,
                "min_tick_step": 0.005,
            },
            {
                "title": f"next pool_L {top_n_label} excess",
                "ylabel": "bps",
                "column": "next_internal_excess_bps",
                "default_ylim": (0.0, 12.0),
                "tick_step": 2.0,
                "tick_decimals": None,
                "label_decimals": 1,
                "adaptive_ylim": True,
                "include_zero": True,
                "target_ticks": 6,
                "min_tick_step": 1.0,
            },
        ],
        output_path=overlay_acceptance_svg,
        pools=acceptance_series,
    )

    net_alpha_cumulative_data.to_csv(
        net_alpha_cumulative_csv,
        index=False,
        float_format="%.6f",
    )
    net_alpha_cumulative_plot_data = add_cumulative_percent_display_columns(
        net_alpha_cumulative_data
    )
    cumulative_title = f"{title_prefix} fee {realized_fee_bps:g}bps 池内{top_n_label}隔夜净收益累和"
    cumulative_net_values = _panel_values(
        net_alpha_cumulative_plot_data,
        pools=top_cumulative_series,
        column="next_cumulative_net_return_pct",
    )
    market_alpha_values = _panel_values(
        net_alpha_cumulative_plot_data,
        pools=alpha_cumulative_series,
        column="next_cumulative_alpha_vs_market_pct",
    )
    write_two_panel_line_svg(
        net_alpha_cumulative_plot_data,
        title=cumulative_title,
        panels=[
            {
                "title": CUMULATIVE_NET_RETURN_PANEL_TITLE,
                "ylabel": "%",
                "column": "next_cumulative_net_return_pct",
                "default_ylim": line_axis(cumulative_net_values),
                "tick_step": line_step(cumulative_net_values),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
            {
                "title": CUMULATIVE_MARKET_ALPHA_PANEL_TITLE,
                "ylabel": "%",
                "column": "next_cumulative_alpha_vs_market_pct",
                "default_ylim": line_axis(market_alpha_values),
                "tick_step": line_step(market_alpha_values),
                "tick_decimals": None,
                "fixed_ylim": True,
            },
        ],
        output_path=net_alpha_cumulative_svg,
        pools=top_cumulative_series,
        x_label_mode="years_only",
        line_width=2.1,
        line_marker_count=0,
    )

    trace = {
        "backtests_root": str(backtests_root),
        "output_dir": str(output_dir),
        "pool": pool,
        "cumulative_decision_normalizer": CUMULATIVE_DECISION_NORMALIZER,
        "return_bps_denominator": RETURN_BPS_DENOMINATOR,
        "next_close_capital_divisor": NEXT_CLOSE_CAPITAL_DIVISOR,
        "realized_fee_bps": realized_fee_bps,
        "top_n": top_n,
        "pool_turnover_path": str(pool_turnover_path) if pool_turnover_path else None,
        "pool_fee_mode": pool_fee_mode,
        "daily_cumulative_semantics": (
            "next-close labels span entry day to next trading day's close, so cumulative "
            "acceptance divides next-close bps by next_close_capital_divisor before "
            "linear cumulative summation"
        ),
        "overlay_acceptance": {
            "figure_title": f"{title_prefix} short rank IC和next pool_L 超额",
            "panels": [
                "short universe rank IC",
                f"next {pool} {top_n_label} excess",
            ],
            "reason": (
                "short pool excess is omitted because A-share T+1 makes short-horizon "
                "cash PnL non-tradable; next IC is omitted because this model is not "
                "trained to rank next-day returns directly"
            ),
            "baseline_run_id": baseline_run_id,
        },
        "baseline_pool_cumulative": {
            "enabled": include_baseline_pool_cumulative,
            "run_id": baseline_run_id,
            "pool": pool,
            "key": "baseline_pool_l",
        },
        "market_cumulative": {
            "enabled": True,
            "run_id": baseline_run_id,
            "pool": "universe",
            "key": "market",
            "definition": "universe pool_next_mean_bps divided by next_close_capital_divisor and cumulatively summed",
        },
        "directions": [
            {"key": item.key, "label": item.label, "run_id": item.run_id}
            for item in plot_directions
        ],
        "plotted_series": {
            "overlay_acceptance": ["baseline", *series],
            "cumulative_top": list(top_cumulative_series),
            "cumulative_market_alpha": list(alpha_cumulative_series),
        },
        "figures": {
            "overlay_acceptance": str(overlay_acceptance_svg),
            "net_alpha_cumulative": str(net_alpha_cumulative_svg),
        },
        "plot_data": {
            "overlay_acceptance": str(overlay_acceptance_csv),
            "net_alpha_cumulative": str(net_alpha_cumulative_csv),
        },
        "cumulative_acceptance": {
            "figure_title": cumulative_title,
            "panels": [
                CUMULATIVE_NET_RETURN_PANEL_TITLE,
                CUMULATIVE_MARKET_ALPHA_PANEL_TITLE,
            ],
            "market_series": "full A-share market average overnight return",
            "background_series": "pool_L background overnight return after pool_fee_bps",
            "reason": "short cumulative is omitted because this workflow cannot trade T+0",
            "unit": "%",
            "source_unit": "bps",
            "fee_bps_per_trade": realized_fee_bps,
            "absolute_definition": (
                "top panel plots market, pool background, baseline selected TopN, and "
                "comparison selected TopN cumulative next-close returns. Pool/model lines "
                "subtract their realized fee before dividing by next_close_capital_divisor "
                "and cumulative summation; market uses universe pool_next_mean_bps without "
                "a trading fee. Figure axis displays cumulative bps divided by 100 as percent"
            ),
            "background_definition": (
                "pool_L background overnight return minus pool_fee_bps; pool fee uses "
                "equal-weight stock-pool membership turnover when available"
            ),
            "pool_turnover_source": "see pool_turnover_source column in cumulative plot data",
            "market_alpha_definition": (
                "bottom panel plots pool/background and model capital-adjusted cumulative "
                "net bps minus full-market capital-adjusted cumulative bps, displayed as "
                "percent"
            ),
            "accumulation_definition": (
                "capital-adjusted cumulative net bps = cumsum(daily_net_bps / "
                "next_close_capital_divisor)"
            ),
        },
        "source_files": source_files(backtests_root, plot_directions),
    }
    trace["baseline_universe_cumulative"] = {
        "enabled": True,
        "requested_by_cli": include_baseline_universe_cumulative,
        "run_id": baseline_run_id,
        "pool": "universe",
        "key": "baseline_universe",
        "used_as": "market source",
        "panels": ["next"],
    }
    write_json(trace_path, trace, ensure_ascii=True)

    return {
        "overlay_acceptance_plot_data": str(overlay_acceptance_csv),
        "overlay_acceptance_figure": str(overlay_acceptance_svg),
        "net_alpha_cumulative_plot_data": str(net_alpha_cumulative_csv),
        "net_alpha_cumulative_figure": str(net_alpha_cumulative_svg),
        "trace": str(trace_path),
    }
