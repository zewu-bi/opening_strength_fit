from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CUMULATIVE_DECISION_NORMALIZER = 1.0
DEFAULT_REALIZED_FEE_BPS = 8.0
RETURN_BPS_DENOMINATOR = 10_000.0
NEXT_CLOSE_CAPITAL_DIVISOR = 2.0
DEFAULT_POOL_FEE_MODE = "stock_pool_membership"
POOL_FEE_MODES = ("round_trip", "stock_pool_membership", "summary_estimate")
CAPACITY_CUMULATIVE_FEE_COLUMNS = (
    "capacity_fee_bps_per_trade",
    "capacity_audit_fee_bps_per_trade",
    "capacity_additional_fee_bps_per_trade",
    "capacity_additional_fee_bps",
)
CUMULATIVE_PLOT_VALUE_COLUMNS = (
    "decision_groups",
    "clocks",
    "candidate_rows",
    "selected_rows",
    "capacity_decision_groups",
    "capacity_daily_capital_fraction",
    "capacity_total_notional",
    "capacity_decision_notional",
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
)
REALIZED_SOURCE_VALUE_COLUMNS = (
    "decision_groups",
    "clocks",
    "candidate_rows",
    "selected_rows",
    "pool_short_mean_bps",
    "selected_short_mean_bps",
    "short_internal_excess_bps",
    "pool_next_mean_bps",
    "selected_next_mean_bps",
    "next_internal_excess_bps",
)


@dataclass(frozen=True)
class DirectionSpec:
    key: str
    run_id: str
    label: str


DEFAULT_DIRECTIONS = (
    DirectionSpec(
        key="xs_relative",
        run_id="lgbm_delay2_36m_2022_2025_pool_l_xs_relative_v1",
        label="xs_relative",
    ),
    DirectionSpec(
        key="hist_surprise",
        run_id="lgbm_delay2_36m_2022_2025_fullxs_hist_same_minute_surprise_v1",
        label="hist_same_minute_surprise",
    ),
    DirectionSpec(
        key="path_shape",
        run_id="lgbm_delay2_36m_2022_2025_fullxs_path_shape_confirm_v1",
        label="path_shape_confirm",
    ),
    DirectionSpec(
        key="clock_segment",
        run_id="lgbm_delay2_36m_2022_2025_fullxs_clock_segment_lgbm_v1",
        label="clock_segment_lgbm",
    ),
)


def _read_pool_rows(path: Path, required: set[str], pool: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    item = frame.loc[frame["pool"].astype(str).eq(pool)].copy()
    if item.empty:
        raise ValueError(f"{path} has no rows for pool {pool!r}")
    return item


def load_horizon_plot_data(
    *,
    backtests_root: Path,
    directions: tuple[DirectionSpec, ...],
    pool: str,
    horizon: str,
) -> pd.DataFrame:
    value_columns = {
        "short": ("short_internal_excess_bps", "short_rank_ic"),
        "next": ("next_internal_excess_bps", "next_rank_ic"),
    }
    if horizon not in value_columns:
        raise ValueError(f"unknown horizon: {horizon}")
    required = {"test_month", "pool", *value_columns[horizon]}

    frames = []
    for direction in directions:
        path = backtests_root / direction.run_id / f"{horizon}_excess_rank_ic_plot_data.csv"
        item = _read_pool_rows(path, required, pool)
        item["pool"] = direction.key
        item["pool_label"] = direction.label
        item["variant"] = direction.label
        frames.append(
            item[["test_month", "variant", "pool", "pool_label", *value_columns[horizon]]]
        )
    combined = pd.concat(frames, ignore_index=True)
    return sort_month_major(combined)


def load_realized_cumulative_plot_data(
    *,
    backtests_root: Path,
    directions: tuple[DirectionSpec, ...],
    pool: str,
    include_baseline_pool: bool,
    include_baseline_universe: bool,
    baseline_run_id: str,
    baseline_label: str = "baseline",
    fee_bps: float,
    pool_turnover_path: str | Path | None = None,
    pool_fee_mode: str = DEFAULT_POOL_FEE_MODE,
) -> pd.DataFrame:
    if pool_fee_mode not in POOL_FEE_MODES:
        raise ValueError(f"unknown pool_fee_mode {pool_fee_mode!r}; expected {POOL_FEE_MODES}")
    required = {"pool", "date", *REALIZED_SOURCE_VALUE_COLUMNS[2:]}
    pool_turnover_by_date = (
        try_load_pool_turnover_by_date(_resolve_pool_turnover_path(pool, pool_turnover_path))
        if pool_fee_mode == "stock_pool_membership"
        else None
    )
    sources: list[tuple[DirectionSpec, str, bool, pd.Series | None]] = []
    if include_baseline_pool:
        sources.append(
            (
                DirectionSpec("baseline_pool_l", baseline_run_id, baseline_label),
                pool,
                False,
                pool_turnover_by_date,
            )
        )
    if include_baseline_universe:
        sources.append(
            (
                DirectionSpec("baseline_universe", baseline_run_id, f"{baseline_label} universe"),
                "universe",
                True,
                None,
            )
        )
    sources.extend((direction, pool, False, pool_turnover_by_date) for direction in directions)
    frames = [
        _load_one_realized_plot_data(
            path=backtests_root / direction.run_id / "daily_pool_internal_summary.csv",
            source_pool=source_pool,
            key=direction.key,
            label=direction.label,
            required=required,
            fee_bps=fee_bps,
            pool_turnover_by_date=turnover,
            pool_fee_mode=pool_fee_mode,
            next_only=next_only,
        )
        for direction, source_pool, next_only, turnover in sources
    ]

    combined = pd.concat(frames, ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"], errors="coerce")
    combined = combined.dropna(subset=["week_start"]).sort_values(["pool", "week_start"])
    numeric_columns = (
        "decision_groups",
        "short_net_return_bps",
        "next_net_return_bps",
        "pool_next_net_return_bps",
        "next_internal_excess_bps",
    )
    combined[list(numeric_columns)] = combined[list(numeric_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    capital_fraction = 1.0 / NEXT_CLOSE_CAPITAL_DIVISOR
    combined["capacity_decision_groups"] = combined["decision_groups"]
    combined["capacity_daily_capital_fraction"] = capital_fraction
    combined["capacity_total_notional"] = pd.NA
    combined["capacity_decision_notional"] = pd.NA
    combined["next_capital_net_return_bps"] = combined["next_net_return_bps"] * capital_fraction
    combined["pool_next_capital_net_return_bps"] = (
        combined["pool_next_net_return_bps"] * capital_fraction
    )
    combined["next_capital_internal_excess_bps"] = (
        combined["next_internal_excess_bps"] * capital_fraction
    )
    cumulative_columns = (
        ("short_cumulative_net_return_bps", "short_net_return_bps"),
        ("next_cumulative_net_return_bps", "next_capital_net_return_bps"),
        ("pool_next_cumulative_net_return_bps", "pool_next_capital_net_return_bps"),
        (
            "next_cumulative_internal_excess_return_bps",
            "next_capital_internal_excess_bps",
        ),
    )
    for index in combined.groupby("pool", sort=False).groups.values():
        for target, source in cumulative_columns:
            combined.loc[index, target] = cumulative_sum_bps(combined.loc[index, source])
    combined["week_start"] = combined["week_start"].dt.strftime("%Y-%m-%d")
    return combined


def load_capacity_cumulative_plot_data(
    *,
    backtests_root: Path,
    capacity_directions: tuple[DirectionSpec, ...],
    pool: str,
    capacity_total_notional: float | None = None,
    summary_filename: str = "capacity_acceptance_daily_summary.csv",
    source_label: str = "capacity",
) -> pd.DataFrame:
    required = {
        "pool",
        "date",
        "capacity_decision_groups",
        "selected_rows",
        "target_notional",
        "capacity_daily_capital_fraction",
        "fee_bps",
        "next_net_return_bps",
    }
    frames = []
    for direction in capacity_directions:
        path = backtests_root / direction.run_id / summary_filename
        if not path.exists():
            raise ValueError(
                f"{source_label} cumulative mode requires {summary_filename} "
                f"for {direction.key!r}; missing {path}"
            )
        item = _read_pool_rows(path, required, pool)
        item["pool"] = direction.key
        item["pool_label"] = direction.label
        item["variant"] = direction.label
        item["week_start"] = pd.to_datetime(item["date"], errors="coerce")
        item = item.dropna(subset=["week_start"]).sort_values("week_start")
        for column in (
            "capacity_decision_groups",
            "selected_rows",
            "target_notional",
            "capacity_daily_capital_fraction",
            "fee_bps",
            "gross_next_return_bps",
            "next_net_return_bps",
            "next_capital_net_return_bps",
            "next_net_pnl",
        ):
            if column in item.columns:
                item[column] = pd.to_numeric(item[column], errors="coerce")
        if capacity_total_notional is not None:
            item["capacity_total_notional"] = float(capacity_total_notional)
        elif "capacity_total_notional" not in item.columns:
            item["capacity_total_notional"] = pd.NA
        item["capacity_total_notional"] = pd.to_numeric(
            item["capacity_total_notional"],
            errors="coerce",
        )
        item["capacity_decision_notional"] = pd.to_numeric(
            item["target_notional"], errors="coerce"
        ) / pd.to_numeric(item["capacity_decision_groups"], errors="coerce").replace(0, pd.NA)
        missing_fraction = item["capacity_daily_capital_fraction"].isna()
        if missing_fraction.any():
            item.loc[missing_fraction, "capacity_daily_capital_fraction"] = (
                pd.to_numeric(item.loc[missing_fraction, "target_notional"], errors="coerce")
                / item.loc[missing_fraction, "capacity_total_notional"]
            )
        if "next_capital_net_return_bps" not in item.columns:
            item["next_capital_net_return_bps"] = (
                item["next_net_return_bps"] * item["capacity_daily_capital_fraction"]
            )
        if "next_net_pnl" not in item.columns:
            item["next_net_pnl"] = (
                item["next_capital_net_return_bps"]
                / RETURN_BPS_DENOMINATOR
                * item["capacity_total_notional"]
            )
        for column in CUMULATIVE_PLOT_VALUE_COLUMNS:
            if column not in item.columns:
                item[column] = pd.NA
        item["decision_groups"] = item["capacity_decision_groups"]
        item["clocks"] = item["capacity_decision_groups"]
        item["selected_next_mean_bps"] = item.get(
            "gross_next_return_bps",
            item["next_net_return_bps"] + item["fee_bps"],
        )
        item["selected_fee_bps"] = item["fee_bps"]
        item["pool_turnover_source"] = "capacity_acceptance_pool_source_pending"
        item["pool_fee_bps"] = 0.0
        item["next_cumulative_net_return_bps"] = cumulative_sum_bps(
            item["next_capital_net_return_bps"]
        )
        item["next_cumulative_net_pnl"] = cumulative_sum_bps(item["next_net_pnl"])
        item["week_start"] = item["week_start"].dt.strftime("%Y-%m-%d")
        frames.append(
            item[
                [
                    "pool",
                    "pool_label",
                    "week_start",
                    *CUMULATIVE_PLOT_VALUE_COLUMNS,
                    "target_notional",
                    "variant",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _load_one_realized_plot_data(
    *,
    path: Path,
    source_pool: str,
    key: str,
    label: str,
    required: set[str],
    fee_bps: float,
    pool_turnover_by_date: pd.Series | None,
    pool_fee_mode: str,
    next_only: bool = False,
) -> pd.DataFrame:
    item = _read_pool_rows(path, required, source_pool)
    item["week_start"] = pd.to_datetime(item["date"], errors="coerce")
    item["pool"] = key
    item["pool_label"] = label
    item["variant"] = label
    item["fee_bps_per_trade"] = float(fee_bps)
    if "decision_groups" not in item.columns:
        item["decision_groups"] = pd.NA
    if "clocks" not in item.columns:
        item["clocks"] = pd.NA

    for column in REALIZED_SOURCE_VALUE_COLUMNS:
        item[column] = pd.to_numeric(item[column], errors="coerce") / CUMULATIVE_DECISION_NORMALIZER
    item["selected_turnover"] = 1.0
    estimated_pool_turnover = estimate_pool_turnover(item)
    item["pool_turnover_source"] = "daily_label_round_trip"
    if pool_fee_mode == "round_trip":
        item["pool_turnover"] = 1.0
    elif pool_fee_mode == "stock_pool_membership" and pool_turnover_by_date is not None:
        date_key = item["week_start"].dt.strftime("%Y-%m-%d")
        realized_pool_turnover = date_key.map(pool_turnover_by_date)
        item["pool_turnover"] = realized_pool_turnover.fillna(estimated_pool_turnover)
        item.loc[realized_pool_turnover.notna(), "pool_turnover_source"] = "stock_pool_membership"
    else:
        item["pool_turnover_source"] = "selected_rows_over_candidate_rows"
        item["pool_turnover"] = estimated_pool_turnover
    item["selected_fee_bps"] = float(fee_bps) * item["selected_turnover"]
    item["pool_fee_bps"] = float(fee_bps) * item["pool_turnover"]
    item["fee_bps"] = item["selected_fee_bps"]
    item["short_net_return_bps"] = item["selected_short_mean_bps"] - item["selected_fee_bps"]
    item["next_net_return_bps"] = item["selected_next_mean_bps"] - item["selected_fee_bps"]
    item["pool_next_net_return_bps"] = item["pool_next_mean_bps"] - item["pool_fee_bps"]
    if next_only:
        for column in (
            "pool_short_mean_bps",
            "selected_short_mean_bps",
            "short_internal_excess_bps",
            "short_net_return_bps",
        ):
            item[column] = pd.NA
    return item[
        [
            "pool",
            "pool_label",
            "week_start",
            *REALIZED_SOURCE_VALUE_COLUMNS,
            "fee_bps_per_trade",
            "selected_turnover",
            "pool_turnover",
            "pool_turnover_source",
            "selected_fee_bps",
            "pool_fee_bps",
            "fee_bps",
            "short_net_return_bps",
            "next_net_return_bps",
            "pool_next_net_return_bps",
            "variant",
        ]
    ]


def estimate_pool_turnover(frame: pd.DataFrame) -> pd.Series:
    candidate_rows = pd.to_numeric(frame["candidate_rows"], errors="coerce")
    selected_rows = pd.to_numeric(frame["selected_rows"], errors="coerce")
    turnover = selected_rows / candidate_rows.replace(0.0, pd.NA)
    return turnover.clip(lower=0.0, upper=1.0).fillna(0.0)


def try_load_pool_turnover_by_date(path: str | Path | None) -> pd.Series | None:
    if path is None:
        return None
    try:
        return load_pool_turnover_by_date(path)
    except (Exception, SystemExit):
        return None


def load_pool_turnover_by_date(path: str | Path) -> pd.Series:
    from opening_strength_fit.stock_pool import load_stock_pool

    pool = load_stock_pool(path).astype(bool).sort_index()
    if pool.empty:
        return pd.Series(dtype="float64")
    counts = pool.sum(axis=1).astype(float)
    weights = pool.astype(float).div(counts.replace(0.0, pd.NA), axis=0).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1) * 0.5
    turnover.iloc[0] = 1.0 if counts.iloc[0] > 0 else 0.0
    turnover.index = pd.to_datetime(turnover.index, errors="coerce").strftime("%Y-%m-%d")
    turnover.name = "pool_turnover"
    return turnover.astype("float64")


def _resolve_pool_turnover_path(
    pool: str,
    pool_turnover_path: str | Path | None,
) -> str | Path | None:
    if pool_turnover_path in (None, ""):
        return None
    if str(pool_turnover_path).lower() != "auto":
        return pool_turnover_path

    from opening_strength_fit.stock_pool import DEFAULT_STOCK_POOL_PATHS

    suffix = pool.removeprefix("pool_").upper()
    return DEFAULT_STOCK_POOL_PATHS.get(suffix)


def cumulative_sum_bps(return_bps: pd.Series) -> pd.Series:
    return pd.to_numeric(return_bps, errors="coerce").fillna(0.0).cumsum()


def sort_month_major(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["_mean_order"] = data["test_month"].astype(str).eq("Mean").astype("int8")
    data = data.sort_values(["_mean_order", "test_month", "pool"]).drop(columns="_mean_order")
    return data.reset_index(drop=True)


def source_files(
    backtests_root: Path, directions: tuple[DirectionSpec, ...]
) -> dict[str, dict[str, str]]:
    filenames = {
        "short": "short_excess_rank_ic_plot_data.csv",
        "next": "next_excess_rank_ic_plot_data.csv",
        "daily_cumulative": "daily_cumulative_plot_data.csv",
        "daily_realized": "daily_pool_internal_summary.csv",
    }
    return {
        direction.key: {
            key: str(backtests_root / direction.run_id / filename)
            for key, filename in filenames.items()
        }
        for direction in directions
    }
