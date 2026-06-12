from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CUMULATIVE_DECISION_NORMALIZER = 1000.0
DEFAULT_REALIZED_FEE_BPS = 5.0


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
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        item = frame.loc[frame["pool"].astype(str).eq(pool)].copy()
        if item.empty:
            raise ValueError(f"{path} has no rows for pool {pool!r}")
        item["pool"] = direction.key
        item["pool_label"] = direction.label
        item["variant"] = direction.label
        frames.append(item[["test_month", "variant", "pool", "pool_label", *value_columns[horizon]]])
    combined = pd.concat(frames, ignore_index=True)
    return sort_month_major(combined)


def load_cumulative_plot_data(
    *,
    backtests_root: Path,
    directions: tuple[DirectionSpec, ...],
    pool: str,
    include_baseline_pool: bool,
    include_baseline_universe: bool,
    baseline_run_id: str,
) -> pd.DataFrame:
    required = {
        "pool",
        "week_start",
        "short_internal_excess_bps",
        "next_internal_excess_bps",
        "short_cumulative_internal_excess_bps",
        "next_cumulative_internal_excess_bps",
    }
    frames = []
    if include_baseline_pool:
        frames.append(
            _load_one_cumulative_plot_data(
                path=backtests_root / baseline_run_id / "daily_cumulative_plot_data.csv",
                source_pool=pool,
                key="baseline_pool_l",
                label="baseline pool_L",
                required=required,
            )
        )
    if include_baseline_universe:
        frames.append(
            _load_one_cumulative_plot_data(
                path=backtests_root / baseline_run_id / "daily_cumulative_plot_data.csv",
                source_pool="universe",
                key="baseline_universe",
                label="baseline universe",
                required=required,
                next_only=True,
            )
        )
    for direction in directions:
        path = backtests_root / direction.run_id / "daily_cumulative_plot_data.csv"
        frames.append(
            _load_one_cumulative_plot_data(
                path=path,
                source_pool=pool,
                key=direction.key,
                label=direction.label,
                required=required,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"], errors="coerce")
    combined = combined.dropna(subset=["week_start"]).sort_values(["pool", "week_start"])
    combined = _normalize_cumulative_decision_bps(combined)
    combined["week_start"] = combined["week_start"].dt.strftime("%Y-%m-%d")
    return combined


def load_realized_cumulative_plot_data(
    *,
    backtests_root: Path,
    directions: tuple[DirectionSpec, ...],
    pool: str,
    include_baseline_pool: bool,
    include_baseline_universe: bool,
    baseline_run_id: str,
    fee_bps: float,
) -> pd.DataFrame:
    required = {
        "pool",
        "date",
        "pool_short_mean_bps",
        "selected_short_mean_bps",
        "short_internal_excess_bps",
        "pool_next_mean_bps",
        "selected_next_mean_bps",
        "next_internal_excess_bps",
    }
    frames = []
    if include_baseline_pool:
        frames.append(
            _load_one_realized_plot_data(
                path=backtests_root / baseline_run_id / "daily_pool_internal_summary.csv",
                source_pool=pool,
                key="baseline_pool_l",
                label="baseline pool_L",
                required=required,
                fee_bps=fee_bps,
            )
        )
    if include_baseline_universe:
        frames.append(
            _load_one_realized_plot_data(
                path=backtests_root / baseline_run_id / "daily_pool_internal_summary.csv",
                source_pool="universe",
                key="baseline_universe",
                label="baseline universe",
                required=required,
                fee_bps=fee_bps,
                next_only=True,
            )
        )
    for direction in directions:
        frames.append(
            _load_one_realized_plot_data(
                path=backtests_root / direction.run_id / "daily_pool_internal_summary.csv",
                source_pool=pool,
                key=direction.key,
                label=direction.label,
                required=required,
                fee_bps=fee_bps,
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"], errors="coerce")
    combined = combined.dropna(subset=["week_start"]).sort_values(["pool", "week_start"])
    for _, index in combined.groupby("pool", sort=False).groups.items():
        item = combined.loc[index].sort_values("week_start")
        short_values = pd.to_numeric(item["short_net_return_bps"], errors="coerce")
        next_values = pd.to_numeric(item["next_net_return_bps"], errors="coerce")
        combined.loc[item.index, "short_cumulative_net_return_bps"] = short_values.cumsum()
        combined.loc[item.index, "next_cumulative_net_return_bps"] = next_values.cumsum()
    combined["week_start"] = combined["week_start"].dt.strftime("%Y-%m-%d")
    return combined


def _load_one_cumulative_plot_data(
    *,
    path: Path,
    source_pool: str,
    key: str,
    label: str,
    required: set[str],
    next_only: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    item = frame.loc[frame["pool"].astype(str).eq(source_pool)].copy()
    if item.empty:
        raise ValueError(f"{path} has no rows for pool {source_pool!r}")
    item["pool"] = key
    item["pool_label"] = label
    item["variant"] = label
    if next_only:
        item["short_internal_excess_bps"] = pd.NA
        item["short_cumulative_internal_excess_bps"] = pd.NA
    return item[
        [
            "pool",
            "pool_label",
            "week_start",
            "short_internal_excess_bps",
            "next_internal_excess_bps",
            "variant",
            "short_cumulative_internal_excess_bps",
            "next_cumulative_internal_excess_bps",
        ]
    ]


def _load_one_realized_plot_data(
    *,
    path: Path,
    source_pool: str,
    key: str,
    label: str,
    required: set[str],
    fee_bps: float,
    next_only: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    item = frame.loc[frame["pool"].astype(str).eq(source_pool)].copy()
    if item.empty:
        raise ValueError(f"{path} has no rows for pool {source_pool!r}")
    item["week_start"] = pd.to_datetime(item["date"], errors="coerce")
    item["pool"] = key
    item["pool_label"] = label
    item["variant"] = label
    item["fee_bps_per_trade"] = float(fee_bps)

    for column in (
        "pool_short_mean_bps",
        "selected_short_mean_bps",
        "short_internal_excess_bps",
        "pool_next_mean_bps",
        "selected_next_mean_bps",
        "next_internal_excess_bps",
    ):
        item[column] = pd.to_numeric(item[column], errors="coerce") / CUMULATIVE_DECISION_NORMALIZER
    item["fee_bps"] = float(fee_bps) / CUMULATIVE_DECISION_NORMALIZER
    item["short_net_return_bps"] = item["selected_short_mean_bps"] - item["fee_bps"]
    item["next_net_return_bps"] = item["selected_next_mean_bps"] - item["fee_bps"]
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
            "pool_short_mean_bps",
            "selected_short_mean_bps",
            "short_internal_excess_bps",
            "pool_next_mean_bps",
            "selected_next_mean_bps",
            "next_internal_excess_bps",
            "fee_bps_per_trade",
            "fee_bps",
            "short_net_return_bps",
            "next_net_return_bps",
            "variant",
        ]
    ]


def _normalize_cumulative_decision_bps(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    columns = [
        "short_internal_excess_bps",
        "next_internal_excess_bps",
        "short_cumulative_internal_excess_bps",
        "next_cumulative_internal_excess_bps",
    ]
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce") / CUMULATIVE_DECISION_NORMALIZER
    return data


def relative_to_baseline_cumulative_data(
    cumulative_data: pd.DataFrame,
    *,
    directions: tuple[DirectionSpec, ...],
    baseline_key: str,
) -> pd.DataFrame:
    data = cumulative_data.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], errors="coerce")
    data = data.dropna(subset=["week_start"])
    baseline = data.loc[data["pool"].astype(str).eq(baseline_key)].copy()
    if baseline.empty:
        raise ValueError(f"cumulative data has no baseline rows for {baseline_key!r}")

    baseline = baseline[
        [
            "week_start",
            "short_internal_excess_bps",
            "next_internal_excess_bps",
        ]
    ].rename(
        columns={
            "short_internal_excess_bps": "baseline_short_internal_excess_bps",
            "next_internal_excess_bps": "baseline_next_internal_excess_bps",
        }
    )

    frames = []
    for direction in directions:
        item = data.loc[data["pool"].astype(str).eq(direction.key)].copy()
        if item.empty:
            raise ValueError(f"cumulative data has no rows for direction {direction.key!r}")
        item = item.merge(baseline, on="week_start", how="inner")
        if item.empty:
            raise ValueError(
                f"direction {direction.key!r} has no dates in common with baseline {baseline_key!r}"
            )
        item["short_relative_excess_bps"] = (
            item["short_internal_excess_bps"] - item["baseline_short_internal_excess_bps"]
        )
        item["next_relative_excess_bps"] = (
            item["next_internal_excess_bps"] - item["baseline_next_internal_excess_bps"]
        )
        item = item.sort_values("week_start")
        item["short_cumulative_relative_excess_bps"] = item["short_relative_excess_bps"].cumsum()
        item["next_cumulative_relative_excess_bps"] = item["next_relative_excess_bps"].cumsum()
        item["pool"] = direction.key
        item["pool_label"] = direction.label
        item["variant"] = direction.label
        frames.append(
            item[
                [
                    "pool",
                    "pool_label",
                    "week_start",
                    "short_relative_excess_bps",
                    "next_relative_excess_bps",
                    "variant",
                    "short_cumulative_relative_excess_bps",
                    "next_cumulative_relative_excess_bps",
                ]
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    return combined


def relative_to_baseline_year_data(
    *,
    backtests_root: Path,
    directions: tuple[DirectionSpec, ...],
    pool: str,
    baseline_run_id: str,
) -> pd.DataFrame:
    baseline_year = _load_year_summary(
        backtests_root / baseline_run_id / "pool_internal_year_summary.csv",
        pool=pool,
    )
    baseline_mean = _load_summary_row(
        backtests_root / baseline_run_id / "pool_internal_summary.csv",
        pool=pool,
    )
    frames = []
    for direction in directions:
        year = _load_year_summary(
            backtests_root / direction.run_id / "pool_internal_year_summary.csv",
            pool=pool,
        )
        merged = year.merge(
            baseline_year,
            on="test_month",
            how="inner",
            suffixes=("", "_baseline"),
        )
        if merged.empty:
            raise ValueError(f"{direction.run_id} has no yearly rows in common with baseline")
        item = pd.DataFrame(
            {
                "test_month": merged["test_month"],
                "pool": direction.key,
                "pool_label": direction.label,
                "variant": direction.label,
                "short_relative_excess_bps": (
                    merged["short_internal_excess_bps"]
                    - merged["short_internal_excess_bps_baseline"]
                ),
                "next_relative_excess_bps": (
                    merged["next_internal_excess_bps"]
                    - merged["next_internal_excess_bps_baseline"]
                ),
            }
        )
        direction_mean = _load_summary_row(
            backtests_root / direction.run_id / "pool_internal_summary.csv",
            pool=pool,
        )
        mean_item = pd.DataFrame(
            {
                "test_month": ["Mean"],
                "pool": [direction.key],
                "pool_label": [direction.label],
                "variant": [direction.label],
                "short_relative_excess_bps": [
                    direction_mean["short_internal_excess_bps"]
                    - baseline_mean["short_internal_excess_bps"]
                ],
                "next_relative_excess_bps": [
                    direction_mean["next_internal_excess_bps"]
                    - baseline_mean["next_internal_excess_bps"]
                ],
            }
        )
        frames.append(pd.concat([item, mean_item], ignore_index=True))
    return sort_month_major(pd.concat(frames, ignore_index=True))


def load_yearly_net_alpha_data(
    *,
    backtests_root: Path,
    directions: tuple[DirectionSpec, ...],
    pool: str,
    include_baseline_pool: bool,
    baseline_run_id: str,
    fee_bps: float,
) -> pd.DataFrame:
    frames = []
    if include_baseline_pool:
        frames.append(
            _load_yearly_net_alpha_rows(
                year_path=backtests_root / baseline_run_id / "pool_internal_year_summary.csv",
                summary_path=backtests_root / baseline_run_id / "pool_internal_summary.csv",
                pool=pool,
                key="baseline_pool_l",
                label="baseline pool_L",
                fee_bps=fee_bps,
            )
        )
    for direction in directions:
        frames.append(
            _load_yearly_net_alpha_rows(
                year_path=backtests_root / direction.run_id / "pool_internal_year_summary.csv",
                summary_path=backtests_root / direction.run_id / "pool_internal_summary.csv",
                pool=pool,
                key=direction.key,
                label=direction.label,
                fee_bps=fee_bps,
            )
        )
    return sort_month_major(pd.concat(frames, ignore_index=True))


def _load_yearly_net_alpha_rows(
    *,
    year_path: Path,
    summary_path: Path,
    pool: str,
    key: str,
    label: str,
    fee_bps: float,
) -> pd.DataFrame:
    required = {
        "pool",
        "year",
        "pool_next_mean_bps",
        "selected_next_mean_bps",
        "next_internal_excess_bps",
    }
    frame = pd.read_csv(year_path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{year_path} missing columns: {missing}")
    item = frame.loc[frame["pool"].astype(str).eq(pool)].copy()
    if item.empty:
        raise ValueError(f"{year_path} has no rows for pool {pool!r}")
    item["test_month"] = item["year"].astype(str)

    summary = _load_summary_row(summary_path, pool=pool)
    mean_item = pd.DataFrame(
        {
            "test_month": ["Mean"],
            "pool_next_mean_bps": [summary["pool_next_mean_bps"]],
            "selected_next_mean_bps": [summary["selected_next_mean_bps"]],
            "next_internal_excess_bps": [summary["next_internal_excess_bps"]],
        }
    )
    out = pd.concat(
        [
            item[
                [
                    "test_month",
                    "pool_next_mean_bps",
                    "selected_next_mean_bps",
                    "next_internal_excess_bps",
                ]
            ],
            mean_item,
        ],
        ignore_index=True,
    )
    out["pool"] = key
    out["pool_label"] = label
    out["variant"] = label
    out["fee_bps"] = float(fee_bps)
    out["next_net_return_bps"] = out["selected_next_mean_bps"] - out["fee_bps"]
    out["next_alpha_bps"] = out["next_net_return_bps"] - out["pool_next_mean_bps"]
    return out[
        [
            "test_month",
            "variant",
            "pool",
            "pool_label",
            "pool_next_mean_bps",
            "selected_next_mean_bps",
            "fee_bps",
            "next_net_return_bps",
            "next_alpha_bps",
        ]
    ]


def _load_year_summary(path: Path, *, pool: str) -> pd.DataFrame:
    required = {"pool", "year", "short_internal_excess_bps", "next_internal_excess_bps"}
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    item = frame.loc[frame["pool"].astype(str).eq(pool)].copy()
    if item.empty:
        raise ValueError(f"{path} has no rows for pool {pool!r}")
    item["test_month"] = item["year"].astype(str)
    return item[["test_month", "short_internal_excess_bps", "next_internal_excess_bps"]]


def _load_summary_row(path: Path, *, pool: str) -> pd.Series:
    required = {"pool", "short_internal_excess_bps", "next_internal_excess_bps"}
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    item = frame.loc[frame["pool"].astype(str).eq(pool)].copy()
    if item.empty:
        raise ValueError(f"{path} has no rows for pool {pool!r}")
    return item.iloc[0]


def line_axis(values: pd.Series) -> tuple[float, float]:
    from opening_strength_fit.pool_internal_plot_svg import nice_line_axis

    axis, _ = nice_line_axis(values, include_zero=True, target_ticks=9)
    return axis


def line_step(values: pd.Series) -> float:
    from opening_strength_fit.pool_internal_plot_svg import nice_line_axis

    _, step = nice_line_axis(values, include_zero=True, target_ticks=9)
    return step


def sort_month_major(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["_mean_order"] = data["test_month"].astype(str).eq("Mean").astype("int8")
    data = data.sort_values(["_mean_order", "test_month", "pool"]).drop(columns="_mean_order")
    return data.reset_index(drop=True)


def source_files(backtests_root: Path, directions: tuple[DirectionSpec, ...]) -> dict[str, dict[str, str]]:
    return {
        direction.key: {
            "short": str(backtests_root / direction.run_id / "short_excess_rank_ic_plot_data.csv"),
            "next": str(backtests_root / direction.run_id / "next_excess_rank_ic_plot_data.csv"),
            "daily_cumulative": str(
                backtests_root / direction.run_id / "daily_cumulative_plot_data.csv"
            ),
            "daily_realized": str(
                backtests_root / direction.run_id / "daily_pool_internal_summary.csv"
            ),
        }
        for direction in directions
    }
