from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, NEXT_CLOSE_LABEL_COL
from opening_strength_fit.io import frame_columns, read_frame
from opening_strength_fit.schema import normalize_decision_keys

RETURN_BPS_DENOMINATOR = 10_000.0
DEFAULT_CAPACITY_TOTAL_NOTIONAL = 1_000_000_000.0
DEFAULT_CAPACITY_SLICES = 20.0
DEFAULT_CAPACITY_DECISION_NOTIONAL = DEFAULT_CAPACITY_TOTAL_NOTIONAL / DEFAULT_CAPACITY_SLICES
DEFAULT_CAPACITY_LABEL_COL = NEXT_CLOSE_LABEL_COL

SELECTED_REQUIRED_COLUMNS = (
    "pool",
    "date",
    "symbol",
    "decision_target_timestamp",
    "allocated_notional",
    "target_notional",
)


def add_daily_return_columns(daily: pd.DataFrame, capacity_total_notional: float) -> pd.DataFrame:
    total = float(capacity_total_notional)
    daily["week_start"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily["capacity_total_notional"] = total
    daily["capacity_daily_capital_fraction"] = daily["target_notional"] / total
    daily["gross_next_return_bps"] = (
        daily["gross_pnl"] / daily["target_notional"] * RETURN_BPS_DENOMINATOR
    )
    daily["fee_bps"] = daily["fee_pnl"] / daily["target_notional"] * RETURN_BPS_DENOMINATOR
    daily["next_net_return_bps"] = (
        daily["net_pnl"] / daily["target_notional"] * RETURN_BPS_DENOMINATOR
    )
    daily["next_capital_net_return_bps"] = daily["net_pnl"] / total * RETURN_BPS_DENOMINATOR
    daily["next_net_pnl"] = daily["net_pnl"]
    return daily


normalize_key_columns = partial(normalize_decision_keys, key_columns=KEY_COLUMNS)


def read_required_frames(
    paths: tuple[str | Path, ...],
    *,
    required: tuple[str, ...],
) -> pd.DataFrame:
    frames = []
    required_set = set(required)
    for raw in paths:
        path = Path(raw)
        available = frame_columns(path)
        missing = sorted(required_set - available)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        frame = read_frame(path, columns=list(required))
        frames.append(frame)
    if not frames:
        raise ValueError("no input files supplied")
    return pd.concat(frames, ignore_index=True)


def load_capacity_selected(paths: tuple[str | Path, ...]) -> pd.DataFrame:
    selected = read_required_frames(paths, required=SELECTED_REQUIRED_COLUMNS)
    selected = normalize_key_columns(selected)
    for column in ("allocated_notional", "target_notional"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected.dropna(subset=["allocated_notional", "target_notional"]).copy()


def load_label_frame(
    paths: tuple[str | Path, ...],
    *,
    label_col: str,
    dates: set[str],
) -> pd.DataFrame:
    labels = read_required_frames(
        paths,
        required=(*KEY_COLUMNS, label_col),
    )
    labels = normalize_key_columns(labels)
    labels = labels.loc[labels["date"].isin(dates)].copy()
    labels[label_col] = pd.to_numeric(labels[label_col], errors="coerce")
    labels = labels.dropna(subset=[label_col])
    if labels.empty:
        raise ValueError("label inputs did not match any selected capacity dates")
    return labels.drop_duplicates(list(KEY_COLUMNS), keep="last")


def attach_selected_returns(
    selected: pd.DataFrame, labels: pd.DataFrame, label_col: str, fee_bps: float, source: str
) -> pd.DataFrame:
    if selected.empty:
        merged = selected.copy()
        for column in ("_gross_pnl", "_fee_pnl", "_net_pnl"):
            merged[column] = pd.Series(dtype="float64")
        return merged
    merged = selected.merge(
        labels[list(KEY_COLUMNS) + [label_col]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    missing = int(merged[label_col].isna().sum())
    if missing:
        raise ValueError(f"{source} selected rows missing {label_col}: {missing}")
    merged["_gross_pnl"] = merged["allocated_notional"] * merged[label_col]
    merged["_fee_pnl"] = merged["allocated_notional"] * float(fee_bps) / RETURN_BPS_DENOMINATOR
    merged["_net_pnl"] = merged["_gross_pnl"] - merged["_fee_pnl"]
    return merged


def summarize_daily_allocations(
    merged: pd.DataFrame,
    group_targets: pd.DataFrame,
    *,
    extra_aggregations: dict[str, tuple[str, str]] | None = None,
    fill_empty: bool = False,
) -> pd.DataFrame:
    targets = group_targets.groupby(["pool", "date"], sort=False).agg(
        capacity_decision_groups=("decision_target_timestamp", "size"),
        target_notional=("group_target_notional", "sum"),
    )
    aggregations = {
        "gross_pnl": ("_gross_pnl", "sum"),
        "fee_pnl": ("_fee_pnl", "sum"),
        "net_pnl": ("_net_pnl", "sum"),
        "selected_allocated_notional": ("allocated_notional", "sum"),
        "selected_rows": ("allocated_notional", "size"),
        **(extra_aggregations or {}),
    }
    daily = targets.join(merged.groupby(["pool", "date"], sort=False).agg(**aggregations))
    if fill_empty:
        daily = daily.fillna(dict.fromkeys(aggregations, 0.0))
    return daily.reset_index()


def summarize_capacity_acceptance(
    selected: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    capacity_total_notional: float,
    fee_bps: float,
    label_col: str = DEFAULT_CAPACITY_LABEL_COL,
) -> pd.DataFrame:
    if capacity_total_notional <= 0:
        raise ValueError("capacity_total_notional must be positive")
    merged = attach_selected_returns(selected, labels, label_col, fee_bps, "capacity")
    group_targets = (
        merged.groupby(["pool", "date", "decision_target_timestamp"], sort=False)
        .agg(group_target_notional=("target_notional", "first"))
        .reset_index()
    )
    daily = add_daily_return_columns(
        summarize_daily_allocations(merged, group_targets),
        capacity_total_notional,
    )
    return daily[
        [
            "pool",
            "date",
            "week_start",
            "capacity_decision_groups",
            "selected_rows",
            "selected_allocated_notional",
            "target_notional",
            "capacity_total_notional",
            "capacity_daily_capital_fraction",
            "gross_next_return_bps",
            "fee_bps",
            "next_net_return_bps",
            "next_capital_net_return_bps",
            "next_net_pnl",
            "gross_pnl",
            "fee_pnl",
        ]
    ]


def summarize_acceptance_overall(
    daily: pd.DataFrame,
    **extra: tuple[str, str],
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    aggregations = {
        "days": ("date", "nunique"),
        "capacity_decision_groups": ("capacity_decision_groups", "sum"),
        "mean_selected_rows": ("selected_rows", "mean"),
        "mean_target_notional": ("target_notional", "mean"),
        **extra,
        "mean_capacity_daily_capital_fraction": ("capacity_daily_capital_fraction", "mean"),
        "mean_next_net_return_bps": ("next_net_return_bps", "mean"),
        "final_next_cumulative_net_return_bps": ("next_capital_net_return_bps", "sum"),
        "total_net_pnl": ("next_net_pnl", "sum"),
    }
    return daily.groupby("pool", sort=False).agg(**aggregations).reset_index()


summarize_capacity_acceptance_overall = summarize_acceptance_overall
