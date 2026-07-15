from __future__ import annotations

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


def normalize_key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return normalize_decision_keys(frame, key_columns=KEY_COLUMNS)


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
    merged = selected.merge(
        labels[list(KEY_COLUMNS) + [label_col]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    missing_label_rows = int(merged[label_col].isna().sum())
    if missing_label_rows:
        raise ValueError(f"capacity selected rows missing {label_col}: {missing_label_rows}")

    merged["_gross_pnl"] = merged["allocated_notional"] * merged[label_col]
    merged["_fee_pnl"] = merged["allocated_notional"] * float(fee_bps) / RETURN_BPS_DENOMINATOR
    merged["_net_pnl"] = merged["_gross_pnl"] - merged["_fee_pnl"]
    group_targets = (
        merged.groupby(["pool", "date", "decision_target_timestamp"], sort=False)
        .agg(group_target_notional=("target_notional", "first"))
        .reset_index()
    )
    daily_targets = group_targets.groupby(["pool", "date"], sort=False).agg(
        capacity_decision_groups=("decision_target_timestamp", "size"),
        target_notional=("group_target_notional", "sum"),
    )
    daily = merged.groupby(["pool", "date"], sort=False).agg(
        gross_pnl=("_gross_pnl", "sum"),
        fee_pnl=("_fee_pnl", "sum"),
        net_pnl=("_net_pnl", "sum"),
        selected_allocated_notional=("allocated_notional", "sum"),
        selected_rows=("allocated_notional", "size"),
    )
    daily = daily.join(daily_targets, how="left").reset_index()
    daily["week_start"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily["capacity_total_notional"] = float(capacity_total_notional)
    daily["capacity_daily_capital_fraction"] = daily["target_notional"] / float(
        capacity_total_notional
    )
    daily["gross_next_return_bps"] = (
        daily["gross_pnl"] / daily["target_notional"] * RETURN_BPS_DENOMINATOR
    )
    daily["fee_bps"] = daily["fee_pnl"] / daily["target_notional"] * RETURN_BPS_DENOMINATOR
    daily["next_net_return_bps"] = (
        daily["net_pnl"] / daily["target_notional"] * RETURN_BPS_DENOMINATOR
    )
    daily["next_capital_net_return_bps"] = (
        daily["net_pnl"] / float(capacity_total_notional) * RETURN_BPS_DENOMINATOR
    )
    daily["next_net_pnl"] = daily["net_pnl"]
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


def summarize_capacity_acceptance_overall(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    out = daily.groupby("pool", sort=False).agg(
        days=("date", "nunique"),
        capacity_decision_groups=("capacity_decision_groups", "sum"),
        mean_selected_rows=("selected_rows", "mean"),
        mean_target_notional=("target_notional", "mean"),
        mean_capacity_daily_capital_fraction=("capacity_daily_capital_fraction", "mean"),
        mean_next_net_return_bps=("next_net_return_bps", "mean"),
        final_next_cumulative_net_return_bps=("next_capital_net_return_bps", "sum"),
        total_net_pnl=("next_net_pnl", "sum"),
    )
    return out.reset_index()
