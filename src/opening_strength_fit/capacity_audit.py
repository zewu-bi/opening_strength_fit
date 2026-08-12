from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS
from opening_strength_fit.feature_utils import (
    finite_numeric,
)
from opening_strength_fit.feature_utils import (
    weighted_share_stats as _share_stats,
)
from opening_strength_fit.schema import normalize_decision_keys

GROUP_COLS = ("date", "decision_target_timestamp")


@dataclass(frozen=True)
class CapacityConstraints:
    target_notional: float
    score_col: str = "prediction"
    capacity_notional_col: str = "turnover_diff_30t"
    capacity_volume_col: str = ""
    capacity_price_col: str = "ask_price_1"
    max_participation_rate: float = 0.10
    max_symbol_weight: float = 0.01
    min_trade_notional: float = 0.0
    max_names: int = 0
    ask_depth_levels: int = 0
    ask_depth_participation_rate: float = 0.25
    allow_decision_depth_fallback: bool = False
    industry_col: str = ""
    max_industry_weight: float = 0.0


normalize_capacity_frame = partial(
    normalize_decision_keys,
    key_columns=KEY_COLUMNS,
    drop_missing=True,
)


def _positive_notional(price: pd.Series, volume: pd.Series) -> pd.Series:
    price = finite_numeric(price)
    volume = finite_numeric(volume)
    return (price * volume).where(price.gt(0) & volume.gt(0), 0.0)


def ask_depth_pairs(
    frame: pd.DataFrame,
    *,
    levels: int,
    allow_decision_depth_fallback: bool,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for level in range(1, int(levels) + 1):
        entry_pair = (f"entry_ask_price_{level}", f"entry_ask_volume_{level}")
        decision_pair = (f"ask_price_{level}", f"ask_volume_{level}")
        if entry_pair[0] in frame.columns and entry_pair[1] in frame.columns:
            pairs.append(entry_pair)
        elif (
            allow_decision_depth_fallback
            and decision_pair[0] in frame.columns
            and decision_pair[1] in frame.columns
        ):
            pairs.append(decision_pair)
        else:
            break
    return pairs


def add_capacity_limits(frame: pd.DataFrame, constraints: CapacityConstraints) -> pd.DataFrame:
    target = float(constraints.target_notional)
    if target <= 0:
        raise ValueError("target_notional must be positive")
    if constraints.max_participation_rate < 0:
        raise ValueError("max_participation_rate must be >= 0")
    if constraints.max_symbol_weight < 0:
        raise ValueError("max_symbol_weight must be >= 0")
    if constraints.max_industry_weight < 0:
        raise ValueError("max_industry_weight must be >= 0")

    out = frame.copy()
    out[constraints.score_col] = finite_numeric(out[constraints.score_col])
    if constraints.capacity_price_col and constraints.capacity_price_col in out.columns:
        out["_capacity_price"] = finite_numeric(out[constraints.capacity_price_col])
    else:
        out["_capacity_price"] = np.nan

    limit_parts: list[pd.Series] = []
    if constraints.max_symbol_weight > 0:
        limit_parts.append(
            pd.Series(target * float(constraints.max_symbol_weight), index=out.index)
        )
    else:
        limit_parts.append(pd.Series(target, index=out.index))

    if constraints.max_participation_rate > 0:
        if constraints.capacity_notional_col and constraints.capacity_notional_col in out.columns:
            capacity = finite_numeric(out[constraints.capacity_notional_col])
        elif (
            constraints.capacity_volume_col
            and constraints.capacity_volume_col in out.columns
            and constraints.capacity_price_col in out.columns
        ):
            capacity = finite_numeric(out[constraints.capacity_volume_col]) * finite_numeric(
                out[constraints.capacity_price_col],
            )
        else:
            capacity = pd.Series(np.nan, index=out.index)
        capacity = capacity.where(capacity.gt(0))
        out["_capacity_notional"] = capacity
        out["_capacity_limit_notional"] = capacity * float(constraints.max_participation_rate)
        limit_parts.append(out["_capacity_limit_notional"])
    else:
        out["_capacity_notional"] = np.nan
        out["_capacity_limit_notional"] = np.nan

    if constraints.ask_depth_levels > 0:
        pairs = ask_depth_pairs(
            out,
            levels=int(constraints.ask_depth_levels),
            allow_decision_depth_fallback=constraints.allow_decision_depth_fallback,
        )
        if len(pairs) < int(constraints.ask_depth_levels):
            out["_ask_depth_notional"] = np.nan
            out["_ask_depth_limit_notional"] = np.nan
            limit_parts.append(pd.Series(np.nan, index=out.index))
        else:
            depth = sum(_positive_notional(out[price], out[volume]) for price, volume in pairs)
            out["_ask_depth_notional"] = depth
            out["_ask_depth_limit_notional"] = depth * float(
                constraints.ask_depth_participation_rate
            )
            limit_parts.append(out["_ask_depth_limit_notional"])
    else:
        out["_ask_depth_notional"] = np.nan
        out["_ask_depth_limit_notional"] = np.nan

    limits = pd.concat(limit_parts, axis=1).min(axis=1, skipna=False)
    out["_row_limit_notional"] = limits.where(limits.gt(0), 0.0)
    return out


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _finite_values(values: pd.Series) -> pd.Series:
    return finite_numeric(values).dropna()


def _series_mean(values: pd.Series) -> float:
    finite = _finite_values(values)
    return float(finite.mean()) if not finite.empty else float("nan")


def _series_quantile(values: pd.Series, quantile: float) -> float:
    finite = _finite_values(values)
    return float(finite.quantile(quantile)) if not finite.empty else float("nan")


def _series_max(values: pd.Series) -> float:
    finite = _finite_values(values)
    return float(finite.max()) if not finite.empty else float("nan")


def _optional_row_float(row: pd.Series, column: str) -> float:
    value = row[column]
    return float(value) if pd.notna(value) else float("nan")


def capacity_selection_record(
    row: pd.Series,
    *,
    pool: str,
    constraints: CapacityConstraints,
    rank: int,
    allocated: float,
    industry_value: str = "",
    extra_allocations: dict[str, object] | None = None,
    symbol: object | None = None,
    audit_depth: bool = True,
) -> dict[str, object]:
    target = float(constraints.target_notional)
    capacity = _optional_row_float(row, "_capacity_notional")
    record = {
        "pool": pool,
        "test_month": pd.Timestamp(row["date"]).to_period("M").strftime("%Y-%m"),
        "date": row["date"],
        "decision_target_timestamp": row["decision_target_timestamp"],
        "clock": pd.Timestamp(row["decision_target_timestamp"]).strftime("%H:%M"),
        "rank": rank,
        "symbol": row["symbol"] if symbol is None else symbol,
        "score": float(row[constraints.score_col]),
        "capacity_price": _optional_row_float(row, "_capacity_price"),
        "target_notional": target,
        "allocated_notional": allocated,
        **(extra_allocations or {}),
        "target_weight": allocated / target,
        "row_limit_notional": float(row["_row_limit_notional"]),
        "capacity_notional": capacity,
        "capacity_limit_notional": _optional_row_float(row, "_capacity_limit_notional"),
        "capacity_participation_rate": allocated / capacity if capacity > 0 else float("nan"),
    }
    if audit_depth:
        depth = _optional_row_float(row, "_ask_depth_notional")
        record.update(
            ask_depth_notional=depth,
            ask_depth_limit_notional=_optional_row_float(row, "_ask_depth_limit_notional"),
            ask_depth_participation_rate=(allocated / depth if depth > 0 else float("nan")),
            industry=industry_value,
        )
    return record


_selected_record = capacity_selection_record


def _fast_selected_rows(
    viable: pd.DataFrame,
    *,
    pool: str,
    constraints: CapacityConstraints,
) -> pd.DataFrame:
    target = float(constraints.target_notional)
    if constraints.max_names > 0:
        viable = viable.head(int(constraints.max_names)).copy()
    if viable.empty:
        return pd.DataFrame()
    limits = finite_numeric(viable["_row_limit_notional"]).clip(lower=0.0)
    cumulative_before = limits.cumsum().shift(fill_value=0.0)
    room = (target - cumulative_before).clip(lower=0.0)
    allocated = pd.concat([limits, room], axis=1).min(axis=1)
    selected = viable.loc[allocated.gt(0)].copy()
    if selected.empty:
        return selected
    selected["allocated_notional"] = allocated.loc[selected.index].astype("float64")
    selected["rank"] = range(1, len(selected) + 1)
    selected["target_notional"] = target
    selected["target_weight"] = selected["allocated_notional"] / target
    selected["test_month"] = (
        pd.to_datetime(selected["date"], errors="coerce").dt.to_period("M").astype(str)
    )
    selected["clock"] = pd.to_datetime(
        selected["decision_target_timestamp"],
        errors="coerce",
    ).dt.strftime("%H:%M")
    selected["pool"] = pool
    selected["score"] = finite_numeric(selected[constraints.score_col])
    selected["capacity_price"] = finite_numeric(selected["_capacity_price"])
    selected["capacity_participation_rate"] = selected["allocated_notional"] / finite_numeric(
        selected["_capacity_notional"]
    )
    selected["ask_depth_participation_rate"] = selected["allocated_notional"] / finite_numeric(
        selected["_ask_depth_notional"]
    )
    selected["industry"] = ""
    return selected[
        [
            "pool",
            "test_month",
            "date",
            "decision_target_timestamp",
            "clock",
            "rank",
            "symbol",
            "score",
            "capacity_price",
            "target_notional",
            "allocated_notional",
            "target_weight",
            "_row_limit_notional",
            "_capacity_notional",
            "_capacity_limit_notional",
            "capacity_participation_rate",
            "_ask_depth_notional",
            "_ask_depth_limit_notional",
            "ask_depth_participation_rate",
            "industry",
        ]
    ].rename(
        columns={
            "_row_limit_notional": "row_limit_notional",
            "_capacity_notional": "capacity_notional",
            "_capacity_limit_notional": "capacity_limit_notional",
            "_ask_depth_notional": "ask_depth_notional",
            "_ask_depth_limit_notional": "ask_depth_limit_notional",
        }
    )


def _allocate_group(
    group: pd.DataFrame,
    *,
    pool: str,
    constraints: CapacityConstraints,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    target = float(constraints.target_notional)
    remaining = target
    selected_rows: list[dict[str, object]] = []
    industry_used: dict[str, float] = {}
    ranked = group.sort_values([constraints.score_col, "symbol"], ascending=[False, True])
    viable = ranked.loc[ranked["_row_limit_notional"].gt(0)].copy()

    if (
        not constraints.industry_col
        and constraints.max_industry_weight <= 0
        and constraints.min_trade_notional <= 0
    ):
        selected = _fast_selected_rows(viable, pool=pool, constraints=constraints)
        selected_rows = selected.to_dict("records")
        selected_for_metrics = selected
    else:
        selected_for_metrics = pd.DataFrame()

        for _, row in viable.iterrows():
            if constraints.max_names > 0 and len(selected_rows) >= int(constraints.max_names):
                break
            limit = min(float(row["_row_limit_notional"]), remaining)
            industry_value = ""
            if constraints.industry_col and constraints.industry_col in ranked.columns:
                industry_value = str(row[constraints.industry_col])
                if constraints.max_industry_weight > 0:
                    industry_cap = target * float(constraints.max_industry_weight)
                    industry_room = industry_cap - industry_used.get(industry_value, 0.0)
                    limit = min(limit, max(0.0, industry_room))
            if constraints.min_trade_notional > 0 and limit < float(constraints.min_trade_notional):
                continue
            if limit <= 0:
                continue

            allocated = min(limit, remaining)
            remaining -= allocated
            if industry_value:
                industry_used[industry_value] = industry_used.get(industry_value, 0.0) + allocated
            selected_rows.append(
                _selected_record(
                    row,
                    pool=pool,
                    constraints=constraints,
                    rank=len(selected_rows) + 1,
                    allocated=allocated,
                    industry_value=industry_value,
                )
            )
            if remaining <= 0:
                break
        selected_for_metrics = pd.DataFrame(selected_rows)

    selected = selected_for_metrics
    allocated_notional = float(selected["allocated_notional"].sum()) if not selected.empty else 0.0
    filled = allocated_notional >= target * (1.0 - 1e-9)
    max_rank_reached = int(len(selected_rows))
    symbol_stats = (
        _share_stats(selected["symbol"], selected["allocated_notional"])
        if not selected.empty
        else _share_stats(pd.Series(dtype=object), pd.Series(dtype=float))
    )
    row = {
        "pool": pool,
        "test_month": pd.Timestamp(group["date"].iloc[0]).to_period("M").strftime("%Y-%m"),
        "date": group["date"].iloc[0],
        "decision_target_timestamp": group["decision_target_timestamp"].iloc[0],
        "clock": pd.Timestamp(group["decision_target_timestamp"].iloc[0]).strftime("%H:%M"),
        "candidate_rows": int(len(group)),
        "viable_rows": int(len(viable)),
        "selected_rows": int(len(selected_rows)),
        "target_notional": target,
        "allocated_notional": allocated_notional,
        "cash_notional": max(0.0, target - allocated_notional),
        "fill_ratio": allocated_notional / target,
        "filled": filled,
        "max_rank_reached": max_rank_reached,
        "top_depth_to_target": float(max_rank_reached) if filled else float("nan"),
        "capacity_exhausted_depth": float(max_rank_reached) if not filled else float("nan"),
        "max_symbol_weight": symbol_stats["max_share"],
        "top5_symbol_weight": symbol_stats["top5_share"],
        "symbol_hhi": symbol_stats["hhi"],
        "effective_symbols": symbol_stats["effective_count"],
        "max_capacity_participation_rate": float(
            finite_numeric(
                selected.get("capacity_participation_rate", pd.Series(dtype=float))
            ).max()
        )
        if not selected.empty
        else float("nan"),
        "max_ask_depth_participation_rate": float(
            finite_numeric(
                selected.get("ask_depth_participation_rate", pd.Series(dtype=float))
            ).max()
        )
        if not selected.empty
        else float("nan"),
    }
    if constraints.industry_col and "industry" in selected:
        industry_stats = _share_stats(selected["industry"], selected["allocated_notional"])
        row.update(
            {
                "selected_industries": int(industry_stats["unique"]),
                "max_industry_weight": industry_stats["max_share"],
                "top5_industry_weight": industry_stats["top5_share"],
                "industry_hhi": industry_stats["hhi"],
                "effective_industries": industry_stats["effective_count"],
            }
        )
    return selected_rows, row


def build_capacity_portfolios(
    frame: pd.DataFrame,
    constraints: CapacityConstraints,
    *,
    pool: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = normalize_capacity_frame(frame)
    work = add_capacity_limits(work, constraints)
    work = work.dropna(subset=[constraints.score_col]).copy()
    selected_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for _, group in work.groupby(list(GROUP_COLS), sort=False):
        selected, metrics = _allocate_group(group, pool=pool, constraints=constraints)
        selected_rows.extend(selected)
        metric_rows.append(metrics)
    return pd.DataFrame(selected_rows), pd.DataFrame(metric_rows)


def _aggregate_metrics(group: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, item in group.groupby(by, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        target = float(item["target_notional"].sum())
        allocated = float(item["allocated_notional"].sum())
        filled = (
            item["filled"].fillna(False).astype(bool)
            if "filled" in item.columns
            else item["fill_ratio"].ge(1.0 - 1e-9)
        )
        row = {column: value for column, value in zip(by, key_values, strict=True)}
        row.update(
            {
                "groups": int(len(item)),
                "months": int(item["test_month"].nunique()),
                "candidate_rows": float(item["candidate_rows"].mean()),
                "viable_rows": float(item["viable_rows"].mean()),
                "selected_rows": float(item["selected_rows"].mean()),
                "target_notional": float(item["target_notional"].mean()),
                "allocated_notional": float(item["allocated_notional"].mean()),
                "fill_ratio": _safe_div(allocated, target),
                "filled_groups": int(filled.sum()),
                "unfilled_groups": int((~filled).sum()),
                "fill_success_rate": _safe_div(float(filled.sum()), float(len(item))),
                "min_fill_ratio": float(item["fill_ratio"].min()),
                "mean_top_depth_to_target": _series_mean(item["top_depth_to_target"]),
                "p50_top_depth_to_target": _series_quantile(item["top_depth_to_target"], 0.50),
                "p90_top_depth_to_target": _series_quantile(item["top_depth_to_target"], 0.90),
                "p95_top_depth_to_target": _series_quantile(item["top_depth_to_target"], 0.95),
                "max_top_depth_to_target": _series_max(item["top_depth_to_target"]),
                "mean_rank_reached": _series_mean(item["max_rank_reached"]),
                "max_rank_reached": _series_max(item["max_rank_reached"]),
                "max_symbol_weight": float(item["max_symbol_weight"].max()),
                "mean_effective_symbols": float(item["effective_symbols"].mean()),
                "max_capacity_participation_rate": float(
                    item["max_capacity_participation_rate"].max()
                ),
                "max_ask_depth_participation_rate": float(
                    item["max_ask_depth_participation_rate"].max()
                ),
            }
        )
        if "max_industry_weight" in item.columns:
            row.update(
                {
                    "max_industry_weight": float(item["max_industry_weight"].max()),
                    "mean_effective_industries": float(item["effective_industries"].mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


summarize_capacity_groups = partial(_aggregate_metrics, by=["pool"])
summarize_capacity_months = partial(_aggregate_metrics, by=["pool", "test_month"])
summarize_capacity_daily = partial(_aggregate_metrics, by=["pool", "date"])
