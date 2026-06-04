from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.dataset import build_labeled_feature_frame, load_ticks
from opening_strength_fit.labels import safe_price_return


def _normalize_time(value: object) -> str:
    text = str(value)
    if len(text) == 5:
        text = f"{text}:00"
    return text


def _handle_missing_column(
    column: str,
    *,
    run_label: str,
    constraint: str,
    policy: str,
) -> bool:
    message = f"{run_label}: missing column {column!r} for {constraint}"
    if policy == "error":
        raise SystemExit(message)
    if policy == "warn":
        print(f"constraint_warning: {message}")
    return False


def _has_column(
    frame: pd.DataFrame,
    column: str,
    *,
    run_label: str,
    constraint: str,
    policy: str,
) -> bool:
    if column in frame.columns:
        return True
    return _handle_missing_column(
        column,
        run_label=run_label,
        constraint=constraint,
        policy=policy,
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def slot_weight(*, top_n: int, max_symbol_weight: float) -> float:
    if top_n <= 0:
        raise SystemExit("--top-n must be positive")
    weight = 1.0 / float(top_n)
    if max_symbol_weight and max_symbol_weight > 0:
        weight = min(weight, float(max_symbol_weight))
    return weight


def _positive_notional(price: pd.Series, volume: pd.Series) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce")
    volume = pd.to_numeric(volume, errors="coerce")
    return (price * volume).where(price.gt(0) & volume.gt(0), 0.0)


def ask_depth_column_pairs(
    frame: pd.DataFrame,
    *,
    levels: int,
    allow_decision_fallback: bool,
    run_label: str,
    missing_policy: str,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for level in range(1, int(levels) + 1):
        entry_price_col = f"entry_ask_price_{level}"
        entry_volume_col = f"entry_ask_volume_{level}"
        decision_price_col = f"ask_price_{level}"
        decision_volume_col = f"ask_volume_{level}"
        if entry_price_col in frame.columns and entry_volume_col in frame.columns:
            pairs.append((entry_price_col, entry_volume_col))
            continue
        if (
            allow_decision_fallback
            and decision_price_col in frame.columns
            and decision_volume_col in frame.columns
        ):
            pairs.append((decision_price_col, decision_volume_col))
            continue
        _handle_missing_column(
            f"{entry_price_col}/{entry_volume_col}",
            run_label=run_label,
            constraint="entry ask-depth execution",
            policy=missing_policy,
        )
        break
    return pairs


def add_ask_depth_execution_columns(
    frame: pd.DataFrame,
    *,
    pairs: list[tuple[str, str]],
    target_notional: float,
    participation_rate: float,
    fill_mode: str,
) -> pd.DataFrame:
    work = frame.copy()
    usable_rate = float(participation_rate)
    if usable_rate <= 0 or usable_rate > 1:
        raise SystemExit("--ask-depth-participation-rate must be in (0, 1]")

    level_notional = []
    for price_col, volume_col in pairs:
        level_notional.append(_positive_notional(work[price_col], work[volume_col]))
    if not level_notional:
        return work

    depth_notional = sum(level_notional)
    usable_depth = depth_notional * usable_rate
    work["_ask_depth_levels"] = len(pairs)
    work["_ask_depth_notional"] = depth_notional
    work["_ask_depth_usable_notional"] = usable_depth
    work["_ask_depth_target_notional"] = float(target_notional)

    target = float(target_notional)
    if target <= 0:
        raise SystemExit("--capital-per-cycle must be positive when --ask-depth-levels is set")

    if fill_mode == "scale":
        fill_ratio = (usable_depth / target).clip(lower=0.0, upper=1.0)
        work["_depth_fill_ratio"] = fill_ratio.where(fill_ratio.notna(), 0.0)
        return work

    enough_depth = usable_depth.ge(target)
    work["_depth_fill_ratio"] = enough_depth.astype("float64")
    if fill_mode != "sweep" or not enough_depth.any():
        return work

    price_matrix = (
        work[[price_col for price_col, _ in pairs]]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype="float64")
    )
    volume_matrix = (
        work[[volume_col for _, volume_col in pairs]]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype="float64")
    )
    valid = (price_matrix > 0) & (volume_matrix > 0)
    level_capacity = np.where(valid, price_matrix * volume_matrix * usable_rate, 0.0)

    remaining = np.full(len(work), target, dtype="float64")
    spent = np.zeros(len(work), dtype="float64")
    shares = np.zeros(len(work), dtype="float64")
    for level_idx in range(level_capacity.shape[1]):
        take = np.minimum(remaining, level_capacity[:, level_idx])
        price = price_matrix[:, level_idx]
        level_shares = np.divide(
            take,
            price,
            out=np.zeros_like(take),
            where=price > 0,
        )
        shares += level_shares
        spent += take
        remaining -= take

    sweep_price = np.divide(
        spent,
        shares,
        out=np.full_like(spent, np.nan),
        where=shares > 0,
    )
    work["_sweep_buy_price"] = sweep_price
    if "buy_price" in work.columns:
        buy_price = pd.to_numeric(work["buy_price"], errors="coerce")
        work["_depth_price_impact_bps"] = 10_000.0 * safe_price_return(
            pd.Series(sweep_price, index=work.index),
            buy_price,
        )
    return work


def clock_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"2000-01-01 {_normalize_time(value)}")


def load_predictions(path: Path, *, score_col: str, label_col: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"prediction parquet not found: {path}")
    frame = pd.read_parquet(path)
    time_col = "decision_target_timestamp" if "decision_target_timestamp" in frame else "timestamp"
    required = {"date", "symbol", time_col, score_col, label_col}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work["symbol"] = work["symbol"].astype(str)
    work["_decision_ts"] = pd.to_datetime(work[time_col], errors="coerce")
    work["entry_time"] = work["_decision_ts"].dt.strftime("%H:%M:%S")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    valid = work["date"].notna() & work["entry_time"].notna()
    valid &= work[score_col].notna() & work[label_col].notna()
    if "valid_label" in work:
        valid &= work["valid_label"].astype(bool)
    return work.loc[valid].copy()


def _looks_labeled_context(frame: pd.DataFrame) -> bool:
    return {"date", "symbol", "timestamp", "label"}.issubset(frame.columns)


def load_replay_context(
    path: str,
    *,
    kind: str,
    entry_times: list[str],
    entry_tick_delay: int,
    entry_max_gap_seconds: int | None,
    decision_max_lag_seconds: int | None,
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    frame = load_ticks(path)
    if kind == "labeled" or (kind == "auto" and _looks_labeled_context(frame)):
        return frame
    if kind not in {"auto", "raw_ticks"}:
        raise SystemExit(f"unknown context kind {kind!r}; expected auto, raw_ticks, or labeled")
    return build_labeled_feature_frame(
        frame,
        entry_tick_delay=int(entry_tick_delay),
        entry_max_gap_seconds=entry_max_gap_seconds,
        sample_mode="decision_points",
        decision_times=entry_times,
        decision_max_lag_seconds=decision_max_lag_seconds,
    )


def enrich_predictions_with_context(
    predictions: pd.DataFrame,
    context: pd.DataFrame,
    *,
    run_label: str,
    label_col: str,
    context_label_mode: str,
) -> pd.DataFrame:
    if context.empty:
        return predictions
    context_time_col = (
        "decision_target_timestamp"
        if "decision_target_timestamp" in context.columns
        else "timestamp"
    )
    required = {"date", "symbol", context_time_col}
    missing = required.difference(context.columns)
    if missing:
        raise SystemExit(
            f"context input missing required columns for replay enrichment: {sorted(missing)}"
        )

    left = predictions.copy()
    ctx = context.copy()
    ctx["date"] = pd.to_datetime(ctx["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    ctx["symbol"] = ctx["symbol"].astype(str)
    ctx["_decision_ts"] = pd.to_datetime(ctx[context_time_col], errors="coerce")
    ctx["_context_matched"] = True

    exclude = {"label", "prediction", "valid_label"}
    value_columns = [
        column
        for column in ctx.columns
        if column not in {"date", "symbol", "_decision_ts"}
        and column not in exclude
        and not str(column).startswith("_")
    ]
    ctx = ctx.dropna(subset=["date", "symbol", "_decision_ts"])
    ctx = ctx.sort_values(["date", "symbol", "_decision_ts"]).drop_duplicates(
        ["date", "symbol", "_decision_ts"],
        keep="last",
    )
    before_columns = set(left.columns)
    merged = left.merge(
        ctx[["date", "symbol", "_decision_ts", "_context_matched", *value_columns]],
        on=["date", "symbol", "_decision_ts"],
        how="left",
        suffixes=("", "_context"),
    )
    if "label" in ctx.columns and label_col in merged.columns:
        label_context = ctx[["date", "symbol", "_decision_ts", "label"]].rename(
            columns={"label": "_context_label"}
        )
        merged = merged.merge(
            label_context,
            on=["date", "symbol", "_decision_ts"],
            how="left",
        )
        context_label = pd.to_numeric(merged["_context_label"], errors="coerce")
        if context_label_mode == "replace":
            merged["_prediction_label"] = merged[label_col]
            merged[label_col] = context_label.combine_first(merged[label_col])
        elif context_label_mode == "fill":
            merged["_prediction_label"] = merged[label_col]
            merged[label_col] = pd.to_numeric(
                merged[label_col],
                errors="coerce",
            ).combine_first(context_label)
        elif context_label_mode != "keep":
            raise SystemExit(f"unknown context label mode: {context_label_mode!r}")
        merged = merged.drop(columns=["_context_label"])
    for column in value_columns:
        context_column = f"{column}_context"
        if context_column in merged.columns:
            merged[column] = merged[column].combine_first(merged[context_column])
            merged = merged.drop(columns=[context_column])

    matched = merged["_context_matched"].fillna(False).astype(bool).sum()
    merged = merged.drop(columns=["_context_matched"])
    added = sorted(set(merged.columns).difference(before_columns))
    print(
        f"{run_label}: replay_context_enriched "
        f"matched_rows={int(matched):,}/{len(merged):,} added_columns={len(added)} "
        f"label_mode={context_label_mode}"
    )
    return merged


def _apply_quality_filters(
    work: pd.DataFrame,
    mask: pd.Series,
    *,
    run_label: str,
    tradable_statuses: set[str],
    require_entry_status: bool,
    max_decision_lag_seconds: float | None,
    max_entry_tick_gap_seconds: float | None,
    max_spread_bps: float | None,
    min_limit_up_room_bps: float | None,
    min_ask_volume_1: float | None,
    min_bid_volume_1: float | None,
    missing_policy: str,
) -> pd.Series:
    numeric_filters = (
        ("decision_lag_seconds", max_decision_lag_seconds, "max decision lag", "le"),
        ("entry_max_tick_gap_seconds", max_entry_tick_gap_seconds, "max entry tick gap", "le"),
        ("spread_bps", max_spread_bps, "max spread", "le"),
        ("ask1_to_limit_up_bps", min_limit_up_room_bps, "limit-up room", "ge"),
        ("ask_volume_1", min_ask_volume_1, "minimum ask volume", "ge"),
        ("bid_volume_1", min_bid_volume_1, "minimum bid volume", "ge"),
    )
    for column, threshold, label, operator in numeric_filters:
        if threshold is None:
            continue
        if not _has_column(
            work,
            column,
            run_label=run_label,
            constraint=label,
            policy=missing_policy,
        ):
            continue
        values = _numeric(work, column)
        mask &= values.le(float(threshold)) if operator == "le" else values.ge(float(threshold))

    if tradable_statuses:
        if _has_column(
            work,
            "status",
            run_label=run_label,
            constraint="tradable status",
            policy=missing_policy,
        ):
            mask &= work["status"].astype(str).str.upper().isin(tradable_statuses)
        if "entry_status" in work.columns:
            mask &= work["entry_status"].astype(str).str.upper().isin(tradable_statuses)
        elif require_entry_status:
            _handle_missing_column(
                "entry_status",
                run_label=run_label,
                constraint="entry tradable status",
                policy=missing_policy,
            )
    return mask


def _capacity_notional(
    work: pd.DataFrame,
    *,
    run_label: str,
    capacity_notional_col: str,
    capacity_volume_col: str,
    capacity_price_col: str,
    missing_policy: str,
) -> pd.Series | None:
    if capacity_notional_col and capacity_notional_col in work.columns:
        return _numeric(work, capacity_notional_col)
    if capacity_volume_col:
        has_volume = _has_column(
            work,
            capacity_volume_col,
            run_label=run_label,
            constraint="capacity volume",
            policy=missing_policy,
        )
        has_price = _has_column(
            work,
            capacity_price_col,
            run_label=run_label,
            constraint="capacity price",
            policy=missing_policy,
        )
        if has_volume and has_price:
            return _numeric(work, capacity_volume_col) * _numeric(
                work,
                capacity_price_col,
            )
    elif capacity_notional_col:
        _handle_missing_column(
            capacity_notional_col,
            run_label=run_label,
            constraint="capacity notional",
            policy=missing_policy,
        )
    return None


def _apply_capacity_filter(
    work: pd.DataFrame,
    mask: pd.Series,
    *,
    run_label: str,
    top_n: int,
    capacity_notional_col: str,
    capacity_volume_col: str,
    capacity_price_col: str,
    min_capacity_notional: float,
    max_participation_rate: float,
    capital_per_cycle: float,
    max_symbol_weight: float,
    missing_policy: str,
) -> pd.Series:
    if float(min_capacity_notional) <= 0 and float(max_participation_rate) <= 0:
        return mask
    capacity = _capacity_notional(
        work,
        run_label=run_label,
        capacity_notional_col=capacity_notional_col,
        capacity_volume_col=capacity_volume_col,
        capacity_price_col=capacity_price_col,
        missing_policy=missing_policy,
    )
    if capacity is None:
        return mask

    work["_capacity_notional"] = capacity
    mask &= capacity.notna() & capacity.gt(0)
    if float(min_capacity_notional) > 0:
        mask &= capacity.ge(float(min_capacity_notional))
    if float(max_participation_rate) > 0:
        if float(capital_per_cycle) <= 0:
            raise SystemExit(
                "--capital-per-cycle must be positive when --max-participation-rate is set"
            )
        target_notional = float(capital_per_cycle) * slot_weight(
            top_n=top_n,
            max_symbol_weight=max_symbol_weight,
        )
        work["_capacity_target_notional"] = target_notional
        mask &= (capacity * float(max_participation_rate)).ge(target_notional)
    return mask


def _apply_ask_depth_filter(
    work: pd.DataFrame,
    mask: pd.Series,
    *,
    run_label: str,
    top_n: int,
    ask_depth_levels: int,
    ask_depth_participation_rate: float,
    ask_depth_fill_mode: str,
    allow_decision_depth_fallback: bool,
    capital_per_cycle: float,
    max_symbol_weight: float,
    missing_policy: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if int(ask_depth_levels) <= 0:
        return work, mask
    if float(capital_per_cycle) <= 0:
        raise SystemExit("--capital-per-cycle must be positive when --ask-depth-levels is set")
    target_notional = float(capital_per_cycle) * slot_weight(
        top_n=top_n,
        max_symbol_weight=max_symbol_weight,
    )
    pairs = ask_depth_column_pairs(
        work,
        levels=int(ask_depth_levels),
        allow_decision_fallback=allow_decision_depth_fallback,
        run_label=run_label,
        missing_policy=missing_policy,
    )
    if len(pairs) < int(ask_depth_levels):
        raise SystemExit(
            f"{run_label}: ask-depth execution requested {int(ask_depth_levels)} "
            f"levels but only found {len(pairs)} usable level(s). Provide "
            "--context-input or predictions with entry_ask_price/entry_ask_volume "
            "context columns, reduce --ask-depth-levels, or use "
            "--allow-decision-depth-fallback for non-delay diagnostics."
        )
    if not pairs:
        return work, mask

    work = add_ask_depth_execution_columns(
        work,
        pairs=pairs,
        target_notional=target_notional,
        participation_rate=float(ask_depth_participation_rate),
        fill_mode=ask_depth_fill_mode,
    )
    if ask_depth_fill_mode in {"filter", "sweep"}:
        mask &= work["_ask_depth_usable_notional"].ge(target_notional)
    elif ask_depth_fill_mode == "scale":
        mask &= work["_depth_fill_ratio"].gt(0)
    else:
        raise SystemExit(f"unknown ask depth fill mode: {ask_depth_fill_mode!r}")
    return work, mask


def apply_static_constraints(
    frame: pd.DataFrame,
    *,
    run_label: str,
    top_n: int,
    fee_bps: float,
    slippage_bps: float,
    tradable_statuses: set[str],
    require_entry_status: bool,
    max_decision_lag_seconds: float | None,
    max_entry_tick_gap_seconds: float | None,
    max_spread_bps: float | None,
    min_limit_up_room_bps: float | None,
    min_ask_volume_1: float | None,
    min_bid_volume_1: float | None,
    capacity_notional_col: str,
    capacity_volume_col: str,
    capacity_price_col: str,
    min_capacity_notional: float,
    max_participation_rate: float,
    capital_per_cycle: float,
    ask_depth_levels: int,
    ask_depth_participation_rate: float,
    ask_depth_fill_mode: str,
    allow_decision_depth_fallback: bool,
    max_symbol_weight: float,
    missing_policy: str,
) -> pd.DataFrame:
    work = frame.copy()
    mask = pd.Series(True, index=work.index)

    mask = _apply_quality_filters(
        work,
        mask,
        run_label=run_label,
        tradable_statuses=tradable_statuses,
        require_entry_status=require_entry_status,
        max_decision_lag_seconds=max_decision_lag_seconds,
        max_entry_tick_gap_seconds=max_entry_tick_gap_seconds,
        max_spread_bps=max_spread_bps,
        min_limit_up_room_bps=min_limit_up_room_bps,
        min_ask_volume_1=min_ask_volume_1,
        min_bid_volume_1=min_bid_volume_1,
        missing_policy=missing_policy,
    )
    mask = _apply_capacity_filter(
        work,
        mask,
        run_label=run_label,
        top_n=top_n,
        capacity_notional_col=capacity_notional_col,
        capacity_volume_col=capacity_volume_col,
        capacity_price_col=capacity_price_col,
        min_capacity_notional=min_capacity_notional,
        max_participation_rate=max_participation_rate,
        capital_per_cycle=capital_per_cycle,
        max_symbol_weight=max_symbol_weight,
        missing_policy=missing_policy,
    )
    work, mask = _apply_ask_depth_filter(
        work,
        mask,
        run_label=run_label,
        top_n=top_n,
        ask_depth_levels=ask_depth_levels,
        ask_depth_participation_rate=ask_depth_participation_rate,
        ask_depth_fill_mode=ask_depth_fill_mode,
        allow_decision_depth_fallback=allow_decision_depth_fallback,
        capital_per_cycle=capital_per_cycle,
        max_symbol_weight=max_symbol_weight,
        missing_policy=missing_policy,
    )

    total_cost_bps = float(fee_bps) + float(slippage_bps)
    work["_cost_bps"] = total_cost_bps
    return work.loc[mask].copy()
