from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS

GROUP_COLS = ("date", "decision_target_timestamp")


@dataclass(frozen=True)
class ExposureSpec:
    column: str
    category: str


DEFAULT_EXPOSURES = (
    ExposureSpec("buy_price", "price"),
    ExposureSpec("mid_price", "price"),
    ExposureSpec("ask_price_1", "price"),
    ExposureSpec("spread_bps", "liquidity"),
    ExposureSpec("ask_volume_1", "liquidity"),
    ExposureSpec("bid_volume_1", "liquidity"),
    ExposureSpec("ask_depth_10", "liquidity"),
    ExposureSpec("bid_depth_10", "liquidity"),
    ExposureSpec("depth_imbalance_1", "microstructure"),
    ExposureSpec("depth_imbalance_10", "microstructure"),
    ExposureSpec("preopen_volume", "activity"),
    ExposureSpec("preopen_turnover", "activity"),
    ExposureSpec("volume_diff_1t", "activity"),
    ExposureSpec("volume_diff_3t", "activity"),
    ExposureSpec("volume_diff_10t", "activity"),
    ExposureSpec("volume_diff_30t", "activity"),
    ExposureSpec("turnover_diff_1t", "activity"),
    ExposureSpec("turnover_diff_3t", "activity"),
    ExposureSpec("turnover_diff_10t", "activity"),
    ExposureSpec("turnover_diff_30t", "activity"),
    ExposureSpec("return_vs_prev_close", "momentum"),
    ExposureSpec("return_vs_open", "momentum"),
    ExposureSpec("return_10t", "momentum"),
    ExposureSpec("return_30t", "momentum"),
    ExposureSpec("market_cap", "size"),
    ExposureSpec("float_market_cap", "size"),
    ExposureSpec("adv20", "liquidity"),
    ExposureSpec("turnover_rate", "liquidity"),
    ExposureSpec("volatility_20d", "volatility"),
)

_DEFAULT_CATEGORY_BY_COLUMN = {spec.column: spec.category for spec in DEFAULT_EXPOSURES}


def normalize_audit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"],
        errors="coerce",
    )
    return out.dropna(subset=list(KEY_COLUMNS)).copy()


def category_for_exposure(column: str) -> str:
    if column in _DEFAULT_CATEGORY_BY_COLUMN:
        return _DEFAULT_CATEGORY_BY_COLUMN[column]
    lowered = column.lower()
    if "industry" in lowered or "sector" in lowered:
        return "industry"
    if "market_cap" in lowered or lowered in {"mktcap", "size"}:
        return "size"
    if "adv" in lowered or "amount" in lowered or "turnover" in lowered:
        return "liquidity"
    if "volume" in lowered or "trade" in lowered:
        return "activity"
    if "return" in lowered or "momentum" in lowered or "reversal" in lowered:
        return "momentum"
    if "vol" in lowered or "sigma" in lowered:
        return "volatility"
    if "price" in lowered or lowered.endswith("_px"):
        return "price"
    return "custom"


def exposure_specs(columns: Iterable[str]) -> list[ExposureSpec]:
    seen: set[str] = set()
    specs: list[ExposureSpec] = []
    for column in columns:
        name = str(column).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        specs.append(ExposureSpec(name, category_for_exposure(name)))
    return specs


def active_default_exposures(available_columns: Iterable[str]) -> list[ExposureSpec]:
    available = set(available_columns)
    return [spec for spec in DEFAULT_EXPOSURES if spec.column in available]


def finite_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def finite_mean(values: pd.Series) -> float:
    numeric = finite_numeric(values).dropna()
    return float(numeric.mean()) if len(numeric) else float("nan")


def finite_pearson(left: pd.Series, right: pd.Series) -> float:
    values = pd.DataFrame({"left": finite_numeric(left), "right": finite_numeric(right)}).dropna()
    if len(values) < 2:
        return float("nan")
    if values["left"].nunique(dropna=True) < 2 or values["right"].nunique(dropna=True) < 2:
        return float("nan")
    left_values = values["left"].to_numpy(dtype=float)
    right_values = values["right"].to_numpy(dtype=float)
    return float(np.corrcoef(left_values, right_values)[0, 1])


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    numeric = finite_numeric(values)
    if weights is None:
        return finite_mean(numeric)
    weight_values = finite_numeric(weights).clip(lower=0.0)
    mask = numeric.notna() & weight_values.notna() & weight_values.gt(0.0)
    if not mask.any():
        return float("nan")
    denom = float(weight_values.loc[mask].sum())
    if denom <= 0.0:
        return float("nan")
    return float((numeric.loc[mask] * weight_values.loc[mask]).sum() / denom)


def weighted_share(mask: pd.Series, weights: pd.Series | None = None) -> float:
    if weights is None:
        valid = mask.dropna()
        return float(valid.mean()) if len(valid) else float("nan")
    weight_values = finite_numeric(weights).clip(lower=0.0)
    valid = mask.notna() & weight_values.notna() & weight_values.gt(0.0)
    if not valid.any():
        return float("nan")
    denom = float(weight_values.loc[valid].sum())
    if denom <= 0.0:
        return float("nan")
    truth = mask.fillna(False).astype(bool)
    return float(weight_values.loc[valid & truth].sum() / denom)


def add_selection_columns(
    frame: pd.DataFrame,
    *,
    score_col: str,
    top_n: int,
    selection_col: str = "",
    weight_col: str = "",
) -> pd.DataFrame:
    if top_n < 1 and not selection_col and not weight_col:
        raise SystemExit("--top-n must be >= 1")

    out = frame.copy()
    if score_col in out.columns:
        out[score_col] = finite_numeric(out[score_col])

    if selection_col:
        if selection_col not in out.columns:
            raise SystemExit(f"selection column is missing: {selection_col}")
        values = out[selection_col]
        if pd.api.types.is_bool_dtype(values):
            out["_selected"] = values.fillna(False).astype(bool)
        else:
            out["_selected"] = finite_numeric(values).fillna(0.0).gt(0.0)
    elif weight_col:
        if weight_col not in out.columns:
            raise SystemExit(f"weight column is missing: {weight_col}")
        out["_selected"] = finite_numeric(out[weight_col]).fillna(0.0).gt(0.0)
    else:
        if score_col not in out.columns:
            raise SystemExit(f"score column is missing: {score_col}")
        out = out.dropna(subset=[score_col]).copy()
        out["_score_rank"] = out.groupby(list(GROUP_COLS), sort=False)[score_col].rank(
            ascending=False,
            method="first",
        )
        out["_selected"] = out["_score_rank"].le(top_n)

    if weight_col:
        out["_selection_weight"] = finite_numeric(out[weight_col]).clip(lower=0.0)
    else:
        out["_selection_weight"] = 1.0
    out.loc[~out["_selected"], "_selection_weight"] = 0.0
    return out


def _exposure_row(
    *,
    pool: str,
    date: object,
    timestamp: object,
    group: pd.DataFrame,
    selected: pd.DataFrame,
    spec: ExposureSpec,
    score_col: str,
    score_ranks: pd.Series | None,
    compute_score_corr: bool,
) -> dict[str, object]:
    values = finite_numeric(group[spec.column])
    selected_values = finite_numeric(selected[spec.column])
    selected_weights = selected["_selection_weight"] if "_selection_weight" in selected else None
    candidate_nonnull = int(values.notna().sum())
    selected_nonnull = int(selected_values.notna().sum())
    candidate_mean = finite_mean(values)
    candidate_std = float(values.std(ddof=0)) if candidate_nonnull else float("nan")
    selected_mean = weighted_mean(selected_values, selected_weights)
    ranks = values.rank(method="average", pct=True)
    selected_ranks = ranks.reindex(selected.index)
    selected_mean_rank = weighted_mean(selected_ranks, selected_weights)

    if np.isfinite(candidate_std) and candidate_std > 0:
        selected_z = (selected_values - candidate_mean) / candidate_std
        selected_mean_z = weighted_mean(selected_z, selected_weights)
    else:
        selected_mean_z = float("nan")
    selected_top_decile = selected_ranks.ge(0.90).where(selected_ranks.notna())
    selected_bottom_decile = selected_ranks.le(0.10).where(selected_ranks.notna())

    score_corr = (
        finite_pearson(score_ranks, ranks)
        if compute_score_corr and score_ranks is not None
        else float("nan")
    )
    clock = pd.Timestamp(timestamp).strftime("%H:%M")
    return {
        "pool": pool,
        "test_month": pd.Timestamp(date).to_period("M").strftime("%Y-%m"),
        "date": str(date),
        "decision_target_timestamp": pd.Timestamp(timestamp),
        "clock": clock,
        "category": spec.category,
        "exposure": spec.column,
        "candidate_rows": int(len(group)),
        "selected_rows": int(len(selected)),
        "candidate_nonnull": candidate_nonnull,
        "selected_nonnull": selected_nonnull,
        "candidate_mean": candidate_mean,
        "selected_mean": selected_mean,
        "selected_minus_candidate": selected_mean - candidate_mean,
        "candidate_median": float("nan"),
        "selected_median": float("nan"),
        "candidate_p10": float("nan"),
        "candidate_p90": float("nan"),
        "selected_mean_rank": selected_mean_rank,
        "selected_rank_deviation": selected_mean_rank - 0.5,
        "selected_mean_z": selected_mean_z,
        "selected_top_decile_share": weighted_share(selected_top_decile, selected_weights),
        "selected_bottom_decile_share": weighted_share(selected_bottom_decile, selected_weights),
        "score_exposure_spearman": score_corr,
    }


def exposure_group_metrics(
    frame: pd.DataFrame,
    specs: list[ExposureSpec],
    *,
    pool: str,
    score_col: str = "prediction",
    top_n: int = 100,
    selection_col: str = "",
    weight_col: str = "",
    compute_score_corr: bool = True,
) -> pd.DataFrame:
    if not specs:
        return pd.DataFrame()
    work = add_selection_columns(
        normalize_audit_frame(frame),
        score_col=score_col,
        top_n=top_n,
        selection_col=selection_col,
        weight_col=weight_col,
    )
    rows: list[dict[str, object]] = []
    for (date, timestamp), group in work.groupby(list(GROUP_COLS), sort=False):
        selected = group.loc[group["_selected"]].copy()
        score_ranks = (
            finite_numeric(group[score_col]).rank(method="average", pct=True)
            if score_col in group.columns
            else None
        )
        for spec in specs:
            if spec.column not in group.columns:
                continue
            rows.append(
                _exposure_row(
                    pool=pool,
                    date=date,
                    timestamp=timestamp,
                    group=group,
                    selected=selected,
                    spec=spec,
                    score_col=score_col,
                    score_ranks=score_ranks,
                    compute_score_corr=compute_score_corr,
                )
            )
    return pd.DataFrame(rows)


def _weighted_group_shares(
    values: pd.Series,
    weights: pd.Series,
) -> tuple[pd.Series, float]:
    weights = finite_numeric(weights).clip(lower=0.0)
    valid = values.notna() & weights.notna() & weights.gt(0.0)
    total = float(weights.loc[valid].sum())
    if total <= 0.0:
        return pd.Series(dtype="float64"), 0.0
    shares = weights.loc[valid].groupby(values.loc[valid].astype(str)).sum() / total
    return shares.astype("float64"), total


def industry_group_metrics(
    frame: pd.DataFrame,
    *,
    industry_col: str,
    pool: str,
    score_col: str = "prediction",
    top_n: int = 100,
    selection_col: str = "",
    weight_col: str = "",
) -> pd.DataFrame:
    if not industry_col or industry_col not in frame.columns:
        return pd.DataFrame()
    work = add_selection_columns(
        normalize_audit_frame(frame),
        score_col=score_col,
        top_n=top_n,
        selection_col=selection_col,
        weight_col=weight_col,
    )
    rows: list[dict[str, object]] = []
    for (date, timestamp), group in work.groupby(list(GROUP_COLS), sort=False):
        selected = group.loc[group["_selected"]].copy()
        candidate_weights = pd.Series(1.0, index=group.index)
        selected_weights = (
            selected["_selection_weight"] if "_selection_weight" in selected else None
        )
        if selected_weights is None:
            selected_weights = pd.Series(1.0, index=selected.index)
        candidate_shares, candidate_valid_weight = _weighted_group_shares(
            group[industry_col],
            candidate_weights,
        )
        selected_shares, selected_valid_weight = _weighted_group_shares(
            selected[industry_col],
            selected_weights,
        )
        industries = sorted(set(candidate_shares.index) | set(selected_shares.index))
        clock = pd.Timestamp(timestamp).strftime("%H:%M")
        for industry in industries:
            candidate_share = float(candidate_shares.get(industry, 0.0))
            selected_share = float(selected_shares.get(industry, 0.0))
            rows.append(
                {
                    "pool": pool,
                    "test_month": pd.Timestamp(date).to_period("M").strftime("%Y-%m"),
                    "date": str(date),
                    "decision_target_timestamp": pd.Timestamp(timestamp),
                    "clock": clock,
                    "industry_col": industry_col,
                    "industry": industry,
                    "candidate_rows": int(len(group)),
                    "selected_rows": int(len(selected)),
                    "candidate_industry_weight": candidate_valid_weight,
                    "selected_industry_weight": selected_valid_weight,
                    "candidate_share": candidate_share,
                    "selected_share": selected_share,
                    "active_share": selected_share - candidate_share,
                    "abs_active_share": abs(selected_share - candidate_share),
                }
            )
    return pd.DataFrame(rows)


def summarize_exposure_groups(
    group_metrics: pd.DataFrame,
    by: list[str],
    *,
    month_col: str = "test_month",
) -> pd.DataFrame:
    if group_metrics.empty:
        return pd.DataFrame()
    out = (
        group_metrics.groupby(by, sort=False, dropna=False)
        .agg(
            groups=("selected_rows", "size"),
            months=(month_col, "nunique"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            candidate_nonnull=("candidate_nonnull", "mean"),
            selected_nonnull=("selected_nonnull", "mean"),
            candidate_mean=("candidate_mean", "mean"),
            selected_mean=("selected_mean", "mean"),
            selected_minus_candidate=("selected_minus_candidate", "mean"),
            candidate_median=("candidate_median", "mean"),
            selected_median=("selected_median", "mean"),
            selected_mean_rank=("selected_mean_rank", "mean"),
            selected_rank_deviation=("selected_rank_deviation", "mean"),
            selected_mean_z=("selected_mean_z", "mean"),
            selected_top_decile_share=("selected_top_decile_share", "mean"),
            selected_bottom_decile_share=("selected_bottom_decile_share", "mean"),
            score_exposure_spearman=("score_exposure_spearman", "mean"),
        )
        .reset_index()
    )
    out["abs_selected_mean_z"] = out["selected_mean_z"].abs()
    out["abs_rank_deviation"] = out["selected_rank_deviation"].abs()
    return out


def summarize_industry_groups(industry_metrics: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if industry_metrics.empty:
        return pd.DataFrame()
    return (
        industry_metrics.groupby(by, sort=False, dropna=False)
        .agg(
            groups=("selected_rows", "size"),
            months=("test_month", "nunique"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            candidate_share=("candidate_share", "mean"),
            selected_share=("selected_share", "mean"),
            active_share=("active_share", "mean"),
            abs_active_share=("abs_active_share", "mean"),
            max_active_share=("active_share", "max"),
            min_active_share=("active_share", "min"),
        )
        .reset_index()
        .sort_values(by + ["abs_active_share"], ascending=[True] * len(by) + [False])
    )


def category_summary(exposure_summary: pd.DataFrame) -> pd.DataFrame:
    if exposure_summary.empty:
        return pd.DataFrame()
    return (
        exposure_summary.groupby(["pool", "category"], sort=False, dropna=False)
        .agg(
            exposures=("exposure", "nunique"),
            max_abs_selected_mean_z=("abs_selected_mean_z", "max"),
            mean_abs_selected_mean_z=("abs_selected_mean_z", "mean"),
            max_abs_rank_deviation=("abs_rank_deviation", "max"),
            mean_abs_rank_deviation=("abs_rank_deviation", "mean"),
            mean_selected_top_decile_share=("selected_top_decile_share", "mean"),
            mean_selected_bottom_decile_share=("selected_bottom_decile_share", "mean"),
        )
        .reset_index()
    )


def _share_stats(values: pd.Series, weights: pd.Series) -> dict[str, float]:
    weights = finite_numeric(weights).clip(lower=0.0)
    valid = values.notna() & weights.notna() & weights.gt(0.0)
    if not valid.any():
        return {
            "unique": 0.0,
            "max_share": float("nan"),
            "top5_share": float("nan"),
            "hhi": float("nan"),
            "effective_count": float("nan"),
        }
    shares = weights.loc[valid].groupby(values.loc[valid].astype(str)).sum()
    shares = shares / shares.sum()
    hhi = float((shares**2).sum())
    return {
        "unique": float(len(shares)),
        "max_share": float(shares.max()),
        "top5_share": float(shares.sort_values(ascending=False).head(5).sum()),
        "hhi": hhi,
        "effective_count": float(1.0 / hhi) if hhi > 0 else float("nan"),
    }


def daily_concentration(
    frame: pd.DataFrame,
    *,
    pool: str,
    score_col: str = "prediction",
    top_n: int = 100,
    selection_col: str = "",
    weight_col: str = "",
    industry_col: str = "",
) -> pd.DataFrame:
    work = add_selection_columns(
        normalize_audit_frame(frame),
        score_col=score_col,
        top_n=top_n,
        selection_col=selection_col,
        weight_col=weight_col,
    )
    rows: list[dict[str, object]] = []
    for date, group in work.groupby("date", sort=False):
        selected = group.loc[group["_selected"]].copy()
        weights = selected["_selection_weight"] if not selected.empty else pd.Series(dtype=float)
        symbol_stats = _share_stats(selected["symbol"], weights)
        row = {
            "pool": pool,
            "test_month": pd.Timestamp(date).to_period("M").strftime("%Y-%m"),
            "date": str(date),
            "decision_clocks": int(group["decision_target_timestamp"].nunique()),
            "candidate_rows": int(len(group)),
            "candidate_symbols": int(group["symbol"].nunique()),
            "selected_rows": int(len(selected)),
            "selected_symbols": int(symbol_stats["unique"]),
            "selected_repeat_rate": (
                1.0 - symbol_stats["unique"] / len(selected) if len(selected) else float("nan")
            ),
            "selected_symbol_max_share": symbol_stats["max_share"],
            "selected_symbol_top5_share": symbol_stats["top5_share"],
            "selected_symbol_hhi": symbol_stats["hhi"],
            "selected_effective_symbols": symbol_stats["effective_count"],
        }
        if industry_col and industry_col in selected.columns:
            industry_stats = _share_stats(selected[industry_col], weights)
            candidate_industry = group[industry_col].dropna().astype(str)
            row.update(
                {
                    "candidate_industries": int(candidate_industry.nunique()),
                    "selected_industries": int(industry_stats["unique"]),
                    "selected_industry_max_share": industry_stats["max_share"],
                    "selected_industry_top5_share": industry_stats["top5_share"],
                    "selected_industry_hhi": industry_stats["hhi"],
                    "selected_effective_industries": industry_stats["effective_count"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_concentration(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    agg: dict[str, tuple[str, str]] = {
        "days": ("date", "nunique"),
        "decision_clocks": ("decision_clocks", "mean"),
        "candidate_rows": ("candidate_rows", "mean"),
        "candidate_symbols": ("candidate_symbols", "mean"),
        "selected_rows": ("selected_rows", "mean"),
        "selected_symbols": ("selected_symbols", "mean"),
        "selected_repeat_rate": ("selected_repeat_rate", "mean"),
        "selected_symbol_max_share": ("selected_symbol_max_share", "mean"),
        "selected_symbol_top5_share": ("selected_symbol_top5_share", "mean"),
        "selected_symbol_hhi": ("selected_symbol_hhi", "mean"),
        "selected_effective_symbols": ("selected_effective_symbols", "mean"),
    }
    for column in (
        "candidate_industries",
        "selected_industries",
        "selected_industry_max_share",
        "selected_industry_top5_share",
        "selected_industry_hhi",
        "selected_effective_industries",
    ):
        if column in daily.columns:
            agg[column] = (column, "mean")
    return daily.groupby("pool", sort=False).agg(**agg).reset_index()
