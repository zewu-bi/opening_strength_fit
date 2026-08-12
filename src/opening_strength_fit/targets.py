from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from opening_strength_fit.candidates import opening_candidate_mask
from opening_strength_fit.evaluation import resolve_group_cols
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns

DEFAULT_HEAT_NEUTRALIZE_COLUMNS = (
    "ask_price_1",
    "bid_price_1",
    "mid_price",
    "spread_bps",
    "ask1_to_limit_up_bps",
    "return_vs_prev_close",
    "return_vs_open",
    "preopen_return_vs_prev_close",
    "return_1t",
    "return_3t",
    "return_10t",
    "return_30t",
    "volume",
    "turnover",
    "preopen_volume",
    "preopen_turnover",
    "volume_diff_1t",
    "turnover_diff_1t",
    "volume_diff_3t",
    "turnover_diff_3t",
    "volume_diff_10t",
    "turnover_diff_10t",
    "volume_diff_30t",
    "turnover_diff_30t",
    "trade_vwap_1t",
    "trade_vwap_3t",
    "trade_vwap_10t",
    "trade_vwap_30t",
)


def _valid_mask(
    frame: pd.DataFrame,
    *,
    label_col: str,
    valid_col: str,
) -> pd.Series:
    label = pd.to_numeric(frame[label_col], errors="coerce")
    mask = label.notna() & np.isfinite(label)
    if valid_col in frame.columns:
        mask &= frame[valid_col].astype(bool)
    return mask


def _truthy_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0.0).ne(0.0)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "on", "pass", "passed"})


def _as_column_tuple(columns: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if columns is None:
        return DEFAULT_HEAT_NEUTRALIZE_COLUMNS
    return tuple(str(column).strip() for column in columns if str(column).strip())


def _rank_centered(values: pd.Series) -> pd.Series:
    valid_count = int(values.notna().sum())
    if valid_count < 2:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    rank_pct = values.rank(method="average", pct=True)
    rank_mean = (valid_count + 1.0) / (2.0 * valid_count)
    return rank_pct - rank_mean


def _zscore(values: pd.Series, *, std_epsilon: float) -> pd.Series:
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= float(std_epsilon):
        return pd.Series(np.nan, index=values.index, dtype="float64")
    return (values - mean) / std


def _cross_sectional_transformed_label(
    out: pd.DataFrame,
    *,
    values: pd.Series,
    valid: pd.Series,
    group_cols: tuple[str, ...],
    transform: str,
    min_group_size: int,
    std_epsilon: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    values = (
        pd.to_numeric(values, errors="coerce")
        .astype("float64")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )
    transform = transform.strip().lower().replace("-", "_")
    valid = valid & values.notna()

    component = pd.Series(np.nan, index=out.index, dtype="float64")
    mean = pd.Series(np.nan, index=out.index, dtype="float64")
    median = pd.Series(np.nan, index=out.index, dtype="float64")
    std = pd.Series(np.nan, index=out.index, dtype="float64")
    count = pd.Series(np.nan, index=out.index, dtype="float64")
    rank_pct = pd.Series(np.nan, index=out.index, dtype="float64")
    if not valid.any():
        return component, mean, median, std, count, rank_pct

    valid_frame = out.loc[valid, [*group_cols]].copy()
    valid_frame["_value"] = values.loc[valid].to_numpy()
    grouped = valid_frame.groupby(list(group_cols), sort=False)["_value"]

    mean.loc[valid] = grouped.transform("mean").to_numpy()
    median.loc[valid] = grouped.transform("median").to_numpy()
    std.loc[valid] = grouped.transform(lambda group: group.std(ddof=0)).to_numpy()
    count.loc[valid] = grouped.transform("count").to_numpy()
    rank_pct.loc[valid] = grouped.rank(method="average", pct=True).to_numpy()

    usable = valid & count.ge(int(min_group_size))
    if transform == "raw":
        component.loc[usable] = values.loc[usable]
    elif transform in {"demean", "center"}:
        component.loc[usable] = values.loc[usable] - mean.loc[usable]
    elif transform == "zscore":
        zscore_usable = usable & std.gt(float(std_epsilon))
        component.loc[zscore_usable] = (
            values.loc[zscore_usable] - mean.loc[zscore_usable]
        ) / std.loc[zscore_usable]
    elif transform == "rank_pct":
        component.loc[usable] = rank_pct.loc[usable]
    elif transform == "rank_centered":
        rank_mean = (count.loc[usable] + 1.0) / (2.0 * count.loc[usable])
        component.loc[usable] = rank_pct.loc[usable] - rank_mean
    else:
        raise SystemExit(
            "mixed label transform must be raw, demean, zscore, rank_pct, "
            f"or rank_centered; got {transform!r}"
        )
    return component, mean, median, std, count, rank_pct


def _transformed_exposure(
    values: pd.Series,
    *,
    transform: str,
    std_epsilon: float,
) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").astype("float64")
    if transform == "rank_centered":
        return _rank_centered(values)
    if transform == "zscore":
        return _zscore(values, std_epsilon=std_epsilon)
    if transform == "center":
        return values - values.mean()
    raise SystemExit(
        f"unknown heat neutralization transform {transform!r}; "
        "expected rank_centered, zscore, or center"
    )


def _ridge_fitted(x: np.ndarray, y: np.ndarray, *, ridge_alpha: float) -> np.ndarray:
    if x.size == 0 or x.shape[1] == 0:
        return np.zeros_like(y, dtype="float64")
    alpha = float(ridge_alpha)
    xtx = x.T @ x
    if alpha > 0.0:
        xtx = xtx + np.eye(xtx.shape[0], dtype="float64") * alpha
    xty = x.T @ y
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    return x @ beta


def _add_heat_neutral_target(
    out: pd.DataFrame,
    *,
    raw_label: pd.Series,
    valid: pd.Series,
    usable: pd.Series,
    group_cols: tuple[str, ...],
    exposure_cols: tuple[str, ...],
    group_mean: pd.Series,
    neutralization_strength: float,
    neutralization_ridge_alpha: float,
    neutralization_transform: str,
    min_neutralize_cols: int,
    std_epsilon: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if not 0.0 <= float(neutralization_strength) <= 1.0:
        raise SystemExit("neutralization_strength must be between 0 and 1")

    present_exposure_cols = tuple(column for column in exposure_cols if column in out.columns)
    if not present_exposure_cols:
        requested = ", ".join(exposure_cols)
        raise SystemExit(
            f"heat_neutral target has no usable exposure columns; requested: {requested}"
        )

    target = pd.Series(np.nan, index=out.index, dtype="float64")
    heat_fitted = pd.Series(np.nan, index=out.index, dtype="float64")
    heat_residual = pd.Series(np.nan, index=out.index, dtype="float64")
    exposure_count = pd.Series(np.nan, index=out.index, dtype="float64")

    group_frame = out.loc[valid, [*group_cols]].copy()
    grouped_indices = group_frame.groupby(list(group_cols), sort=False).groups
    for indices in grouped_indices.values():
        group_index = pd.Index(indices)
        group_index = group_index[usable.loc[group_index].to_numpy()]
        if len(group_index) == 0:
            continue

        y = (raw_label.loc[group_index] - group_mean.loc[group_index]).astype("float64")
        exposure_parts = []
        for column in present_exposure_cols:
            transformed = _transformed_exposure(
                out.loc[group_index, column],
                transform=neutralization_transform,
                std_epsilon=std_epsilon,
            )
            if transformed.notna().sum() < 2:
                continue
            filled = transformed.fillna(0.0).astype("float64")
            if filled.nunique(dropna=True) < 2:
                continue
            exposure_parts.append(filled.to_numpy())

        if len(exposure_parts) < int(min_neutralize_cols):
            fitted = np.zeros(len(group_index), dtype="float64")
        else:
            x = np.column_stack(exposure_parts).astype("float64")
            fitted = _ridge_fitted(
                x,
                y.to_numpy(dtype="float64"),
                ridge_alpha=neutralization_ridge_alpha,
            )

        residual = y.to_numpy(dtype="float64") - fitted
        target.loc[group_index] = (
            y.to_numpy(dtype="float64") - float(neutralization_strength) * fitted
        )
        heat_fitted.loc[group_index] = fitted
        heat_residual.loc[group_index] = residual
        exposure_count.loc[group_index] = float(len(exposure_parts))

    return target, heat_fitted, heat_residual, exposure_count


def _has_guard_definition(
    *,
    min_values: Mapping[str, object] | None,
    max_values: Mapping[str, object] | None,
    rank_min_values: Mapping[str, object] | None,
    rank_max_values: Mapping[str, object] | None,
) -> bool:
    return bool(min_values or max_values or rank_min_values or rank_max_values)


def _guard_pass_mask(
    out: pd.DataFrame,
    *,
    guard_pass_col: str,
    min_values: Mapping[str, object] | None,
    max_values: Mapping[str, object] | None,
    rank_min_values: Mapping[str, object] | None,
    rank_max_values: Mapping[str, object] | None,
    rank_group_cols: Sequence[str],
    rank_method: str,
) -> pd.Series:
    if guard_pass_col and guard_pass_col in out.columns:
        return _truthy_series(out[guard_pass_col])
    if not _has_guard_definition(
        min_values=min_values,
        max_values=max_values,
        rank_min_values=rank_min_values,
        rank_max_values=rank_max_values,
    ):
        if guard_pass_col:
            raise SystemExit(
                f"guard_shrunk target missing guard_pass_col {guard_pass_col!r}; "
                "provide the column or target_cache guard thresholds"
            )
        raise SystemExit(
            "guard_shrunk target requires guard_pass_col or target_cache guard thresholds"
        )
    return opening_candidate_mask(
        out,
        min_values=min_values,
        max_values=max_values,
        rank_min_values=rank_min_values,
        rank_max_values=rank_max_values,
        rank_group_cols=rank_group_cols,
        rank_method=rank_method,
    )


def _guard_risk_score(
    out: pd.DataFrame,
    *,
    rank_min_values: Mapping[str, object] | None,
    rank_max_values: Mapping[str, object] | None,
    rank_group_cols: Sequence[str],
    rank_method: str,
    normalization: str,
) -> tuple[pd.Series, pd.Series]:
    rank_min = {
        str(column): float(value)
        for column, value in (rank_min_values or {}).items()
        if value not in (None, "")
    }
    rank_max = {
        str(column): float(value)
        for column, value in (rank_max_values or {}).items()
        if value not in (None, "")
    }
    columns = tuple(sorted(set(rank_min) | set(rank_max)))
    if not columns:
        raise SystemExit("guard_risk_shrunk target requires guard_risk_rank_min/max")

    group_cols = resolve_group_cols(out, tuple(rank_group_cols))
    if not group_cols:
        raise SystemExit("guard risk ranks need at least one available group column")

    components = []
    for column in columns:
        if column not in out.columns:
            raise SystemExit(f"guard risk missing required column: {column}")
        values = pd.to_numeric(out[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        rank_pct = values.groupby([out[col] for col in group_cols]).rank(
            method=rank_method,
            pct=True,
        )
        risks = []
        if column in rank_min:
            threshold = rank_min[column]
            denominator = threshold if threshold > 0.0 else 1.0
            risks.append(((threshold - rank_pct) / denominator).clip(lower=0.0, upper=1.0))
        if column in rank_max:
            threshold = rank_max[column]
            denominator = 1.0 - threshold if threshold < 1.0 else 1.0
            risks.append(((rank_pct - threshold) / denominator).clip(lower=0.0, upper=1.0))
        if risks:
            component = pd.concat(risks, axis=1).max(axis=1)
            components.append(component.fillna(0.0).astype("float64"))

    if not components:
        raise SystemExit("guard risk config produced no usable components")

    component_frame = pd.concat(components, axis=1)
    risk_sum = component_frame.sum(axis=1).astype("float64")
    component_count = component_frame.gt(0.0).sum(axis=1).astype("float64")
    normalization = normalization.strip().lower().replace("-", "_")
    if normalization == "mean":
        risk = risk_sum / float(len(components))
    elif normalization == "sum":
        risk = risk_sum
    elif normalization == "max":
        risk = component_frame.max(axis=1).astype("float64")
    else:
        raise SystemExit("guard_risk_normalization must be mean, sum, or max")
    return risk.clip(lower=0.0), component_count


def add_cross_sectional_target_label(
    frame: pd.DataFrame,
    *,
    mode: str = "demean",
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    label_col: str = "label",
    target_col: str = "label",
    raw_label_col: str = "label_raw",
    valid_col: str = "valid_label",
    min_group_size: int = 2,
    std_epsilon: float = 1e-12,
    neutralize_cols: tuple[str, ...] | list[str] | None = None,
    neutralization_strength: float = 1.0,
    neutralization_ridge_alpha: float = 1.0,
    neutralization_transform: str = "rank_centered",
    min_neutralize_cols: int = 1,
    guard_shrink_penalty: float = 0.5,
    guard_pass_col: str = "next_flip_guard_10t_pass",
    guard_min_values: Mapping[str, object] | None = None,
    guard_max_values: Mapping[str, object] | None = None,
    guard_rank_min_values: Mapping[str, object] | None = None,
    guard_rank_max_values: Mapping[str, object] | None = None,
    guard_rank_group_cols: Sequence[str] | None = None,
    guard_rank_method: str = "average",
    guard_risk_lambda: float = 1.0,
    guard_risk_rank_min_values: Mapping[str, object] | None = None,
    guard_risk_rank_max_values: Mapping[str, object] | None = None,
    guard_risk_normalization: str = "mean",
    long_label_col: str = "alpha_return_next_close",
    long_label_weight: float = 0.10,
    short_label_transform: str = "zscore",
    long_label_transform: str = "zscore",
) -> pd.DataFrame:
    """Replace or add a cross-sectionally aligned label.

    The original raw label is preserved in ``raw_label_col``. Group statistics
    are computed only from valid rows, and invalid rows keep a missing target.
    """

    if label_col not in frame.columns:
        raise SystemExit(f"missing label column: {label_col}")

    mode = mode.strip().lower().replace("-", "_")
    if mode not in {
        "raw",
        "demean",
        "zscore",
        "rank_pct",
        "rank_centered",
        "heat_neutral",
        "guard_shrunk",
        "guard_risk_shrunk",
        "mixed",
    }:
        raise SystemExit(
            "unknown target mode "
            f"{mode!r}; expected raw, demean, zscore, rank_pct, "
            "rank_centered, heat_neutral, guard_shrunk, guard_risk_shrunk, "
            "or mixed"
        )
    neutralization_transform = neutralization_transform.strip().lower().replace("-", "_")
    if not 0.0 <= float(guard_shrink_penalty) <= 1.0:
        raise SystemExit("guard_shrink_penalty must be between 0 and 1")
    if float(guard_risk_lambda) < 0.0:
        raise SystemExit("guard_risk_lambda must be non-negative")
    if not np.isfinite(float(long_label_weight)):
        raise SystemExit("long_label_weight must be finite")

    out = ensure_timestamp_columns(standardize_columns(frame)).copy()
    resolved_group_cols = resolve_group_cols(out, group_cols)
    if not resolved_group_cols:
        raise SystemExit(f"none of the requested group columns exist: {group_cols}")

    raw_label = pd.to_numeric(out[label_col], errors="coerce").astype("float64")
    if raw_label_col not in out.columns:
        out[raw_label_col] = raw_label

    valid = _valid_mask(out, label_col=label_col, valid_col=valid_col)
    (
        target,
        group_mean,
        group_median,
        group_std,
        group_count,
        rank_pct,
    ) = _cross_sectional_transformed_label(
        out,
        values=raw_label,
        valid=valid,
        group_cols=resolved_group_cols,
        transform=(
            mode if mode in {"raw", "demean", "zscore", "rank_pct", "rank_centered"} else "raw"
        ),
        min_group_size=min_group_size,
        std_epsilon=std_epsilon,
    )
    usable = valid & group_count.ge(int(min_group_size))
    if mode == "heat_neutral":
        (
            target,
            heat_fitted,
            heat_residual,
            heat_exposure_count,
        ) = _add_heat_neutral_target(
            out,
            raw_label=raw_label,
            valid=valid,
            usable=usable,
            group_cols=resolved_group_cols,
            exposure_cols=_as_column_tuple(neutralize_cols),
            group_mean=group_mean,
            neutralization_strength=neutralization_strength,
            neutralization_ridge_alpha=neutralization_ridge_alpha,
            neutralization_transform=neutralization_transform,
            min_neutralize_cols=min_neutralize_cols,
            std_epsilon=std_epsilon,
        )
        out["label_xs_heat_fitted"] = heat_fitted
        out["label_xs_heat_residual"] = heat_residual
        out["label_xs_heat_exposure_count"] = heat_exposure_count
    elif mode in {"guard_shrunk", "guard_risk_shrunk"}:
        rank_group_cols = tuple(guard_rank_group_cols or resolved_group_cols)
        if mode == "guard_shrunk":
            guard_pass = _guard_pass_mask(
                out,
                guard_pass_col=guard_pass_col,
                min_values=guard_min_values,
                max_values=guard_max_values,
                rank_min_values=guard_rank_min_values,
                rank_max_values=guard_rank_max_values,
                rank_group_cols=rank_group_cols,
                rank_method=guard_rank_method,
            )
            dirty = ~guard_pass.astype(bool)
            shrink_rate = float(guard_shrink_penalty) * dirty.astype("float64")
            out["label_xs_guard_pass"] = guard_pass.astype("int8")
            out["label_xs_guard_dirty"] = dirty.astype("int8")
        else:
            risk, risk_component_count = _guard_risk_score(
                out,
                rank_min_values=guard_risk_rank_min_values,
                rank_max_values=guard_risk_rank_max_values,
                rank_group_cols=rank_group_cols,
                rank_method=guard_rank_method,
                normalization=guard_risk_normalization,
            )
            shrink_rate = float(guard_risk_lambda) * risk
            out["label_xs_guard_risk"] = risk.where(usable)
            out["label_xs_guard_risk_component_count"] = risk_component_count.where(usable)
        positive_excess = (raw_label - group_median).clip(lower=0.0)
        shrink = shrink_rate * positive_excess
        shrink_mask = usable & positive_excess.gt(0.0) & shrink_rate.gt(0.0)
        target.loc[shrink_mask] = raw_label.loc[shrink_mask] - shrink.loc[shrink_mask]
        out["label_xs_guard_positive_excess"] = positive_excess.where(usable)
        out["label_xs_guard_shrink"] = shrink.where(shrink_mask, 0.0).where(usable)
    elif mode == "mixed":
        if long_label_col not in out.columns:
            raise SystemExit(f"mixed target missing long label column: {long_label_col}")
        target[:] = np.nan
        long_label = (
            pd.to_numeric(out[long_label_col], errors="coerce")
            .astype("float64")
            .replace([np.inf, -np.inf], np.nan)
        )
        mixed_valid = valid & long_label.notna()
        (
            short_component,
            _short_mean,
            _short_median,
            _short_std,
            _short_count,
            _short_rank_pct,
        ) = _cross_sectional_transformed_label(
            out,
            values=raw_label,
            valid=mixed_valid,
            group_cols=resolved_group_cols,
            transform=short_label_transform,
            min_group_size=min_group_size,
            std_epsilon=std_epsilon,
        )
        (
            long_component,
            long_mean,
            _long_median,
            long_std,
            long_count,
            long_rank_pct,
        ) = _cross_sectional_transformed_label(
            out,
            values=long_label,
            valid=mixed_valid,
            group_cols=resolved_group_cols,
            transform=long_label_transform,
            min_group_size=min_group_size,
            std_epsilon=std_epsilon,
        )
        mixed_usable = short_component.notna() & long_component.notna()
        target.loc[mixed_usable] = (
            short_component.loc[mixed_usable]
            + float(long_label_weight) * long_component.loc[mixed_usable]
        )
        out["label_xs_short_component"] = short_component
        out["label_xs_long_raw"] = long_label
        out["label_xs_long_mean"] = long_mean
        out["label_xs_long_std"] = long_std
        out["label_xs_long_count"] = long_count
        out["label_xs_long_rank_pct"] = long_rank_pct
        out["label_xs_long_component"] = long_component
        out["label_xs_mixed_long_weight"] = float(long_label_weight)

    out["label_xs_mean"] = group_mean
    out["label_xs_median"] = group_median
    out["label_xs_std"] = group_std
    out["label_xs_count"] = group_count
    out["label_xs_rank_pct"] = rank_pct
    out[target_col] = target
    if valid_col in out.columns:
        out[valid_col] = out[valid_col].astype(bool) & out[target_col].notna()
    else:
        out[valid_col] = out[target_col].notna()
    return out


def target_label_summary(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    raw_label_col: str = "label_raw",
    valid_col: str = "valid_label",
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
) -> dict[str, object]:
    valid = _valid_mask(frame, label_col=label_col, valid_col=valid_col)
    label = pd.to_numeric(frame.loc[valid, label_col], errors="coerce")
    raw = (
        pd.to_numeric(frame.loc[valid, raw_label_col], errors="coerce")
        if raw_label_col in frame.columns
        else pd.Series(dtype="float64")
    )
    resolved_group_cols = resolve_group_cols(frame, group_cols)
    groups = (
        int(frame.loc[valid].groupby(list(resolved_group_cols), sort=False).ngroups)
        if resolved_group_cols and valid.any()
        else 0
    )
    return {
        "rows": int(len(frame)),
        "valid_rows": int(valid.sum()),
        "groups": groups,
        "label_mean": float(label.mean()) if len(label) else float("nan"),
        "label_std": float(label.std(ddof=0)) if len(label) else float("nan"),
        "raw_label_mean": float(raw.mean()) if len(raw) else float("nan"),
        "raw_label_std": float(raw.std(ddof=0)) if len(raw) else float("nan"),
    }
