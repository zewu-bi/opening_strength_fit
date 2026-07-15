"""Feature transforms and training-derived standardization for Torch models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.feature_transforms.cross_sectional import (
    transform_cross_sectional_feature_values,
)
from opening_strength_fit.feature_transforms.mechanism import (
    mechanismized_feature_value_reference_columns,
    transform_mechanismized_feature_values,
    transform_mechanismized_v2_feature_values,
    transform_mechanismized_v3_feature_values,
)

_GLOBAL_STANDARDIZATION_MODES = {
    "",
    "global",
    "global_zscore",
    "train",
    "train_zscore",
    "feature_zscore",
}
_SYMBOL_TRAIN_STANDARDIZATION_MODES = {
    "symbol_train_zscore",
    "per_symbol_train_zscore",
    "symbol_zscore",
    "symbol_history_zscore",
    "self_history_zscore",
}


def _normalize_feature_standardization(value: str) -> str:
    mode = str(value or "global_zscore").strip().lower().replace("-", "_")
    if mode in _GLOBAL_STANDARDIZATION_MODES:
        return "global_zscore"
    if mode in _SYMBOL_TRAIN_STANDARDIZATION_MODES:
        return "symbol_train_zscore"
    raise SystemExit(
        "model.feature_standardization must be one of global_zscore or symbol_train_zscore"
    )


_NO_FEATURE_VALUE_TRANSFORMS = {"", "none", "identity", "raw", "off", "false"}
_CROSS_SECTIONAL_FEATURE_VALUE_TRANSFORMS = {
    "cross_sectional_demean": "demean",
    "xs_demean": "demean",
    "cross_sectional_zscore": "zscore",
    "xs_zscore": "zscore",
    "cross_sectional_robust_zscore": "robust_zscore",
    "xs_robust_zscore": "robust_zscore",
    "cross_sectional_rank_pct": "rank_pct",
    "xs_rank_pct": "rank_pct",
    "cross_sectional_rank": "rank",
    "xs_rank": "rank",
    "cross_sectional_rank_centered": "rank_centered",
    "cross_sectional_rank_centered_inplace": "rank_centered",
    "xs_rank_centered": "rank_centered",
    "rank_centered": "rank_centered",
}
_MECHANISMIZED_FEATURE_VALUE_TRANSFORMS = {
    "mechanismized_cross_sectional_rank_centered": "rank_centered",
    "mechanismized_xs_rank_centered": "rank_centered",
    "mechanismized_rank_centered": "rank_centered",
    "mechanismized_dimensionless": "rank_centered",
    "mechanismized_dimensionless_328": "rank_centered",
    "mechanism_aware_cross_sectional_rank_centered": "rank_centered",
    "mechanism_aware_xs_rank_centered": "rank_centered",
    "mechanism_aware_rank_centered": "rank_centered",
    "mechanismized_cross_sectional_zscore": "zscore",
    "mechanismized_xs_zscore": "zscore",
    "mechanismized_zscore": "zscore",
    "mechanismized_only": "none",
    "mechanism_aware_only": "none",
}
_MECHANISMIZED_V2_FEATURE_VALUE_TRANSFORMS = {
    "mechanismized_v2_cross_sectional_robust_zscore": "robust_zscore",
    "mechanismized_v2_xs_robust_zscore": "robust_zscore",
    "mechanismized_v2_robust_zscore": "robust_zscore",
    "mechanismized_v2_dimensionless": "robust_zscore",
    "mechanismized_v2_dimensionless_328": "robust_zscore",
    "mechanismized_v2_cross_sectional_zscore": "zscore",
    "mechanismized_v2_xs_zscore": "zscore",
    "mechanismized_v2_zscore": "zscore",
    "mechanismized_v2_cross_sectional_rank_centered": "rank_centered",
    "mechanismized_v2_xs_rank_centered": "rank_centered",
    "mechanismized_v2_rank_centered": "rank_centered",
    "mechanismized_v2_only": "none",
    "mechanism_aware_v2_cross_sectional_robust_zscore": "robust_zscore",
    "mechanism_aware_v2_dimensionless": "robust_zscore",
    "mechanism_aware_v2_only": "none",
}
_MECHANISMIZED_V3_FEATURE_VALUE_TRANSFORMS = {
    "mechanismized_v3_cross_sectional_robust_zscore": "robust_zscore",
    "mechanismized_v3_xs_robust_zscore": "robust_zscore",
    "mechanismized_v3_robust_zscore": "robust_zscore",
    "mechanismized_v3_dimensionless": "none",
    "mechanismized_v3_dimensionless_328": "none",
    "mechanismized_v3_cross_sectional_zscore": "zscore",
    "mechanismized_v3_xs_zscore": "zscore",
    "mechanismized_v3_zscore": "zscore",
    "mechanismized_v3_cross_sectional_rank_centered": "rank_centered",
    "mechanismized_v3_xs_rank_centered": "rank_centered",
    "mechanismized_v3_rank_centered": "rank_centered",
    "mechanismized_v3_only": "none",
    "mechanism_aware_v3_cross_sectional_robust_zscore": "robust_zscore",
    "mechanism_aware_v3_dimensionless": "none",
    "mechanism_aware_v3_only": "none",
}


def _normalize_feature_value_transform(value: str) -> str:
    mode = str(value or "none").strip().lower().replace("-", "_")
    if mode in _NO_FEATURE_VALUE_TRANSFORMS:
        return "none"
    if mode in _CROSS_SECTIONAL_FEATURE_VALUE_TRANSFORMS:
        return f"cross_sectional_{_CROSS_SECTIONAL_FEATURE_VALUE_TRANSFORMS[mode]}"
    if mode in _MECHANISMIZED_V3_FEATURE_VALUE_TRANSFORMS:
        return f"mechanismized_v3_{_MECHANISMIZED_V3_FEATURE_VALUE_TRANSFORMS[mode]}"
    if mode in _MECHANISMIZED_V2_FEATURE_VALUE_TRANSFORMS:
        return f"mechanismized_v2_{_MECHANISMIZED_V2_FEATURE_VALUE_TRANSFORMS[mode]}"
    if mode in _MECHANISMIZED_FEATURE_VALUE_TRANSFORMS:
        return f"mechanismized_{_MECHANISMIZED_FEATURE_VALUE_TRANSFORMS[mode]}"
    raise SystemExit(
        "features.feature_value_transform must be none, cross_sectional_demean, "
        "cross_sectional_zscore, cross_sectional_robust_zscore, cross_sectional_rank_pct, "
        "cross_sectional_rank, cross_sectional_rank_centered, "
        "mechanismized_cross_sectional_rank_centered, mechanismized_v2_dimensionless_328, "
        "or mechanismized_v3_dimensionless_328"
    )


def _feature_value_transform_mode(normalized: str) -> str:
    return normalized.removeprefix("cross_sectional_")


def _torch_feature_value_frame(
    frame: pd.DataFrame,
    features: list[str],
    *,
    feature_value_transform: str,
    group_cols: tuple[str, ...],
    rank_method: str,
    tick_size: float = 0.01,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    mode = _normalize_feature_value_transform(feature_value_transform)
    support_columns = (
        mechanismized_feature_value_reference_columns() if mode.startswith("mechanismized_") else ()
    )
    required = list(dict.fromkeys([*extra_columns, *group_cols, *features, *support_columns]))
    transform_required = [*group_cols, *features] if mode != "none" else features
    missing = [column for column in transform_required if column and column not in frame.columns]
    if missing and mode != "none":
        raise SystemExit(f"features.feature_value_transform requires columns: {missing[:5]}")
    available = [column for column in required if column and column in frame.columns]
    model_frame = frame.loc[:, available].copy()
    if mode == "none":
        return model_frame
    if mode.startswith("mechanismized_v3_"):
        return transform_mechanismized_v3_feature_values(
            model_frame,
            columns=tuple(features),
            group_cols=group_cols,
            rank_method=rank_method,
            tick_size=float(tick_size),
            cross_sectional_mode=mode.removeprefix("mechanismized_v3_"),
        )
    if mode.startswith("mechanismized_v2_"):
        return transform_mechanismized_v2_feature_values(
            model_frame,
            columns=tuple(features),
            group_cols=group_cols,
            rank_method=rank_method,
            tick_size=float(tick_size),
            cross_sectional_mode=mode.removeprefix("mechanismized_v2_"),
        )
    if mode.startswith("mechanismized_"):
        return transform_mechanismized_feature_values(
            model_frame,
            columns=tuple(features),
            group_cols=group_cols,
            rank_method=rank_method,
            tick_size=float(tick_size),
            cross_sectional_mode=mode.removeprefix("mechanismized_"),
        )
    return transform_cross_sectional_feature_values(
        model_frame,
        columns=tuple(features),
        group_cols=group_cols,
        mode=_feature_value_transform_mode(mode),
        rank_method=rank_method,
    )


def _standardized_float_matrix(
    frame: pd.DataFrame,
    features: list[str],
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    group_col: str = "symbol",
    group_keys: np.ndarray | None = None,
    group_mean: np.ndarray | None = None,
    group_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = (
        frame[features]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )
    if mean is None:
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(values, axis=0).astype("float32")
        mean = np.where(np.isfinite(mean), mean, 0.0).astype("float32")
    if scale is None:
        with np.errstate(invalid="ignore"):
            scale = np.nanstd(values, axis=0).astype("float32")
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0).astype("float32")
    if group_keys is None or group_mean is None or group_scale is None:
        values -= mean
        values /= scale
    else:
        if group_col not in frame.columns:
            raise SystemExit(
                f"model.feature_standardization='symbol_train_zscore' requires {group_col!r}"
            )
        key_to_index = {str(key): index for index, key in enumerate(group_keys)}
        grouped_indices = frame.groupby(frame[group_col].astype(str), sort=False).indices
        for key, row_positions in grouped_indices.items():
            group_index = key_to_index.get(str(key))
            if group_index is None:
                center = mean
                denominator = scale
            else:
                center = group_mean[group_index]
                denominator = group_scale[group_index]
            values[row_positions] -= center
            values[row_positions] /= denominator
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return values, mean.astype("float32"), scale.astype("float32")


def _fit_symbol_train_standardization(
    frame: pd.DataFrame,
    features: list[str],
    *,
    group_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if group_col not in frame.columns:
        raise SystemExit(
            f"model.feature_standardization='symbol_train_zscore' requires {group_col!r}"
        )
    values = frame[features].replace([np.inf, -np.inf], np.nan)
    global_mean = values.mean(axis=0, skipna=True).to_numpy(dtype="float32", copy=True)
    global_scale = values.std(axis=0, skipna=True, ddof=0).to_numpy(dtype="float32", copy=True)
    global_mean = np.where(np.isfinite(global_mean), global_mean, 0.0).astype("float32")
    global_scale = np.where(
        np.isfinite(global_scale) & (global_scale > 0.0),
        global_scale,
        1.0,
    ).astype("float32")

    grouped = values.groupby(frame[group_col].astype(str), sort=True)
    group_mean_frame = grouped.mean()
    group_scale_frame = grouped.std(ddof=0).reindex(group_mean_frame.index)
    group_keys = group_mean_frame.index.astype(str).to_numpy()
    group_mean = group_mean_frame.to_numpy(dtype="float32", copy=True)
    group_scale = group_scale_frame.to_numpy(dtype="float32", copy=True)
    group_mean = np.where(np.isfinite(group_mean), group_mean, global_mean).astype("float32")
    group_scale = np.where(
        np.isfinite(group_scale) & (group_scale > 0.0),
        group_scale,
        global_scale,
    ).astype("float32")
    return global_mean, global_scale, group_keys, group_mean, group_scale
