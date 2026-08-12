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


def _aliases(prefix: str, groups: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {f"{prefix}{alias}": mode for mode, aliases in groups.items() for alias in aliases}


_RANK_ALIASES = ("cross_sectional_rank_centered", "xs_rank_centered", "rank_centered")
_ZSCORE_ALIASES = ("cross_sectional_zscore", "xs_zscore", "zscore")
_ROBUST_ALIASES = ("cross_sectional_robust_zscore", "xs_robust_zscore", "robust_zscore")
_CROSS_SECTIONAL_FEATURE_VALUE_TRANSFORMS = _aliases(
    "",
    {
        "demean": ("cross_sectional_demean", "xs_demean"),
        "zscore": ("cross_sectional_zscore", "xs_zscore"),
        "robust_zscore": ("cross_sectional_robust_zscore", "xs_robust_zscore"),
        "rank_pct": ("cross_sectional_rank_pct", "xs_rank_pct"),
        "rank": ("cross_sectional_rank", "xs_rank"),
        "rank_centered": (*_RANK_ALIASES, "cross_sectional_rank_centered_inplace"),
    },
)
_MECHANISMIZED_FEATURE_VALUE_TRANSFORMS = _aliases(
    "mechanismized_",
    {
        "rank_centered": (*_RANK_ALIASES, "dimensionless", "dimensionless_328"),
        "zscore": _ZSCORE_ALIASES,
        "none": ("none", "only"),
    },
) | _aliases(
    "mechanism_aware_",
    {"rank_centered": _RANK_ALIASES, "none": ("only",)},
)
_MECHANISMIZED_V2_FEATURE_VALUE_TRANSFORMS = _aliases(
    "mechanismized_v2_",
    {
        "robust_zscore": (*_ROBUST_ALIASES, "dimensionless", "dimensionless_328"),
        "zscore": _ZSCORE_ALIASES,
        "rank_centered": _RANK_ALIASES,
        "none": ("none", "only"),
    },
) | _aliases(
    "mechanism_aware_v2_",
    {"robust_zscore": ("cross_sectional_robust_zscore", "dimensionless"), "none": ("only",)},
)
_MECHANISMIZED_V3_FEATURE_VALUE_TRANSFORMS = _aliases(
    "mechanismized_v3_",
    {
        "robust_zscore": _ROBUST_ALIASES,
        "zscore": _ZSCORE_ALIASES,
        "rank_centered": _RANK_ALIASES,
        "none": ("dimensionless", "dimensionless_328", "none", "only"),
    },
) | _aliases(
    "mechanism_aware_v3_",
    {"robust_zscore": ("cross_sectional_robust_zscore",), "none": ("dimensionless", "only")},
)


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
    if mode == "none":
        missing_features = [column for column in features if column not in frame.columns]
        if missing_features:
            raise SystemExit(f"model features are missing columns: {missing_features[:5]}")
        # A published model-ready frame already contains the final float32 feature
        # values. Returning it directly avoids duplicating the full wide training set.
        return frame
    available = [column for column in required if column and column in frame.columns]
    model_frame = frame.loc[:, available].copy()
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
    row_mask: pd.Series | np.ndarray | None = None,
    column_block_size: int = 16,
    stats_row_block_size: int = 131_072,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_values = None if row_mask is None else np.asarray(row_mask, dtype=bool)
    if mask_values is not None and len(mask_values) != len(frame):
        raise ValueError("row_mask must have the same length as frame")
    row_count = len(frame) if mask_values is None else int(mask_values.sum())
    values = np.empty((row_count, len(features)), dtype=np.float32)
    column_block_size = max(1, int(column_block_size))
    for start in range(0, len(features), column_block_size):
        end = min(start + column_block_size, len(features))
        columns = features[start:end]
        block = frame.loc[:, columns] if mask_values is None else frame.loc[mask_values, columns]
        values[:, start:end] = block.to_numpy(dtype=np.float32, copy=False)

    stats_row_block_size = max(1, int(stats_row_block_size))
    if mean is None:
        sums = np.zeros(len(features), dtype=np.float64)
        counts = np.zeros(len(features), dtype=np.int64)
        for start in range(0, len(values), stats_row_block_size):
            block = values[start : start + stats_row_block_size]
            finite = np.isfinite(block)
            sums += np.where(finite, block, 0.0).sum(axis=0, dtype=np.float64)
            counts += finite.sum(axis=0, dtype=np.int64)
        mean64 = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums),
            where=counts > 0,
        )
        mean = mean64.astype("float32")
    if scale is None:
        squared_deviations = np.zeros(len(features), dtype=np.float64)
        counts = np.zeros(len(features), dtype=np.int64)
        mean64 = mean.astype(np.float64)
        for start in range(0, len(values), stats_row_block_size):
            block = values[start : start + stats_row_block_size]
            finite = np.isfinite(block)
            centered = block.astype(np.float64) - mean64
            centered[~finite] = 0.0
            squared_deviations += np.einsum("ij,ij->j", centered, centered)
            counts += finite.sum(axis=0, dtype=np.int64)
        variance = np.divide(
            squared_deviations,
            counts,
            out=np.zeros_like(squared_deviations),
            where=counts > 0,
        )
        scale = np.sqrt(variance).astype("float32")
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0).astype("float32")
    if group_keys is None or group_mean is None or group_scale is None:
        for start in range(0, len(values), stats_row_block_size):
            block = values[start : start + stats_row_block_size]
            block -= mean
            block /= scale
            np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
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
