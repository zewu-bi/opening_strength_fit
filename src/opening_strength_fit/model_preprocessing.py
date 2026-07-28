from __future__ import annotations

import pandas as pd

from opening_strength_fit.feature_transforms.mechanism import (
    mechanismized_v3_changed_feature_columns,
)
from opening_strength_fit.torch_model.preprocessing import (
    _normalize_feature_value_transform,
    _torch_feature_value_frame,
)

_REPLACE_OUTPUT_MODES = {"", "replace", "inplace", "in_place", "transformed_only"}
_APPEND_OUTPUT_MODES = {
    "append",
    "raw_plus_transformed",
    "raw_and_transformed",
    "dual",
}


def _normalize_feature_value_transform_output(value: str) -> str:
    mode = str(value or "replace").strip().lower().replace("-", "_")
    if mode in _REPLACE_OUTPUT_MODES:
        return "replace"
    if mode in _APPEND_OUTPUT_MODES:
        return "append"
    raise SystemExit(
        "features.feature_value_transform_output must be replace or raw_plus_transformed"
    )


def lightgbm_feature_value_frame(
    frame: pd.DataFrame,
    source_features: list[str],
    *,
    feature_value_transform: str,
    feature_value_transform_output: str = "replace",
    feature_value_transform_prefix: str = "mech_v3_",
    group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    rank_method: str = "average",
    tick_size: float = 0.01,
    extra_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Build LightGBM inputs with the same value transform used by Torch models."""

    normalized_transform = _normalize_feature_value_transform(feature_value_transform)
    output_mode = _normalize_feature_value_transform_output(feature_value_transform_output)
    if normalized_transform == "none":
        if output_mode == "append":
            raise SystemExit("raw_plus_transformed requires a non-identity feature_value_transform")
        return frame, list(source_features)

    if output_mode == "replace":
        transformed = _torch_feature_value_frame(
            frame,
            source_features,
            feature_value_transform=normalized_transform,
            group_cols=group_cols,
            rank_method=rank_method,
            tick_size=tick_size,
            extra_columns=extra_columns,
        )
        return transformed, list(source_features)

    if not normalized_transform.startswith("mechanismized_v3_"):
        raise SystemExit("raw_plus_transformed currently supports only mechanismized_v3 transforms")
    prefix = str(feature_value_transform_prefix or "").strip()
    if not prefix:
        raise SystemExit("features.feature_value_transform_prefix must be non-empty")

    changed_features = list(mechanismized_v3_changed_feature_columns(tuple(source_features)))
    if not changed_features:
        raise SystemExit("mechanismized_v3 append mode found no materially changed features")
    transformed = _torch_feature_value_frame(
        frame,
        changed_features,
        feature_value_transform=normalized_transform,
        group_cols=group_cols,
        rank_method=rank_method,
        tick_size=tick_size,
        extra_columns=(),
    )
    derived_names = [f"{prefix}{column}" for column in changed_features]
    collisions = [column for column in derived_names if column in frame.columns]
    if collisions:
        raise SystemExit(
            f"feature_value_transform_prefix collides with input columns: {collisions[:5]}"
        )

    retained = list(
        dict.fromkeys(
            column
            for column in (*extra_columns, *source_features)
            if column and column in frame.columns
        )
    )
    derived = transformed.loc[:, changed_features].copy()
    derived.columns = derived_names
    out = pd.concat([frame.loc[:, retained].copy(), derived], axis=1)
    return out, [*source_features, *derived_names]


__all__ = [
    "_normalize_feature_value_transform_output",
    "lightgbm_feature_value_frame",
]
