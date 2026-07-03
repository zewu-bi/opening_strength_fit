from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.evaluation import resolve_group_cols
from opening_strength_fit.feature_utils import safe_divide
from opening_strength_fit.model_torch import _torch_mlp_score
from opening_strength_fit.model_types import (
    PREDICTION_CONTEXT_COLUMNS,
    ClockSegmentPredictionModel,
    EnsemblePredictionModel,
    RidgePredictionModel,
    TorchMLPPredictionModel,
)


def _normalized_weights(weights: list[float], size: int) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype="float64")
    if len(weights) != size:
        raw = np.ones(size, dtype="float64")
    else:
        raw = np.asarray(weights, dtype="float64")
        raw = np.where(np.isfinite(raw), raw, 0.0)
    total = float(raw.sum())
    if total == 0.0:
        return np.ones(size, dtype="float64") / size
    return raw / total


def _model_score(
    model: RidgePredictionModel | TorchMLPPredictionModel, frame: pd.DataFrame
) -> np.ndarray:
    missing = set(model.features) - set(frame.columns)
    if missing:
        raise SystemExit(f"prediction frame is missing features: {sorted(missing)[:5]}")
    if isinstance(model, TorchMLPPredictionModel):
        return _torch_mlp_score(model, frame)
    x = frame[model.features].replace([np.inf, -np.inf], np.nan)
    return np.asarray(model.pipeline.predict(x), dtype="float64")


def _group_relative_score(
    scores: pd.Series,
    frame: pd.DataFrame,
    *,
    mode: str,
    group_cols: tuple[str, ...],
) -> pd.Series:
    available_group_cols = resolve_group_cols(frame, group_cols)
    if not available_group_cols:
        if mode == "rank":
            return scores.rank(method="average", pct=True)
        mean = scores.mean()
        std = scores.std()
        return pd.Series(safe_divide(scores - mean, std), index=scores.index)

    grouped = scores.groupby([frame[column] for column in available_group_cols], sort=False)
    if mode == "rank":
        return grouped.rank(method="average", pct=True)
    centered = scores - grouped.transform("mean")
    return pd.Series(safe_divide(centered, grouped.transform("std")), index=scores.index)


def _ensemble_score(model: EnsemblePredictionModel, frame: pd.DataFrame) -> np.ndarray:
    if not model.models:
        raise SystemExit("ensemble model has no members")
    combine_mode = model.combine_mode.strip().lower()
    member_scores = []
    for member in model.models:
        scores = pd.Series(_model_score(member, frame), index=frame.index)
        if combine_mode in {"rank", "rank_mean", "rank_pct"}:
            scores = _group_relative_score(
                scores,
                frame,
                mode="rank",
                group_cols=model.rank_group_cols,
            )
        elif combine_mode in {"rank_centered", "centered_rank"}:
            scores = (
                _group_relative_score(
                    scores,
                    frame,
                    mode="rank",
                    group_cols=model.rank_group_cols,
                )
                - 0.5
            )
        elif combine_mode in {"zscore", "zscore_mean"}:
            scores = _group_relative_score(
                scores,
                frame,
                mode="zscore",
                group_cols=model.rank_group_cols,
            )
        elif combine_mode not in {"raw", "mean", "raw_mean"}:
            raise SystemExit(
                "model.combine_mode for ensemble must be raw/rank/rank_centered/zscore"
            )
        member_scores.append(scores.to_numpy(dtype="float64"))
    weights = _normalized_weights(model.weights, len(member_scores))
    stacked = np.vstack(member_scores)
    return np.average(stacked, axis=0, weights=weights)


def _frame_clock(frame: pd.DataFrame) -> pd.Series:
    if "decision_time" in frame.columns:
        raw = frame["decision_time"].astype(str)
        extracted = raw.str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False).fillna("")
        return extracted.map(_normalize_clock_value)
    time_col = "decision_target_timestamp" if "decision_target_timestamp" in frame else "timestamp"
    return pd.to_datetime(frame[time_col], errors="coerce").dt.strftime("%H:%M:%S").fillna("")


def _normalize_clock_value(value: str) -> str:
    parts = str(value).split(":")
    if len(parts) == 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
    if len(parts) >= 3:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(float(parts[2])):02d}"
    return ""


def _clock_segment_score(model: ClockSegmentPredictionModel, frame: pd.DataFrame) -> np.ndarray:
    scores = np.full(len(frame), np.nan, dtype="float64")
    clock = _frame_clock(frame)
    assigned = pd.Series(False, index=frame.index)
    for _name, clocks, segment_model in model.segment_models:
        mask = clock.isin(set(clocks)) & ~assigned
        if not bool(mask.any()):
            continue
        scores[mask.to_numpy()] = _model_score(segment_model, frame.loc[mask])
        assigned.loc[mask] = True
    if assigned.all():
        return scores
    if model.fallback_model is None:
        missing_clocks = sorted(clock.loc[~assigned].dropna().unique())
        raise SystemExit(f"clock-segment model has no segment for clocks: {missing_clocks}")
    missing = ~assigned
    scores[missing.to_numpy()] = _model_score(model.fallback_model, frame.loc[missing])
    return scores


def predict_frame(
    model: (
        RidgePredictionModel
        | EnsemblePredictionModel
        | ClockSegmentPredictionModel
        | TorchMLPPredictionModel
    ),
    frame: pd.DataFrame,
) -> pd.DataFrame:
    missing = set(model.features) - set(frame.columns)
    if missing:
        raise SystemExit(f"prediction frame is missing features: {sorted(missing)[:5]}")

    columns = [
        column
        for column in (
            "date",
            "symbol",
            "timestamp",
            "decision_time",
            "decision_target_timestamp",
            "decision_lag_seconds",
            "label",
            model.target_col,
        )
        if column in frame
    ]
    columns = list(dict.fromkeys(columns))
    columns.extend(
        column for column in PREDICTION_CONTEXT_COLUMNS if column in frame and column not in columns
    )
    out = frame[columns].copy()
    if isinstance(model, EnsemblePredictionModel):
        out["prediction"] = _ensemble_score(model, frame)
    elif isinstance(model, ClockSegmentPredictionModel):
        out["prediction"] = _clock_segment_score(model, frame)
    else:
        out["prediction"] = _model_score(model, frame)
    if "valid_label" in frame.columns:
        out["valid_label"] = frame["valid_label"].to_numpy()
    return out
