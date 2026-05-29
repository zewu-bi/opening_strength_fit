from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from opening_strength_fit.config import (
    config_float,
    config_int,
    config_optional_int,
    config_str,
    config_value,
)
from opening_strength_fit.labels import finite_numeric_series
from opening_strength_fit.model import RidgePredictionModel, fit_lightgbm_frame
from opening_strength_fit.training import _feature_filters_from_config, _feature_limit


KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")


def section_value(config: dict, section: str, fallback_section: str, key: str, default):
    return config_value(
        config,
        section,
        key,
        config_value(config, fallback_section, key, default),
    )


def section_int(
    config: dict,
    section: str,
    fallback_section: str,
    key: str,
    default: int,
) -> int:
    return int(section_value(config, section, fallback_section, key, default))


def section_float(
    config: dict,
    section: str,
    fallback_section: str,
    key: str,
    default: float,
) -> float:
    return float(section_value(config, section, fallback_section, key, default))


def section_str(
    config: dict,
    section: str,
    fallback_section: str,
    key: str,
    default: str,
) -> str:
    return str(section_value(config, section, fallback_section, key, default))


def section_optional_int(
    config: dict,
    section: str,
    fallback_section: str,
    key: str,
    default: int | None = None,
) -> int | None:
    value = section_value(config, section, fallback_section, key, default)
    return None if value in (None, "") else int(value)


def fit_lgbm_config_section(
    train: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    section: str,
    target_col: str,
    sample_weight_col: str = "",
    random_state_default: int = 7,
) -> tuple[RidgePredictionModel, dict[str, int]]:
    return fit_lightgbm_frame(
        train,
        feature_limit=_feature_limit(args, config),
        target_col=target_col,
        sample_weight_col=sample_weight_col,
        feature_filters=_feature_filters_from_config(config),
        n_estimators=section_int(config, section, "model", "n_estimators", 300),
        learning_rate=section_float(config, section, "model", "learning_rate", 0.03),
        num_leaves=section_int(config, section, "model", "num_leaves", 63),
        max_depth=section_int(config, section, "model", "max_depth", -1),
        min_child_samples=section_int(
            config,
            section,
            "model",
            "min_child_samples",
            200,
        ),
        subsample=section_float(config, section, "model", "subsample", 1.0),
        colsample_bytree=section_float(
            config,
            section,
            "model",
            "colsample_bytree",
            1.0,
        ),
        reg_alpha=section_float(config, section, "model", "reg_alpha", 0.0),
        reg_lambda=section_float(config, section, "model", "reg_lambda", 0.0),
        random_state=section_int(
            config,
            section,
            "model",
            "random_state",
            random_state_default,
        ),
        n_jobs=section_int(config, section, "model", "n_jobs", -1),
        device_type=section_str(config, section, "model", "device_type", "cpu"),
        max_bin=section_optional_int(config, section, "model", "max_bin", None),
        gpu_use_dp=False,
    )


def predict_model_score(model: RidgePredictionModel, frame: pd.DataFrame) -> np.ndarray:
    missing = set(model.features) - set(frame.columns)
    if missing:
        raise SystemExit(f"prediction frame is missing features: {sorted(missing)[:5]}")
    x = frame[model.features].replace([np.inf, -np.inf], np.nan)
    return model.pipeline.predict(x)


def add_group_rank(frame: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    groupers = [frame["date"], frame["decision_target_timestamp"]]
    frame[rank_col] = (
        pd.to_numeric(frame[score_col], errors="coerce")
        .groupby(groupers)
        .rank(method="average", pct=True)
    )
    return frame


def alpha_conditioned_reversal_risk(
    labeled: pd.DataFrame,
    config: dict,
    *,
    target_form: str | None = None,
    next_rank_max: float | None = None,
    candidate_alpha_rank_min: float | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    if "alpha_return_next_close" not in labeled.columns:
        raise SystemExit("alpha-conditioned target requires alpha_return_next_close")
    if "candidate_alpha_rank" not in labeled.columns:
        raise SystemExit("alpha-conditioned target requires candidate_alpha_rank")

    groupers = [labeled["date"], labeled["decision_target_timestamp"]]
    alpha_rank = finite_numeric_series(labeled["candidate_alpha_rank"])
    next_rank = finite_numeric_series(labeled["alpha_return_next_close"]).groupby(
        groupers
    ).rank(method="average", pct=True)
    candidate_min = (
        config_float(config, "risk_layer", "candidate_alpha_rank_min", 0.80)
        if candidate_alpha_rank_min is None
        else float(candidate_alpha_rank_min)
    )
    candidate = alpha_rank.ge(candidate_min)

    form = (
        config_str(config, "risk_layer", "target_form", "binary_next_low")
        if target_form is None
        else target_form
    ).strip().lower()
    rank_max = (
        config_float(config, "risk_layer", "next_rank_max", 0.40)
        if next_rank_max is None
        else float(next_rank_max)
    )
    if form in {"binary", "hard", "binary_next_low"}:
        risk = next_rank.le(rank_max).astype("float64")
    elif form in {"gap", "next_gap", "next_rank_gap"}:
        risk = ((rank_max - next_rank) / rank_max).clip(lower=0.0, upper=1.0)
    else:
        raise SystemExit(
            f"unknown [risk_layer].target_form={form!r}; expected binary_next_low or next_rank_gap"
        )

    risk = risk.where(candidate, 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)

    sample_weight: pd.Series | None = None
    non_candidate_weight = config_float(
        config,
        "risk_layer",
        "non_candidate_weight",
        0.05,
    )
    candidate_weight = config_float(config, "risk_layer", "candidate_weight", 1.0)
    if non_candidate_weight != 1.0 or candidate_weight != 1.0:
        sample_weight = pd.Series(
            np.where(candidate, candidate_weight, non_candidate_weight),
            index=labeled.index,
            dtype="float64",
        )
    return risk, candidate.astype("bool"), sample_weight


def add_alpha_conditioned_risk_targets(
    train: pd.DataFrame,
    config: dict,
    *,
    gap_target_col: str = "target_alpha_conditioned_gap_risk",
    binary_target_col: str = "target_alpha_conditioned_binary_risk",
    sample_weight_col: str = "risk_sample_weight",
    candidate_col: str = "target_alpha_conditioned_candidate",
    copy_frame: bool = True,
) -> pd.DataFrame:
    out = train.copy() if copy_frame else train
    candidate_min = config_float(config, "risk_layer", "candidate_alpha_rank_min", 0.80)
    gap_rank_max = config_float(config, "risk_layer", "gap_next_rank_max", 0.50)
    binary_rank_max = config_float(config, "risk_layer", "binary_next_rank_max", 0.40)

    gap_risk, candidate, sample_weight = alpha_conditioned_reversal_risk(
        out,
        config,
        target_form="next_rank_gap",
        next_rank_max=gap_rank_max,
        candidate_alpha_rank_min=candidate_min,
    )
    binary_risk, _, _ = alpha_conditioned_reversal_risk(
        out,
        config,
        target_form="binary_next_low",
        next_rank_max=binary_rank_max,
        candidate_alpha_rank_min=candidate_min,
    )
    out[gap_target_col] = gap_risk
    out[binary_target_col] = binary_risk
    out[candidate_col] = candidate
    out[sample_weight_col] = sample_weight if sample_weight is not None else 1.0
    return out
