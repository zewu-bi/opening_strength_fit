from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from opening_strength_fit.evaluation import resolve_group_cols
from opening_strength_fit.schema import PRICE_LEVELS


NON_FEATURE_COLUMNS = {
    "date",
    "year",
    "month",
    "minute_bucket",
    "time",
    "symbol",
    "timestamp",
    "decision_time",
    "decision_target_timestamp",
    "decision_lag_seconds",
    "entry_timestamp",
    "entry_delay_ticks",
    "entry_delay_seconds",
    "entry_max_tick_gap_seconds",
    "entry_lag_seconds",
    "entry_status",
    "label",
    "label_raw",
    "label_xs_mean",
    "label_xs_std",
    "label_xs_count",
    "label_xs_rank_pct",
    "target_label",
    "gross_label",
    "valid_label",
    "buy_price",
    "sell_vwap",
    "sell_volume",
    "sell_turnover",
    "alpha_return_next_close",
    "candidate_alpha_score",
    "candidate_alpha_rank",
    "alpha_conditioning_prediction",
    "prediction",
    "sample_weight",
    "risk_sample_weight",
}
LEAKY_PREFIXES = (
    "label_xs_",
    "target_",
    "timestamp_sell_",
    "volume_sell_",
    "turnover_sell_",
    "timestamp_entry",
    "entry_ask_price_",
    "entry_ask_volume_",
)

ENTRY_ASK_CONTEXT_COLUMNS = tuple(
    column
    for level in PRICE_LEVELS
    for column in (f"entry_ask_price_{level}", f"entry_ask_volume_{level}")
)

PREDICTION_CONTEXT_COLUMNS = (
    "status",
    "entry_status",
    "entry_timestamp",
    "entry_delay_ticks",
    "entry_delay_seconds",
    "entry_max_tick_gap_seconds",
    "gross_label",
    "buy_price",
    "volume",
    "turnover",
    "ask_price_1",
    "bid_price_1",
    "ask_volume_1",
    "bid_volume_1",
    "mid_price",
    "spread_bps",
    "ask1_to_limit_up_bps",
    "ask_depth_10",
    "bid_depth_10",
    "depth_imbalance_1",
    "depth_imbalance_10",
    "volume_diff_1t",
    "volume_diff_3t",
    "volume_diff_10t",
    "volume_diff_30t",
    "turnover_diff_1t",
    "turnover_diff_3t",
    "turnover_diff_10t",
    "turnover_diff_30t",
    "preopen_volume",
    "preopen_turnover",
    "return_10t",
    "return_30t",
    "preopen_return_vs_prev_close",
    *ENTRY_ASK_CONTEXT_COLUMNS,
)


@dataclass
class RidgePredictionModel:
    features: list[str]
    alpha: float
    pipeline: Pipeline
    model_name: str = "ridge"
    target_col: str = "label"


def _match_patterns(column: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, column) for pattern in patterns)


def _match_prefixes(column: str, prefixes: tuple[str, ...]) -> bool:
    return bool(prefixes) and column.startswith(prefixes)


def feature_columns(
    df: pd.DataFrame,
    limit: int | None = None,
    *,
    include_columns: tuple[str, ...] = (),
    include_prefixes: tuple[str, ...] = (),
    include_patterns: tuple[str, ...] = (),
    drop_columns: tuple[str, ...] = (),
    drop_prefixes: tuple[str, ...] = (),
    drop_patterns: tuple[str, ...] = (),
) -> list[str]:
    numeric_columns = df.select_dtypes(include=[np.number, "bool"]).columns
    include_columns_set = set(include_columns)
    drop_columns_set = set(drop_columns)
    has_include_filter = bool(
        include_columns_set or include_prefixes or include_patterns
    )
    features = []
    for column in numeric_columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in LEAKY_PREFIXES):
            continue
        if column in drop_columns_set:
            continue
        if _match_prefixes(str(column), drop_prefixes):
            continue
        if drop_patterns and _match_patterns(str(column), drop_patterns):
            continue
        if has_include_filter and not (
            column in include_columns_set
            or _match_prefixes(str(column), include_prefixes)
            or _match_patterns(str(column), include_patterns)
        ):
            continue
        features.append(str(column))
    if limit is not None:
        features = features[:limit]
    return features


def _clean_xy(
    df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise SystemExit(f"missing model target column: {target_col}")
    target = pd.to_numeric(df[target_col], errors="coerce")
    mask = target.notna() & np.isfinite(target)
    if "valid_label" in df.columns:
        mask &= df["valid_label"].fillna(False).astype(bool)
    if not bool(mask.any()):
        raise SystemExit("empty labeled frame after filtering valid labels")
    x = df.loc[mask, features].replace([np.inf, -np.inf], np.nan)
    y = target.loc[mask].astype("float64")
    return x, y


def fit_ridge_frame(
    train: pd.DataFrame,
    *,
    alpha: float = 1.0,
    feature_limit: int | None = None,
    target_col: str = "label",
    feature_filters: dict[str, tuple[str, ...]] | None = None,
) -> tuple[RidgePredictionModel, dict[str, int]]:
    features = feature_columns(train, feature_limit, **(feature_filters or {}))
    if not features:
        raise SystemExit("no numeric feature columns found")

    x, y = _clean_xy(train, features, target_col=target_col)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipeline.fit(x, y)
    stats = {
        "rows": len(x),
        "dates": int(train.loc[x.index, "date"].nunique()),
        "symbols": int(train.loc[x.index, "symbol"].nunique()),
        "features": len(features),
    }
    return (
        RidgePredictionModel(
            features=features,
            alpha=alpha,
            pipeline=pipeline,
            model_name="ridge",
            target_col=target_col,
        ),
        stats,
    )


def fit_gbm_frame(
    train: pd.DataFrame,
    *,
    feature_limit: int | None = None,
    target_col: str = "label",
    sample_weight_col: str = "",
    feature_filters: dict[str, tuple[str, ...]] | None = None,
    max_iter: int = 100,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.0,
    random_state: int = 7,
) -> tuple[RidgePredictionModel, dict[str, int]]:
    features = feature_columns(train, feature_limit, **(feature_filters or {}))
    if sample_weight_col:
        features = [column for column in features if column != sample_weight_col]
    if not features:
        raise SystemExit("no numeric feature columns found")

    x, y = _clean_xy(train, features, target_col=target_col)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "gbm",
                HistGradientBoostingRegressor(
                    max_iter=int(max_iter),
                    learning_rate=float(learning_rate),
                    max_leaf_nodes=int(max_leaf_nodes),
                    l2_regularization=float(l2_regularization),
                    random_state=int(random_state),
                ),
            ),
        ]
    )
    pipeline.fit(x, y)
    stats = {
        "rows": len(x),
        "dates": int(train.loc[x.index, "date"].nunique()),
        "symbols": int(train.loc[x.index, "symbol"].nunique()),
        "features": len(features),
    }
    return (
        RidgePredictionModel(
            features=features,
            alpha=float("nan"),
            pipeline=pipeline,
            model_name="gbm",
            target_col=target_col,
        ),
        stats,
    )


def fit_lightgbm_frame(
    train: pd.DataFrame,
    *,
    feature_limit: int | None = None,
    target_col: str = "label",
    sample_weight_col: str = "",
    feature_filters: dict[str, tuple[str, ...]] | None = None,
    n_estimators: int = 300,
    learning_rate: float = 0.03,
    num_leaves: int = 63,
    max_depth: int = -1,
    min_child_samples: int = 200,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    random_state: int = 7,
    n_jobs: int = -1,
    device_type: str = "cpu",
    max_bin: int | None = None,
    gpu_use_dp: bool = False,
) -> tuple[RidgePredictionModel, dict[str, int]]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise SystemExit(
            "model.name='lightgbm' requires the lightgbm package. "
            "Install project dependencies or rebuild the training image."
        ) from exc

    features = feature_columns(train, feature_limit, **(feature_filters or {}))
    if not features:
        raise SystemExit("no numeric feature columns found")

    x, y = _clean_xy(train, features, target_col=target_col)
    sample_weight = None
    if sample_weight_col:
        if sample_weight_col not in train.columns:
            raise SystemExit(f"missing sample weight column: {sample_weight_col}")
        sample_weight = (
            pd.to_numeric(train.loc[x.index, sample_weight_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0)
        )
    lightgbm_params = {
        "objective": "regression",
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "min_child_samples": int(min_child_samples),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_alpha": float(reg_alpha),
        "reg_lambda": float(reg_lambda),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "verbosity": -1,
    }
    device_type = str(device_type or "cpu").strip().lower()
    if device_type not in {"", "auto"}:
        lightgbm_params["device_type"] = device_type
    if max_bin is not None and int(max_bin) > 0:
        lightgbm_params["max_bin"] = int(max_bin)
    if device_type in {"gpu", "cuda"}:
        lightgbm_params["gpu_use_dp"] = bool(gpu_use_dp)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("lightgbm", LGBMRegressor(**lightgbm_params)),
        ]
    )
    fit_params = (
        {"lightgbm__sample_weight": sample_weight.to_numpy(dtype="float64")}
        if sample_weight is not None
        else {}
    )
    pipeline.fit(x, y, **fit_params)
    stats = {
        "rows": len(x),
        "dates": int(train.loc[x.index, "date"].nunique()),
        "symbols": int(train.loc[x.index, "symbol"].nunique()),
        "features": len(features),
    }
    if sample_weight is not None:
        stats["sample_weight_mean"] = float(sample_weight.mean())
        stats["sample_weight_zero_rate"] = float((sample_weight <= 0.0).mean())
    return (
        RidgePredictionModel(
            features=features,
            alpha=float("nan"),
            pipeline=pipeline,
            model_name=f"lightgbm_{device_type or 'cpu'}",
            target_col=target_col,
        ),
        stats,
    )


def predict_frame(model: RidgePredictionModel, frame: pd.DataFrame) -> pd.DataFrame:
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
        column
        for column in PREDICTION_CONTEXT_COLUMNS
        if column in frame and column not in columns
    )
    out = frame[columns].copy()
    x = frame[model.features].replace([np.inf, -np.inf], np.nan)
    out["prediction"] = model.pipeline.predict(x)
    if "valid_label" in frame.columns:
        out["valid_label"] = frame["valid_label"].to_numpy()
    return out


def corr(a: pd.Series, b: pd.Series, method: str) -> float:
    valid = a.notna() & b.notna()
    a = a.loc[valid]
    b = b.loc[valid]
    if len(a) < 2 or a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return float("nan")
    return float(a.corr(b, method=method))


def ir(mean: float, std: float) -> float:
    if pd.isna(std) or std == 0:
        return float("nan")
    return mean / std


def daily_prediction_metrics(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    score_col: str = "prediction",
) -> pd.DataFrame:
    frame = df.loc[df[label_col].notna() & df[score_col].notna()].copy()
    return (
        frame.groupby("date")
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "ic": corr(group[label_col], group[score_col], "pearson"),
                    "rank_ic": corr(group[label_col], group[score_col], "spearman"),
                    "mean_label": float(group[label_col].mean()),
                    "win_rate": float((group[label_col] > 0).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def grouped_prediction_metrics(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    score_col: str = "prediction",
    group_cols: tuple[str, ...] = ("date",),
) -> pd.DataFrame:
    frame = df.loc[df[label_col].notna() & df[score_col].notna()].copy()
    available_group_cols = resolve_group_cols(frame, group_cols)
    if not available_group_cols:
        return pd.DataFrame(
            [
                {
                    "rows": len(frame),
                    "ic": corr(frame[label_col], frame[score_col], "pearson"),
                    "rank_ic": corr(frame[label_col], frame[score_col], "spearman"),
                    "mean_label": (
                        float(frame[label_col].mean()) if len(frame) else float("nan")
                    ),
                    "win_rate": (
                        float((frame[label_col] > 0).mean())
                        if len(frame)
                        else float("nan")
                    ),
                }
            ]
        )
    return (
        frame.groupby(list(available_group_cols))
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "ic": corr(group[label_col], group[score_col], "pearson"),
                    "rank_ic": corr(group[label_col], group[score_col], "spearman"),
                    "mean_label": float(group[label_col].mean()),
                    "win_rate": float((group[label_col] > 0).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def evaluate_prediction_frame(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    score_col: str = "prediction",
    group_cols: tuple[str, ...] = ("date",),
) -> dict[str, object]:
    frame = df.loc[df[label_col].notna() & df[score_col].notna()].copy()
    resolved_group_cols = resolve_group_cols(frame, group_cols)
    daily = daily_prediction_metrics(frame, label_col=label_col, score_col=score_col)
    grouped = grouped_prediction_metrics(
        frame,
        label_col=label_col,
        score_col=score_col,
        group_cols=resolved_group_cols,
    )
    ic_mean = float(daily["ic"].mean())
    ic_std = float(daily["ic"].std())
    rank_ic_mean = float(daily["rank_ic"].mean())
    rank_ic_std = float(daily["rank_ic"].std())
    group_ic_mean = float(grouped["ic"].mean())
    group_ic_std = float(grouped["ic"].std())
    group_rank_ic_mean = float(grouped["rank_ic"].mean())
    group_rank_ic_std = float(grouped["rank_ic"].std())
    ic_grouping = ",".join(resolved_group_cols) if resolved_group_cols else "global"
    sample_grain = (
        "date x symbol x decision_time"
        if "decision_time" in frame.columns
        else "date x symbol x opening_tick"
    )
    return {
        "rows": len(frame),
        "dates": int(frame["date"].nunique()) if "date" in frame.columns else 0,
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        "sample_grain": sample_grain,
        "ic_grouping": ic_grouping,
        "ic_groups": int(len(grouped)),
        "overall_ic": corr(frame[label_col], frame[score_col], "pearson"),
        "overall_rank_ic": corr(frame[label_col], frame[score_col], "spearman"),
        "group_ic_mean": group_ic_mean,
        "group_ic_std": group_ic_std,
        "group_ic_ir": ir(group_ic_mean, group_ic_std),
        "group_rank_ic_mean": group_rank_ic_mean,
        "group_rank_ic_std": group_rank_ic_std,
        "group_rank_ic_ir": ir(group_rank_ic_mean, group_rank_ic_std),
        "daily_ic_mean": ic_mean,
        "daily_ic_std": ic_std,
        "daily_ic_ir": ir(ic_mean, ic_std),
        "daily_rank_ic_mean": rank_ic_mean,
        "daily_rank_ic_std": rank_ic_std,
        "daily_rank_ic_ir": ir(rank_ic_mean, rank_ic_std),
        "mean_label": float(frame[label_col].mean()) if len(frame) else float("nan"),
        "win_rate": float((frame[label_col] > 0).mean()) if len(frame) else float("nan"),
    }
