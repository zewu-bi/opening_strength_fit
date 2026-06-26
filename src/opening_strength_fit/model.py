from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from opening_strength_fit.evaluation import resolve_group_cols
from opening_strength_fit.features import safe_divide
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


@dataclass
class EnsemblePredictionModel:
    features: list[str]
    alpha: float
    models: list[RidgePredictionModel]
    weights: list[float]
    combine_mode: str = "rank"
    rank_group_cols: tuple[str, ...] = ("date", "decision_target_timestamp")
    model_name: str = "ensemble"
    target_col: str = "label"


@dataclass
class ClockSegmentPredictionModel:
    features: list[str]
    segment_models: list[tuple[str, tuple[str, ...], RidgePredictionModel]]
    fallback_model: RidgePredictionModel | None = None
    model_name: str = "clock_segment"
    target_col: str = "label"


@dataclass
class TorchMLPPredictionModel:
    features: list[str]
    module: Any
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    device: str
    batch_size: int
    model_name: str = "torch_mlp"
    target_col: str = "label"


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "model.name='torch_mlp' requires PyTorch. Rebuild the training image with "
            "INSTALL_TORCH_CUDA=1 or install torch in the runtime environment."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


class _TorchMLPModule:
    @staticmethod
    def build(
        *,
        input_dim: int,
        hidden_layers: tuple[int, ...],
        dropout: float,
        activation: str,
        architecture: str = "mlp",
    ):
        _torch, nn, _loader, _dataset = _import_torch()
        activation_name = activation.strip().lower()
        activation_factories = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation_name not in activation_factories:
            raise SystemExit(
                "model.activation for torch_mlp must be one of relu, gelu, silu, swish, tanh"
            )
        architecture_name = architecture.strip().lower() or "mlp"
        if architecture_name == "wide_deep_residual":
            if len(hidden_layers) != 1:
                raise SystemExit(
                    "model.architecture='wide_deep_residual' expects exactly one hidden layer"
                )
            hidden_dim = int(hidden_layers[0])
            if hidden_dim <= 0:
                raise SystemExit("model.hidden_layers for torch_mlp must contain positive ints")
            return _WideDeepResidualMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=float(dropout),
                activation=activation_factories[activation_name],
                nn=nn,
            )
        if architecture_name not in {"mlp", "sequential"}:
            raise SystemExit("model.architecture for torch_mlp must be mlp or wide_deep_residual")
        layers = []
        current_dim = int(input_dim)
        for width in hidden_layers:
            width = int(width)
            if width <= 0:
                raise SystemExit("model.hidden_layers for torch_mlp must contain positive ints")
            layers.append(nn.Linear(current_dim, width))
            layers.append(activation_factories[activation_name]())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = width
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)


def _WideDeepResidualMLP(
    *,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    activation,
    nn,
):
    class WideDeepResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_branch = nn.Linear(input_dim, 1)
            self.nonlinear_hidden = nn.Linear(input_dim, hidden_dim)
            self.nonlinear_norm = nn.LayerNorm(hidden_dim)
            self.nonlinear_activation = activation()
            self.nonlinear_dropout = nn.Dropout(dropout)
            self.nonlinear_output = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.nonlinear_output.weight)
            nn.init.zeros_(self.nonlinear_output.bias)

        def forward(self, x):
            linear = self.linear_branch(x)
            nonlinear = self.nonlinear_hidden(x)
            nonlinear = self.nonlinear_norm(nonlinear)
            nonlinear = self.nonlinear_activation(nonlinear)
            nonlinear = self.nonlinear_dropout(nonlinear)
            nonlinear = self.nonlinear_output(nonlinear)
            return linear + nonlinear

    return WideDeepResidualMLP()


def _torch_device(device: str, torch) -> str:
    requested = device.strip().lower() or "auto"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("model.device requests CUDA, but torch.cuda.is_available() is false")
    if requested in {"gpu", "cuda"}:
        return "cuda"
    if requested in {"cpu"} or requested.startswith("cuda:"):
        return requested
    raise SystemExit("model.device for torch_mlp must be auto, cpu, cuda, gpu, or cuda:<index>")


def _torch_loss(loss: str, torch, reduction: str = "none"):
    name = loss.strip().lower()
    if name in {"mse", "l2", "mean_squared_error"}:
        return torch.nn.MSELoss(reduction=reduction)
    if name in {"huber", "smooth_l1", "smoothl1"}:
        return torch.nn.SmoothL1Loss(reduction=reduction)
    raise SystemExit("model.loss for torch_mlp must be mse or huber")


def _standardized_float_matrix(
    frame: pd.DataFrame,
    features: list[str],
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
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
    values -= mean
    values /= scale
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return values, mean.astype("float32"), scale.astype("float32")


def fit_torch_mlp_frame(
    train: pd.DataFrame,
    *,
    feature_limit: int | None = None,
    target_col: str = "label",
    sample_weight_col: str = "",
    feature_filters: dict[str, tuple[str, ...]] | None = None,
    hidden_layers: tuple[int, ...] = (512, 256, 128),
    architecture: str = "mlp",
    dropout: float = 0.1,
    activation: str = "relu",
    batch_size: int = 32768,
    predict_batch_size: int | None = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 8,
    validation_fraction: float = 0.02,
    validation_max_rows: int = 250_000,
    early_stopping_patience: int = 2,
    loss: str = "mse",
    device: str = "auto",
    random_state: int = 7,
    num_workers: int = 0,
) -> tuple[TorchMLPPredictionModel, dict[str, int | float | str | bool]]:
    torch, _nn, DataLoader, TensorDataset = _import_torch()
    features = feature_columns(train, feature_limit, **(feature_filters or {}))
    if sample_weight_col:
        features = [column for column in features if column != sample_weight_col]
    if not features:
        raise SystemExit("no numeric feature columns found")

    x_frame, y_series = _clean_xy(train, features, target_col=target_col)
    x_values, feature_mean, feature_scale = _standardized_float_matrix(x_frame, features)
    y_values = y_series.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

    sample_weight = None
    if sample_weight_col:
        if sample_weight_col not in train.columns:
            raise SystemExit(f"missing sample weight column: {sample_weight_col}")
        sample_weight = (
            pd.to_numeric(train.loc[x_frame.index, sample_weight_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=np.float32)
            .reshape(-1, 1)
        )

    if len(x_values) < 2:
        raise SystemExit("torch_mlp needs at least two valid training rows")
    torch.manual_seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))
    rng = np.random.default_rng(int(random_state))

    resolved_device = _torch_device(device, torch)
    module = _TorchMLPModule.build(
        input_dim=len(features),
        hidden_layers=tuple(int(value) for value in hidden_layers),
        dropout=float(dropout),
        activation=activation,
        architecture=architecture,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    criterion = _torch_loss(loss, torch, reduction="none")

    x_tensor = torch.from_numpy(x_values)
    y_tensor = torch.from_numpy(y_values)
    tensors = [x_tensor, y_tensor]
    if sample_weight is not None:
        weight_tensor = torch.from_numpy(sample_weight)
        tensors.append(weight_tensor)
    dataset = TensorDataset(*tensors)

    n_rows = len(dataset)
    validation_rows = int(n_rows * max(0.0, float(validation_fraction)))
    if validation_max_rows > 0:
        validation_rows = min(validation_rows, int(validation_max_rows))
    validation_rows = max(0, min(validation_rows, n_rows - 1))
    if validation_rows > 0:
        validation_indices = rng.choice(n_rows, size=validation_rows, replace=False)
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[validation_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        validation_dataset = torch.utils.data.Subset(dataset, validation_indices)
    else:
        train_dataset = dataset
        validation_dataset = None

    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(str(resolved_device).startswith("cuda")),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
        if validation_dataset is not None
        else None
    )

    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    patience_used = 0
    epochs_trained = 0
    final_train_loss = float("nan")
    final_validation_loss = float("nan")
    max_epochs = int(max_epochs)
    for epoch in range(1, max_epochs + 1):
        module.train()
        total_loss = 0.0
        total_weight = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_x = batch[0].to(resolved_device, non_blocking=True)
            batch_y = batch[1].to(resolved_device, non_blocking=True)
            raw_loss = criterion(module(batch_x), batch_y)
            if len(batch) > 2:
                batch_w = batch[2].to(resolved_device, non_blocking=True)
                raw_loss = raw_loss * batch_w
                denom = torch.clamp(batch_w.sum(), min=1.0)
            else:
                denom = torch.tensor(float(batch_y.numel()), device=resolved_device)
            loss_value = raw_loss.sum() / denom
            loss_value.backward()
            optimizer.step()
            total_loss += float(raw_loss.sum().detach().cpu())
            total_weight += float(denom.detach().cpu())
        final_train_loss = total_loss / total_weight if total_weight else float("nan")

        if validation_loader is None:
            final_validation_loss = final_train_loss
        else:
            module.eval()
            total_loss = 0.0
            total_weight = 0.0
            with torch.no_grad():
                for batch in validation_loader:
                    batch_x = batch[0].to(resolved_device, non_blocking=True)
                    batch_y = batch[1].to(resolved_device, non_blocking=True)
                    raw_loss = criterion(module(batch_x), batch_y)
                    if len(batch) > 2:
                        batch_w = batch[2].to(resolved_device, non_blocking=True)
                        raw_loss = raw_loss * batch_w
                        denom = torch.clamp(batch_w.sum(), min=1.0)
                    else:
                        denom = torch.tensor(float(batch_y.numel()), device=resolved_device)
                    total_loss += float(raw_loss.sum().detach().cpu())
                    total_weight += float(denom.detach().cpu())
            final_validation_loss = total_loss / total_weight if total_weight else float("nan")

        epochs_trained = epoch
        if final_validation_loss < best_validation:
            best_validation = final_validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in module.state_dict().items()
            }
            patience_used = 0
        else:
            patience_used += 1
            if int(early_stopping_patience) >= 0 and patience_used > int(early_stopping_patience):
                break

    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()

    device_name = ""
    if str(resolved_device).startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.device(resolved_device))
    stats: dict[str, int | float | str | bool] = {
        "rows": len(x_values),
        "dates": int(train.loc[x_frame.index, "date"].nunique()),
        "symbols": int(train.loc[x_frame.index, "symbol"].nunique()),
        "features": len(features),
        "device": resolved_device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_name": device_name,
        "epochs_trained": int(epochs_trained),
        "best_epoch": int(best_epoch),
        "train_loss": float(final_train_loss),
        "validation_loss": float(best_validation),
        "validation_rows": int(validation_rows),
    }
    if sample_weight is not None:
        stats["sample_weight_mean"] = float(sample_weight.mean())
        stats["sample_weight_zero_rate"] = float((sample_weight <= 0.0).mean())
    return (
        TorchMLPPredictionModel(
            features=features,
            module=module,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            device=resolved_device,
            batch_size=int(predict_batch_size or batch_size),
            target_col=target_col,
        ),
        stats,
    )


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
    has_include_filter = bool(include_columns_set or include_prefixes or include_patterns)
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


def _torch_mlp_score(model: TorchMLPPredictionModel, frame: pd.DataFrame) -> np.ndarray:
    torch, _nn, _loader, _dataset = _import_torch()
    module = model.module.to(model.device)
    module.eval()
    scores = np.empty(len(frame), dtype="float64")
    batch_size = max(1, int(model.batch_size))
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            end = min(start + batch_size, len(frame))
            x_values, _mean, _scale = _standardized_float_matrix(
                frame.iloc[start:end],
                model.features,
                mean=model.feature_mean,
                scale=model.feature_scale,
            )
            batch_x = torch.from_numpy(x_values).to(model.device, non_blocking=True)
            batch_scores = module(batch_x).detach().cpu().numpy().reshape(-1)
            scores[start:end] = batch_scores.astype("float64")
    return scores


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
                    "mean_label": (float(frame[label_col].mean()) if len(frame) else float("nan")),
                    "win_rate": (
                        float((frame[label_col] > 0).mean()) if len(frame) else float("nan")
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
