from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.model_features import _clean_xy, feature_columns
from opening_strength_fit.model_types import TorchMLPPredictionModel


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
