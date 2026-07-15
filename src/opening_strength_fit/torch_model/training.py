"""Torch MLP fitting, loss and device selection, and training diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.model_features import _clean_xy, feature_columns
from opening_strength_fit.model_types import TorchMLPPredictionModel
from opening_strength_fit.torch_model.architectures import _import_torch, _TorchMLPModule
from opening_strength_fit.torch_model.preprocessing import (
    _fit_symbol_train_standardization,
    _normalize_feature_standardization,
    _normalize_feature_value_transform,
    _standardized_float_matrix,
    _torch_feature_value_frame,
)


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


def _torch_gate_diagnostics(
    module,
    x_values: np.ndarray,
    *,
    device: str,
    torch,
    rng: np.random.Generator,
    max_rows: int,
    batch_size: int,
) -> dict[str, object]:
    if not hasattr(module, "gate_values"):
        return {}
    max_rows = max(0, int(max_rows))
    if max_rows <= 0 or len(x_values) == 0:
        return {}
    if len(x_values) > max_rows:
        selected = rng.choice(len(x_values), size=max_rows, replace=False)
        selected.sort()
        values = x_values[selected]
    else:
        values = x_values

    group_names = tuple(str(name) for name in getattr(module, "group_names", ()))
    group_dims = tuple(int(dim) for dim in getattr(module, "group_dims", ()))
    group_feature_counts = tuple(
        int(count) for count in getattr(module, "group_feature_counts", ())
    )
    if not group_names:
        return {}

    count = 0
    sums = np.zeros(len(group_names), dtype="float64")
    sums_sq = np.zeros(len(group_names), dtype="float64")
    mins = np.full(len(group_names), np.inf, dtype="float64")
    maxs = np.full(len(group_names), -np.inf, dtype="float64")
    module.eval()
    gate_batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(values), gate_batch_size):
            end = min(start + gate_batch_size, len(values))
            batch_x = torch.from_numpy(values[start:end]).to(device, non_blocking=True)
            gates = module.gate_values(batch_x).detach().cpu().numpy().astype("float64")
            count += len(gates)
            sums += gates.sum(axis=0)
            sums_sq += np.square(gates).sum(axis=0)
            mins = np.minimum(mins, gates.min(axis=0))
            maxs = np.maximum(maxs, gates.max(axis=0))

    means = sums / float(count)
    variances = np.maximum(sums_sq / float(count) - np.square(means), 0.0)
    stds = np.sqrt(variances)
    groups = []
    for index, name in enumerate(group_names):
        groups.append(
            {
                "name": name,
                "features": (
                    group_feature_counts[index] if index < len(group_feature_counts) else None
                ),
                "embedding_dim": group_dims[index] if index < len(group_dims) else None,
                "gate_mean": float(means[index]),
                "gate_std": float(stds[index]),
                "gate_min": float(mins[index]),
                "gate_max": float(maxs[index]),
            }
        )
    return {
        "gate_diagnostics_rows": int(count),
        "gate_groups": groups,
    }


def fit_torch_mlp_frame(
    train: pd.DataFrame,
    *,
    feature_limit: int | None = None,
    target_col: str = "label",
    sample_weight_col: str = "",
    feature_filters: dict[str, tuple[str, ...]] | None = None,
    hidden_layers: tuple[int, ...] = (512, 256, 128),
    architecture: str = "mlp",
    group_embedding_dim: int = 48,
    group_embedding_dims: dict[str, int] | None = None,
    fusion_dim: int = 256,
    block_hidden_dim: int = 512,
    num_blocks: int = 2,
    transformer_heads: int = 4,
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
    gate_diagnostics_max_rows: int = 200_000,
    feature_standardization: str = "global_zscore",
    feature_standardization_group_col: str = "symbol",
    feature_value_transform: str = "none",
    feature_value_transform_group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    feature_value_transform_rank_method: str = "average",
    feature_value_transform_tick_size: float = 0.01,
) -> tuple[TorchMLPPredictionModel, dict[str, object]]:
    torch, _nn, DataLoader, TensorDataset = _import_torch()
    features = feature_columns(train, feature_limit, **(feature_filters or {}))
    if sample_weight_col:
        features = [column for column in features if column != sample_weight_col]
    if not features:
        raise SystemExit("no numeric feature columns found")

    standardization_mode = _normalize_feature_standardization(feature_standardization)
    standardization_group_col = str(feature_standardization_group_col or "symbol")
    value_transform_mode = _normalize_feature_value_transform(feature_value_transform)
    value_transform_group_cols = tuple(
        str(column)
        for column in (feature_value_transform_group_cols or ("date", "decision_target_timestamp"))
        if str(column).strip()
    )
    value_transform_rank_method = str(feature_value_transform_rank_method or "average")
    value_transform_tick_size = float(feature_value_transform_tick_size)
    model_input = _torch_feature_value_frame(
        train,
        features,
        feature_value_transform=value_transform_mode,
        group_cols=value_transform_group_cols,
        rank_method=value_transform_rank_method,
        tick_size=value_transform_tick_size,
        extra_columns=(target_col, "valid_label", standardization_group_col),
    )
    x_frame, y_series = _clean_xy(model_input, features, target_col=target_col)
    standardization_group_keys = None
    standardization_group_mean = None
    standardization_group_scale = None
    if standardization_mode == "symbol_train_zscore":
        if standardization_group_col not in model_input.columns:
            raise SystemExit(
                "model.feature_standardization='symbol_train_zscore' requires "
                f"{standardization_group_col!r}"
            )
        standardization_frame = model_input.loc[
            x_frame.index,
            [standardization_group_col, *features],
        ]
        (
            feature_mean,
            feature_scale,
            standardization_group_keys,
            standardization_group_mean,
            standardization_group_scale,
        ) = _fit_symbol_train_standardization(
            standardization_frame,
            features,
            group_col=standardization_group_col,
        )
        x_values, feature_mean, feature_scale = _standardized_float_matrix(
            standardization_frame,
            features,
            mean=feature_mean,
            scale=feature_scale,
            group_col=standardization_group_col,
            group_keys=standardization_group_keys,
            group_mean=standardization_group_mean,
            group_scale=standardization_group_scale,
        )
    else:
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
        feature_names=tuple(features),
        group_embedding_dim=int(group_embedding_dim),
        group_embedding_dims=group_embedding_dims or {},
        fusion_dim=int(fusion_dim),
        block_hidden_dim=int(block_hidden_dim),
        num_blocks=int(num_blocks),
        transformer_heads=int(transformer_heads),
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
    stats: dict[str, object] = {
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
        "feature_standardization": standardization_mode,
        "standardization_group_col": standardization_group_col
        if standardization_mode == "symbol_train_zscore"
        else "",
        "standardization_groups": int(len(standardization_group_keys))
        if standardization_group_keys is not None
        else 0,
        "feature_value_transform": value_transform_mode,
        "feature_value_transform_group_cols": list(value_transform_group_cols)
        if value_transform_mode != "none"
        else [],
        "feature_value_transform_rank_method": value_transform_rank_method
        if value_transform_mode != "none"
        else "",
        "feature_value_transform_tick_size": value_transform_tick_size
        if value_transform_mode.startswith("mechanismized_")
        else "",
    }
    if sample_weight is not None:
        stats["sample_weight_mean"] = float(sample_weight.mean())
        stats["sample_weight_zero_rate"] = float((sample_weight <= 0.0).mean())
    diagnostics = _torch_gate_diagnostics(
        module,
        x_values,
        device=resolved_device,
        torch=torch,
        rng=rng,
        max_rows=int(gate_diagnostics_max_rows),
        batch_size=min(int(predict_batch_size or batch_size), 32768),
    )
    stats.update(diagnostics)
    return (
        TorchMLPPredictionModel(
            features=features,
            module=module,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            feature_standardization=standardization_mode,
            standardization_group_col=standardization_group_col,
            standardization_group_keys=standardization_group_keys,
            standardization_group_mean=standardization_group_mean,
            standardization_group_scale=standardization_group_scale,
            feature_value_transform=value_transform_mode,
            feature_value_transform_group_cols=value_transform_group_cols,
            feature_value_transform_rank_method=value_transform_rank_method,
            feature_value_transform_tick_size=value_transform_tick_size,
            device=resolved_device,
            batch_size=int(predict_batch_size or batch_size),
            diagnostics=diagnostics or None,
            target_col=target_col,
        ),
        stats,
    )
