"""Torch MLP fitting, loss and device selection, and training diagnostics."""

from __future__ import annotations

import time

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


def _normalize_training_tensor_storage(value: str) -> str:
    name = str(value or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "cuda": "cuda_resident",
        "cuda_resident": "cuda_resident",
        "gpu": "cuda_resident",
        "gpu_resident": "cuda_resident",
        "host": "host_vectorized",
        "host_vectorized": "host_vectorized",
        "cpu": "host_vectorized",
    }
    if name not in aliases:
        raise SystemExit(
            "model.training_tensor_storage for torch_mlp must be auto, "
            "cuda_resident, or host_vectorized"
        )
    return aliases[name]


def _resolve_training_tensor_storage(
    requested: str,
    *,
    device: str,
    required_bytes: int,
    free_bytes: int,
    reserve_bytes: int,
) -> str:
    """Resolve the bulk tensor location without importing Torch in unit tests."""
    mode = _normalize_training_tensor_storage(requested)
    cuda_device = str(device).startswith("cuda")
    if not cuda_device:
        if mode == "cuda_resident":
            raise SystemExit("model.training_tensor_storage='cuda_resident' requires a CUDA device")
        return "host_vectorized"
    if mode == "host_vectorized":
        return mode

    fits = int(free_bytes) - int(required_bytes) >= int(reserve_bytes)
    if fits:
        return "cuda_resident"
    if mode == "auto":
        return "host_vectorized"

    gib = float(1024**3)
    raise SystemExit(
        "model.training_tensor_storage='cuda_resident' does not fit: "
        f"requires {required_bytes / gib:.2f} GiB of training tensors with "
        f"{reserve_bytes / gib:.2f} GiB reserved, but only {free_bytes / gib:.2f} GiB "
        "is currently free"
    )


def _consume_torch_loader_seed(torch) -> int:
    """Match the global CPU RNG draw made when a DataLoader iterator is created."""
    return int(torch.empty((), dtype=torch.int64).random_().item())


def _torch_random_sampler_order(torch, length: int):
    """Match DataLoader(RandomSampler) order while keeping indices as one tensor."""
    _consume_torch_loader_seed(torch)
    sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    generator.manual_seed(sampler_seed)
    return torch.randperm(int(length), generator=generator)


def _gate_diagnostic_sample(
    x_values: np.ndarray,
    *,
    rng: np.random.Generator,
    max_rows: int,
) -> np.ndarray:
    max_rows = max(0, int(max_rows))
    if max_rows <= 0 or len(x_values) == 0:
        return x_values[:0]
    if len(x_values) <= max_rows:
        return x_values
    selected = rng.choice(len(x_values), size=max_rows, replace=False)
    selected.sort()
    return np.ascontiguousarray(x_values[selected])


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
    training_tensor_storage: str = "auto",
    cuda_resident_reserve_gib: float = 8.0,
    gate_diagnostics_max_rows: int = 200_000,
    feature_standardization: str = "global_zscore",
    feature_standardization_group_col: str = "symbol",
    feature_value_transform: str = "none",
    feature_value_transform_group_cols: tuple[str, ...] = ("date", "decision_target_timestamp"),
    feature_value_transform_rank_method: str = "average",
    feature_value_transform_tick_size: float = 0.01,
) -> tuple[TorchMLPPredictionModel, dict[str, object]]:
    fit_started = time.monotonic()
    torch, _nn, _DataLoader, _TensorDataset = _import_torch()
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
    standardization_group_keys = None
    standardization_group_mean = None
    standardization_group_scale = None
    if standardization_mode == "symbol_train_zscore":
        x_frame, y_series = _clean_xy(model_input, features, target_col=target_col)
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
        selected_rows: pd.Index | pd.Series = x_frame.index
    else:
        target = pd.to_numeric(model_input[target_col], errors="coerce")
        valid_rows = target.notna() & np.isfinite(target)
        if "valid_label" in model_input.columns:
            valid_rows &= model_input["valid_label"].fillna(False).astype(bool)
        if not bool(valid_rows.any()):
            raise SystemExit("empty labeled frame after filtering valid labels")
        x_values, feature_mean, feature_scale = _standardized_float_matrix(
            model_input,
            features,
            row_mask=valid_rows,
        )
        y_series = target.loc[valid_rows].astype("float64")
        selected_rows = valid_rows
    y_values = y_series.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

    sample_weight = None
    if sample_weight_col:
        if sample_weight_col not in train.columns:
            raise SystemExit(f"missing sample weight column: {sample_weight_col}")
        sample_weight = (
            pd.to_numeric(train.loc[selected_rows, sample_weight_col], errors="coerce")
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

    n_rows = len(x_values)
    validation_rows = int(n_rows * max(0.0, float(validation_fraction)))
    if validation_max_rows > 0:
        validation_rows = min(validation_rows, int(validation_max_rows))
    validation_rows = max(0, min(validation_rows, n_rows - 1))
    if validation_rows > 0:
        validation_indices = rng.choice(n_rows, size=validation_rows, replace=False)
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[validation_indices] = False
        train_indices = np.flatnonzero(train_mask)
    else:
        validation_indices = np.empty(0, dtype=np.int64)
        train_indices = np.arange(n_rows, dtype=np.int64)

    diagnostic_values = _gate_diagnostic_sample(
        x_values,
        rng=rng,
        max_rows=int(gate_diagnostics_max_rows),
    )
    date_count = int(train.loc[selected_rows, "date"].nunique())
    symbol_count = int(train.loc[selected_rows, "symbol"].nunique())
    sample_weight_mean = None
    sample_weight_zero_rate = None
    if sample_weight is not None:
        sample_weight_mean = float(sample_weight.mean())
        sample_weight_zero_rate = float((sample_weight <= 0.0).mean())

    # Include the two int64 index vectors used for the split and epoch order. The
    # configured reserve covers model activations, optimizer state, CUDA context,
    # and temporary batch tensors.
    required_tensor_bytes = int(
        x_values.nbytes
        + y_values.nbytes
        + (sample_weight.nbytes if sample_weight is not None else 0)
        + (n_rows + len(train_indices)) * np.dtype(np.int64).itemsize
    )
    reserve_bytes = max(0, int(float(cuda_resident_reserve_gib) * 1024**3))
    cuda_free_bytes = 0
    cuda_total_bytes = 0
    if str(resolved_device).startswith("cuda"):
        cuda_free_bytes, cuda_total_bytes = (
            int(value) for value in torch.cuda.mem_get_info(torch.device(resolved_device))
        )
    resolved_storage = _resolve_training_tensor_storage(
        training_tensor_storage,
        device=resolved_device,
        required_bytes=required_tensor_bytes,
        free_bytes=cuda_free_bytes,
        reserve_bytes=reserve_bytes,
    )
    cuda_resident = resolved_storage == "cuda_resident"
    gib = float(1024**3)
    print(
        "\ntorch_training_storage:"
        f"\n  rows: {n_rows}"
        f"\n  features: {len(features)}"
        f"\n  train_rows: {len(train_indices)}"
        f"\n  validation_rows: {validation_rows}"
        f"\n  device: {resolved_device}"
        f"\n  storage: {resolved_storage}"
        f"\n  tensor_gib: {required_tensor_bytes / gib:.2f}"
        f"\n  cuda_free_gib: {cuda_free_bytes / gib:.2f}"
        f"\n  cuda_total_gib: {cuda_total_bytes / gib:.2f}"
        f"\n  reserve_gib: {reserve_bytes / gib:.2f}"
        f"\n  preparation_seconds: {time.monotonic() - fit_started:.1f}"
    )

    storage_started = time.monotonic()
    x_tensor = torch.from_numpy(x_values)
    y_tensor = torch.from_numpy(y_values)
    weight_tensor = torch.from_numpy(sample_weight) if sample_weight is not None else None
    train_indices_tensor = torch.from_numpy(train_indices)
    validation_indices_tensor = torch.from_numpy(validation_indices)
    if cuda_resident:
        x_tensor = x_tensor.to(resolved_device)
        y_tensor = y_tensor.to(resolved_device)
        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(resolved_device)
        train_indices_tensor = train_indices_tensor.to(resolved_device)
        validation_indices_tensor = validation_indices_tensor.to(resolved_device)
        del x_values, y_values, train_indices, validation_indices
        if sample_weight is not None:
            del sample_weight
    storage_seconds = time.monotonic() - storage_started
    print(f"  storage_transfer_seconds: {storage_seconds:.1f}")

    def take_batch(tensor, indices):
        values = tensor.index_select(0, indices)
        if not cuda_resident and str(resolved_device) != "cpu":
            values = values.to(resolved_device)
        return values

    def batch_loss_tensors(indices):
        batch_x = take_batch(x_tensor, indices)
        batch_y = take_batch(y_tensor, indices)
        batch_w = take_batch(weight_tensor, indices) if weight_tensor is not None else None
        return batch_x, batch_y, batch_w

    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    patience_used = 0
    epochs_trained = 0
    final_train_loss = float("nan")
    final_validation_loss = float("nan")
    epoch_seconds: list[float] = []
    training_started = time.monotonic()
    max_epochs = int(max_epochs)
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.monotonic()
        module.train()
        total_loss_tensor = torch.zeros((), dtype=torch.float64, device=resolved_device)
        total_weight_tensor = torch.zeros((), dtype=torch.float64, device=resolved_device)
        epoch_order = _torch_random_sampler_order(torch, len(train_indices_tensor))
        if cuda_resident:
            epoch_order = epoch_order.to(resolved_device)
        for start in range(0, len(epoch_order), int(batch_size)):
            positions = epoch_order[start : start + int(batch_size)]
            indices = train_indices_tensor.index_select(0, positions)
            optimizer.zero_grad(set_to_none=True)
            batch_x, batch_y, batch_w = batch_loss_tensors(indices)
            raw_loss = criterion(module(batch_x), batch_y)
            if batch_w is not None:
                raw_loss = raw_loss * batch_w
                denom = torch.clamp(batch_w.sum(), min=1.0)
            else:
                denom = raw_loss.new_tensor(float(batch_y.numel()))
            raw_loss_sum = raw_loss.sum()
            loss_value = raw_loss_sum / denom
            loss_value.backward()
            optimizer.step()
            total_loss_tensor += raw_loss_sum.detach().to(dtype=torch.float64)
            total_weight_tensor += denom.detach().to(dtype=torch.float64)
        train_totals = torch.stack((total_loss_tensor, total_weight_tensor)).cpu().tolist()
        final_train_loss = (
            float(train_totals[0] / train_totals[1]) if train_totals[1] else float("nan")
        )

        if validation_rows <= 0:
            final_validation_loss = final_train_loss
        else:
            _consume_torch_loader_seed(torch)
            module.eval()
            total_loss_tensor = torch.zeros((), dtype=torch.float64, device=resolved_device)
            total_weight_tensor = torch.zeros((), dtype=torch.float64, device=resolved_device)
            with torch.no_grad():
                for start in range(0, validation_rows, int(batch_size)):
                    indices = validation_indices_tensor[start : start + int(batch_size)]
                    batch_x, batch_y, batch_w = batch_loss_tensors(indices)
                    raw_loss = criterion(module(batch_x), batch_y)
                    if batch_w is not None:
                        raw_loss = raw_loss * batch_w
                        denom = torch.clamp(batch_w.sum(), min=1.0)
                    else:
                        denom = raw_loss.new_tensor(float(batch_y.numel()))
                    total_loss_tensor += raw_loss.sum().to(dtype=torch.float64)
                    total_weight_tensor += denom.to(dtype=torch.float64)
            validation_totals = torch.stack((total_loss_tensor, total_weight_tensor)).cpu().tolist()
            final_validation_loss = (
                float(validation_totals[0] / validation_totals[1])
                if validation_totals[1]
                else float("nan")
            )

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
        epoch_seconds.append(time.monotonic() - epoch_started)
        print(
            "torch_epoch:"
            f" epoch={epoch}/{max_epochs}"
            f" train_loss={final_train_loss:.9g}"
            f" validation_loss={final_validation_loss:.9g}"
            f" best_epoch={best_epoch}"
            f" seconds={epoch_seconds[-1]:.1f}"
        )
        if patience_used:
            print(f"  early_stopping_patience_used: {patience_used}")
        if patience_used:
            if int(early_stopping_patience) >= 0 and patience_used > int(early_stopping_patience):
                break

    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()

    device_name = ""
    if str(resolved_device).startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.device(resolved_device))
    stats: dict[str, object] = {
        "rows": n_rows,
        "dates": date_count,
        "symbols": symbol_count,
        "features": len(features),
        "device": resolved_device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_name": device_name,
        "epochs_trained": int(epochs_trained),
        "best_epoch": int(best_epoch),
        "train_loss": float(final_train_loss),
        "validation_loss": float(best_validation),
        "validation_rows": int(validation_rows),
        "training_tensor_storage": resolved_storage,
        "training_tensor_required_bytes": required_tensor_bytes,
        "cuda_memory_free_bytes_before_storage": cuda_free_bytes,
        "cuda_memory_total_bytes": cuda_total_bytes,
        "cuda_resident_reserve_bytes": reserve_bytes,
        "vectorized_index_batches": True,
        "num_workers_ignored": int(num_workers),
        "training_preparation_seconds": float(training_started - fit_started),
        "training_storage_transfer_seconds": float(storage_seconds),
        "training_epoch_seconds": epoch_seconds,
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
    if sample_weight_mean is not None:
        stats["sample_weight_mean"] = sample_weight_mean
        stats["sample_weight_zero_rate"] = sample_weight_zero_rate
    diagnostics = _torch_gate_diagnostics(
        module,
        diagnostic_values,
        device=resolved_device,
        torch=torch,
        rng=rng,
        max_rows=len(diagnostic_values),
        batch_size=min(int(predict_batch_size or batch_size), 32768),
    )
    x_tensor = y_tensor = train_indices_tensor = validation_indices_tensor = None
    weight_tensor = None
    if str(resolved_device).startswith("cuda"):
        torch.cuda.empty_cache()
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
