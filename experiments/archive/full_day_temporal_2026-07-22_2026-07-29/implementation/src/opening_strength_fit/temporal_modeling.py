from __future__ import annotations

import copy
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.temporal_dataset import (
    aligned_sequence_validity,
    daily_rank_ic,
    load_sequence,
    prepare_day_inputs,
    sequence_mask_path,
    target_rank,
    target_values,
    top_n_excess,
    universe_mask,
)


def build_temporal_model(
    architecture: str,
    *,
    input_channels: int,
    sequence_length: int,
    hidden_width: int,
    dropout: float,
):
    import torch
    from torch import nn

    class FlatMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_channels * sequence_length, hidden_width),
                nn.LayerNorm(hidden_width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_width, hidden_width // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_width // 2, 1),
            )

        def forward(self, values, time_valid=None):
            return self.network(values).squeeze(-1)

    class LinearPath(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_channels * sequence_length, 1),
            )

        def forward(self, values, time_valid=None):
            return self.network(values).squeeze(-1)

    class ResidualTemporalBlock(nn.Module):
        def __init__(self, width: int, dilation: int) -> None:
            super().__init__()
            padding = 2 * dilation
            groups = 8 if width % 8 == 0 else 1
            self.network = nn.Sequential(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=5,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.GroupNorm(groups, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(width, width, kernel_size=1),
                nn.Dropout(dropout),
            )
            self.activation = nn.GELU()

        def forward(self, values):
            return self.activation(values + self.network(values))

    class TemporalConvNet(nn.Module):
        def __init__(self, *, attention: bool) -> None:
            super().__init__()
            self.attention = attention
            self.stem = nn.Sequential(
                nn.Conv1d(input_channels, hidden_width, kernel_size=5, padding=2),
                nn.GELU(),
            )
            self.blocks = nn.Sequential(
                *(ResidualTemporalBlock(hidden_width, dilation) for dilation in (1, 2, 4, 8))
            )
            self.attention_score = nn.Conv1d(hidden_width, 1, kernel_size=1) if attention else None
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_width),
                nn.Linear(hidden_width, hidden_width // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_width // 2, 1),
            )

        def forward(self, values, time_valid=None):
            hidden = self.blocks(self.stem(values))
            if time_valid is None:
                time_valid = torch.ones(
                    hidden.shape[0],
                    hidden.shape[2],
                    dtype=torch.bool,
                    device=hidden.device,
                )
            if self.attention_score is not None:
                logits = self.attention_score(hidden).squeeze(1)
                logits = logits.masked_fill(~time_valid, -1e4)
                weights = torch.softmax(logits, dim=1)
            else:
                weights = time_valid.to(hidden.dtype)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = torch.sum(hidden * weights.unsqueeze(1), dim=2)
            return self.head(pooled).squeeze(-1)

    normalized = str(architecture).strip().lower()
    if normalized == "linear":
        return LinearPath()
    if normalized == "mlp":
        return FlatMLP()
    if normalized == "tcn":
        return TemporalConvNet(attention=False)
    if normalized == "tcn_attention":
        return TemporalConvNet(attention=True)
    raise ValueError(
        f"unsupported temporal architecture={architecture!r}; "
        "expected linear, mlp, tcn, or tcn_attention"
    )


def _loss_function(
    name: str,
    *,
    head_fraction: float,
    huber_delta: float,
    device: str,
):
    import torch
    from torch import nn

    normalized = str(name).strip().lower()
    if normalized == "mse":
        return nn.MSELoss()
    if normalized == "huber":
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        return nn.HuberLoss(delta=float(huber_delta))
    if normalized == "head_bce":
        if not 0 < head_fraction < 0.5:
            raise ValueError("head_fraction must be in (0, 0.5) for head_bce")
        positive_weight = torch.tensor(
            (1.0 - head_fraction) / head_fraction,
            dtype=torch.float32,
            device=device,
        )
        return nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    raise ValueError(f"unsupported loss={name!r}; expected mse, huber, or head_bce")


def _training_target(
    ranked_target: np.ndarray,
    *,
    loss_name: str,
    head_fraction: float,
) -> np.ndarray:
    if str(loss_name).strip().lower() != "head_bce":
        return ranked_target
    threshold = 1.0 - 2.0 * float(head_fraction)
    output = np.full(len(ranked_target), np.nan, dtype=np.float32)
    eligible = np.isfinite(ranked_target)
    output[eligible] = (ranked_target[eligible] >= threshold).astype(np.float32)
    return output


def _batch_indices(indices: np.ndarray, batch_size: int) -> Sequence[np.ndarray]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _model_scores(model, inputs, time_valid, *, device: str, batch_size: int) -> np.ndarray:
    import torch

    output = np.full(len(inputs), np.nan, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for indices in _batch_indices(np.arange(len(inputs)), batch_size):
            x = torch.from_numpy(inputs[indices]).to(device=device, non_blocking=True)
            valid = torch.from_numpy(time_valid[indices]).to(
                device=device,
                non_blocking=True,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.startswith("cuda"),
            ):
                scores = model(x, valid)
            output[indices] = scores.float().cpu().numpy()
    return output


def _load_sequence_inputs(
    path: Path,
    *,
    value_mode: str,
    latest_clocks: Mapping[str, str],
    raw_scale: float,
    raw_scales: Mapping[str, float] | None,
    input_mask_sequence_root: Path | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    arrays = load_sequence(path)
    input_valid_override = None
    if input_mask_sequence_root is not None:
        mask_path = sequence_mask_path(path, input_mask_sequence_root)
        if not mask_path.exists():
            raise SystemExit(f"input mask sequence shard is missing: {mask_path}")
        input_valid_override = aligned_sequence_validity(arrays, load_sequence(mask_path))
    inputs, time_valid = prepare_day_inputs(
        arrays,
        value_mode=value_mode,
        latest_clocks=latest_clocks,
        raw_scale=raw_scale,
        raw_scales=raw_scales,
        input_valid_override=input_valid_override,
    )
    return arrays, inputs, time_valid


def _fit_target_winsor_bounds(
    paths: Sequence[Path],
    *,
    train_universe: str,
    target_mode: str,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float] | None:
    normalized = str(target_mode).strip().lower()
    if not normalized.endswith("_winsor"):
        return None
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("target winsor quantiles must satisfy 0 <= lower < upper <= 1")
    base_mode = normalized.removesuffix("_winsor")
    parts: list[np.ndarray] = []
    for path in paths:
        arrays = load_sequence(path)
        mask = universe_mask(arrays, train_universe)
        transformed = target_values(
            np.asarray(arrays["target"], dtype=np.float32),
            universe_mask=mask,
            mode=base_mode,
        )
        finite = transformed[np.isfinite(transformed)]
        if len(finite):
            parts.append(finite)
    if not parts:
        raise SystemExit("cannot estimate target winsor bounds from an empty training target")
    pooled = np.concatenate(parts).astype(np.float64, copy=False)
    lower, upper = np.quantile(pooled, [lower_quantile, upper_quantile])
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise SystemExit(f"invalid fitted target winsor bounds: {lower}, {upper}")
    return float(lower), float(upper)


def evaluate_temporal_model(
    model,
    paths: Sequence[Path],
    *,
    device: str,
    batch_size: int,
    value_mode: str,
    latest_clocks: Mapping[str, str],
    raw_scale: float,
    evaluation_universe: str,
    top_n: int,
    include_predictions: bool,
    raw_scales: Mapping[str, float] | None = None,
    target_mode: str = "rank",
    target_winsor_bounds: tuple[float, float] | None = None,
    input_mask_sequence_root: Path | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    losses: list[float] = []
    rank_ics: list[float] = []
    top_excesses: list[float] = []
    prediction_parts: list[pd.DataFrame] = []
    for path in paths:
        arrays, inputs, time_valid = _load_sequence_inputs(
            path,
            value_mode=value_mode,
            latest_clocks=latest_clocks,
            raw_scale=raw_scale,
            raw_scales=raw_scales,
            input_mask_sequence_root=input_mask_sequence_root,
        )
        scores = _model_scores(
            model,
            inputs,
            time_valid,
            device=device,
            batch_size=batch_size,
        )
        raw_target = np.asarray(arrays["target"], dtype=np.float32)
        eval_mask = universe_mask(arrays, evaluation_universe)
        ranked_target = target_rank(raw_target, universe_mask=eval_mask)
        model_target = target_values(
            raw_target,
            universe_mask=eval_mask,
            mode=target_mode,
            winsor_bounds=target_winsor_bounds,
        )
        eligible = eval_mask & np.isfinite(model_target)
        if eligible.any():
            losses.append(float(np.mean((scores[eligible] - model_target[eligible]) ** 2)))
        rank_ics.append(daily_rank_ic(scores, raw_target, eligible))
        top_excesses.append(top_n_excess(scores, raw_target, eligible, top_n=top_n))
        if include_predictions:
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "date": path.parent.name.removeprefix("date="),
                        "symbol": arrays["symbols"].astype(str),
                        "score": scores,
                        "target": raw_target,
                        "target_rank": ranked_target,
                        "model_target": model_target,
                        "evaluation_eligible": eligible.astype(np.int8),
                        "stock_pool_member": arrays["pool_member"].astype(np.int8),
                    }
                )
            )
    metrics = {
        "loss": float(np.nanmean(losses)) if losses else float("nan"),
        "daily_rank_ic": float(np.nanmean(rank_ics)) if rank_ics else float("nan"),
        "daily_rank_ic_std": float(np.nanstd(rank_ics, ddof=1))
        if len(rank_ics) > 1
        else float("nan"),
        "top_n_excess": float(np.nanmean(top_excesses)) if top_excesses else float("nan"),
        "days": float(len(paths)),
    }
    predictions = (
        pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    )
    return metrics, predictions


def fit_temporal_model(
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    *,
    architecture: str,
    device: str,
    value_mode: str,
    latest_clocks: Mapping[str, str],
    raw_scale: float,
    train_universe: str,
    evaluation_universe: str,
    hidden_width: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    head_fraction: float,
    selection_metric: str,
    patience: int,
    top_n: int,
    seed: int,
    raw_scales: Mapping[str, float] | None = None,
    huber_delta: float = 0.25,
    target_mode: str = "rank",
    target_winsor_lower_quantile: float = 0.01,
    target_winsor_upper_quantile: float = 0.99,
    input_mask_sequence_root: Path | None = None,
) -> tuple[object, pd.DataFrame, dict[str, object]]:
    import torch

    if not train_paths:
        raise SystemExit("temporal model has no training sequence shards")
    if not validation_paths:
        raise SystemExit("temporal model has no validation sequence shards")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    _, sample_inputs, _ = _load_sequence_inputs(
        train_paths[0],
        value_mode=value_mode,
        latest_clocks=latest_clocks,
        raw_scale=raw_scale,
        raw_scales=raw_scales,
        input_mask_sequence_root=input_mask_sequence_root,
    )
    model = build_temporal_model(
        architecture,
        input_channels=sample_inputs.shape[1],
        sequence_length=sample_inputs.shape[2],
        hidden_width=hidden_width,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = _loss_function(
        loss_name,
        head_fraction=head_fraction,
        huber_delta=huber_delta,
        device=device,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    history: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    normalized_selection_metric = str(selection_metric).strip().lower()
    if normalized_selection_metric not in {"loss", "daily_rank_ic", "top_n_excess"}:
        raise ValueError("selection_metric must be loss, daily_rank_ic, or top_n_excess")
    target_winsor_bounds = _fit_target_winsor_bounds(
        train_paths,
        train_universe=train_universe,
        target_mode=target_mode,
        lower_quantile=target_winsor_lower_quantile,
        upper_quantile=target_winsor_upper_quantile,
    )
    minimize_selection = normalized_selection_metric == "loss"
    best_value = np.inf if minimize_selection else -np.inf
    stale_epochs = 0
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        path_order = list(train_paths)
        random.Random(seed + epoch).shuffle(path_order)
        epoch_loss_sum = 0.0
        epoch_samples = 0
        for path in path_order:
            arrays, inputs, time_valid = _load_sequence_inputs(
                path,
                value_mode=value_mode,
                latest_clocks=latest_clocks,
                raw_scale=raw_scale,
                raw_scales=raw_scales,
                input_mask_sequence_root=input_mask_sequence_root,
            )
            train_mask = universe_mask(arrays, train_universe)
            transformed_target = target_values(
                np.asarray(arrays["target"], dtype=np.float32),
                universe_mask=train_mask,
                mode=target_mode,
                winsor_bounds=target_winsor_bounds,
            )
            training_target = _training_target(
                transformed_target,
                loss_name=loss_name,
                head_fraction=head_fraction,
            )
            indices = np.flatnonzero(
                train_mask & np.isfinite(training_target) & time_valid.any(axis=1)
            )
            np.random.default_rng(seed + epoch).shuffle(indices)
            for batch_indices in _batch_indices(indices, batch_size):
                x = torch.from_numpy(inputs[batch_indices]).to(
                    device=device,
                    non_blocking=True,
                )
                valid = torch.from_numpy(time_valid[batch_indices]).to(
                    device=device,
                    non_blocking=True,
                )
                y = torch.from_numpy(training_target[batch_indices]).to(
                    device=device,
                    non_blocking=True,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=device.startswith("cuda"),
                ):
                    predictions = model(x, valid)
                    loss = criterion(predictions, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss_sum += float(loss.detach().cpu()) * len(batch_indices)
                epoch_samples += len(batch_indices)

        validation, _ = evaluate_temporal_model(
            model,
            validation_paths,
            device=device,
            batch_size=batch_size,
            value_mode=value_mode,
            latest_clocks=latest_clocks,
            raw_scale=raw_scale,
            raw_scales=raw_scales,
            evaluation_universe=evaluation_universe,
            top_n=top_n,
            include_predictions=False,
            target_mode=target_mode,
            target_winsor_bounds=target_winsor_bounds,
            input_mask_sequence_root=input_mask_sequence_root,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": epoch_loss_sum / max(1, epoch_samples),
            "train_samples": epoch_samples,
            "validation_loss": validation["loss"],
            "validation_daily_rank_ic": validation["daily_rank_ic"],
            "validation_top_n_excess": validation["top_n_excess"],
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={record['train_loss']:.6f} "
            f"validation_ic={record['validation_daily_rank_ic']:.6f} "
            f"validation_top_excess_bps="
            f"{float(record['validation_top_n_excess']) * 10000:.3f}",
            flush=True,
        )
        selection_value = float(validation[normalized_selection_metric])
        improved = (
            selection_value < best_value if minimize_selection else selection_value > best_value
        )
        if np.isfinite(selection_value) and improved:
            best_value = selection_value
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return (
        model,
        pd.DataFrame(history),
        {
            "selection_metric": normalized_selection_metric,
            "best_validation_selection_value": float(best_value),
            "target_mode": str(target_mode).strip().lower(),
            "target_winsor_bounds": list(target_winsor_bounds)
            if target_winsor_bounds is not None
            else None,
            "epochs_completed": len(history),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
    )
