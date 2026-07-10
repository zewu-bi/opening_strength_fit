from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.features import (
    mechanismized_feature_value_reference_columns,
    transform_cross_sectional_feature_values,
    transform_mechanismized_feature_values,
    transform_mechanismized_v2_feature_values,
    transform_mechanismized_v3_feature_values,
)
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
        feature_names: tuple[str, ...] = (),
        group_embedding_dim: int = 48,
        group_embedding_dims: dict[str, int] | None = None,
        fusion_dim: int = 256,
        block_hidden_dim: int = 512,
        num_blocks: int = 2,
        transformer_heads: int = 4,
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
        grouped_feature_names = feature_names or tuple(
            f"feature_{index}" for index in range(input_dim)
        )
        if architecture_name in {
            "grouped_residual",
            "grouped_residual_gelu",
            "grouped_residual_mlp",
        }:
            return _GroupedResidualMLP(
                feature_names=grouped_feature_names,
                group_embedding_dim=group_embedding_dim,
                fusion_dim=fusion_dim,
                block_hidden_dim=block_hidden_dim,
                num_blocks=num_blocks,
                dropout=float(dropout),
                activation=activation_factories[activation_name],
                torch=_torch,
                nn=nn,
            )
        if architecture_name in {"grouped_gated", "grouped_gated_mlp"}:
            return _GroupedGatedMLP(
                feature_names=grouped_feature_names,
                group_embedding_dim=group_embedding_dim,
                fusion_dim=fusion_dim,
                dropout=float(dropout),
                activation=activation_factories[activation_name],
                torch=_torch,
                nn=nn,
            )
        if architecture_name in {"grouped_gated_v2", "mechanism_grouped_gated"}:
            return _GroupedGatedV2MLP(
                feature_names=grouped_feature_names,
                group_embedding_dim=group_embedding_dim,
                group_embedding_dims=group_embedding_dims or {},
                fusion_dim=fusion_dim,
                dropout=float(dropout),
                activation=activation_factories[activation_name],
                torch=_torch,
                nn=nn,
            )
        if architecture_name in {"grouped_cross", "grouped_cross_mlp"}:
            return _GroupedCrossMLP(
                feature_names=grouped_feature_names,
                group_embedding_dim=group_embedding_dim,
                fusion_dim=fusion_dim,
                block_hidden_dim=block_hidden_dim,
                num_blocks=num_blocks,
                dropout=float(dropout),
                activation=activation_factories[activation_name],
                torch=_torch,
                nn=nn,
            )
        if architecture_name in {
            "group_token_transformer",
            "grouped_transformer",
            "group_transformer",
        }:
            return _GroupTokenTransformerMLP(
                feature_names=grouped_feature_names,
                group_embedding_dim=group_embedding_dim,
                fusion_dim=fusion_dim,
                block_hidden_dim=block_hidden_dim,
                num_blocks=num_blocks,
                transformer_heads=transformer_heads,
                dropout=float(dropout),
                activation_name=activation_name,
                activation=activation_factories[activation_name],
                torch=_torch,
                nn=nn,
            )
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
            raise SystemExit(
                "model.architecture for torch_mlp must be mlp, wide_deep_residual, "
                "grouped_residual, grouped_gated, grouped_gated_v2, grouped_cross, "
                "or group_token_transformer"
            )
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


def _positive_int(value: int, name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise SystemExit(f"model.{name} for torch_mlp must be a positive int")
    return resolved


def _feature_group_name(feature: str) -> str:
    name = feature.lower()
    if name.startswith(("hist_surprise_", "path_shape_")):
        return "hist_path"
    if name.startswith("preopen_") or name in {"exch_time_offset_us", "ask1_to_limit_up_bps"}:
        return "preopen"
    if (
        name.startswith(
            (
                "volume_diff_",
                "turnover_diff_",
                "trade_vwap_",
                "postopen_v2_trade_turnover_to_depth_notional_",
                "postopen_v2_trade_volume_to_ask_depth10_",
                "postopen_v2_trade_volume_to_bid_depth10_",
            )
        )
        or name == "trade_num"
    ):
        return "activity"
    if (
        "spread" in name
        or "depth" in name
        or "imbalance" in name
        or "gap_curve" in name
        or "gap_max" in name
        or "queue_replenish" in name
        or name.startswith(
            (
                "ask_volume_",
                "bid_volume_",
                "ask_count_",
                "bid_count_",
                "ask_gap_",
                "bid_gap_",
            )
        )
        or name in {"total_ask_volume", "total_bid_volume", "total_ask_count", "total_bid_count"}
    ):
        return "liquidity"
    if (
        name.startswith(("return_", "postopen_v2_mid_price_from_open_"))
        or "price" in name
        or "vwap_vs" in name
        or name in {"mid_price", "ask_price_1", "bid_price_1", "avg_ask_price", "avg_bid_price"}
    ):
        return "price_momentum"
    return "other"


def _feature_group_name_v2(feature: str) -> str:
    name = feature.lower()
    if name.startswith("hist_surprise_"):
        return "historical_surprise"
    if name.startswith("path_shape_"):
        return "path_shape_confirmation"
    if name.startswith("preopen_") or name == "exch_time_offset_us":
        return "preopen_auction"
    if name in {
        "ask1_to_limit_up_bps",
        "ask_price_1",
        "bid_price_1",
        "mid_price",
        "avg_ask_price",
        "avg_bid_price",
    }:
        return "limit_price_state"
    if (
        name.startswith(("trade_vwap_", "postopen_v2_trade_vwap_vs_"))
        or name.startswith("postopen_v2_trade_turnover_to_depth_notional_")
        or name.startswith("postopen_v2_trade_volume_to_ask_depth10_")
        or name.startswith("postopen_v2_trade_volume_to_bid_depth10_")
    ):
        return "trade_price_impact"
    if (
        name.startswith(("volume_diff_", "turnover_diff_", "postopen_volume_"))
        or name == "trade_num"
    ):
        return "trade_activity"
    if (
        name.startswith(("return_", "postopen_mid_price_"))
        or name.startswith("postopen_ask_price_")
        or name.startswith("postopen_bid_price_")
        or name.startswith("postopen_v2_mid_price_from_open_")
        or name.startswith("postopen_v2_ask_price_1_from_open_")
        or name.startswith("postopen_v2_bid_price_1_from_open_")
        or name in {"return_vs_open", "return_vs_prev_close"}
    ):
        return "postopen_price_path"
    if (
        name.startswith("depth_imbalance_")
        or name.startswith("postopen_v2_depth_imbalance_")
        or "queue_replenish" in name
    ):
        return "book_imbalance_pressure"
    if (
        "spread" in name
        or name.startswith(("ask_gap_", "bid_gap_"))
        or "gap_curve" in name
        or "gap_max" in name
        or "depth_concentration" in name
        or "ask1_share_depth" in name
        or "bid1_share_depth" in name
    ):
        return "book_shape_spread_gap"
    if (
        name.startswith(("postopen_ask_volume_", "postopen_bid_volume_"))
        or name.startswith("postopen_v2_ask_depth_")
        or name.startswith("postopen_v2_bid_depth_")
    ):
        return "postopen_liquidity_change"
    if (
        name.startswith(
            (
                "ask_volume_",
                "bid_volume_",
                "ask_count_",
                "bid_count_",
                "ask_depth_",
                "bid_depth_",
            )
        )
        or name in {"total_ask_volume", "total_bid_volume", "total_ask_count", "total_bid_count"}
    ):
        return "book_depth_level"
    return "other"


def _feature_group_indices(feature_names: tuple[str, ...]) -> list[tuple[str, list[int]]]:
    ordered_groups = [
        "activity",
        "hist_path",
        "liquidity",
        "price_momentum",
        "preopen",
        "other",
    ]
    groups = {name: [] for name in ordered_groups}
    for index, feature in enumerate(feature_names):
        groups.setdefault(_feature_group_name(feature), []).append(index)
    return [(name, groups[name]) for name in ordered_groups if groups.get(name)]


def _feature_group_indices_v2(feature_names: tuple[str, ...]) -> list[tuple[str, list[int]]]:
    ordered_groups = [
        "preopen_auction",
        "limit_price_state",
        "book_depth_level",
        "book_shape_spread_gap",
        "book_imbalance_pressure",
        "trade_activity",
        "trade_price_impact",
        "postopen_price_path",
        "postopen_liquidity_change",
        "historical_surprise",
        "path_shape_confirmation",
        "other",
    ]
    groups = {name: [] for name in ordered_groups}
    for index, feature in enumerate(feature_names):
        groups.setdefault(_feature_group_name_v2(feature), []).append(index)
    return [(name, groups[name]) for name in ordered_groups if groups.get(name)]


def _group_encoder(
    *,
    input_dim: int,
    group_embedding_dim: int,
    fusion_dim: int,
    dropout: float,
    activation,
    nn,
):
    hidden_dim = max(group_embedding_dim, min(fusion_dim, max(32, input_dim * 2)))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, group_embedding_dim),
        activation(),
    )


def _grouped_encoders(
    *,
    feature_names: tuple[str, ...],
    group_embedding_dim: int,
    group_embedding_dims: dict[str, int] | None = None,
    fusion_dim: int,
    dropout: float,
    activation,
    nn,
    v2_groups: bool = False,
):
    groups = _feature_group_indices_v2(feature_names) if v2_groups else _feature_group_indices(
        feature_names
    )
    if not groups:
        raise SystemExit("grouped torch_mlp architectures need at least one feature")
    group_names = tuple(name for name, _indices in groups)
    group_indices = tuple(tuple(indices) for _name, indices in groups)
    dims = tuple(
        int((group_embedding_dims or {}).get(name, group_embedding_dim))
        for name in group_names
    )
    if any(dim <= 0 for dim in dims):
        raise SystemExit("model.group_embedding_dims values must be positive ints")
    encoders = nn.ModuleList(
        [
            _group_encoder(
                input_dim=len(indices),
                group_embedding_dim=dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
                activation=activation,
                nn=nn,
            )
            for (_name, indices), dim in zip(groups, dims, strict=True)
        ]
    )
    return group_names, group_indices, dims, encoders


def _encode_groups(x, group_indices, encoders):
    return [
        encoder(x[:, list(indices)])
        for indices, encoder in zip(group_indices, encoders, strict=True)
    ]


def _GroupedResidualMLP(
    *,
    feature_names: tuple[str, ...],
    group_embedding_dim: int,
    fusion_dim: int,
    block_hidden_dim: int,
    num_blocks: int,
    dropout: float,
    activation,
    torch,
    nn,
):
    group_embedding_dim = _positive_int(group_embedding_dim, "group_embedding_dim")
    fusion_dim = _positive_int(fusion_dim, "fusion_dim")
    block_hidden_dim = _positive_int(block_hidden_dim, "block_hidden_dim")
    num_blocks = _positive_int(num_blocks, "num_blocks")

    class GroupedResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.group_names, self.group_indices, self.group_dims, self.encoders = (
                _grouped_encoders(
                    feature_names=feature_names,
                    group_embedding_dim=group_embedding_dim,
                    fusion_dim=fusion_dim,
                    dropout=dropout,
                    activation=activation,
                    nn=nn,
                )
            )
            total_dim = sum(self.group_dims)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, fusion_dim),
                activation(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(fusion_dim),
                        nn.Linear(fusion_dim, block_hidden_dim),
                        activation(),
                        nn.Dropout(dropout),
                        nn.Linear(block_hidden_dim, fusion_dim),
                    )
                    for _ in range(num_blocks)
                ]
            )
            self.output = nn.Linear(fusion_dim, 1)

        def forward(self, x):
            embeddings = _encode_groups(x, self.group_indices, self.encoders)
            z = self.fusion(torch.cat(embeddings, dim=1))
            for block in self.blocks:
                z = z + block(z)
            return self.output(z)

    return GroupedResidualMLP()


def _GroupedGatedMLP(
    *,
    feature_names: tuple[str, ...],
    group_embedding_dim: int,
    fusion_dim: int,
    dropout: float,
    activation,
    torch,
    nn,
):
    group_embedding_dim = _positive_int(group_embedding_dim, "group_embedding_dim")
    fusion_dim = _positive_int(fusion_dim, "fusion_dim")

    class GroupedGatedMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.group_names, self.group_indices, self.group_dims, self.encoders = (
                _grouped_encoders(
                    feature_names=feature_names,
                    group_embedding_dim=group_embedding_dim,
                    fusion_dim=fusion_dim,
                    dropout=dropout,
                    activation=activation,
                    nn=nn,
                )
            )
            self.group_count = len(self.group_indices)
            total_dim = sum(self.group_dims)
            self.context = nn.Sequential(nn.Linear(total_dim, fusion_dim), activation())
            self.gate = nn.Sequential(nn.Linear(fusion_dim, self.group_count), nn.Sigmoid())
            self.group_feature_counts = tuple(len(indices) for indices in self.group_indices)
            head_hidden = max(32, fusion_dim // 2)
            self.head = nn.Sequential(
                nn.LayerNorm(total_dim),
                nn.Linear(total_dim, fusion_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, head_hidden),
                activation(),
                nn.Linear(head_hidden, 1),
            )

        def forward(self, x):
            embeddings = _encode_groups(x, self.group_indices, self.encoders)
            stacked = torch.stack(embeddings, dim=1)
            flat = stacked.flatten(start_dim=1)
            gates = self.gate(self.context(flat)).unsqueeze(-1)
            return self.head((stacked * gates).flatten(start_dim=1))

        def gate_values(self, x):
            embeddings = _encode_groups(x, self.group_indices, self.encoders)
            flat = torch.stack(embeddings, dim=1).flatten(start_dim=1)
            return self.gate(self.context(flat))

    return GroupedGatedMLP()


def _GroupedGatedV2MLP(
    *,
    feature_names: tuple[str, ...],
    group_embedding_dim: int,
    group_embedding_dims: dict[str, int],
    fusion_dim: int,
    dropout: float,
    activation,
    torch,
    nn,
):
    group_embedding_dim = _positive_int(group_embedding_dim, "group_embedding_dim")
    fusion_dim = _positive_int(fusion_dim, "fusion_dim")

    class GroupedGatedV2MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.group_names, self.group_indices, self.group_dims, self.encoders = (
                _grouped_encoders(
                    feature_names=feature_names,
                    group_embedding_dim=group_embedding_dim,
                    group_embedding_dims=group_embedding_dims,
                    fusion_dim=fusion_dim,
                    dropout=dropout,
                    activation=activation,
                    nn=nn,
                    v2_groups=True,
                )
            )
            self.group_count = len(self.group_indices)
            self.group_feature_counts = tuple(len(indices) for indices in self.group_indices)
            total_dim = sum(self.group_dims)
            self.context = nn.Sequential(nn.Linear(total_dim, fusion_dim), activation())
            self.gate = nn.Sequential(nn.Linear(fusion_dim, self.group_count), nn.Sigmoid())
            head_hidden = max(32, fusion_dim // 2)
            self.head = nn.Sequential(
                nn.LayerNorm(total_dim),
                nn.Linear(total_dim, fusion_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, head_hidden),
                activation(),
                nn.Linear(head_hidden, 1),
            )

        def _embeddings_and_flat(self, x):
            embeddings = _encode_groups(x, self.group_indices, self.encoders)
            return embeddings, torch.cat(embeddings, dim=1)

        def gate_values(self, x):
            _embeddings, flat = self._embeddings_and_flat(x)
            return self.gate(self.context(flat))

        def forward(self, x):
            embeddings, flat = self._embeddings_and_flat(x)
            gates = self.gate(self.context(flat))
            weighted = [
                embedding * gates[:, index : index + 1]
                for index, embedding in enumerate(embeddings)
            ]
            return self.head(torch.cat(weighted, dim=1))

    return GroupedGatedV2MLP()


def _GroupedCrossMLP(
    *,
    feature_names: tuple[str, ...],
    group_embedding_dim: int,
    fusion_dim: int,
    block_hidden_dim: int,
    num_blocks: int,
    dropout: float,
    activation,
    torch,
    nn,
):
    group_embedding_dim = _positive_int(group_embedding_dim, "group_embedding_dim")
    fusion_dim = _positive_int(fusion_dim, "fusion_dim")
    block_hidden_dim = _positive_int(block_hidden_dim, "block_hidden_dim")
    num_blocks = _positive_int(num_blocks, "num_blocks")

    class GroupedCrossMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.group_names, self.group_indices, self.group_dims, self.encoders = (
                _grouped_encoders(
                    feature_names=feature_names,
                    group_embedding_dim=group_embedding_dim,
                    fusion_dim=fusion_dim,
                    dropout=dropout,
                    activation=activation,
                    nn=nn,
                )
            )
            total_dim = sum(self.group_dims)
            self.cross_layers = nn.ModuleList(
                [nn.Linear(total_dim, total_dim) for _ in range(num_blocks)]
            )
            self.deep = nn.Sequential(
                nn.Linear(total_dim, fusion_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim),
                activation(),
            )
            self.output = nn.Sequential(
                nn.LayerNorm(total_dim + fusion_dim),
                nn.Linear(total_dim + fusion_dim, block_hidden_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(block_hidden_dim, 1),
            )

        def forward(self, x):
            x0 = torch.cat(_encode_groups(x, self.group_indices, self.encoders), dim=1)
            cross = x0
            for layer in self.cross_layers:
                cross = x0 * layer(cross) + cross
            deep = self.deep(x0)
            return self.output(torch.cat([cross, deep], dim=1))

    return GroupedCrossMLP()


def _GroupTokenTransformerMLP(
    *,
    feature_names: tuple[str, ...],
    group_embedding_dim: int,
    fusion_dim: int,
    block_hidden_dim: int,
    num_blocks: int,
    transformer_heads: int,
    dropout: float,
    activation_name: str,
    activation,
    torch,
    nn,
):
    group_embedding_dim = _positive_int(group_embedding_dim, "group_embedding_dim")
    fusion_dim = _positive_int(fusion_dim, "fusion_dim")
    block_hidden_dim = _positive_int(block_hidden_dim, "block_hidden_dim")
    num_blocks = _positive_int(num_blocks, "num_blocks")
    transformer_heads = _positive_int(transformer_heads, "transformer_heads")
    if group_embedding_dim % transformer_heads != 0:
        raise SystemExit("model.group_embedding_dim must be divisible by model.transformer_heads")

    class GroupTokenTransformerMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.group_names, self.group_indices, self.group_dims, self.encoders = (
                _grouped_encoders(
                    feature_names=feature_names,
                    group_embedding_dim=group_embedding_dim,
                    fusion_dim=fusion_dim,
                    dropout=dropout,
                    activation=activation,
                    nn=nn,
                )
            )
            transformer_activation = "gelu" if activation_name == "gelu" else "relu"
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=group_embedding_dim,
                nhead=transformer_heads,
                dim_feedforward=block_hidden_dim,
                dropout=dropout,
                activation=transformer_activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
            self.head = nn.Sequential(
                nn.LayerNorm(group_embedding_dim),
                nn.Linear(group_embedding_dim, fusion_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, 1),
            )

        def forward(self, x):
            tokens = torch.stack(_encode_groups(x, self.group_indices, self.encoders), dim=1)
            tokens = self.transformer(tokens)
            return self.head(tokens.mean(dim=1))

    return GroupTokenTransformerMLP()


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
_CROSS_SECTIONAL_FEATURE_VALUE_TRANSFORMS = {
    "cross_sectional_demean": "demean",
    "xs_demean": "demean",
    "cross_sectional_zscore": "zscore",
    "xs_zscore": "zscore",
    "cross_sectional_robust_zscore": "robust_zscore",
    "xs_robust_zscore": "robust_zscore",
    "cross_sectional_rank_pct": "rank_pct",
    "xs_rank_pct": "rank_pct",
    "cross_sectional_rank": "rank",
    "xs_rank": "rank",
    "cross_sectional_rank_centered": "rank_centered",
    "cross_sectional_rank_centered_inplace": "rank_centered",
    "xs_rank_centered": "rank_centered",
    "rank_centered": "rank_centered",
}
_MECHANISMIZED_FEATURE_VALUE_TRANSFORMS = {
    "mechanismized_cross_sectional_rank_centered": "rank_centered",
    "mechanismized_xs_rank_centered": "rank_centered",
    "mechanismized_rank_centered": "rank_centered",
    "mechanismized_dimensionless": "rank_centered",
    "mechanismized_dimensionless_328": "rank_centered",
    "mechanism_aware_cross_sectional_rank_centered": "rank_centered",
    "mechanism_aware_xs_rank_centered": "rank_centered",
    "mechanism_aware_rank_centered": "rank_centered",
    "mechanismized_cross_sectional_zscore": "zscore",
    "mechanismized_xs_zscore": "zscore",
    "mechanismized_zscore": "zscore",
    "mechanismized_only": "none",
    "mechanism_aware_only": "none",
}
_MECHANISMIZED_V2_FEATURE_VALUE_TRANSFORMS = {
    "mechanismized_v2_cross_sectional_robust_zscore": "robust_zscore",
    "mechanismized_v2_xs_robust_zscore": "robust_zscore",
    "mechanismized_v2_robust_zscore": "robust_zscore",
    "mechanismized_v2_dimensionless": "robust_zscore",
    "mechanismized_v2_dimensionless_328": "robust_zscore",
    "mechanismized_v2_cross_sectional_zscore": "zscore",
    "mechanismized_v2_xs_zscore": "zscore",
    "mechanismized_v2_zscore": "zscore",
    "mechanismized_v2_cross_sectional_rank_centered": "rank_centered",
    "mechanismized_v2_xs_rank_centered": "rank_centered",
    "mechanismized_v2_rank_centered": "rank_centered",
    "mechanismized_v2_only": "none",
    "mechanism_aware_v2_cross_sectional_robust_zscore": "robust_zscore",
    "mechanism_aware_v2_dimensionless": "robust_zscore",
    "mechanism_aware_v2_only": "none",
}
_MECHANISMIZED_V3_FEATURE_VALUE_TRANSFORMS = {
    "mechanismized_v3_cross_sectional_robust_zscore": "robust_zscore",
    "mechanismized_v3_xs_robust_zscore": "robust_zscore",
    "mechanismized_v3_robust_zscore": "robust_zscore",
    "mechanismized_v3_dimensionless": "none",
    "mechanismized_v3_dimensionless_328": "none",
    "mechanismized_v3_cross_sectional_zscore": "zscore",
    "mechanismized_v3_xs_zscore": "zscore",
    "mechanismized_v3_zscore": "zscore",
    "mechanismized_v3_cross_sectional_rank_centered": "rank_centered",
    "mechanismized_v3_xs_rank_centered": "rank_centered",
    "mechanismized_v3_rank_centered": "rank_centered",
    "mechanismized_v3_only": "none",
    "mechanism_aware_v3_cross_sectional_robust_zscore": "robust_zscore",
    "mechanism_aware_v3_dimensionless": "none",
    "mechanism_aware_v3_only": "none",
}


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
        mechanismized_feature_value_reference_columns()
        if mode.startswith("mechanismized_")
        else ()
    )
    required = list(dict.fromkeys([*extra_columns, *group_cols, *features, *support_columns]))
    transform_required = [*group_cols, *features] if mode != "none" else features
    missing = [column for column in transform_required if column and column not in frame.columns]
    if missing and mode != "none":
        raise SystemExit(
            "features.feature_value_transform requires columns: "
            f"{missing[:5]}"
        )
    available = [column for column in required if column and column in frame.columns]
    model_frame = frame.loc[:, available].copy()
    if mode == "none":
        return model_frame
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
    if group_keys is None or group_mean is None or group_scale is None:
        values -= mean
        values /= scale
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


def _torch_mlp_score(model: TorchMLPPredictionModel, frame: pd.DataFrame) -> np.ndarray:
    torch, _nn, _loader, _dataset = _import_torch()
    module = model.module.to(model.device)
    module.eval()
    scores = np.empty(len(frame), dtype="float64")
    batch_size = max(1, int(model.batch_size))
    score_frame = _torch_feature_value_frame(
        frame,
        model.features,
        feature_value_transform=model.feature_value_transform,
        group_cols=model.feature_value_transform_group_cols,
        rank_method=model.feature_value_transform_rank_method,
        tick_size=model.feature_value_transform_tick_size,
        extra_columns=(model.standardization_group_col,),
    )
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            end = min(start + batch_size, len(frame))
            x_values, _mean, _scale = _standardized_float_matrix(
                score_frame.iloc[start:end],
                model.features,
                mean=model.feature_mean,
                scale=model.feature_scale,
                group_col=model.standardization_group_col,
                group_keys=model.standardization_group_keys
                if model.feature_standardization == "symbol_train_zscore"
                else None,
                group_mean=model.standardization_group_mean
                if model.feature_standardization == "symbol_train_zscore"
                else None,
                group_scale=model.standardization_group_scale
                if model.feature_standardization == "symbol_train_zscore"
                else None,
            )
            batch_x = torch.from_numpy(x_values).to(model.device, non_blocking=True)
            batch_scores = module(batch_x).detach().cpu().numpy().reshape(-1)
            scores[start:end] = batch_scores.astype("float64")
    return scores
