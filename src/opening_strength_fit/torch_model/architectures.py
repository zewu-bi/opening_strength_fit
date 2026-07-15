"""Lazy Torch imports, feature grouping, and neural network architectures."""

from __future__ import annotations


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
    if name.startswith(("preopen_", "auction_")) or name == "exch_time_offset_us":
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
    if name.startswith(
        (
            "ask_volume_",
            "bid_volume_",
            "ask_count_",
            "bid_count_",
            "ask_depth_",
            "bid_depth_",
        )
    ) or name in {"total_ask_volume", "total_bid_volume", "total_ask_count", "total_bid_count"}:
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
    groups = (
        _feature_group_indices_v2(feature_names)
        if v2_groups
        else _feature_group_indices(feature_names)
    )
    if not groups:
        raise SystemExit("grouped torch_mlp architectures need at least one feature")
    group_names = tuple(name for name, _indices in groups)
    group_indices = tuple(tuple(indices) for _name, indices in groups)
    dims = tuple(
        int((group_embedding_dims or {}).get(name, group_embedding_dim)) for name in group_names
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
