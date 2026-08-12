"""Batched inference for fitted Torch MLP models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opening_strength_fit.model_types import TorchMLPPredictionModel
from opening_strength_fit.torch_model.architectures import _import_torch
from opening_strength_fit.torch_model.preprocessing import (
    _standardized_float_matrix,
    _torch_feature_value_frame,
)


def _torch_mlp_score(model: TorchMLPPredictionModel, frame: pd.DataFrame) -> np.ndarray:
    torch, _nn = _import_torch()
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
