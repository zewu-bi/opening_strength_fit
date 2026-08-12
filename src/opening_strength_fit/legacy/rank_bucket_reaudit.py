"""Compatibility API for the superseded rank-bucket reaudit module."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.legacy import top1000_rank_bucket_diagnostics as current
from opening_strength_fit.legacy.multiscale_bucket_diag import spearman_rank_ic

TOP_N = 1000
BUCKET_COUNTS = current.BUCKET_COUNTS
build_bucket_diagnostics = current.build_bucket_diagnostics
curve_rank_ic = current.curve_rank_ic
finalize_curve_data = current.finalize_curve_data
summarize_ic = current.summarize_ic
plot_return_curves = current.plot_bucket_curves


def numpy_average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    group_start = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    starts = np.flatnonzero(group_start)
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype="float64")
    ranks[order] = ((starts + ends + 1.0) / 2.0)[np.cumsum(group_start) - 1]
    return ranks


def pearson_ic(scores: np.ndarray, outcomes: np.ndarray) -> float:
    valid = np.isfinite(scores) & np.isfinite(outcomes)
    if valid.sum() < 3:
        return math.nan
    scores, outcomes = scores[valid], outcomes[valid]
    if np.ptp(scores) == 0 or np.ptp(outcomes) == 0:
        return math.nan
    return float(np.corrcoef(scores, outcomes)[0, 1])


def numpy_rank_ic(scores: np.ndarray, outcomes: np.ndarray) -> float:
    valid = np.isfinite(scores) & np.isfinite(outcomes)
    if valid.sum() < 3:
        return math.nan
    return pearson_ic(numpy_average_ranks(scores[valid]), numpy_average_ranks(outcomes[valid]))


@dataclass(frozen=True)
class RankBucketReauditConfig:
    prediction_root: Path
    next_label_root: Path
    pool_path: str
    output_dir: Path
    run_ids: dict[str, str]
    months: list[str]


def build_group_ic_diagnostics(frame: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    out = current.build_group_ic(frame, variant=variant, top_n=TOP_N).rename(
        columns={"spearman_realized_rank_ic": "rank_ic"}
    )
    independent, excess = [], []
    for _, group in frame.groupby(current.ms.GROUP_COLS, sort=False, observed=True):
        for scoped in (group, group.iloc[:TOP_N], group.iloc[:100]):
            scores = scoped["prediction"].to_numpy(dtype="float64", copy=False)
            labels = scoped["alpha_return_next_close"].to_numpy(dtype="float64", copy=False)
            independent.append(numpy_rank_ic(scores, labels))
            excess.append(
                spearman_rank_ic(
                    scores,
                    scoped["excess_bps"].to_numpy(dtype="float64", copy=False),
                )
            )
    out["independent_rank_ic"] = independent
    out["excess_rank_ic"] = excess
    out["implementation_abs_diff"] = np.abs(out["rank_ic"] - out["independent_rank_ic"])
    out["reverse_sign_abs_error"] = np.abs(out["rank_ic"] + out["reverse_score_rank_ic"])
    out["label_excess_abs_diff"] = np.abs(out["rank_ic"] - out["excess_rank_ic"])
    return out


def run_rank_bucket_reaudit(config: RankBucketReauditConfig) -> None:
    for variant, run_id in config.run_ids.items():
        output_dir = config.output_dir if len(config.run_ids) == 1 else config.output_dir / variant
        current.run_rank_bucket(
            prediction_root=config.prediction_root,
            next_label_root=config.next_label_root,
            prediction_next_label_col="",
            pool_path=config.pool_path,
            output_dir=output_dir,
            variant=variant,
            run_id=run_id,
            months=config.months,
            top_n=TOP_N,
        )
