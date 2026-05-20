from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


RULE_COLUMNS = {
    "open_momentum_30t": "return_30t",
    "open_momentum_10t": "return_10t",
    "turnover_speed_30t": "turnover_diff_30t",
    "volume_speed_30t": "volume_diff_30t",
    "depth_imbalance_10": "depth_imbalance_10",
    "depth_imbalance_1": "depth_imbalance_1",
    "auction_return": "preopen_return_vs_prev_close",
    "auction_turnover": "preopen_turnover",
    "limit_up_room": "ask1_to_limit_up_bps",
}

NEGATED_RULE_COLUMNS = {
    "tight_spread": "spread_bps",
}


def available_rule_scores(
    frame: pd.DataFrame,
    *,
    rules: Iterable[str] | None = None,
) -> dict[str, pd.Series]:
    requested = set(rules) if rules else set(RULE_COLUMNS) | set(NEGATED_RULE_COLUMNS)
    scores: dict[str, pd.Series] = {}
    for name, column in RULE_COLUMNS.items():
        if name in requested and column in frame.columns:
            scores[name] = frame[column]
    for name, column in NEGATED_RULE_COLUMNS.items():
        if name in requested and column in frame.columns:
            scores[name] = -frame[column]
    return scores


def rule_prediction_frame(
    frame: pd.DataFrame,
    *,
    rule_name: str,
    score: pd.Series,
) -> pd.DataFrame:
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
            "valid_label",
        )
        if column in frame.columns
    ]
    out = frame[columns].copy()
    out["rule_name"] = rule_name
    out["prediction"] = score.to_numpy()
    return out
