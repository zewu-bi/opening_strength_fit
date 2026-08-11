from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.legacy import multiscale_bucket_diag as ms

TOP1000_SCORE_BUCKETS = tuple(range(1, 11))
TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS = 100
TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS = 3000
TOP1000_RETURN_HISTOGRAM_Y_LIMITS = (1e2, 3e5)


def load_ranked_pool_shard(
    *,
    pred_path: Path,
    labels: pd.DataFrame | None,
    pool: pd.DataFrame,
    prediction_next_label_col: str,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    if not prediction_next_label_col:
        if labels is None:
            raise ValueError("separate next-close labels are required")
        return ms.load_ranked_pool_shard(pred_path=pred_path, labels=labels, pool=pool)

    columns = [*ms.PREDICTION_COLS, prediction_next_label_col]
    pred = pd.read_parquet(pred_path, columns=columns).rename(
        columns={prediction_next_label_col: "alpha_return_next_close"}
    )
    pred["date"] = ms.normalize_date(pred["date"])
    pred = pred.loc[ms.stock_pool_membership_mask(pred, pool)].copy()
    pred_rows = len(pred)
    duplicate_keys = int(pred.duplicated(ms.PREDICTION_COLS[:3]).sum())
    if duplicate_keys:
        raise ValueError(f"invalid predictions for {pred_path}: duplicates={duplicate_keys}")
    missing_labels = int(pred["alpha_return_next_close"].isna().sum())
    if missing_labels:
        raise ValueError(
            f"invalid embedded labels for {pred_path}: missing_labels={missing_labels}"
        )

    frame = pred
    frame["pool_mean"] = frame.groupby(ms.GROUP_COLS, observed=True)[
        "alpha_return_next_close"
    ].transform("mean")
    frame["excess_bps"] = (frame["alpha_return_next_close"] - frame["pool_mean"]) * 10000.0
    frame["realized_pool_pct"] = frame.groupby(ms.GROUP_COLS, observed=True)[
        "alpha_return_next_close"
    ].rank(ascending=False, method="average", pct=True)
    frame["realized_pool_top5"] = frame["realized_pool_pct"] <= 0.05
    frame["realized_pool_top10"] = frame["realized_pool_pct"] <= 0.10
    frame = frame.sort_values(
        ms.GROUP_COLS + ["prediction"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    grouped = frame.groupby(ms.GROUP_COLS, sort=False, observed=True)
    frame["score_rank"] = grouped.cumcount() + 1
    frame["group_size"] = grouped["symbol"].transform("size")
    frame["month"] = frame["date"].str.slice(0, 7)
    return frame, {
        "prediction_pool_rows": pred_rows,
        "joined_rows": len(frame),
        "groups": int(grouped.ngroups),
        "duplicate_keys": duplicate_keys,
        "missing_labels": missing_labels,
        "next_label_source": f"prediction:{prediction_next_label_col}",
    }
