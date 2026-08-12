from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from opening_strength_fit.schema import ensure_timestamp_columns

DEFAULT_FEE_BPS = (0.0, 5.0, 10.0, 13.0)


def add_label_audit_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    out = ensure_timestamp_columns(frame)
    timestamp = pd.to_datetime(out["timestamp"])
    out["year"] = timestamp.dt.year.astype(int)
    out["month"] = timestamp.dt.strftime("%Y-%m")
    out["minute_bucket"] = timestamp.dt.strftime("%H:%M")
    return out


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    if "valid_label" in frame.columns:
        return frame["valid_label"].astype(bool)
    return frame["label"].notna() & np.isfinite(frame["label"])


def _quality_rates(group: pd.DataFrame) -> dict[str, float]:
    rates: dict[str, float] = {}
    if "sell_volume" in group.columns:
        rates["zero_sell_volume_ratio"] = float((group["sell_volume"] <= 0).mean())
    if "sell_turnover" in group.columns:
        rates["nonpositive_sell_turnover_ratio"] = float((group["sell_turnover"] <= 0).mean())
    price_col = "buy_price" if "buy_price" in group.columns else "ask_price_1"
    if price_col in group.columns:
        rates["invalid_buy_price_ratio"] = float(
            (group[price_col].isna() | (group[price_col] <= 0)).mean()
        )
    return rates


def summarize_label_distribution(
    frame: pd.DataFrame,
    *,
    fee_bps_values: Iterable[float] = DEFAULT_FEE_BPS,
    group_cols: tuple[str, ...] = ("year", "month", "minute_bucket"),
) -> pd.DataFrame:
    work = add_label_audit_buckets(frame)
    missing = [column for column in group_cols if column not in work.columns]
    if missing:
        raise SystemExit(f"label audit group columns missing: {missing}")

    base_label_col = "gross_label" if "gross_label" in work.columns else "label"
    rows = []
    for fee_bps in fee_bps_values:
        fee_label = work[base_label_col] - float(fee_bps) / 10_000.0
        audit = work.assign(_fee_label=fee_label, _valid_label=_valid_mask(work))
        for keys, group in audit.groupby(list(group_cols), dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            valid = group["_valid_label"] & group["_fee_label"].notna()
            labels = group.loc[valid, "_fee_label"].astype("float64")
            row: dict[str, object] = dict(zip(group_cols, keys, strict=False))
            row.update(
                {
                    "fee_bps": float(fee_bps),
                    "rows": int(len(group)),
                    "valid_labels": int(valid.sum()),
                    "valid_label_ratio": float(valid.mean()) if len(group) else np.nan,
                    "label_mean": float(labels.mean()) if len(labels) else np.nan,
                    "label_std": float(labels.std()) if len(labels) else np.nan,
                    "label_min": float(labels.min()) if len(labels) else np.nan,
                    "label_p1": float(labels.quantile(0.01)) if len(labels) else np.nan,
                    "label_p5": float(labels.quantile(0.05)) if len(labels) else np.nan,
                    "label_p50": float(labels.quantile(0.50)) if len(labels) else np.nan,
                    "label_p95": float(labels.quantile(0.95)) if len(labels) else np.nan,
                    "label_p99": float(labels.quantile(0.99)) if len(labels) else np.nan,
                    "label_max": float(labels.max()) if len(labels) else np.nan,
                }
            )
            row.update(_quality_rates(group))
            rows.append(row)
    return pd.DataFrame(rows)
