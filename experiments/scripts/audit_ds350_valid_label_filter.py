from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.raw_source import read_daily_limit_flags
from opening_strength_fit.schema import normalize_decision_keys_preserving_rows
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "symbol", "decision_target_timestamp"]


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_decision_keys_preserving_rows(frame)
    for column in ("label_short", "label_next_close", "target_label"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
        out[f"{column}_finite"] = np.isfinite(out[column])
    out["valid_label"] = out["label_short_finite"] & out["target_label_finite"]
    return out


def _summary(frame: pd.DataFrame, *, horizon: str, year: int, pool_lag: int | None) -> dict:
    valid = frame["valid_label"]
    grouped = frame.groupby(["date", "decision_target_timestamp"], sort=False)
    counts = grouped["valid_label"].agg(["size", "sum"])
    invalid = ~valid
    limit = frame["final_up_limit"].fillna(False).astype(bool)
    short_missing = ~frame["label_short_finite"]
    next_missing = ~frame["label_next_close_finite"]
    both_missing = short_missing & next_missing
    short_only_missing = short_missing & ~next_missing
    next_only_missing = ~short_missing & next_missing
    target_only_missing = ~short_missing & ~next_missing & ~frame["target_label_finite"]

    def reason_pct(mask: pd.Series) -> float:
        return float(mask.mean() * 100.0)

    def reason_limit_share(mask: pd.Series) -> float:
        return float(limit.loc[mask].mean() * 100.0) if mask.any() else 0.0

    return {
        "horizon": horizon,
        "year": year,
        "pool_lag": "all_a" if pool_lag is None else pool_lag,
        "rows": int(len(frame)),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int(invalid.sum()),
        "invalid_pct": float(invalid.mean() * 100.0),
        "short_missing_pct": float((~frame["label_short_finite"]).mean() * 100.0),
        "next_close_missing_pct": float((~frame["label_next_close_finite"]).mean() * 100.0),
        "target_missing_pct": float((~frame["target_label_finite"]).mean() * 100.0),
        "both_short_next_missing_pct": reason_pct(both_missing),
        "short_only_missing_pct": reason_pct(short_only_missing),
        "next_only_missing_pct": reason_pct(next_only_missing),
        "target_only_missing_pct": reason_pct(target_only_missing),
        "groups": int(len(counts)),
        "groups_with_any_invalid_pct": float((counts["sum"] < counts["size"]).mean() * 100.0),
        "groups_with_zero_valid_pct": float(counts["sum"].eq(0).mean() * 100.0),
        "mean_candidates_per_group": float(counts["size"].mean()),
        "mean_valid_candidates_per_group": float(counts["sum"].mean()),
        "invalid_final_up_limit_share_pct": float(limit.loc[invalid].mean() * 100.0)
        if invalid.any()
        else 0.0,
        "valid_final_up_limit_share_pct": float(limit.loc[valid].mean() * 100.0)
        if valid.any()
        else 0.0,
        "both_missing_final_up_limit_share_pct": reason_limit_share(both_missing),
        "short_only_missing_final_up_limit_share_pct": reason_limit_share(short_only_missing),
        "next_only_missing_final_up_limit_share_pct": reason_limit_share(next_only_missing),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    root = Path("/mnt/output/opening_strength_fit")
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    rows: list[dict[str, object]] = []
    invalid_dates: list[pd.DataFrame] = []
    for horizon, suffix in (("1m", "h1m_v2"), ("close", "hclose_v1")):
        label_root = root / f"datasets/opening_0931_0940_labels_{suffix}"
        for year in range(2022, 2026):
            frame = _normalize(
                pd.read_parquet(
                    label_root / f"year={year}/labels.parquet",
                    columns=[*KEYS, "label_short", "label_next_close", "target_label"],
                )
            )
            frame = frame.merge(
                read_daily_limit_flags(
                    root / "cache/opening_0931_0940_raw_source",
                    year,
                ),
                on=["date", "symbol"],
                how="left",
                validate="many_to_one",
            )
            rows.append(_summary(frame, horizon=horizon, year=year, pool_lag=None))
            for lag in (0, 1):
                work = frame.loc[
                    stock_pool_membership_mask(frame, pool, date_lag_sessions=lag)
                ].copy()
                rows.append(_summary(work, horizon=horizon, year=year, pool_lag=lag))

            by_date = (
                frame.groupby("date", sort=False)[
                    ["valid_label", "label_short_finite", "label_next_close_finite"]
                ]
                .agg(["size", "sum"])
                .reset_index()
            )
            by_date.columns = [
                "date",
                "rows",
                "valid_rows",
                "short_rows",
                "short_valid_rows",
                "next_rows",
                "next_valid_rows",
            ]
            by_date["horizon"] = horizon
            by_date["year"] = year
            by_date["invalid_pct"] = (1.0 - by_date["valid_rows"] / by_date["rows"]) * 100.0
            invalid_dates.append(by_date.loc[by_date["invalid_pct"].gt(0)])

    summary = pd.DataFrame(rows)
    dates = pd.concat(invalid_dates, ignore_index=True)
    summary.to_csv(args.output_dir / "valid_label_summary.csv", index=False)
    dates.to_csv(args.output_dir / "invalid_dates.csv", index=False)
    payload = {
        "summary": rows,
        "worst_invalid_dates": dates.nlargest(30, "invalid_pct").to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_csv(index=False).strip())
    print("WORST_INVALID_DATES")
    print(dates.nlargest(30, "invalid_pct").to_csv(index=False).strip())


if __name__ == "__main__":
    main()
