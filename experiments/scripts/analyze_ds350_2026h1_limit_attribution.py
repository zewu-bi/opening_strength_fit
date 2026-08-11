from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.raw_source import read_daily_limit_flags
from opening_strength_fit.schema import normalize_text_series as _text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

GROUP_KEYS = ["date", "decision_target_timestamp"]
OUTCOMES = {
    "entry_to_same_day_close": "label_short",
    "same_day_close_to_next_close": "close_to_next_close",
    "entry_to_next_close": "label_next_close",
}
TOP_N = 100
EXCLUDED_DATES = ["2026-03-18", "2026-04-22", "2026-05-06"]


def _group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        [*GROUP_KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    top = ordered.groupby(GROUP_KEYS, sort=False).head(TOP_N).copy()
    out = ordered.groupby(GROUP_KEYS, sort=False).agg(
        candidate_rows=("prediction", "size"),
        candidate_limit_rows=("final_up_limit", "sum"),
    )
    out = out.join(
        top.groupby(GROUP_KEYS, sort=False).agg(
            selected_rows=("prediction", "size"),
            selected_limit_rows=("final_up_limit", "sum"),
        )
    )
    out["selected_limit_share_pct"] = out["selected_limit_rows"] / out["selected_rows"] * 100.0
    out["candidate_limit_share_pct"] = out["candidate_limit_rows"] / out["candidate_rows"] * 100.0

    for name, column in OUTCOMES.items():
        candidate_valid = ordered[column].notna()
        selected_valid = top[column].notna()
        candidate_limit = candidate_valid & ordered["final_up_limit"]
        selected_limit = selected_valid & top["final_up_limit"]
        candidate_nonlimit = candidate_valid & ~ordered["final_up_limit"]
        selected_nonlimit = selected_valid & ~top["final_up_limit"]

        candidate = (
            ordered.loc[candidate_valid]
            .groupby(GROUP_KEYS, sort=False)[column]
            .agg(["sum", "mean", "count"])
        )
        selected = (
            top.loc[selected_valid]
            .groupby(GROUP_KEYS, sort=False)[column]
            .agg(["sum", "mean", "count"])
        )
        candidate_limit_sum = (
            ordered.loc[candidate_limit]
            .groupby(GROUP_KEYS, sort=False)[column]
            .sum()
            .reindex(out.index)
            .fillna(0.0)
        )
        selected_limit_sum = (
            top.loc[selected_limit]
            .groupby(GROUP_KEYS, sort=False)[column]
            .sum()
            .reindex(out.index)
            .fillna(0.0)
        )
        candidate_nonlimit_sum = (
            ordered.loc[candidate_nonlimit]
            .groupby(GROUP_KEYS, sort=False)[column]
            .sum()
            .reindex(out.index)
            .fillna(0.0)
        )
        selected_nonlimit_sum = (
            top.loc[selected_nonlimit]
            .groupby(GROUP_KEYS, sort=False)[column]
            .sum()
            .reindex(out.index)
            .fillna(0.0)
        )
        candidate_count = candidate["count"].reindex(out.index)
        selected_count = selected["count"].reindex(out.index)
        candidate_mean = (candidate["sum"] / candidate["count"]).reindex(out.index)
        selected_mean = (selected["sum"] / selected["count"]).reindex(out.index)
        excess = (selected_mean - candidate_mean) * 10_000.0
        limit_contribution = (
            selected_limit_sum / selected_count - candidate_limit_sum / candidate_count
        ) * 10_000.0
        nonlimit_contribution = (
            selected_nonlimit_sum / selected_count - candidate_nonlimit_sum / candidate_count
        ) * 10_000.0
        if not np.allclose(
            excess.to_numpy(),
            (limit_contribution + nonlimit_contribution).to_numpy(),
            rtol=1e-6,
            atol=1e-4,
            equal_nan=True,
        ):
            raise AssertionError(f"{name} limit decomposition did not reconcile")

        filtered = ordered.loc[candidate_nonlimit]
        reselected = filtered.groupby(GROUP_KEYS, sort=False).head(TOP_N)
        filtered_pool = filtered.groupby(GROUP_KEYS, sort=False)[column].mean()
        reselected_mean = reselected.groupby(GROUP_KEYS, sort=False)[column].mean()

        positive5 = selected_valid & top[column].gt(0.05)
        positive5_rows = (
            positive5.groupby([top[key] for key in GROUP_KEYS], sort=False)
            .sum()
            .reindex(out.index)
            .fillna(0)
        )
        positive5_sum = (
            top.loc[positive5]
            .groupby(GROUP_KEYS, sort=False)[column]
            .sum()
            .reindex(out.index)
            .fillna(0.0)
        )

        out[f"{name}_pool_raw_bps"] = candidate_mean * 10_000.0
        out[f"{name}_selected_raw_bps"] = selected_mean * 10_000.0
        out[f"{name}_excess_bps"] = excess
        out[f"{name}_limit_excess_contribution_bps"] = limit_contribution
        out[f"{name}_nonlimit_excess_contribution_bps"] = nonlimit_contribution
        out[f"{name}_reselected_no_limit_excess_bps"] = (reselected_mean - filtered_pool) * 10_000.0
        out[f"{name}_positive5_share_pct"] = positive5_rows / selected_count * 100.0
        out[f"{name}_positive5_gross_contribution_bps"] = positive5_sum / selected_count * 10_000.0
        out[f"{name}_selected_valid_rows"] = selected_count

    return out.reset_index()


def _summary(metrics: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {
        "groups": int(len(metrics)),
        "dates": int(metrics["date"].nunique()),
        "selected_limit_share_pct": float(metrics["selected_limit_share_pct"].mean()),
        "candidate_limit_share_pct": float(metrics["candidate_limit_share_pct"].mean()),
    }
    for name in OUTCOMES:
        out[name] = {
            key: float(metrics[f"{name}_{key}"].mean())
            for key in (
                "pool_raw_bps",
                "selected_raw_bps",
                "excess_bps",
                "limit_excess_contribution_bps",
                "nonlimit_excess_contribution_bps",
                "reselected_no_limit_excess_bps",
                "positive5_share_pct",
                "positive5_gross_contribution_bps",
                "selected_valid_rows",
            )
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/output/opening_strength_fit"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/output/opening_strength_fit/audits/ds350_2026h1_holdout_return_scope_v3"
        ),
    )
    args = parser.parse_args()
    prediction_path = (
        args.root / "nn/holdout/nn_ds350_w0931_hclose_train2023_2025_test2026h1_purge1_v1/"
        "predictions_unfiltered.parquet"
    )
    predictions = pd.read_parquet(
        prediction_path,
        columns=[
            "date",
            "symbol",
            "decision_target_timestamp",
            "prediction",
            "label_short",
            "label_next_close",
        ],
    )
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    predictions["symbol"] = _text(predictions["symbol"])
    predictions["decision_target_timestamp"] = pd.to_datetime(
        predictions["decision_target_timestamp"], errors="coerce"
    )
    predictions["close_to_next_close"] = (
        1.0 + pd.to_numeric(predictions["label_next_close"], errors="coerce")
    ) / (1.0 + pd.to_numeric(predictions["label_short"], errors="coerce")) - 1.0
    predictions = predictions.loc[~predictions["date"].isin(EXCLUDED_DATES)]
    work = predictions.merge(
        read_daily_limit_flags(
            args.root / "cache/opening_0931_0940_raw_source",
            2026,
        ),
        on=["date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    if work["final_up_limit"].isna().any():
        raise RuntimeError("daily limit reference did not cover every prediction row")
    work["final_up_limit"] = work["final_up_limit"].astype(bool)

    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    pool_l_mask = stock_pool_membership_mask(work, pool, date_lag_sessions=0)
    scopes = {
        "all_a": work,
        "pool_L_same_day": work.loc[pool_l_mask].copy(),
    }
    metric_parts = []
    summaries = {}
    monthlies = {}
    for scope, scoped_work in scopes.items():
        if scoped_work.empty:
            summaries[scope] = {"status": "empty"}
            monthlies[scope] = {}
            continue
        scope_metrics = _group_metrics(scoped_work)
        scope_metrics["scope"] = scope
        scope_metrics["month"] = pd.to_datetime(scope_metrics["date"]).dt.to_period("M").astype(str)
        metric_parts.append(scope_metrics)
        summaries[scope] = _summary(scope_metrics)
        monthlies[scope] = {
            month: _summary(part) for month, part in scope_metrics.groupby("month", sort=True)
        }
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics["month"] = pd.to_datetime(metrics["date"]).dt.to_period("M").astype(str)
    result = {
        "status": "ok",
        "scope": "strict 2026H1; excludes dates preceding a ClickHouse tick gap",
        "excluded_dates": EXCLUDED_DATES,
        "return_definitions": {
            "entry_to_same_day_close": "entry at decision time + 6 seconds to same-day close",
            "same_day_close_to_next_close": "same-day close to next trading-day close; compounded from labels",
            "entry_to_next_close": "entry at decision time + 6 seconds to next trading-day close",
        },
        "selection_scopes": {
            "all_a": "all eligible A-share candidates; no stock-pool overlay",
            "pool_L_same_day": "Pool L members on the same date (date_lag_sessions=0)",
        },
        "stock_pool": {
            "path": DEFAULT_STOCK_POOL_PATHS["L"],
            "first_date": str(pool.index.min()),
            "last_date": str(pool.index.max()),
            "dates": int(len(pool)),
            "symbols": int(len(pool.columns)),
            "matched_rows": int(pool_l_mask.sum()),
            "all_rows": int(len(work)),
        },
        "summary": summaries,
        "monthly": monthlies,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.output_dir / "group_metrics.parquet", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
