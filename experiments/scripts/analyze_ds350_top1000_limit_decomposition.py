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

GROUP_KEYS = ["date", "decision_target_timestamp"]
TOP_N = 1000
BUCKET_SIZE = 100


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_decision_keys_preserving_rows(frame)
    out["outcome"] = pd.to_numeric(out.pop("label_next_close"), errors="coerce").astype("float64")
    return out


def _bucket_metrics(frame: pd.DataFrame, *, remove_limits_before_ranking: bool) -> pd.DataFrame:
    ordered = frame.sort_values(
        [*GROUP_KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    if remove_limits_before_ranking:
        ordered = ordered.loc[~ordered["final_up_limit"]].copy()
    ordered["rank"] = ordered.groupby(GROUP_KEYS, sort=False).cumcount() + 1
    top = ordered.loc[ordered["rank"].le(TOP_N)].copy()
    top["bucket"] = ((top["rank"] - 1) // BUCKET_SIZE + 1).astype("int8")

    candidate = ordered.groupby(GROUP_KEYS, sort=False).agg(
        candidate_rows=("outcome", "count"),
        candidate_sum=("outcome", "sum"),
    )
    candidate_limit = (
        ordered.loc[ordered["final_up_limit"]]
        .groupby(GROUP_KEYS, sort=False)["outcome"]
        .sum()
        .reindex(candidate.index)
        .fillna(0.0)
    )
    candidate_nonlimit = (
        ordered.loc[~ordered["final_up_limit"]]
        .groupby(GROUP_KEYS, sort=False)["outcome"]
        .agg(["sum", "mean"])
        .reindex(candidate.index)
    )

    result = top.groupby([*GROUP_KEYS, "bucket"], sort=False).agg(
        bucket_rows=("outcome", "count"),
        bucket_sum=("outcome", "sum"),
        bucket_mean=("outcome", "mean"),
        bucket_limit_rows=("final_up_limit", "sum"),
    )
    bucket_limit_sum = (
        top.loc[top["final_up_limit"]]
        .groupby([*GROUP_KEYS, "bucket"], sort=False)["outcome"]
        .sum()
        .reindex(result.index)
        .fillna(0.0)
    )
    bucket_nonlimit = (
        top.loc[~top["final_up_limit"]]
        .groupby([*GROUP_KEYS, "bucket"], sort=False)["outcome"]
        .agg(["sum", "mean"])
        .reindex(result.index)
    )
    group_index = result.index.droplevel("bucket")
    candidate_rows = candidate["candidate_rows"].reindex(group_index).to_numpy()
    candidate_mean = (
        (candidate["candidate_sum"] / candidate["candidate_rows"]).reindex(group_index).to_numpy()
    )
    candidate_limit_sum = candidate_limit.reindex(group_index).to_numpy()
    candidate_nonlimit_sum = candidate_nonlimit["sum"].reindex(group_index).to_numpy()
    candidate_nonlimit_mean = candidate_nonlimit["mean"].reindex(group_index).to_numpy()

    result["bucket_excess_bps"] = (result["bucket_mean"].to_numpy() - candidate_mean) * 10_000.0
    result["limit_excess_contribution_bps"] = (
        bucket_limit_sum.to_numpy() / result["bucket_rows"].to_numpy()
        - candidate_limit_sum / candidate_rows
    ) * 10_000.0
    result["nonlimit_excess_contribution_bps"] = (
        bucket_nonlimit["sum"].fillna(0.0).to_numpy() / result["bucket_rows"].to_numpy()
        - candidate_nonlimit_sum / candidate_rows
    ) * 10_000.0
    if not np.allclose(
        result["bucket_excess_bps"],
        result["limit_excess_contribution_bps"] + result["nonlimit_excess_contribution_bps"],
        atol=1e-4,
        rtol=1e-6,
    ):
        raise AssertionError("bucket contribution decomposition did not reconcile")
    result["bucket_limit_share_pct"] = result["bucket_limit_rows"] / result["bucket_rows"] * 100.0
    result["bucket_nonlimit_conditional_excess_bps"] = (
        bucket_nonlimit["mean"].to_numpy() - candidate_nonlimit_mean
    ) * 10_000.0
    result["ranking"] = (
        "exclude_final_limit_then_rerank" if remove_limits_before_ranking else "original"
    )
    return result.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/output/opening_strength_fit/audits/ds350_h1m_top1000_limit_decomposition_v1"
        ),
    )
    args = parser.parse_args()
    root = Path("/mnt/output/opening_strength_fit")
    model_root = root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1/w0931_0940_h1m"
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    files = sorted(model_root.glob("month_*/predictions.parquet"))
    if len(files) != 8:
        raise SystemExit(f"expected eight folds, found {len(files)}")
    references: dict[int, pd.DataFrame] = {}
    parts = []
    missing_reference_rows = 0
    for index, path in enumerate(files, start=1):
        frame = _normalize(
            pd.read_parquet(
                path,
                columns=[
                    "date",
                    "symbol",
                    "decision_target_timestamp",
                    "prediction",
                    "label_next_close",
                ],
            )
        ).dropna(subset=[*GROUP_KEYS, "symbol", "prediction", "outcome"])
        frame = frame.loc[stock_pool_membership_mask(frame, pool, date_lag_sessions=0)].copy()
        year = int(frame["date"].iloc[0][:4])
        references.setdefault(
            year,
            read_daily_limit_flags(
                root / "cache/opening_0931_0940_raw_source",
                year,
            ),
        )
        frame = frame.merge(
            references[year], on=["date", "symbol"], how="left", validate="many_to_one"
        )
        missing_reference_rows += int(frame["final_up_limit"].isna().sum())
        frame["final_up_limit"] = frame["final_up_limit"].fillna(False).astype(bool)
        for remove_limits in (False, True):
            part = _bucket_metrics(frame, remove_limits_before_ranking=remove_limits)
            part["fold"] = path.parent.name.removeprefix("month_")
            parts.append(part)
        print(f"progress fold={index}/8 rows={len(frame)}", flush=True)

    metrics = pd.concat(parts, ignore_index=True)
    summary = (
        metrics.groupby(["ranking", "bucket"], sort=True)
        .agg(
            groups=("bucket_excess_bps", "size"),
            mean_excess_bps=("bucket_excess_bps", "mean"),
            limit_share_pct=("bucket_limit_share_pct", "mean"),
            limit_contribution_bps=("limit_excess_contribution_bps", "mean"),
            nonlimit_contribution_bps=("nonlimit_excess_contribution_bps", "mean"),
            nonlimit_conditional_excess_bps=(
                "bucket_nonlimit_conditional_excess_bps",
                "mean",
            ),
        )
        .reset_index()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.output_dir / "group_bucket_metrics.parquet", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    trace = {
        "status": "ok",
        "scope": "same-day Pool L",
        "score": "09:31-09:40 1m-label model",
        "outcome": "entry+6s to next close (label_next_close; not pure overnight)",
        "buckets": "Top1000 by score, ten consecutive 100-name buckets",
        "event": "same-day final close at upper limit; ex-post attribution only",
        "missing_final_limit_reference_rows_treated_as_nonlimit": missing_reference_rows,
    }
    (args.output_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    print("SUMMARY")
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
