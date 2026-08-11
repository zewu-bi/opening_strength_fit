from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import equal_weighted_period_means
from opening_strength_fit.raw_source import read_daily_limit_flags
from opening_strength_fit.schema import normalize_decision_keys_preserving_rows
from opening_strength_fit.schema import normalize_text_series as _text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

GROUP_KEYS = ["date", "decision_target_timestamp"]
JOIN_KEYS = ["date", "symbol", "decision_target_timestamp"]
TOP_N = 100


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_decision_keys_preserving_rows(frame)
    for column in ("status", "entry_status"):
        if column in out:
            out[column] = _text(out[column]).str.upper()
    for column in (
        "prediction",
        "label",
        "label_short",
        "label_next_close",
        "ask_volume_1",
        "buy_price",
    ):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["valid_label"] = out["valid_label"].fillna(False).astype(bool)
    if "entry_after_cross_section_ready" in out:
        out["entry_after_cross_section_ready"] = (
            out["entry_after_cross_section_ready"].fillna(False).astype(bool)
        )
    return out


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [*GROUP_KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )


def _group_metrics(
    frame: pd.DataFrame,
    *,
    case: str,
    fold: str,
    purge: int,
    pool_lag: int,
) -> pd.DataFrame:
    ordered_all = _ordered(frame)
    ordered_decision = ordered_all.loc[ordered_all["decision_eligible"]]
    ordered_entry = ordered_all.loc[ordered_all["entry_eligible"]]
    ordered_valid = ordered_all.loc[ordered_all["valid_label"]]
    top_causal = ordered_all.groupby(GROUP_KEYS, sort=False).head(TOP_N).copy()
    top_decision = ordered_decision.groupby(GROUP_KEYS, sort=False).head(TOP_N).copy()
    top_entry = ordered_entry.groupby(GROUP_KEYS, sort=False).head(TOP_N).copy()
    top_valid = ordered_valid.groupby(GROUP_KEYS, sort=False).head(TOP_N).copy()

    pool_all = ordered_all.groupby(GROUP_KEYS, sort=False).agg(
        candidate_rows=("prediction", "size"),
        valid_candidate_rows=("valid_label", "sum"),
        pool_short_mean=("label_short", "mean"),
        pool_next_mean=("label_next_close", "mean"),
    )
    pool_valid = ordered_valid.groupby(GROUP_KEYS, sort=False).agg(
        valid_pool_short_mean=("label_short", "mean"),
        valid_pool_next_mean=("label_next_close", "mean"),
    )
    pool_entry = ordered_entry.groupby(GROUP_KEYS, sort=False).agg(
        entry_candidate_rows=("prediction", "size"),
        entry_pool_short_mean=("label_short", "mean"),
        entry_pool_next_mean=("label_next_close", "mean"),
    )
    pool_decision = ordered_decision.groupby(GROUP_KEYS, sort=False).agg(
        decision_candidate_rows=("prediction", "size"),
        decision_pool_short_mean=("label_short", "mean"),
        decision_pool_next_mean=("label_next_close", "mean"),
    )

    def selected_metrics(top: pd.DataFrame, prefix: str) -> pd.DataFrame:
        work = top.copy()
        work["short_zero"] = work["label_short"].fillna(0.0)
        work["next_zero"] = work["label_next_close"].fillna(0.0)
        work["short_missing"] = work["label_short"].isna()
        work["next_missing"] = work["label_next_close"].isna()
        return work.groupby(GROUP_KEYS, sort=False).agg(
            **{
                f"{prefix}_rows": ("prediction", "size"),
                f"{prefix}_valid_rows": ("valid_label", "sum"),
                f"{prefix}_short_mean": ("label_short", "mean"),
                f"{prefix}_next_mean": ("label_next_close", "mean"),
                f"{prefix}_short_zero_mean": ("short_zero", "mean"),
                f"{prefix}_next_zero_mean": ("next_zero", "mean"),
                f"{prefix}_short_missing_rows": ("short_missing", "sum"),
                f"{prefix}_next_missing_rows": ("next_missing", "sum"),
                f"{prefix}_limit_rows": ("final_up_limit", "sum"),
            }
        )

    out = (
        pool_all.join(pool_valid, how="left")
        .join(pool_decision, how="left")
        .join(pool_entry, how="left")
        .join(selected_metrics(top_causal, "causal"), how="left")
        .join(selected_metrics(top_decision, "decision_filter"), how="left")
        .join(selected_metrics(top_entry, "entry_filter"), how="left")
        .join(selected_metrics(top_valid, "valid_filter"), how="left")
        .reset_index()
    )
    valid_keys = top_valid[JOIN_KEYS].assign(in_valid_top=True)
    causal_overlap = top_causal.merge(valid_keys, on=JOIN_KEYS, how="left", validate="one_to_one")
    overlap = (
        causal_overlap["in_valid_top"]
        .fillna(False)
        .groupby([causal_overlap[column] for column in GROUP_KEYS], sort=False)
        .mean()
        .rename("causal_vs_valid_top100_overlap")
        .reset_index()
    )
    out = out.merge(overlap, on=GROUP_KEYS, how="left", validate="one_to_one")
    decision_overlap = top_decision.merge(
        valid_keys, on=JOIN_KEYS, how="left", validate="one_to_one"
    )
    decision_overlap = (
        decision_overlap["in_valid_top"]
        .fillna(False)
        .groupby([decision_overlap[column] for column in GROUP_KEYS], sort=False)
        .mean()
        .rename("decision_vs_valid_top100_overlap")
        .reset_index()
    )
    out = out.merge(decision_overlap, on=GROUP_KEYS, how="left", validate="one_to_one")
    entry_overlap = top_entry.merge(valid_keys, on=JOIN_KEYS, how="left", validate="one_to_one")
    entry_overlap = (
        entry_overlap["in_valid_top"]
        .fillna(False)
        .groupby([entry_overlap[column] for column in GROUP_KEYS], sort=False)
        .mean()
        .rename("entry_vs_valid_top100_overlap")
        .reset_index()
    )
    out = out.merge(entry_overlap, on=GROUP_KEYS, how="left", validate="one_to_one")
    out["case"] = case
    out["fold"] = fold
    out["purge_train_sessions"] = purge
    out["pool_lag"] = pool_lag
    out["quarter"] = pd.to_datetime(out["date"]).dt.to_period("Q").astype(str)
    out["candidate_invalid_pct"] = (
        1.0 - out["valid_candidate_rows"] / out["candidate_rows"]
    ) * 100.0
    for prefix in ("causal", "decision_filter", "entry_filter", "valid_filter"):
        out[f"{prefix}_invalid_pct"] = (
            1.0 - out[f"{prefix}_valid_rows"] / out[f"{prefix}_rows"]
        ) * 100.0
        out[f"{prefix}_short_missing_pct"] = (
            out[f"{prefix}_short_missing_rows"] / out[f"{prefix}_rows"] * 100.0
        )
        out[f"{prefix}_next_missing_pct"] = (
            out[f"{prefix}_next_missing_rows"] / out[f"{prefix}_rows"] * 100.0
        )
        out[f"{prefix}_limit_share_pct"] = (
            out[f"{prefix}_limit_rows"] / out[f"{prefix}_rows"] * 100.0
        )
        pool_short = {
            "causal": "pool_short_mean",
            "decision_filter": "decision_pool_short_mean",
            "entry_filter": "entry_pool_short_mean",
            "valid_filter": "valid_pool_short_mean",
        }[prefix]
        pool_next = {
            "causal": "pool_next_mean",
            "decision_filter": "decision_pool_next_mean",
            "entry_filter": "entry_pool_next_mean",
            "valid_filter": "valid_pool_next_mean",
        }[prefix]
        out[f"{prefix}_short_excess_bps"] = (
            out[f"{prefix}_short_mean"] - out[pool_short]
        ) * 10_000.0
        out[f"{prefix}_next_excess_bps"] = (out[f"{prefix}_next_mean"] - out[pool_next]) * 10_000.0
        out[f"{prefix}_short_excess_missing_zero_bps"] = (
            out[f"{prefix}_short_zero_mean"] - out[pool_short]
        ) * 10_000.0
        out[f"{prefix}_next_excess_missing_zero_bps"] = (
            out[f"{prefix}_next_zero_mean"] - out[pool_next]
        ) * 10_000.0
    return out


def _quarter_equal(metrics: pd.DataFrame) -> pd.DataFrame:
    id_columns = ["case", "purge_train_sessions", "pool_lag"]
    metric_columns = [
        "candidate_rows",
        "valid_candidate_rows",
        "decision_candidate_rows",
        "entry_candidate_rows",
        "candidate_invalid_pct",
        "causal_vs_valid_top100_overlap",
        "decision_vs_valid_top100_overlap",
        "entry_vs_valid_top100_overlap",
        "causal_invalid_pct",
        "causal_short_missing_pct",
        "causal_next_missing_pct",
        "causal_limit_share_pct",
        "causal_short_excess_bps",
        "causal_next_excess_bps",
        "causal_short_excess_missing_zero_bps",
        "causal_next_excess_missing_zero_bps",
        "decision_filter_invalid_pct",
        "decision_filter_short_missing_pct",
        "decision_filter_next_missing_pct",
        "decision_filter_limit_share_pct",
        "decision_filter_short_excess_bps",
        "decision_filter_next_excess_bps",
        "decision_filter_short_excess_missing_zero_bps",
        "decision_filter_next_excess_missing_zero_bps",
        "entry_filter_invalid_pct",
        "entry_filter_short_missing_pct",
        "entry_filter_next_missing_pct",
        "entry_filter_limit_share_pct",
        "entry_filter_short_excess_bps",
        "entry_filter_next_excess_bps",
        "entry_filter_short_excess_missing_zero_bps",
        "entry_filter_next_excess_missing_zero_bps",
        "valid_filter_limit_share_pct",
        "valid_filter_short_excess_bps",
        "valid_filter_next_excess_bps",
    ]
    return equal_weighted_period_means(
        metrics,
        by=id_columns,
        period_column="quarter",
        value_columns=metric_columns,
    )


def _fold_boundary_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    first_date = out.groupby(["case", "purge_train_sessions", "pool_lag", "fold"], sort=False)[
        "date"
    ].transform("min")
    out["scope"] = np.where(out["date"].eq(first_date), "first_test_day", "remaining_days")
    metric_columns = [
        "causal_short_excess_bps",
        "causal_short_excess_missing_zero_bps",
        "decision_filter_short_excess_bps",
        "decision_filter_short_excess_missing_zero_bps",
        "entry_filter_short_excess_bps",
        "entry_filter_short_excess_missing_zero_bps",
        "valid_filter_short_excess_bps",
        "causal_next_excess_bps",
        "decision_filter_next_excess_bps",
        "entry_filter_next_excess_bps",
        "valid_filter_next_excess_bps",
    ]
    return out.groupby(["case", "purge_train_sessions", "pool_lag", "scope"], as_index=False)[
        metric_columns
    ].mean()


def _reproduction_metrics(
    audit: pd.DataFrame,
    original: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    case: str,
    fold: str,
) -> dict[str, object]:
    left = audit.loc[audit["valid_label"], [*JOIN_KEYS, "prediction"]].rename(
        columns={"prediction": "audit_prediction"}
    )
    right = original[[*JOIN_KEYS, "prediction"]].rename(
        columns={"prediction": "original_prediction"}
    )
    joined = left.merge(right, on=JOIN_KEYS, how="inner", validate="one_to_one")
    delta = joined["audit_prediction"] - joined["original_prediction"]
    pool_work = joined.loc[stock_pool_membership_mask(joined, pool, date_lag_sessions=0)].copy()
    audit_top = (
        _ordered(pool_work.rename(columns={"audit_prediction": "prediction"}))
        .groupby(GROUP_KEYS, sort=False)
        .head(TOP_N)
    )
    original_top = (
        _ordered(pool_work.rename(columns={"original_prediction": "prediction"}))
        .groupby(GROUP_KEYS, sort=False)
        .head(TOP_N)
    )
    overlap = (
        audit_top[JOIN_KEYS]
        .merge(
            original_top[JOIN_KEYS].assign(in_original=True),
            on=JOIN_KEYS,
            how="left",
            validate="one_to_one",
        )["in_original"]
        .fillna(False)
        .mean()
    )
    return {
        "case": case,
        "fold": fold,
        "audit_valid_rows": int(len(left)),
        "original_rows": int(len(right)),
        "matched_rows": int(len(joined)),
        "score_corr": float(joined["audit_prediction"].corr(joined["original_prediction"])),
        "score_mean_abs_delta": float(delta.abs().mean()),
        "score_max_abs_delta": float(delta.abs().max()),
        "pool_lag0_top100_overlap": float(overlap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        choices=("w0931_0940_h1m", "w0931_0940_hclose"),
        help="limit analysis to one or more cases",
    )
    parser.add_argument(
        "--purge",
        action="append",
        type=int,
        choices=(0, 1),
        help="limit analysis to one or more purge settings",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    root = Path("/mnt/output/opening_strength_fit")
    audit_root = root / "nn/audits/ds350_future_info_v1"
    original_root = root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"
    cases = tuple(args.case) if args.case else ("w0931_0940_h1m", "w0931_0940_hclose")
    purges = tuple(args.purge) if args.purge else (0, 1)
    folds = tuple(
        f"month_{month}"
        for month in (
            "2022-01",
            "2022-07",
            "2023-01",
            "2023-07",
            "2024-01",
            "2024-07",
            "2025-01",
            "2025-07",
        )
    )
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    references: dict[int, pd.DataFrame] = {}
    entry_references: dict[str, pd.DataFrame] = {}
    metric_parts: list[pd.DataFrame] = []
    reproduction: list[dict[str, object]] = []
    for case in cases:
        for purge in purges:
            for fold in folds:
                if fold not in entry_references:
                    entry_reference = _normalize(
                        pd.read_parquet(
                            audit_root
                            / "w0931_0940_hclose"
                            / "purge0"
                            / fold
                            / "predictions_unfiltered.parquet",
                            columns=[*JOIN_KEYS, "label_short", "valid_label"],
                        )
                    )
                    entry_reference["entry_eligible"] = entry_reference["label_short"].notna()
                    entry_references[fold] = entry_reference[[*JOIN_KEYS, "entry_eligible"]]
                entry_reference = entry_references[fold]
                path = audit_root / case / f"purge{purge}" / fold / "predictions_unfiltered.parquet"
                frame = _normalize(
                    pd.read_parquet(
                        path,
                        columns=[
                            *JOIN_KEYS,
                            "prediction",
                            "valid_label",
                            "label",
                            "label_short",
                            "label_next_close",
                            "ask_volume_1",
                        ],
                    )
                )
                year = int(frame["date"].dropna().iloc[0][:4])
                references.setdefault(
                    year,
                    read_daily_limit_flags(
                        root / "cache/opening_0931_0940_raw_source",
                        year,
                    ),
                )
                frame = frame.merge(
                    entry_reference,
                    on=JOIN_KEYS,
                    how="left",
                    validate="one_to_one",
                )
                if frame["entry_eligible"].isna().any():
                    raise RuntimeError("close-label entry proxy misses prediction keys")
                frame["entry_eligible"] = frame["entry_eligible"].astype(bool)
                frame["decision_eligible"] = frame["ask_volume_1"].gt(0.0)
                frame = frame.merge(
                    references[year],
                    on=["date", "symbol"],
                    how="left",
                    validate="many_to_one",
                )
                frame["final_up_limit"] = frame["final_up_limit"].fillna(False).astype(bool)
                for lag in (0,):
                    work = frame.loc[
                        stock_pool_membership_mask(frame, pool, date_lag_sessions=lag)
                    ].copy()
                    metric_parts.append(
                        _group_metrics(
                            work,
                            case=case,
                            fold=fold,
                            purge=purge,
                            pool_lag=lag,
                        )
                    )
                if purge == 0:
                    original = _normalize(
                        pd.read_parquet(
                            original_root / case / fold / "predictions.parquet",
                            columns=[*JOIN_KEYS, "prediction", "valid_label"],
                        )
                    )
                    reproduction.append(
                        _reproduction_metrics(frame, original, pool, case=case, fold=fold)
                    )
                del frame
                gc.collect()
                print(f"progress case={case} purge={purge} fold={fold}", flush=True)

    metrics = pd.concat(metric_parts, ignore_index=True)
    summary = _quarter_equal(metrics)
    boundary_summary = _fold_boundary_summary(metrics)
    reproduction_frame = pd.DataFrame(reproduction)
    if reproduction_frame.empty:
        reproduction_summary = pd.DataFrame(
            columns=[
                "case",
                "folds",
                "matched_rows",
                "score_corr_min",
                "score_corr_mean",
                "score_mean_abs_delta_mean",
                "score_max_abs_delta_max",
                "pool_lag0_top100_overlap_mean",
            ]
        )
    else:
        reproduction_summary = reproduction_frame.groupby("case", as_index=False).agg(
            folds=("fold", "size"),
            matched_rows=("matched_rows", "sum"),
            score_corr_min=("score_corr", "min"),
            score_corr_mean=("score_corr", "mean"),
            score_mean_abs_delta_mean=("score_mean_abs_delta", "mean"),
            score_max_abs_delta_max=("score_max_abs_delta", "max"),
            pool_lag0_top100_overlap_mean=("pool_lag0_top100_overlap", "mean"),
        )
    metrics.to_csv(args.output_dir / "group_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "quarter_equal_summary.csv", index=False)
    boundary_summary.to_csv(args.output_dir / "fold_boundary_summary.csv", index=False)
    reproduction_frame.to_csv(args.output_dir / "reproduction_by_fold.csv", index=False)
    reproduction_summary.to_csv(args.output_dir / "reproduction_summary.csv", index=False)
    payload = {
        "selection_definition": {
            "valid_filter": "filter valid_label before Top100 (original implementation)",
            "causal": "select Top100 without inspecting future label availability",
            "decision_filter": "filter by positive transformed ask_volume_1 in the decision-time feature row, then Top100",
            "entry_filter": "sensitivity proxy using same-day-close label availability; this mostly captures +6s entry failures but still contains close availability and is not treated as the strict causal result",
            "missing_zero": "same causal Top100, with unavailable realized returns conservatively set to zero",
        },
        "quarter_equal_summary": summary.to_dict("records"),
        "fold_boundary_summary": boundary_summary.to_dict("records"),
        "reproduction_summary": reproduction_summary.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("QUARTER_EQUAL_SUMMARY")
    print(summary.to_csv(index=False).strip())
    print("REPRODUCTION_SUMMARY")
    print(reproduction_summary.to_csv(index=False).strip())
    print("FOLD_BOUNDARY_SUMMARY")
    print(boundary_summary.to_csv(index=False).strip())


if __name__ == "__main__":
    main()
