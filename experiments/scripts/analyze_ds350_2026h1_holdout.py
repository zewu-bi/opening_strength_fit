from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from opening_strength_fit.schema import normalize_decision_keys_preserving_rows

GROUP_KEYS = ["date", "decision_target_timestamp"]
LABELS = {
    "same_day_close": "label_short",
    "next_close": "label_next_close",
    "mixed_target": "target_label",
}
TOP_N = 100


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_decision_keys_preserving_rows(frame)
    for column in ["prediction", *LABELS.values()]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["valid_label"] = out["valid_label"].fillna(False).astype(bool)
    return out


def _correlation(group: pd.DataFrame, label: str) -> float:
    valid = group["prediction"].notna() & group[label].notna()
    if int(valid.sum()) < 2:
        return float("nan")
    x = group.loc[valid, "prediction"]
    y = group.loc[valid, label]
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="pearson"))


def _group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame["prediction"].notna()].copy()
    work["score_rank"] = work.groupby(GROUP_KEYS, sort=False)["prediction"].rank(
        method="first", ascending=False
    )
    work["selected"] = work["score_rank"].le(TOP_N)
    base = work.groupby(GROUP_KEYS, sort=False).agg(
        candidate_rows=("prediction", "size"),
        selected_rows=("selected", "sum"),
        valid_target_rows=("valid_label", "sum"),
    )
    for name, column in LABELS.items():
        valid = work[column].notna()
        selected = work["selected"]
        pool = work.loc[valid].groupby(GROUP_KEYS, sort=False)[column].mean()
        top = work.loc[valid & selected].groupby(GROUP_KEYS, sort=False)[column].mean()
        top_rows = (valid & selected).groupby([work[key] for key in GROUP_KEYS], sort=False).sum()
        ic = work.groupby(GROUP_KEYS, sort=False).apply(
            _correlation, label=column, include_groups=False
        )
        base[f"{name}_pool_mean"] = pool
        base[f"{name}_top100_mean"] = top
        base[f"{name}_top100_valid_rows"] = top_rows
        base[f"{name}_top100_excess_bps"] = (top - pool) * 10_000.0
        base[f"{name}_ic"] = ic
    return base.reset_index()


def _aggregate(groups: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {
        "groups": int(len(groups)),
        "dates": int(groups["date"].nunique()),
        "candidate_rows_mean": float(groups["candidate_rows"].mean()),
        "selected_rows_mean": float(groups["selected_rows"].mean()),
    }
    for name in LABELS:
        out[name] = {
            "ic": float(groups[f"{name}_ic"].mean()),
            "top100_excess_bps": float(groups[f"{name}_top100_excess_bps"].mean()),
            "top100_raw_bps": float(groups[f"{name}_top100_mean"].mean() * 10_000.0),
            "pool_raw_bps": float(groups[f"{name}_pool_mean"].mean() * 10_000.0),
            "top100_valid_rows_mean": float(groups[f"{name}_top100_valid_rows"].mean()),
        }
    return out


def _monthly(groups: pd.DataFrame, scope: str) -> pd.DataFrame:
    work = groups.copy()
    work["scope"] = scope
    work["month"] = pd.to_datetime(work["date"]).dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for month, part in work.groupby("month", sort=True):
        row: dict[str, object] = {
            "scope": scope,
            "month": month,
            "dates": int(part["date"].nunique()),
            "groups": int(len(part)),
        }
        for name in LABELS:
            row[f"{name}_ic"] = float(part[f"{name}_ic"].mean())
            row[f"{name}_top100_excess_bps"] = float(part[f"{name}_top100_excess_bps"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-root",
        default=(
            "/mnt/output/opening_strength_fit/nn/holdout/"
            "nn_ds350_w0931_hclose_train2023_2025_test2026h1_purge1_v1"
        ),
    )
    parser.add_argument(
        "--dataset-audit",
        default="/mnt/output/opening_strength_fit/audits/ds350_2026_holdout_dataset_v1/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/output/opening_strength_fit/audits/ds350_2026h1_holdout_result_v1",
    )
    args = parser.parse_args()

    training_root = Path(args.training_root)
    predictions_path = training_root / "predictions_unfiltered.parquet"
    metrics_path = training_root / "metrics.json"
    if not predictions_path.exists() or not metrics_path.exists():
        raise RuntimeError(f"training output is incomplete: {training_root}")
    predictions = _normalize(pd.read_parquet(predictions_path))
    predictions = predictions.loc[predictions["date"].between("2026-01-01", "2026-06-30")].copy()

    dataset_audit = json.loads(Path(args.dataset_audit).read_text(encoding="utf-8"))
    excluded_dates = dataset_audit["h1"]["gap_predecessor_dates_to_exclude_for_next_close"]
    full_groups = _group_metrics(predictions)
    strict_predictions = predictions.loc[~predictions["date"].isin(excluded_dates)]
    strict_groups = _group_metrics(strict_predictions)

    training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    train_windows = training_metrics.get("train_stats_by_window", {})
    if len(train_windows) != 1:
        raise RuntimeError(f"expected one holdout window, got {sorted(train_windows)}")
    train_stats = next(iter(train_windows.values()))
    checks = {
        "prediction_start_is_2026h1": predictions["date"].min() == "2026-01-05",
        "prediction_end_is_2026h1": predictions["date"].max() == "2026-06-30",
        "no_predictions_on_tick_gap_dates": not predictions["date"]
        .isin(dataset_audit["h1"]["daily_not_tick"])
        .any(),
        "one_train_session_purged": train_stats.get("purge_train_sessions") == 1,
        "purged_date_is_2025_year_end": train_stats.get("purged_train_dates") == ["2025-12-31"],
        "strict_scope_excludes_three_predecessors": excluded_dates
        == ["2026-03-18", "2026-04-22", "2026-05-06"],
    }
    result = {
        "status": "ok" if all(checks.values()) else "failed",
        "definition": {
            "training": "2023-01 through 2025-12, with 2025-12-31 purged",
            "validation": "2026-01 through 2026-06, never used for fitting or tuning",
            "selection": "top 100 by score within each date x decision clock",
            "excess": "top100 mean return minus all-A candidate-pool mean in the same group",
            "strict_scope": "also excludes dates preceding a ClickHouse tick gap so next-close never skips a trading session",
        },
        "checks": checks,
        "excluded_gap_predecessor_dates": excluded_dates,
        "prediction_rows": int(len(predictions)),
        "strict_prediction_rows": int(len(strict_predictions)),
        "valid_target_rows": int(predictions["valid_label"].sum()),
        "full_source_scope": _aggregate(full_groups),
        "strict_source_gap_adjusted_scope": _aggregate(strict_groups),
        "training_metrics": training_metrics,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    monthly = pd.concat(
        [_monthly(full_groups, "full"), _monthly(strict_groups, "strict")],
        ignore_index=True,
    )
    monthly.to_csv(output_dir / "monthly_metrics.csv", index=False)
    full_groups.to_parquet(output_dir / "group_metrics_full.parquet", index=False)
    strict_groups.to_parquet(output_dir / "group_metrics_strict.parquet", index=False)
    if result["status"] != "ok":
        raise RuntimeError(f"holdout audit failed: {checks}")
    (output_dir / "_SUCCESS").touch()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
