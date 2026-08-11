from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.schema import normalize_decision_keys_preserving_rows

GROUP_KEYS = ["date", "decision_target_timestamp"]
TOP_N = 100
OUTCOMES = (
    "entry_to_same_day_close",
    "same_day_close_to_next_close",
    "entry_to_next_close",
)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_decision_keys_preserving_rows(frame)
    out["entry_to_same_day_close"] = pd.to_numeric(out.pop("label_short"), errors="coerce")
    out["entry_to_next_close"] = pd.to_numeric(out.pop("label_next_close"), errors="coerce")
    out["same_day_close_to_next_close"] = (1.0 + out["entry_to_next_close"]) / (
        1.0 + out["entry_to_same_day_close"]
    ) - 1.0
    return out


def _fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.dropna(subset=[*GROUP_KEYS, "symbol", "prediction"]).sort_values(
        [*GROUP_KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top = ordered.groupby(GROUP_KEYS, sort=False).head(TOP_N)
    base = ordered.groupby(GROUP_KEYS, sort=False).size().rename("candidate_rows").to_frame()
    base["selected_rows"] = top.groupby(GROUP_KEYS, sort=False).size()
    for outcome in OUTCOMES:
        candidate = ordered.groupby(GROUP_KEYS, sort=False)[outcome].mean()
        selected = top.groupby(GROUP_KEYS, sort=False)[outcome].mean()
        base[f"{outcome}_all_a_raw_bps"] = candidate * 10_000.0
        base[f"{outcome}_top100_raw_bps"] = selected * 10_000.0
        base[f"{outcome}_top100_excess_bps"] = (selected - candidate) * 10_000.0
        base[f"{outcome}_selected_valid_rows"] = (
            top.loc[top[outcome].notna()].groupby(GROUP_KEYS, sort=False).size()
        )
    return base.reset_index()


def _stats(values: pd.Series) -> dict[str, float]:
    values = values.dropna()
    std = values.std(ddof=1)
    return {
        "mean_bps": float(values.mean()),
        "day_std_bps": float(std),
        "day_t_stat": float(values.mean() / (std / np.sqrt(len(values)))),
        "positive_day_pct": float(values.gt(0).mean() * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(
            "/mnt/output/opening_strength_fit/nn/"
            "nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1/"
            "w0931_0940_hclose"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/output/opening_strength_fit/audits/ds350_2022_2025_all_a_true_overnight_v1"
        ),
    )
    args = parser.parse_args()

    files = sorted(args.model_root.glob("month_*/predictions.parquet"))
    if len(files) != 8:
        raise SystemExit(f"expected 8 OOS folds, found {len(files)}")
    parts = []
    for index, path in enumerate(files, start=1):
        frame = _normalize(
            pd.read_parquet(
                path,
                columns=[
                    "date",
                    "symbol",
                    "decision_target_timestamp",
                    "prediction",
                    "label_short",
                    "label_next_close",
                ],
            )
        )
        metrics = _fold_metrics(frame)
        metrics["fold"] = path.parent.name.removeprefix("month_")
        parts.append(metrics)
        print(f"progress fold={index}/8 path={path.parent.name} groups={len(metrics)}", flush=True)

    groups = pd.concat(parts, ignore_index=True)
    groups["halfyear"] = (
        pd.to_datetime(groups["date"]).dt.year.astype(str)
        + "H"
        + np.where(pd.to_datetime(groups["date"]).dt.month.le(6), "1", "2")
    )
    daily = (
        groups.groupby(["halfyear", "date"], sort=True)[
            [column for column in groups if column.endswith("_bps")]
        ]
        .mean()
        .reset_index()
    )

    rows = []
    for halfyear, part in daily.groupby("halfyear", sort=True):
        row: dict[str, object] = {
            "halfyear": halfyear,
            "days": int(part["date"].nunique()),
            "groups": int(groups.loc[groups["halfyear"].eq(halfyear)].shape[0]),
        }
        for outcome in OUTCOMES:
            for metric in ("all_a_raw_bps", "top100_raw_bps", "top100_excess_bps"):
                column = f"{outcome}_{metric}"
                values = _stats(part[column])
                for key, value in values.items():
                    row[f"{column}_{key}"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups.to_parquet(args.output_dir / "group_metrics.parquet", index=False)
    daily.to_parquet(args.output_dir / "daily_metrics.parquet", index=False)
    summary.to_csv(args.output_dir / "halfyear_summary.csv", index=False)
    trace = {
        "status": "ok",
        "selection": "Top100 from all eligible A-share predictions; no Pool L overlay",
        "model": "09:31-09:40 close-label max30 grouped-gated model",
        "oos": "eight non-overlapping half-year rolling-OOS folds, 2022H1 through 2025H2",
        "same_day_close_to_next_close": "(1 + label_next_close) / (1 + label_short) - 1",
        "aggregation": "ten decision groups averaged within day, then days equally weighted",
    }
    (args.output_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    keep = [
        "halfyear",
        "days",
        "same_day_close_to_next_close_all_a_raw_bps_mean_bps",
        "same_day_close_to_next_close_top100_raw_bps_mean_bps",
        "same_day_close_to_next_close_top100_excess_bps_mean_bps",
        "same_day_close_to_next_close_top100_excess_bps_day_t_stat",
        "same_day_close_to_next_close_top100_excess_bps_positive_day_pct",
        "entry_to_next_close_top100_excess_bps_mean_bps",
        "entry_to_same_day_close_top100_excess_bps_mean_bps",
    ]
    print("SUMMARY")
    print(summary[keep].to_csv(index=False))


if __name__ == "__main__":
    main()
