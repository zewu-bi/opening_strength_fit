from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS
from opening_strength_fit.io import read_frame
from opening_strength_fit.prediction_frames import normalize_keys, prediction_files
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)
from opening_strength_fit.training_dataset_features import (
    decode_clickhouse_text,
    normalize_clickhouse_date,
)

GROUP_COLUMNS = ["date", "decision_target_timestamp"]
OUTCOMES = ("own_label", "same_day_close", "next_close")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one hard-cutoff/raw-safe leakage kill-test run."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--close-label-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--pool", choices=("none", "L"), default="none")
    parser.add_argument("--pool-date-lag-sessions", type=int, default=0)
    return parser.parse_args()


def load_predictions(root: Path) -> pd.DataFrame:
    frames = [
        read_frame(path, columns=[*KEY_COLUMNS, "prediction", "label"])
        for path in prediction_files(root)
    ]
    out = normalize_keys(pd.concat(frames, ignore_index=True))
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out["own_label"] = pd.to_numeric(out.pop("label"), errors="coerce")
    return out.drop_duplicates(list(KEY_COLUMNS), keep="last")


def load_outcomes(root: Path, years: list[int]) -> pd.DataFrame:
    frames = [
        read_frame(
            root / f"year={year}" / "labels.parquet",
            columns=[*KEY_COLUMNS, "label_short", "label_next_close"],
        )
        for year in years
    ]
    out = normalize_keys(pd.concat(frames, ignore_index=True)).rename(
        columns={"label_short": "same_day_close", "label_next_close": "next_close"}
    )
    for column in ("same_day_close", "next_close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.drop_duplicates(list(KEY_COLUMNS), keep="last")


def load_final_limit(root: Path, years: list[int]) -> pd.DataFrame:
    frames = [
        read_frame(
            root / f"year={year}" / "daily_reference.parquet",
            columns=["TradingDay", "Symbol", "UpdownLimitStatus"],
        )
        for year in years
    ]
    out = pd.concat(frames, ignore_index=True).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "UpdownLimitStatus": "final_limit_status",
        }
    )
    out["date"] = normalize_clickhouse_date(out["date"])
    out["symbol"] = decode_clickhouse_text(out["symbol"])
    out["final_limit_status"] = pd.to_numeric(out["final_limit_status"], errors="coerce")
    return out.drop_duplicates(["date", "symbol"], keep="last")


def rank_ic(group: pd.DataFrame, outcome: str) -> float:
    pair = group[["prediction", outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2 or pair.nunique().min() < 2:
        return float("nan")
    return float(pair["prediction"].corr(pair[outcome], method="spearman"))


def summarize_outcome(
    candidates: pd.DataFrame, selected: pd.DataFrame, outcome: str
) -> dict[str, float | int]:
    candidates = candidates.dropna(subset=[outcome])
    selected = selected.dropna(subset=[outcome])
    candidate_groups = candidates.groupby(GROUP_COLUMNS, sort=False)
    selected_groups = selected.groupby(GROUP_COLUMNS, sort=False)
    excess = (selected_groups[outcome].mean() - candidate_groups[outcome].mean()).dropna()
    ic = candidate_groups.apply(lambda group: rank_ic(group, outcome), include_groups=False)
    return {
        "groups": int(excess.size),
        "rank_ic": float(ic.mean()),
        "top_n_excess_bps": float(excess.mean() * 10_000.0),
    }


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    frame = load_predictions(args.predictions)
    years = sorted(pd.to_numeric(frame["date"].str[:4], errors="raise").unique().astype(int))
    frame = frame.merge(
        load_outcomes(args.close_label_root, years),
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    ).merge(
        load_final_limit(args.raw_root, years),
        on=["date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    if args.pool == "L":
        pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
        frame = frame.loc[
            stock_pool_membership_mask(frame, pool, date_lag_sessions=args.pool_date_lag_sessions)
        ].copy()

    # Keep the selected set identical for all reported outcomes.
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["prediction", *OUTCOMES])
    frame["_score_rank"] = frame.groupby(GROUP_COLUMNS, sort=False)["prediction"].rank(
        method="first", ascending=False
    )
    selected = frame.loc[frame["_score_rank"].le(args.top_n)].copy()
    report = {
        "predictions": str(args.predictions),
        "candidate_universe": args.pool,
        "pool_date_lag_sessions": args.pool_date_lag_sessions if args.pool == "L" else None,
        "top_n": args.top_n,
        "years": years,
        "groups": int(frame.groupby(GROUP_COLUMNS, sort=False).ngroups),
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "selected_final_limit_pct": float(selected["final_limit_status"].eq(1).mean() * 100.0),
        "outcomes": {outcome: summarize_outcome(frame, selected, outcome) for outcome in OUTCOMES},
        "definitions": {
            "same_day_close": "label_short from the close-horizon label artifact",
            "next_close": "label_next_close from the close-horizon label artifact",
            "final_limit": "daily_reference.UpdownLimitStatus == 1; evaluation only",
            "selection": "Top-N within date x decision_target_timestamp",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
