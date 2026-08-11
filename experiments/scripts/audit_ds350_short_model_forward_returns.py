from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.schema import normalize_decision_keys_preserving_rows as _normalize
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "symbol", "decision_target_timestamp"]
GROUPS = ["date", "decision_target_timestamp"]


def _corr(frame: pd.DataFrame, outcome: str, method: str) -> pd.Series:
    work = frame.dropna(subset=["prediction", outcome]).copy()
    if method == "spearman":
        work["_left"] = work.groupby(GROUPS, sort=False)["prediction"].rank(
            method="average", pct=True
        )
        work["_right"] = work.groupby(GROUPS, sort=False)[outcome].rank(method="average", pct=True)
    else:
        work["_left"] = work["prediction"]
        work["_right"] = work[outcome]
    grouped = work.groupby(GROUPS, sort=False)
    left = work["_left"] - grouped["_left"].transform("mean")
    right = work["_right"] - grouped["_right"].transform("mean")
    covariance = (left * right).groupby([work[column] for column in GROUPS], sort=False).mean()
    left_std = (
        (left * left).groupby([work[column] for column in GROUPS], sort=False).mean().pow(0.5)
    )
    right_std = (
        (right * right).groupby([work[column] for column in GROUPS], sort=False).mean().pow(0.5)
    )
    return covariance / (left_std * right_std)


def _group_metrics(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        [*GROUPS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    ordered["selected"] = ordered.groupby(GROUPS, sort=False).cumcount().lt(100)
    outcomes = ["return_1m", "return_close", "return_next_close", "return_close_to_next_close"]
    base = ordered.groupby(GROUPS, sort=False).size().rename("candidate_rows").to_frame()
    base["selected_rows"] = ordered.groupby(GROUPS, sort=False)["selected"].sum()
    for outcome in outcomes:
        valid = ordered.dropna(subset=[outcome])
        selected = valid.loc[valid["selected"]]
        pool_mean = valid.groupby(GROUPS, sort=False)[outcome].mean()
        selected_mean = selected.groupby(GROUPS, sort=False)[outcome].mean()
        selected_count = selected.groupby(GROUPS, sort=False)[outcome].size()
        base[f"{outcome}_pool_mean_bps"] = pool_mean * 10_000.0
        base[f"{outcome}_selected_mean_bps"] = selected_mean * 10_000.0
        base[f"{outcome}_excess_bps"] = (selected_mean - pool_mean) * 10_000.0
        base[f"{outcome}_selected_valid_rows"] = selected_count
        base[f"{outcome}_rank_ic"] = _corr(ordered, outcome, "spearman")
        base[f"{outcome}_pearson_ic"] = _corr(ordered, outcome, "pearson")
    result = base.reset_index()
    result["quarter"] = pd.to_datetime(result["date"]).dt.to_period("Q").astype(str)
    result["scope"] = scope
    return result


def _aggregate(metrics: pd.DataFrame) -> dict[str, object]:
    numeric = [column for column in metrics.select_dtypes(include=[np.number]).columns]
    quarter = metrics.groupby("quarter", sort=False)[numeric].mean()
    values = quarter.mean()
    result: dict[str, object] = {column: float(value) for column, value in values.items()}
    result["groups"] = int(len(metrics))
    result["quarters"] = int(len(quarter))
    for outcome in ("return_1m", "return_close", "return_next_close", "return_close_to_next_close"):
        excess = quarter[f"{outcome}_excess_bps"]
        result[f"{outcome}_positive_quarters"] = int(excess.gt(0).sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, choices=["0931_0940", "1001_1010"])
    args = parser.parse_args()

    root = Path("/mnt/output/opening_strength_fit")
    model_root = root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"
    short_files = sorted((model_root / f"w{args.window}_h1m").glob("month_*/predictions.parquet"))
    close_files = sorted(
        (model_root / f"w{args.window}_hclose").glob("month_*/predictions.parquet")
    )
    if len(short_files) != 8 or len(close_files) != 8:
        raise SystemExit(
            f"expected 8 short and close files, got short={len(short_files)} close={len(close_files)}"
        )

    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    parts: dict[str, list[pd.DataFrame]] = {"universe": [], "pool_L": []}
    coverage: list[dict[str, object]] = []
    for index, (short_path, close_path) in enumerate(
        zip(short_files, close_files, strict=True), start=1
    ):
        short = _normalize(
            pd.read_parquet(
                short_path,
                columns=[
                    *KEYS,
                    "prediction",
                    "label_short",
                    "label_next_close",
                ],
            )
        ).rename(
            columns={
                "label_short": "return_1m",
                "label_next_close": "return_next_close",
            }
        )
        close = _normalize(pd.read_parquet(close_path, columns=[*KEYS, "label_short"])).rename(
            columns={"label_short": "return_close"}
        )
        duplicate_close = int(close.duplicated(KEYS, keep=False).sum())
        if duplicate_close:
            raise SystemExit(f"duplicate close keys: {close_path} rows={duplicate_close}")
        frame = short.merge(close, on=KEYS, how="left", validate="one_to_one", indicator=True)
        frame["return_close_to_next_close"] = (
            1.0 + pd.to_numeric(frame["return_next_close"], errors="coerce")
        ) / (1.0 + pd.to_numeric(frame["return_close"], errors="coerce")) - 1.0
        coverage.append(
            {
                "fold": short_path.parent.name,
                "short_rows": len(short),
                "close_matched_rows": int(frame["_merge"].eq("both").sum()),
                "close_missing_rows": int(frame["_merge"].ne("both").sum()),
            }
        )
        frame = frame.drop(columns="_merge")
        parts["universe"].append(_group_metrics(frame, "universe"))
        in_pool = stock_pool_membership_mask(frame, pool, date_lag_sessions=0)
        parts["pool_L"].append(_group_metrics(frame.loc[in_pool], "pool_L"))
        print(
            f"progress window={args.window} file={index}/8 short_rows={len(short)} "
            f"close_matched={coverage[-1]['close_matched_rows']} pool_rows={int(in_pool.sum())}",
            flush=True,
        )

    result = {
        "window": args.window,
        "score_source": f"w{args.window}_h1m prediction",
        "selection": "Top100 selected once by the 1m-model score; close-invalid rows are not refilled",
        "definitions": {
            "return_1m": "morning entry to 1m-horizon sell VWAP",
            "return_close": "same morning entry to today's close",
            "return_next_close": "same morning entry to next trading day's close",
            "return_close_to_next_close": "today's close to next trading day's close; includes the next day session",
        },
        "coverage": coverage,
        "universe": _aggregate(pd.concat(parts["universe"], ignore_index=True)),
        "pool_L": _aggregate(pd.concat(parts["pool_L"], ignore_index=True)),
    }
    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
