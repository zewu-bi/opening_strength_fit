from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.schema import normalize_date_series as _date
from opening_strength_fit.schema import (
    normalize_decision_keys_preserving_rows as _normalize_predictions,
)
from opening_strength_fit.schema import normalize_text_series as _text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "symbol", "decision_target_timestamp"]
GROUPS = ["date", "decision_target_timestamp"]
OUTCOMES = ["close_to_next_open", "next_open_to_next_close", "close_to_next_close"]


def _reference(root: Path, window: str, year: int) -> tuple[pd.DataFrame, dict[str, str]]:
    raw = root / f"cache/opening_{window}_raw_source/year={year}"
    close = pd.read_parquet(
        raw / "close_reference.parquet",
        columns=["TradingDay", "Symbol", "ClosePrice"],
    ).rename(columns={"TradingDay": "date", "Symbol": "symbol", "ClosePrice": "today_close"})
    daily = pd.read_parquet(
        raw / "daily_reference.parquet",
        columns=["TradingDay", "Symbol", "OpenPrice"],
    ).rename(columns={"TradingDay": "next_date", "Symbol": "symbol", "OpenPrice": "next_open"})
    close["date"] = _date(close["date"])
    close["symbol"] = _text(close["symbol"])
    daily["next_date"] = _date(daily["next_date"])
    daily["symbol"] = _text(daily["symbol"])
    close["today_close"] = pd.to_numeric(close["today_close"], errors="coerce")
    daily["next_open"] = pd.to_numeric(daily["next_open"], errors="coerce")
    close = close.drop_duplicates(["date", "symbol"], keep="last")
    daily = daily.drop_duplicates(["next_date", "symbol"], keep="last")
    dates = sorted(daily["next_date"].dropna().unique())
    next_dates = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    close["next_date"] = close["date"].map(next_dates)
    reference = close.merge(
        daily,
        on=["next_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    return reference[["date", "symbol", "next_date", "today_close", "next_open"]], next_dates


def _metrics(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        [*GROUPS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    ordered["selected"] = ordered.groupby(GROUPS, sort=False).cumcount().lt(100)
    base = ordered.groupby(GROUPS, sort=False).size().rename("candidate_rows").to_frame()
    for outcome in OUTCOMES:
        valid = ordered.dropna(subset=[outcome])
        selected = valid.loc[valid["selected"]]
        pool_mean = valid.groupby(GROUPS, sort=False)[outcome].mean()
        selected_mean = selected.groupby(GROUPS, sort=False)[outcome].mean()
        base[f"{outcome}_pool_mean_bps"] = pool_mean * 10_000.0
        base[f"{outcome}_selected_mean_bps"] = selected_mean * 10_000.0
        base[f"{outcome}_excess_bps"] = (selected_mean - pool_mean) * 10_000.0
        base[f"{outcome}_selected_valid_rows"] = selected.groupby(GROUPS, sort=False).size()
    result = base.reset_index()
    result["quarter"] = pd.to_datetime(result["date"]).dt.to_period("Q").astype(str)
    result["scope"] = scope
    return result


def _aggregate(frame: pd.DataFrame) -> dict[str, object]:
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    quarter = frame.groupby("quarter", sort=False)[numeric].mean()
    result: dict[str, object] = {column: float(value) for column, value in quarter.mean().items()}
    result["groups"] = int(len(frame))
    result["quarters"] = int(len(quarter))
    for outcome in OUTCOMES:
        result[f"{outcome}_positive_quarters"] = int(quarter[f"{outcome}_excess_bps"].gt(0).sum())
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
        raise SystemExit("expected eight short-model and eight close-model prediction files")

    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    references: dict[int, pd.DataFrame] = {}
    parts: dict[str, list[pd.DataFrame]] = {"universe": [], "pool_L": []}
    coverage: list[dict[str, object]] = []
    for index, (short_path, close_path) in enumerate(
        zip(short_files, close_files, strict=True), start=1
    ):
        short = _normalize_predictions(
            pd.read_parquet(
                short_path,
                columns=[*KEYS, "prediction", "label_next_close"],
            )
        ).rename(columns={"label_next_close": "entry_to_next_close"})
        close_label = _normalize_predictions(
            pd.read_parquet(close_path, columns=[*KEYS, "label_short"])
        ).rename(columns={"label_short": "entry_to_close"})
        frame = short.merge(close_label, on=KEYS, how="left", validate="one_to_one")
        frame["close_to_next_close"] = (
            1.0 + pd.to_numeric(frame["entry_to_next_close"], errors="coerce")
        ) / (1.0 + pd.to_numeric(frame["entry_to_close"], errors="coerce")) - 1.0
        year = int(frame["date"].dropna().iloc[0][:4])
        if year not in references:
            references[year], _ = _reference(root, args.window, year)
        frame = frame.merge(
            references[year], on=["date", "symbol"], how="left", validate="many_to_one"
        )
        valid_open = frame["today_close"].gt(0) & frame["next_open"].gt(0)
        frame["close_to_next_open"] = (frame["next_open"] / frame["today_close"] - 1.0).where(
            valid_open
        )
        frame["next_open_to_next_close"] = (
            (1.0 + frame["close_to_next_close"]) / (1.0 + frame["close_to_next_open"]) - 1.0
        ).where(valid_open)
        coverage.append(
            {
                "fold": short_path.parent.name,
                "rows": len(frame),
                "valid_next_open_rows": int(valid_open.sum()),
                "missing_next_open_rows": int((~valid_open).sum()),
            }
        )
        parts["universe"].append(_metrics(frame, "universe"))
        in_pool = stock_pool_membership_mask(frame, pool, date_lag_sessions=0)
        parts["pool_L"].append(_metrics(frame.loc[in_pool], "pool_L"))
        print(
            f"progress window={args.window} file={index}/8 rows={len(frame)} "
            f"valid_next_open={int(valid_open.sum())}",
            flush=True,
        )

    result = {
        "window": args.window,
        "score_source": f"w{args.window}_h1m prediction",
        "selection": "Top100 selected once by 1m-model score; missing next opens are not refilled",
        "definitions": {
            "close_to_next_open": "today tick close to next trading day's official open",
            "next_open_to_next_close": "next trading day's official open to tick close",
            "close_to_next_close": "today tick close to next trading day's tick close",
        },
        "coverage": coverage,
        "universe": _aggregate(pd.concat(parts["universe"], ignore_index=True)),
        "pool_L": _aggregate(pd.concat(parts["pool_L"], ignore_index=True)),
    }
    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
