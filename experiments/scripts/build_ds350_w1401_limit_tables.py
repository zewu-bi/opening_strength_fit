from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from build_ds350_clip_tables import evaluate_model, limit_states

from opening_strength_fit.analysis import KEY_COLUMNS
from opening_strength_fit.io import read_frame
from opening_strength_fit.prediction_frames import normalize_keys
from opening_strength_fit.stock_pool import DEFAULT_STOCK_POOL_PATHS, load_stock_pool
from opening_strength_fit.training_dataset_features import (
    decode_clickhouse_text,
    normalize_clickhouse_date,
)

CASES = (
    ("1m", "w1401_1410_h1m"),
    ("3m", "w1401_1410_h3m"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 14:01 1m/3m baseline and no-limit-training attribution."
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--no-limit-root", type=Path, required=True)
    parser.add_argument("--h1m-label-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def _close_reference(raw_root: Path, years: range) -> pd.DataFrame:
    frames = []
    for year in years:
        path = raw_root / f"year={year}" / "close_reference.parquet"
        frame = read_frame(path, columns=["TradingDay", "Symbol", "ClosePrice"])
        frames.append(frame)
        print(f"read close reference path={path} rows={len(frame)}", flush=True)
    close = pd.concat(frames, ignore_index=True).rename(
        columns={"TradingDay": "date", "Symbol": "symbol", "ClosePrice": "close"}
    )
    close["date"] = normalize_clickhouse_date(close["date"])
    close["symbol"] = decode_clickhouse_text(close["symbol"])
    close["close"] = pd.to_numeric(close["close"], errors="coerce")
    return close.drop_duplicates(["date", "symbol"], keep="last")


def derived_outcome_labels(
    h1m_label_root: Path | str,
    raw_root: Path | str,
    years: range,
) -> pd.DataFrame:
    h1m_label_root = Path(h1m_label_root)
    raw_root = Path(raw_root)
    labels = []
    for year in years:
        path = h1m_label_root / f"year={year}" / "labels.parquet"
        frame = read_frame(path, columns=[*KEY_COLUMNS, "label_next_close"])
        labels.append(frame)
        print(f"read h1m labels path={path} rows={len(frame)}", flush=True)
    out = normalize_keys(pd.concat(labels, ignore_index=True))
    out["next_close"] = pd.to_numeric(out.pop("label_next_close"), errors="coerce")

    # Each raw year carries boundary context, so the union contains the next
    # trading session needed to recover the common clock+6s buy price.
    close = _close_reference(raw_root, range(min(years) - 1, max(years) + 1))
    trading_days = pd.Index(sorted(close["date"].dropna().unique()))
    next_dates = pd.DataFrame({"date": trading_days[:-1], "_next_date": trading_days[1:]})
    out = out.merge(next_dates, on="date", how="left", validate="many_to_one")
    out = out.merge(
        close.rename(columns={"close": "_same_close"}),
        on=["date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    out = out.merge(
        close.rename(columns={"date": "_next_date", "close": "_next_close"}),
        on=["_next_date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    out["same_day_close"] = (1.0 + out["next_close"]) * out["_same_close"] / out[
        "_next_close"
    ] - 1.0
    invalid = (
        ~np.isfinite(out["same_day_close"]) | out["_same_close"].le(0) | out["_next_close"].le(0)
    )
    out.loc[invalid, "same_day_close"] = np.nan
    return out[[*KEY_COLUMNS, "same_day_close", "next_close"]].drop_duplicates(
        list(KEY_COLUMNS), keep="last"
    )


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")

    years = range(2022, 2026)
    labels = derived_outcome_labels(args.h1m_label_root, args.raw_root, years)
    daily = limit_states(args.raw_root, years)
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    display_names = {"baseline": "Baseline", "no_limit": "无涨跌停"}
    results: dict[str, dict[str, object]] = {}
    table_1_rows = []
    table_2_rows = []
    enrichment_rows = []

    for horizon, case in CASES:
        for experiment, root in (
            ("baseline", args.baseline_root / case),
            ("no_limit", args.no_limit_root / case),
        ):
            key = f"{case}_{experiment}"
            print(f"evaluate key={key} root={root}", flush=True)
            result = evaluate_model(
                root,
                labels=labels,
                daily=daily,
                pool=pool,
                top_n=args.top_n,
                rank_ic_outcomes=("own_label",),
            )
            results[key] = result
            own = result["outcomes"]["own_label"]
            close = result["outcomes"]["same_day_close"]
            next_close = result["outcomes"]["next_close"]
            table_1_rows.append(
                {
                    "实验": display_names[experiment],
                    "Label": horizon,
                    "IC": own["rank_ic"],
                    "Label对应超额": own["excess_bps"],
                    "持有到收盘超额": close["excess_bps"],
                    "次日收盘超额": next_close["excess_bps"],
                }
            )
            table_2_rows.append(
                {
                    "实验": display_names[experiment],
                    "Label": horizon,
                    "Label超额": own["excess_bps"],
                    "Label涨停": own["final_limit_contribution_bps"],
                    "Label非涨停": own["non_final_limit_contribution_bps"],
                    "收盘超额": close["excess_bps"],
                    "收盘涨停": close["final_limit_contribution_bps"],
                    "收盘非涨停": close["non_final_limit_contribution_bps"],
                }
            )
            candidate_pct = float(result["candidate_final_limit_pct"])
            selected_pct = float(result["selected_final_limit_pct"])
            enrichment_rows.append(
                {
                    "实验": display_names[experiment],
                    "Label": horizon,
                    "可评估候选涨停率": candidate_pct,
                    "Top100涨停率": selected_pct,
                }
            )
            gc.collect()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table_1 = pd.DataFrame(table_1_rows)
    table_2 = pd.DataFrame(table_2_rows)
    enrichment = pd.DataFrame(enrichment_rows)
    table_1.to_csv(output_dir / "table_1.csv", index=False)
    table_2.to_csv(output_dir / "table_2.csv", index=False)
    enrichment.to_csv(output_dir / "final_limit_rates.csv", index=False)
    (output_dir / "_SUCCESS").touch()
    print("TABLE_1", flush=True)
    print(table_1.to_csv(index=False).strip(), flush=True)
    print("TABLE_2", flush=True)
    print(table_2.to_csv(index=False).strip(), flush=True)
    print("FINAL_LIMIT_RATES", flush=True)
    print(enrichment.to_csv(index=False).strip(), flush=True)


if __name__ == "__main__":
    main()
