from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.schema import normalize_date_series as _date
from opening_strength_fit.schema import normalize_decision_keys_preserving_rows
from opening_strength_fit.schema import normalize_text_series as _text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

GROUP_KEYS = ["date", "decision_target_timestamp"]
TOP_N = 100
OUTCOMES = (
    "close_to_next_open",
    "next_open_to_next_close",
    "close_to_next_close",
)
EXCLUDED_2026_DATES = {"2026-03-18", "2026-04-22", "2026-05-06"}


def _reference(root: Path, year: int) -> pd.DataFrame:
    raw = root / f"cache/opening_0931_0940_raw_source/year={year}"
    daily = pd.read_parquet(
        raw / "daily_reference.parquet",
        columns=[
            "TradingDay",
            "Symbol",
            "ClosePrice",
            "OpenPrice",
            "UpdownLimitStatus",
        ],
    ).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "ClosePrice": "today_close",
            "OpenPrice": "today_open",
            "UpdownLimitStatus": "updown_limit_status",
        }
    )
    daily["date"] = _date(daily["date"])
    daily["symbol"] = _text(daily["symbol"])
    daily["today_close"] = pd.to_numeric(daily["today_close"], errors="coerce")
    daily["today_open"] = pd.to_numeric(daily["today_open"], errors="coerce")
    daily["final_up_limit"] = pd.to_numeric(daily["updown_limit_status"], errors="coerce").eq(1)
    daily = daily.drop_duplicates(["date", "symbol"], keep="last")
    dates = sorted(daily["date"].dropna().unique())
    next_dates = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    current = daily[["date", "symbol", "today_close", "final_up_limit"]].copy()
    current["next_date"] = current["date"].map(next_dates)
    next_open = daily[["date", "symbol", "today_open"]].rename(
        columns={"date": "next_date", "today_open": "next_open"}
    )
    return current.merge(next_open, on=["next_date", "symbol"], how="left", validate="one_to_one")


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_decision_keys_preserving_rows(frame)
    entry_to_close = pd.to_numeric(out.pop("label_short"), errors="coerce")
    entry_to_next_close = pd.to_numeric(out.pop("label_next_close"), errors="coerce")
    out["close_to_next_close"] = (1.0 + entry_to_next_close) / (1.0 + entry_to_close) - 1.0
    return out


def _group_metrics(frame: pd.DataFrame, *, scope: str, sample: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        [*GROUP_KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    ordered["selected"] = ordered.groupby(GROUP_KEYS, sort=False).cumcount().lt(TOP_N)
    top = ordered.loc[ordered["selected"]]
    base = ordered.groupby(GROUP_KEYS, sort=False).size().rename("candidate_rows").to_frame()
    base["selected_rows"] = top.groupby(GROUP_KEYS, sort=False).size()
    base["selected_limit_share_pct"] = (
        top.groupby(GROUP_KEYS, sort=False)["final_up_limit"].mean() * 100.0
    )
    for outcome in OUTCOMES:
        candidate = ordered.loc[ordered[outcome].notna()]
        selected = top.loc[top[outcome].notna()]
        candidate_mean = candidate.groupby(GROUP_KEYS, sort=False)[outcome].mean()
        selected_mean = selected.groupby(GROUP_KEYS, sort=False)[outcome].mean()
        selected_limit = (
            selected.loc[selected["final_up_limit"]]
            .groupby(GROUP_KEYS, sort=False)[outcome]
            .agg(["mean", "count"])
        )
        candidate_limit = (
            candidate.loc[candidate["final_up_limit"]]
            .groupby(GROUP_KEYS, sort=False)[outcome]
            .mean()
        )
        selected_nonlimit = (
            selected.loc[~selected["final_up_limit"]]
            .groupby(GROUP_KEYS, sort=False)[outcome]
            .mean()
        )
        base[f"{outcome}_candidate_raw_bps"] = candidate_mean * 10_000.0
        base[f"{outcome}_selected_raw_bps"] = selected_mean * 10_000.0
        base[f"{outcome}_selected_excess_bps"] = (selected_mean - candidate_mean) * 10_000.0
        base[f"{outcome}_selected_limit_raw_bps"] = selected_limit["mean"] * 10_000.0
        base[f"{outcome}_candidate_limit_raw_bps"] = candidate_limit * 10_000.0
        base[f"{outcome}_selected_limit_vs_limit_bps"] = (
            selected_limit["mean"] - candidate_limit
        ) * 10_000.0
        base[f"{outcome}_selected_nonlimit_raw_bps"] = selected_nonlimit * 10_000.0
        base[f"{outcome}_selected_limit_valid_rows"] = selected_limit["count"]
    result = base.reset_index()
    result["scope"] = scope
    result["sample"] = sample
    result["halfyear"] = (
        pd.to_datetime(result["date"]).dt.year.astype(str)
        + "H"
        + np.where(pd.to_datetime(result["date"]).dt.month.le(6), "1", "2")
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/output/opening_strength_fit/audits/ds350_close_model_next_session_by_limit_v1"
        ),
    )
    args = parser.parse_args()
    root = Path("/mnt/output/opening_strength_fit")
    historical_root = (
        root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1/w0931_0940_hclose"
    )
    holdout = (
        root / "nn/holdout/nn_ds350_w0931_hclose_train2023_2025_test2026h1_purge1_v1/"
        "predictions_unfiltered.parquet"
    )
    inputs = [
        (path, "2022-2025") for path in sorted(historical_root.glob("month_*/predictions.parquet"))
    ]
    inputs.append((holdout, "2026H1"))
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    references: dict[int, pd.DataFrame] = {}
    parts = []
    for index, (path, sample) in enumerate(inputs, start=1):
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
        ).dropna(subset=[*GROUP_KEYS, "symbol", "prediction"])
        if sample == "2026H1":
            frame = frame.loc[~frame["date"].isin(EXCLUDED_2026_DATES)]
        year = int(frame["date"].iloc[0][:4])
        references.setdefault(year, _reference(root, year))
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
        frame["final_up_limit"] = frame["final_up_limit"].fillna(False).astype(bool)
        parts.append(_group_metrics(frame, scope="all_a", sample=sample))
        if sample == "2022-2025":
            in_pool = stock_pool_membership_mask(frame, pool, date_lag_sessions=0)
            parts.append(_group_metrics(frame.loc[in_pool].copy(), scope="pool_L", sample=sample))
        print(
            f"progress input={index}/{len(inputs)} sample={sample} rows={len(frame)} "
            f"valid_next_open={int(valid_open.sum())}",
            flush=True,
        )

    metrics = pd.concat(parts, ignore_index=True)
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"candidate_rows", "selected_rows"}
    ]
    overall = metrics.groupby(["sample", "scope"], sort=True)[numeric].mean().reset_index()
    halfyear = (
        metrics.groupby(["sample", "scope", "halfyear"], sort=True)[numeric].mean().reset_index()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.output_dir / "group_metrics.parquet", index=False)
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)
    halfyear.to_csv(args.output_dir / "halfyear_summary.csv", index=False)
    trace = {
        "status": "ok",
        "score": "09:31-09:40 close-label model",
        "selection": "causal Top100 before inspecting outcome availability",
        "event": "same-day final close at upper limit",
        "close_to_next_open": "same-day official close to next official open",
        "next_open_to_next_close": "next official open to next close",
    }
    (args.output_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "_SUCCESS").touch()
    keep = [
        "sample",
        "scope",
        "selected_limit_share_pct",
        "close_to_next_open_selected_limit_raw_bps",
        "next_open_to_next_close_selected_limit_raw_bps",
        "close_to_next_close_selected_limit_raw_bps",
        "next_open_to_next_close_candidate_limit_raw_bps",
        "next_open_to_next_close_selected_limit_vs_limit_bps",
        "next_open_to_next_close_selected_nonlimit_raw_bps",
        "next_open_to_next_close_selected_raw_bps",
        "next_open_to_next_close_selected_excess_bps",
    ]
    print("OVERALL")
    print(overall[keep].to_csv(index=False))


if __name__ == "__main__":
    main()
