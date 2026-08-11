from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import equal_weighted_period_means, write_analysis_result
from opening_strength_fit.schema import normalize_date_series as _normalize_date
from opening_strength_fit.schema import normalize_decision_keys_preserving_rows
from opening_strength_fit.schema import normalize_text_series as _text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "decision_target_timestamp"]
JOIN_KEYS = ["date", "symbol", "decision_target_timestamp"]
HORIZONS = ("1m", "3m", "10m", "1h", "close")
CONDITIONS = (
    "daily_closes_up_limit",
    "decision_within_100bps",
    "decision_sealed_at_limit",
    "entry_within_100bps",
    "entry_at_limit",
)
TOP_N = 100


def _normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_decision_keys_preserving_rows(frame)
    for column in ("prediction", "label", "mid_price", "spread_bps", "ask_volume_1"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _reference(root: Path, window: str, year: int) -> pd.DataFrame:
    path = root / f"cache/opening_{window}_raw_source/year={year}/daily_reference.parquet"
    frame = pd.read_parquet(
        path,
        columns=[
            "TradingDay",
            "Symbol",
            "ClosePrice",
            "PreClosePrice",
            "STStatus",
            "TradeStatus",
            "UpdownLimitStatus",
        ],
    ).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "ClosePrice": "daily_close",
            "PreClosePrice": "prev_close",
            "STStatus": "st_status",
            "TradeStatus": "daily_trade_status",
            "UpdownLimitStatus": "updown_limit_status",
        }
    )
    frame["date"] = _normalize_date(frame["date"])
    frame["symbol"] = _text(frame["symbol"])
    frame = frame.loc[frame["date"].str.startswith(str(year), na=False)].copy()
    for column in ("daily_close", "prev_close", "st_status", "updown_limit_status"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["daily_trade_status"] = _text(frame["daily_trade_status"]).str.upper()
    frame["daily_closes_up_limit"] = frame["updown_limit_status"].eq(1)

    symbol = frame["symbol"].astype(str)
    st = frame["st_status"].fillna(0).ne(0)
    wide_board = symbol.str.startswith("30") | symbol.str.startswith("68")
    ratio = np.where(st, 0.05, np.where(wide_board, 0.20, 0.10))
    standard = np.floor(frame["prev_close"].to_numpy() * (1.0 + ratio) * 100.0 + 0.50000001) / 100.0
    frame["upper_limit_price"] = pd.Series(standard, index=frame.index).where(
        ~frame["daily_closes_up_limit"], frame["daily_close"]
    )
    # New-listing and relisting rows may not have the standard board limit.
    nonstandard = frame["daily_trade_status"].isin({"NEW", "新股", "N"})
    frame.loc[nonstandard, "upper_limit_price"] = np.nan
    return frame[
        ["date", "symbol", "daily_close", "daily_closes_up_limit", "upper_limit_price"]
    ].drop_duplicates(["date", "symbol"], keep="last")


def _close_authority(path: Path) -> pd.DataFrame:
    close = pd.read_parquet(path, columns=[*JOIN_KEYS, "label"])
    close["date"] = pd.to_datetime(close["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    close["symbol"] = _text(close["symbol"])
    close["decision_target_timestamp"] = pd.to_datetime(
        close["decision_target_timestamp"], errors="coerce"
    )
    close["close_label"] = pd.to_numeric(close.pop("label"), errors="coerce")
    return close.drop_duplicates(JOIN_KEYS, keep="last")


def _add_limit_states(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mid = pd.to_numeric(out["mid_price"], errors="coerce")
    spread = pd.to_numeric(out["spread_bps"], errors="coerce")
    decision_ask = mid * (1.0 + spread / 20_000.0)
    decision_bid = mid * (1.0 - spread / 20_000.0)
    decision_offer = decision_ask.gt(0) & out["ask_volume_1"].gt(0)
    decision_state_price = decision_ask.where(
        decision_offer, decision_bid.where(decision_bid.gt(0))
    )
    close = pd.to_numeric(out["daily_close"], errors="coerce")
    close_label = pd.to_numeric(out["close_label"], errors="coerce")
    entry_price = close / (1.0 + close_label)
    entry_price = entry_price.where(entry_price.gt(0) & np.isfinite(entry_price))
    upper = pd.to_numeric(out["upper_limit_price"], errors="coerce")
    out["decision_ask_reconstructed"] = decision_ask
    out["decision_bid_reconstructed"] = decision_bid
    out["entry_price_reconstructed"] = entry_price
    out["decision_limit_room_bps"] = (upper - decision_ask) / decision_ask * 10_000.0
    out["entry_limit_room_bps"] = (upper - entry_price) / entry_price * 10_000.0
    out["decision_within_100bps"] = decision_offer & out["decision_limit_room_bps"].between(
        -1.0, 100.0
    )
    decision_at_limit = (
        upper.notna()
        & decision_state_price.notna()
        & decision_state_price.sub(upper).abs().le(0.0051)
    )
    out["decision_sealed_at_limit"] = decision_at_limit & ~decision_offer
    out["entry_within_100bps"] = out["entry_limit_room_bps"].between(-1.0, 100.0)
    out["entry_at_limit"] = (
        upper.notna() & entry_price.notna() & entry_price.sub(upper).abs().le(0.0051)
    )
    for column in CONDITIONS:
        out[column] = out[column].fillna(False).astype(bool)
    return out


def _group_metrics(ordered: pd.DataFrame) -> pd.DataFrame:
    top = ordered.groupby(KEYS, sort=False).head(TOP_N)
    filtered = ordered.loc[~ordered["daily_closes_up_limit"]]
    reselected = filtered.groupby(KEYS, sort=False).head(TOP_N)
    base = ordered.groupby(KEYS, sort=False).agg(
        candidate_rows=("label", "size"),
        pool_sum=("label", "sum"),
        pool_count=("label", "count"),
        candidate_final_limit_rows=("daily_closes_up_limit", "sum"),
        candidate_decision_near_rows=("decision_within_100bps", "sum"),
        candidate_decision_sealed_rows=("decision_sealed_at_limit", "sum"),
        candidate_entry_near_rows=("entry_within_100bps", "sum"),
        candidate_entry_at_limit_rows=("entry_at_limit", "sum"),
    )
    selected = top.groupby(KEYS, sort=False).agg(
        selected_rows=("label", "size"),
        selected_sum=("label", "sum"),
        selected_count=("label", "count"),
        selected_final_limit_rows=("daily_closes_up_limit", "sum"),
        selected_decision_near_rows=("decision_within_100bps", "sum"),
        selected_decision_sealed_rows=("decision_sealed_at_limit", "sum"),
        selected_entry_near_rows=("entry_within_100bps", "sum"),
        selected_entry_at_limit_rows=("entry_at_limit", "sum"),
    )
    selected_limit = (
        top.loc[top["daily_closes_up_limit"]]
        .groupby(KEYS, sort=False)
        .agg(
            selected_final_limit_sum=("label", "sum"),
            selected_final_limit_count=("label", "count"),
        )
    )
    filtered_base = filtered.groupby(KEYS, sort=False).agg(
        filtered_pool_sum=("label", "sum"), filtered_pool_count=("label", "count")
    )
    replacement = reselected.groupby(KEYS, sort=False).agg(
        reselected_sum=("label", "sum"), reselected_count=("label", "count")
    )
    result = (
        base.join(selected, how="left")
        .join(selected_limit, how="left")
        .join(filtered_base, how="left")
        .join(replacement, how="left")
        .reset_index()
    )
    count_columns = [
        column for column in result if column.endswith("_rows") or column.endswith("_count")
    ]
    sum_columns = [column for column in result if column.endswith("_sum")]
    result[count_columns] = result[count_columns].fillna(0)
    result[sum_columns] = result[sum_columns].fillna(0.0)
    result["pool_mean"] = result["pool_sum"] / result["pool_count"]
    result["selected_mean"] = result["selected_sum"] / result["selected_count"]
    result["filtered_pool_mean"] = result["filtered_pool_sum"] / result["filtered_pool_count"]
    result["reselected_mean"] = result["reselected_sum"] / result["reselected_count"]
    result["primary_excess_bps"] = (result["selected_mean"] - result["pool_mean"]) * 10_000.0
    for prefix in (
        "final_limit",
        "decision_near",
        "decision_sealed",
        "entry_near",
        "entry_at_limit",
    ):
        result[f"candidate_{prefix}_share_pct"] = (
            result[f"candidate_{prefix}_rows"] / result["candidate_rows"] * 100.0
        )
        result[f"selected_{prefix}_share_pct"] = (
            result[f"selected_{prefix}_rows"] / result["selected_rows"] * 100.0
        )
        result[f"{prefix}_enrichment_x"] = result[f"selected_{prefix}_share_pct"] / result[
            f"candidate_{prefix}_share_pct"
        ].replace(0, np.nan)
    result["selected_final_limit_mean_bps"] = (
        result["selected_final_limit_sum"]
        / result["selected_final_limit_count"].replace(0, np.nan)
        * 10_000.0
    )
    result["selected_final_limit_return_contribution_bps"] = (
        result["selected_final_limit_sum"] / result["selected_rows"] * 10_000.0
    )
    result["selected_final_limit_excess_contribution_bps"] = (
        (
            result["selected_final_limit_sum"]
            - result["selected_final_limit_count"] * result["pool_mean"]
        )
        / result["selected_rows"]
        * 10_000.0
    )
    result["reselected_excess_vs_original_pool_bps"] = (
        result["reselected_mean"] - result["pool_mean"]
    ) * 10_000.0
    result["reselected_excess_vs_filtered_pool_bps"] = (
        result["reselected_mean"] - result["filtered_pool_mean"]
    ) * 10_000.0
    result["quarter"] = pd.to_datetime(result["date"]).dt.to_period("Q").astype(str)
    return result


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "primary_excess_bps",
        "candidate_final_limit_share_pct",
        "selected_final_limit_share_pct",
        "final_limit_enrichment_x",
        "selected_final_limit_mean_bps",
        "selected_final_limit_return_contribution_bps",
        "selected_final_limit_excess_contribution_bps",
        "reselected_excess_vs_original_pool_bps",
        "reselected_excess_vs_filtered_pool_bps",
        "candidate_decision_near_share_pct",
        "selected_decision_near_share_pct",
        "decision_near_enrichment_x",
        "candidate_decision_sealed_share_pct",
        "selected_decision_sealed_share_pct",
        "decision_sealed_enrichment_x",
        "candidate_entry_near_share_pct",
        "selected_entry_near_share_pct",
        "entry_near_enrichment_x",
        "candidate_entry_at_limit_share_pct",
        "selected_entry_at_limit_share_pct",
        "entry_at_limit_enrichment_x",
    ]
    return equal_weighted_period_means(
        metrics,
        by=["label_horizon"],
        period_column="quarter",
        value_columns=columns,
        count_name="groups",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, choices=["0931_0940", "1001_1010"])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    root = Path("/mnt/output/opening_strength_fit")
    model_root = root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    references: dict[int, pd.DataFrame] = {}
    parts: list[pd.DataFrame] = []
    prediction_files = {
        horizon: {
            path.parent.name: path
            for path in sorted(
                (model_root / f"w{args.window}_h{horizon}").glob("month_*/predictions.parquet")
            )
        }
        for horizon in HORIZONS
    }
    if any(len(paths) != 8 for paths in prediction_files.values()):
        raise SystemExit("each horizon must have 8 prediction files")
    fold_names = sorted(prediction_files["close"])

    for fold_index, fold_name in enumerate(fold_names, start=1):
        close_path = prediction_files["close"][fold_name]
        close_state = _normalize_predictions(
            pd.read_parquet(
                close_path,
                columns=[
                    *JOIN_KEYS,
                    "label",
                    "mid_price",
                    "spread_bps",
                    "ask_volume_1",
                ],
            )
        ).rename(columns={"label": "close_label"})
        year = int(close_state["date"].iloc[0][:4])
        references.setdefault(year, _reference(root, args.window, year))
        close_state = close_state.merge(
            references[year], on=["date", "symbol"], how="left", validate="many_to_one"
        )
        close_state = _add_limit_states(close_state)
        state_columns = [*JOIN_KEYS, *CONDITIONS]
        close_state = close_state[state_columns].drop_duplicates(JOIN_KEYS, keep="last")

        for horizon in HORIZONS:
            path = prediction_files[horizon][fold_name]
            frame = _normalize_predictions(
                pd.read_parquet(path, columns=[*JOIN_KEYS, "prediction", "label"])
            ).dropna(subset=[*JOIN_KEYS, "prediction", "label"])
            frame = frame.loc[stock_pool_membership_mask(frame, pool, date_lag_sessions=0)].copy()
            work = frame.merge(close_state, on=JOIN_KEYS, how="left", validate="one_to_one")
            for condition in CONDITIONS:
                work[condition] = work[condition].fillna(False).astype(bool)
            ordered = work.sort_values(
                [*KEYS, "prediction", "symbol"],
                ascending=[True, True, False, True],
                kind="mergesort",
            )
            metrics = _group_metrics(ordered)
            metrics["label_horizon"] = horizon
            parts.append(metrics)
            print(
                f"progress window={args.window} fold={fold_index}/8 label={horizon} "
                f"rows={len(frame)} groups={frame.groupby(KEYS, sort=False).ngroups}",
                flush=True,
            )

    group_metrics = pd.concat(parts, ignore_index=True)
    summary = _summarize(group_metrics)
    trace = {
        "window": args.window,
        "labels": list(HORIZONS),
        "conditions": list(CONDITIONS),
        "pool": "same-day Pool L, lag0",
        "selection": "Top100 by prediction within date x decision_target_timestamp",
        "aggregation": "quarter equal across 16 quarters",
        "decision_ask": "mid_price * (1 + spread_bps / 20000), requires ask_volume_1 > 0",
        "entry_price": "daily tick close / (1 + authoritative same-day-close label)",
        "daily_closes_up_limit": "ex-post outcome, not a point-in-time tradability condition",
    }
    write_analysis_result(
        args.output_dir,
        group_metrics,
        summary,
        metrics_filename="label_limit_trap_group_metrics.parquet",
        summary_filename="label_limit_trap_summary.csv",
        trace=trace,
    )


if __name__ == "__main__":
    main()
