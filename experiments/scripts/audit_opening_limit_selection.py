from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.clickhouse_ticks import get_tick_client
from opening_strength_fit.feature_utils import finite_numeric as finite
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

GROUP_COL = "decision_target_timestamp"
TOP_N = 100
TRADABLE_STATUSES = {"T0", "20", "TRADE"}
SYMBOL_REGEX = r"^(?:(?:00|30)\d{4}\.SZ|(?:60|68)\d{4}\.SH)$"
PREDICTION_COLUMNS = [
    "date",
    "symbol",
    GROUP_COL,
    "decision_time",
    "prediction",
    "label",
    "valid_label",
    "status",
    "entry_status",
    "ask_price_1",
    "ask_volume_1",
    "bid_price_1",
    "bid_volume_1",
    "entry_ask_price_1",
    "entry_ask_volume_1",
    "buy_price",
]
SNAPSHOT_COLUMNS = [
    "date",
    "symbol",
    GROUP_COL,
    "high_price",
    "last_price",
]
NEXT_COLUMNS = ["date", "symbol", GROUP_COL, "alpha_return_next_close"]
CONDITION_COLUMNS = [
    "decision_no_offer",
    "decision_at_limit",
    "decision_at_limit_with_offer",
    "decision_sealed_at_limit",
    "touched_limit_by_decision",
    "touched_limit_but_opened",
    "decision_within_10bps",
    "decision_within_25bps",
    "decision_within_50bps",
    "decision_within_100bps",
    "decision_within_200bps",
    "entry_no_offer",
    "entry_at_limit_with_offer",
    "entry_within_50bps",
    "entry_within_100bps",
    "entry_within_200bps",
    "daily_closes_up_limit",
    "daily_up_limit_not_current_within_100bps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit point-in-time limit-up exposure in opening-model Top100 selections."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labeled-cache", required=True)
    parser.add_argument("--next-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pool", default="L", choices=["L"])
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def read_filtered_parquet(
    path: Path,
    *,
    columns: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=columns,
        filters=[("date", ">=", start_date), ("date", "<=", end_date)],
    )


def daily_reference(
    client: object,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    query = """
select
    toString(TradingDay) as date,
    Symbol as symbol,
    PreClosePrice as prev_close,
    ClosePrice as daily_close,
    HighestPrice as daily_high,
    STStatus as st_status,
    UpdownLimitStatus as updown_limit_status,
    TradeStatus as daily_trade_status
from stock.daily_bar_jy
where TradingDay between {start_date:Date} and {end_date:Date}
  and match(Symbol, {symbol_regex:String})
"""
    return client.query_df(
        query,
        parameters={
            "start_date": start_date,
            "end_date": end_date,
            "symbol_regex": SYMBOL_REGEX,
        },
    )


def add_limit_reference(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    symbol = out["symbol"].astype(str)
    prev_close = finite(out["prev_close"])
    st = finite(out["st_status"]).fillna(0).ne(0)
    wide_board = symbol.str.startswith("30") | symbol.str.startswith("68")
    ratio = np.where(st, 0.05, np.where(wide_board, 0.20, 0.10))
    standard_upper = np.floor(prev_close.to_numpy() * (1.0 + ratio) * 100.0 + 0.50000001) / 100.0
    standard_upper = pd.Series(standard_upper, index=out.index, dtype="float64")

    daily_close = finite(out["daily_close"])
    daily_high = finite(out["daily_high"])
    closes_up = finite(out["updown_limit_status"]).eq(1)
    upper = standard_upper.where(~closes_up, daily_close)

    # IPO/relisting days can have no standard price limit. If the observed daily high
    # exceeds the rule-based upper bound by more than one tick, leave the reference unknown.
    nonstandard = ~closes_up & daily_high.gt(standard_upper + 0.011)
    out["upper_limit_price"] = upper.mask(nonstandard)
    out["daily_closes_up_limit"] = closes_up
    out["standard_limit_reference"] = ~nonstandard & upper.notna()
    return out


def positive(values: pd.Series) -> pd.Series:
    return finite(values).gt(0)


def add_point_in_time_states(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    upper = finite(out["upper_limit_price"])
    ask = finite(out["ask_price_1"])
    ask_volume = finite(out["ask_volume_1"])
    bid = finite(out["bid_price_1"])
    last = finite(out["last_price"])
    high = finite(out["high_price"])
    entry_ask = finite(out["entry_ask_price_1"])
    entry_volume = finite(out["entry_ask_volume_1"])
    buy = finite(out["buy_price"])
    status = out["status"].astype("string")
    entry_status = out["entry_status"].astype("string")

    decision_offer = positive(ask) & positive(ask_volume) & status.isin(TRADABLE_STATUSES)
    entry_offer = (
        positive(entry_ask) & positive(entry_volume) & entry_status.isin(TRADABLE_STATUSES)
    )
    state_price = last.where(positive(last), bid.where(positive(bid), ask))
    executable_quote = ask.where(decision_offer, state_price)
    entry_price = entry_ask.where(entry_offer, buy.where(positive(buy), np.nan))

    out["decision_offer"] = decision_offer
    out["entry_offer"] = entry_offer
    out["decision_state_price"] = state_price
    out["decision_executable_quote"] = executable_quote
    out["entry_executable_price"] = entry_price
    out["decision_limit_room_bps"] = (upper - executable_quote) / executable_quote * 10_000.0
    out["entry_limit_room_bps"] = (upper - entry_price) / entry_price * 10_000.0

    decision_at_limit = (
        upper.notna() & state_price.notna() & state_price.sub(upper).abs().le(0.0051)
    )
    entry_at_limit = upper.notna() & entry_price.notna() & entry_price.sub(upper).abs().le(0.0051)
    touched = upper.notna() & high.notna() & high.ge(upper - 0.0051)
    decision_room = finite(out["decision_limit_room_bps"])
    entry_room = finite(out["entry_limit_room_bps"])

    out["decision_no_offer"] = ~decision_offer & status.isin(TRADABLE_STATUSES)
    out["decision_at_limit"] = decision_at_limit
    out["decision_at_limit_with_offer"] = decision_at_limit & decision_offer
    out["decision_sealed_at_limit"] = decision_at_limit & ~decision_offer
    out["touched_limit_by_decision"] = touched
    out["touched_limit_but_opened"] = touched & ~decision_at_limit
    for threshold in (10, 25, 50, 100, 200):
        out[f"decision_within_{threshold}bps"] = decision_room.between(-1.0, threshold)
    out["entry_no_offer"] = ~entry_offer & entry_status.isin(TRADABLE_STATUSES)
    out["entry_at_limit_with_offer"] = entry_at_limit & entry_offer
    for threshold in (50, 100, 200):
        out[f"entry_within_{threshold}bps"] = entry_room.between(-1.0, threshold)
    out["daily_closes_up_limit"] = out["daily_closes_up_limit"].fillna(False).astype(bool)
    out["daily_up_limit_not_current_within_100bps"] = (
        out["daily_closes_up_limit"] & ~out["decision_within_100bps"]
    )
    for column in CONDITION_COLUMNS:
        out[column] = out[column].fillna(False).astype(bool)
    return out


def group_base(work: pd.DataFrame) -> pd.DataFrame:
    return (
        work.groupby(GROUP_COL, sort=False)
        .agg(
            candidate_rows=("prediction", "size"),
            pool_next_sum=("alpha_return_next_close", "sum"),
            pool_next_count=("alpha_return_next_close", "count"),
            pool_short_sum=("label", "sum"),
            pool_short_count=("label", "count"),
        )
        .reset_index()
    )


def selected_summary(selected: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return (
        selected.groupby(GROUP_COL, sort=False)
        .agg(
            **{
                f"{prefix}_rows": ("prediction", "size"),
                f"{prefix}_next_sum": ("alpha_return_next_close", "sum"),
                f"{prefix}_next_count": ("alpha_return_next_close", "count"),
                f"{prefix}_short_sum": ("label", "sum"),
                f"{prefix}_short_count": ("label", "count"),
            }
        )
        .reset_index()
    )


def safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.div(right.where(right.ne(0)))


def analyze_scope(
    frame: pd.DataFrame,
    *,
    scope: str,
    top_n: int,
) -> pd.DataFrame:
    required = ["prediction", "alpha_return_next_close"]
    if scope == "published_top100":
        required.append("label")
    work = frame.dropna(subset=required).copy()
    if work.empty:
        return pd.DataFrame()
    work = work.sort_values(
        [GROUP_COL, "prediction", "symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    base = group_base(work)
    original = work.groupby(GROUP_COL, sort=False).head(top_n)
    base = base.merge(selected_summary(original, "original"), on=GROUP_COL, how="left")
    outputs: list[pd.DataFrame] = []

    for condition in CONDITION_COLUMNS:
        candidate_condition = (
            work.loc[work[condition]]
            .groupby(GROUP_COL, sort=False)
            .agg(
                candidate_condition_rows=("prediction", "size"),
                candidate_condition_next_sum=("alpha_return_next_close", "sum"),
                candidate_condition_next_count=("alpha_return_next_close", "count"),
            )
            .reset_index()
        )
        selected_condition = (
            original.loc[original[condition]]
            .groupby(GROUP_COL, sort=False)
            .agg(
                selected_condition_rows=("prediction", "size"),
                selected_condition_next_sum=("alpha_return_next_close", "sum"),
                selected_condition_next_count=("alpha_return_next_close", "count"),
            )
            .reset_index()
        )
        filtered = work.loc[~work[condition]]
        filtered_base = (
            filtered.groupby(GROUP_COL, sort=False)
            .agg(
                filtered_candidate_rows=("prediction", "size"),
                filtered_pool_next_sum=("alpha_return_next_close", "sum"),
                filtered_pool_next_count=("alpha_return_next_close", "count"),
            )
            .reset_index()
        )
        reselected = filtered.groupby(GROUP_COL, sort=False).head(top_n)
        reselected_summary = selected_summary(reselected, "reselected")
        metrics = (
            base.merge(candidate_condition, on=GROUP_COL, how="left")
            .merge(selected_condition, on=GROUP_COL, how="left")
            .merge(filtered_base, on=GROUP_COL, how="left")
            .merge(reselected_summary, on=GROUP_COL, how="left")
        )
        count_columns = [
            column for column in metrics if column.endswith("_rows") or column.endswith("_count")
        ]
        metrics[count_columns] = metrics[count_columns].fillna(0)
        sum_columns = [column for column in metrics if column.endswith("_sum")]
        metrics[sum_columns] = metrics[sum_columns].fillna(0.0)
        metrics["candidate_condition_share"] = safe_divide(
            metrics["candidate_condition_rows"], metrics["candidate_rows"]
        )
        metrics["selected_condition_share"] = safe_divide(
            metrics["selected_condition_rows"], metrics["original_rows"]
        )
        metrics["pool_next_bps"] = (
            safe_divide(metrics["pool_next_sum"], metrics["pool_next_count"]) * 10_000.0
        )
        metrics["original_top_next_bps"] = (
            safe_divide(metrics["original_next_sum"], metrics["original_next_count"]) * 10_000.0
        )
        metrics["original_excess_bps"] = metrics["original_top_next_bps"] - metrics["pool_next_bps"]
        metrics["selected_condition_next_bps"] = (
            safe_divide(
                metrics["selected_condition_next_sum"],
                metrics["selected_condition_next_count"],
            )
            * 10_000.0
        )
        metrics["selected_condition_contribution_bps"] = (
            safe_divide(metrics["selected_condition_next_sum"], metrics["original_rows"]) * 10_000.0
        )
        metrics["filtered_pool_next_bps"] = (
            safe_divide(
                metrics["filtered_pool_next_sum"],
                metrics["filtered_pool_next_count"],
            )
            * 10_000.0
        )
        metrics["reselected_top_next_bps"] = (
            safe_divide(metrics["reselected_next_sum"], metrics["reselected_next_count"]) * 10_000.0
        )
        metrics["reselected_excess_vs_original_pool_bps"] = (
            metrics["reselected_top_next_bps"] - metrics["pool_next_bps"]
        )
        metrics["reselected_excess_vs_filtered_pool_bps"] = (
            metrics["reselected_top_next_bps"] - metrics["filtered_pool_next_bps"]
        )
        metrics["condition"] = condition
        metrics["scope"] = scope
        metrics["date"] = pd.to_datetime(metrics[GROUP_COL], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        metrics["clock"] = pd.to_datetime(metrics[GROUP_COL], errors="coerce").dt.strftime("%H:%M")
        metrics["year"] = metrics["date"].str.slice(0, 4)
        outputs.append(metrics)
    return pd.concat(outputs, ignore_index=True)


def selected_event_rows(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    work = frame.dropna(subset=["prediction", "label", "alpha_return_next_close"]).copy()
    work = work.sort_values(
        [GROUP_COL, "prediction", "symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = work.groupby(GROUP_COL, sort=False).head(top_n).copy()
    selected["score_rank"] = selected.groupby(GROUP_COL, sort=False).cumcount() + 1
    event = selected[
        [
            "decision_at_limit",
            "decision_sealed_at_limit",
            "touched_limit_by_decision",
            "decision_within_200bps",
            "entry_at_limit_with_offer",
            "entry_within_200bps",
            "daily_closes_up_limit",
        ]
    ].any(axis=1)
    columns = [
        "date",
        "symbol",
        GROUP_COL,
        "decision_time",
        "score_rank",
        "prediction",
        "label",
        "alpha_return_next_close",
        "status",
        "entry_status",
        "last_price",
        "high_price",
        "ask_price_1",
        "ask_volume_1",
        "bid_price_1",
        "bid_volume_1",
        "entry_ask_price_1",
        "entry_ask_volume_1",
        "buy_price",
        "upper_limit_price",
        "decision_limit_room_bps",
        "entry_limit_room_bps",
        *CONDITION_COLUMNS,
    ]
    return selected.loc[event, columns]


def aggregate_metrics(group_metrics: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in group_metrics.groupby(by, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(by, keys, strict=True))
        candidate_share = group["candidate_condition_rows"].sum() / max(
            group["candidate_rows"].sum(), 1
        )
        selected_share = group["selected_condition_rows"].sum() / max(
            group["original_rows"].sum(), 1
        )
        selected_condition_count = group["selected_condition_next_count"].sum()
        row.update(
            {
                "groups": int(group[GROUP_COL].nunique()),
                "candidate_rows_mean": float(group["candidate_rows"].mean()),
                "selected_rows_mean": float(group["original_rows"].mean()),
                "candidate_condition_count_mean": float(group["candidate_condition_rows"].mean()),
                "selected_condition_count_mean": float(group["selected_condition_rows"].mean()),
                "candidate_condition_share_pct": candidate_share * 100.0,
                "selected_condition_share_pct": selected_share * 100.0,
                "selection_enrichment_x": (
                    selected_share / candidate_share if candidate_share > 0 else np.nan
                ),
                "selected_condition_next_bps": (
                    group["selected_condition_next_sum"].sum() / selected_condition_count * 10_000.0
                    if selected_condition_count > 0
                    else np.nan
                ),
                "selected_condition_contribution_bps": float(
                    group["selected_condition_contribution_bps"].mean()
                ),
                "pool_next_bps": float(group["pool_next_bps"].mean()),
                "original_top_next_bps": float(group["original_top_next_bps"].mean()),
                "original_excess_bps": float(group["original_excess_bps"].mean()),
                "reselected_top_next_bps": float(group["reselected_top_next_bps"].mean()),
                "reselected_excess_vs_original_pool_bps": float(
                    group["reselected_excess_vs_original_pool_bps"].mean()
                ),
                "reselected_excess_vs_filtered_pool_bps": float(
                    group["reselected_excess_vs_filtered_pool_bps"].mean()
                ),
            }
        )
        row["reselected_excess_delta_vs_original_bps"] = (
            row["reselected_excess_vs_original_pool_bps"] - row["original_excess_bps"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_files(root: Path) -> list[Path]:
    return sorted(root.glob("month_*/predictions.parquet"))


def date_bounds(path: Path) -> tuple[str, str]:
    values = pd.read_parquet(path, columns=["date"])["date"].astype(str)
    return values.min(), values.max()


def year_path(root: Path, year: str, pattern: str) -> Path:
    matches = sorted(root.glob(pattern.format(year=year)))
    if len(matches) != 1:
        raise SystemExit(f"expected one {year=} match below {root}, found {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_root = Path(args.predictions)
    labeled_root = Path(args.labeled_cache)
    next_root = Path(args.next_labels)
    files = prediction_files(predictions_root)
    if not files:
        raise SystemExit(f"no prediction files under {predictions_root}")

    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS[args.pool])
    client = get_tick_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 8123)),
        username=os.environ["CLICKHOUSE_USER"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
    )
    metric_parts: list[pd.DataFrame] = []
    selected_event_parts: list[pd.DataFrame] = []
    trace_files: list[dict[str, object]] = []

    for prediction_path in files:
        start_date, end_date = date_bounds(prediction_path)
        year = start_date[:4]
        print(
            f"processing {prediction_path.parent.name}: {start_date}..{end_date}",
            flush=True,
        )
        predictions = read_filtered_parquet(
            prediction_path,
            columns=PREDICTION_COLUMNS,
            start_date=start_date,
            end_date=end_date,
        )
        predictions["date"] = predictions["date"].astype(str)
        predictions[GROUP_COL] = pd.to_datetime(predictions[GROUP_COL], errors="coerce")
        pool_mask = stock_pool_membership_mask(predictions, pool, date_lag_sessions=0)
        predictions = predictions.loc[pool_mask].copy()

        snapshot_path = year_path(
            labeled_root,
            year,
            "opening_{year}_label_v4_clock6_state_unique_mixed_w030_mcap_lag1.parquet",
        )
        snapshot = read_filtered_parquet(
            snapshot_path,
            columns=SNAPSHOT_COLUMNS,
            start_date=start_date,
            end_date=end_date,
        )
        snapshot["date"] = snapshot["date"].astype(str)
        snapshot[GROUP_COL] = pd.to_datetime(snapshot[GROUP_COL], errors="coerce")
        snapshot = snapshot.drop_duplicates(["date", "symbol", GROUP_COL], keep="last")

        next_path = year_path(
            next_root,
            year,
            "opening_{year}_next_close_labels_v1.parquet",
        )
        next_labels = read_filtered_parquet(
            next_path,
            columns=NEXT_COLUMNS,
            start_date=start_date,
            end_date=end_date,
        )
        next_labels["date"] = next_labels["date"].astype(str)
        next_labels[GROUP_COL] = pd.to_datetime(next_labels[GROUP_COL], errors="coerce")
        next_labels = next_labels.drop_duplicates(["date", "symbol", GROUP_COL], keep="last")

        daily = daily_reference(client, start_date=start_date, end_date=end_date)
        daily["date"] = daily["date"].astype(str)
        daily = add_limit_reference(daily)
        daily = daily.drop_duplicates(["date", "symbol"], keep="last")

        frame = (
            predictions.merge(
                snapshot,
                on=["date", "symbol", GROUP_COL],
                how="left",
                validate="one_to_one",
            )
            .merge(
                next_labels,
                on=["date", "symbol", GROUP_COL],
                how="left",
                validate="one_to_one",
            )
            .merge(daily, on=["date", "symbol"], how="left", validate="many_to_one")
        )
        frame = add_point_in_time_states(frame)
        selected_event_parts.append(selected_event_rows(frame, top_n=args.top_n))
        for scope in ("published_top100", "scoreable_pool"):
            metrics = analyze_scope(frame, scope=scope, top_n=args.top_n)
            if not metrics.empty:
                metric_parts.append(metrics)
        trace_files.append(
            {
                "prediction_file": str(prediction_path),
                "start_date": start_date,
                "end_date": end_date,
                "prediction_rows": int(len(predictions)),
                "snapshot_missing_rows": int(frame["last_price"].isna().sum()),
                "next_label_missing_rows": int(frame["alpha_return_next_close"].isna().sum()),
                "upper_limit_missing_rows": int(frame["upper_limit_price"].isna().sum()),
                "published_eligible_rows": int(
                    frame[["prediction", "label", "alpha_return_next_close"]]
                    .notna()
                    .all(axis=1)
                    .sum()
                ),
                "scoreable_rows": int(
                    frame[["prediction", "alpha_return_next_close"]].notna().all(axis=1).sum()
                ),
            }
        )
        del predictions, snapshot, next_labels, daily, frame
        gc.collect()

    group_metrics = pd.concat(metric_parts, ignore_index=True)
    overall = aggregate_metrics(group_metrics, ["scope", "condition"])
    by_clock = aggregate_metrics(group_metrics, ["scope", "condition", "clock"])
    by_year = aggregate_metrics(group_metrics, ["scope", "condition", "year"])
    by_month = aggregate_metrics(
        group_metrics.assign(month=group_metrics["date"].str.slice(0, 7)),
        ["scope", "condition", "month"],
    )
    selected_events = pd.concat(selected_event_parts, ignore_index=True)
    group_metrics.to_parquet(output_dir / "opening_limit_group_metrics.parquet", index=False)
    selected_events.to_parquet(
        output_dir / "opening_limit_selected_events.parquet",
        index=False,
    )
    overall.to_csv(output_dir / "opening_limit_overall.csv", index=False)
    by_clock.to_csv(output_dir / "opening_limit_by_clock.csv", index=False)
    by_year.to_csv(output_dir / "opening_limit_by_year.csv", index=False)
    by_month.to_csv(output_dir / "opening_limit_by_month.csv", index=False)
    trace = {
        "predictions": str(predictions_root),
        "labeled_cache": str(labeled_root),
        "next_labels": str(next_root),
        "pool": args.pool,
        "pool_date_lag_sessions": 0,
        "top_n": args.top_n,
        "scopes": {
            "published_top100": (
                "Exact published evaluation eligibility: prediction, one-minute execution "
                "label, and next-close label are all finite."
            ),
            "scoreable_pool": (
                "Persisted model-score universe: prediction and next-close evaluation label "
                "are finite; one-minute label is not explicitly used here. In these artifacts "
                "every persisted pool-L prediction already has a valid one-minute label, so "
                "this scope is identical to published_top100 and is not a counterfactual "
                "inference pass over invalid-label rows."
            ),
        },
        "definitions": {
            "decision_offer": (
                "Decision-snapshot ask1 price and volume are positive and status is tradable."
            ),
            "entry_offer": (
                "Clock+6s entry ask1 price and volume are positive and entry status is tradable."
            ),
            "upper_limit_price": (
                "ST=5%, ChiNext/STAR=20%, other supported A shares=10%, rounded to one fen; "
                "overridden by exact daily close when UpdownLimitStatus=1. Nonstandard "
                "IPO/relisting rows are marked unknown."
            ),
            "at_limit": "State or executable entry price equals upper limit within half a tick.",
            "touched_limit_by_decision": (
                "Cumulative intraday high observed at the decision snapshot has reached the "
                "upper limit."
            ),
        },
        "files": trace_files,
    }
    (output_dir / "opening_limit_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "_SUCCESS").touch()
    print(f"wrote audit to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
