from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.raw_source import read_daily_limit_flags
from opening_strength_fit.schema import normalize_date_series as _date
from opening_strength_fit.schema import normalize_decision_keys_preserving_rows as _normalize
from opening_strength_fit.schema import normalize_text_series as _text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

ROOT = Path("/mnt/output/opening_strength_fit")
KEYS = ["date", "symbol", "decision_target_timestamp"]
GROUPS = ["date", "decision_target_timestamp"]
TOP_N = 100
EXCLUDED_DATES = {"2026-03-18", "2026-04-22", "2026-05-06"}
TRADABLE_STATUSES = {"T0", "20", "TRADE"}


def iter_common_horizon_predictions(
    *,
    model_root: Path,
    raw_source_root: Path,
    window: str,
    horizons: Sequence[str],
    outcome_columns: Sequence[str],
    required_outcomes: Sequence[str],
    prepare_outcome: Callable[[pd.DataFrame], pd.DataFrame],
    expected_folds: int = 8,
) -> Iterator[tuple[int, str, pd.DataFrame]]:
    """Yield Pool-L prediction/outcome intersections for each fold and horizon."""

    join_keys = list(KEYS)
    files = {
        horizon: {
            path.parent.name: path
            for path in sorted(
                (model_root / f"w{window}_h{horizon}").glob("month_*/predictions.parquet")
            )
        }
        for horizon in horizons
    }
    if any(len(paths) != expected_folds for paths in files.values()):
        raise SystemExit(f"each horizon must have {expected_folds} prediction files")

    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    references: dict[int, pd.DataFrame] = {}
    for fold_index, fold_name in enumerate(sorted(files["close"]), start=1):
        predictions: dict[str, pd.DataFrame] = {}
        for horizon in horizons:
            frame = _normalize(
                pd.read_parquet(files[horizon][fold_name], columns=[*join_keys, "prediction"])
            )
            frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
            frame = frame.dropna(subset=[*join_keys, "prediction"])
            predictions[horizon] = frame.loc[
                stock_pool_membership_mask(frame, pool, date_lag_sessions=0)
            ].copy()

        outcome = prepare_outcome(
            _normalize(
                pd.read_parquet(
                    files["close"][fold_name],
                    columns=[*join_keys, *outcome_columns],
                )
            )
        )
        year = int(outcome["date"].dropna().iloc[0][:4])
        references.setdefault(
            year,
            read_daily_limit_flags(
                raw_source_root,
                year,
                output_column="daily_closes_up_limit",
            ),
        )
        outcome = outcome.merge(
            references[year], on=["date", "symbol"], how="left", validate="many_to_one"
        )

        common = predictions[horizons[0]][join_keys]
        for horizon in horizons[1:]:
            common = common.merge(
                predictions[horizon][join_keys],
                on=join_keys,
                how="inner",
                validate="one_to_one",
            )
        common = common.merge(outcome, on=join_keys, how="inner", validate="one_to_one")
        common = common.dropna(subset=[*required_outcomes, "daily_closes_up_limit"])
        common["daily_closes_up_limit"] = common["daily_closes_up_limit"].astype(bool)
        for horizon in horizons:
            yield (
                fold_index,
                horizon,
                predictions[horizon].merge(
                    common, on=join_keys, how="inner", validate="one_to_one"
                ),
            )


def load_prediction(run_id: str) -> pd.DataFrame:
    path = ROOT / "nn/holdout" / run_id / "predictions_unfiltered.parquet"
    if not path.exists():
        raise SystemExit(f"missing strict holdout predictions: {path}")
    frame = _normalize(pd.read_parquet(path, columns=[*KEYS, "prediction", "label_short"]))
    frame = frame.loc[~frame["date"].isin(EXCLUDED_DATES)].copy()
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame["own_label"] = pd.to_numeric(frame.pop("label_short"), errors="coerce")
    return frame.dropna(subset=[*KEYS, "prediction"])


def _prediction(model: str) -> pd.DataFrame:
    return load_prediction(f"nn_ds350_w0931_h{model}_train2023_2025_test2026h1_purge1_v1")


def _market_reference() -> pd.DataFrame:
    raw = ROOT / "cache/opening_0931_0940_raw_source/year=2026"
    daily = pd.read_parquet(
        raw / "daily_reference.parquet",
        columns=[
            "TradingDay",
            "Symbol",
            "OpenPrice",
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
            "OpenPrice": "official_open",
            "ClosePrice": "official_close",
            "PreClosePrice": "prev_close",
            "STStatus": "st_status",
            "TradeStatus": "trade_status",
            "UpdownLimitStatus": "updown_limit_status",
        }
    )
    daily["date"] = _date(daily["date"])
    daily["symbol"] = _text(daily["symbol"])
    for column in (
        "official_open",
        "official_close",
        "prev_close",
        "st_status",
        "updown_limit_status",
    ):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["trade_status"] = _text(daily["trade_status"]).str.upper()
    daily["final_up_limit"] = daily["updown_limit_status"].eq(1)
    symbol = daily["symbol"].astype(str)
    ratio = np.where(
        daily["st_status"].fillna(0).ne(0),
        0.05,
        np.where(symbol.str.startswith("30") | symbol.str.startswith("68"), 0.20, 0.10),
    )
    standard_limit = (
        np.floor(daily["prev_close"].to_numpy() * (1.0 + ratio) * 100.0 + 0.50000001) / 100.0
    )
    daily["upper_limit_price"] = pd.Series(standard_limit, index=daily.index).where(
        ~daily["final_up_limit"], daily["official_close"]
    )
    daily.loc[daily["trade_status"].isin({"NEW", "新股", "N"}), "upper_limit_price"] = np.nan
    daily = daily.drop_duplicates(["date", "symbol"], keep="last")

    tick_close = pd.read_parquet(
        raw / "close_reference.parquet",
        columns=["TradingDay", "Symbol", "ClosePrice"],
    ).rename(columns={"TradingDay": "date", "Symbol": "symbol", "ClosePrice": "tick_close"})
    tick_close["date"] = _date(tick_close["date"])
    tick_close["symbol"] = _text(tick_close["symbol"])
    tick_close["tick_close"] = pd.to_numeric(tick_close["tick_close"], errors="coerce")
    tick_close = tick_close.drop_duplicates(["date", "symbol"], keep="last")
    daily = daily.merge(tick_close, on=["date", "symbol"], how="left", validate="one_to_one")

    calendar = sorted(daily["date"].dropna().unique())
    next_date = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
    next_state = daily[["date", "symbol", "official_open", "tick_close"]].rename(
        columns={
            "date": "next_date",
            "official_open": "next_open",
            "tick_close": "next_close",
        }
    )
    daily["next_date"] = daily["date"].map(next_date)
    daily = daily.merge(next_state, on=["next_date", "symbol"], how="left", validate="many_to_one")
    return daily[
        [
            "date",
            "symbol",
            "prev_close",
            "tick_close",
            "upper_limit_price",
            "final_up_limit",
            "next_open",
            "next_close",
        ]
    ]


def _close_labels() -> pd.DataFrame:
    path = ROOT / "datasets/opening_0931_0940_labels_hclose_v1/year=2026/labels.parquet"
    out = _normalize(pd.read_parquet(path, columns=[*KEYS, "label_short"]))
    out["return_close"] = pd.to_numeric(out.pop("label_short"), errors="coerce")
    return out.drop_duplicates(KEYS, keep="last")


def _rank_ic(frame: pd.DataFrame, outcome: str) -> float:
    work = frame.dropna(subset=["prediction", outcome]).copy()
    work["x"] = work.groupby(GROUPS, sort=False)["prediction"].rank(method="average", pct=True)
    work["y"] = work.groupby(GROUPS, sort=False)[outcome].rank(method="average", pct=True)
    values = work.groupby(GROUPS, sort=False).apply(
        lambda group: group["x"].corr(group["y"]), include_groups=False
    )
    return float(values.mean())


def _top(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        [*GROUPS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    return ordered.groupby(GROUPS, sort=False).head(TOP_N).copy()


def attach_market_outcomes(
    predictions: pd.DataFrame,
    close_labels: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    frame = predictions.merge(close_labels, on=KEYS, how="left", validate="one_to_one")
    frame = frame.merge(reference, on=["date", "symbol"], how="left", validate="many_to_one")
    frame["final_up_limit"] = frame["final_up_limit"].fillna(False).astype(bool)
    for output, numerator, denominator in (
        ("close_to_next_open", "next_open", "tick_close"),
        ("close_to_next_close", "next_close", "tick_close"),
        ("next_open_to_next_close", "next_close", "next_open"),
    ):
        valid = frame[numerator].gt(0) & frame[denominator].gt(0)
        frame[output] = (frame[numerator] / frame[denominator] - 1.0).where(valid)
    return frame


def _metrics(frame: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    candidate_close = frame.dropna(subset=["return_close"])
    selected_close = selected.dropna(subset=["return_close"])
    pool_group = candidate_close.groupby(GROUPS, sort=False)["return_close"].mean()
    selected_group = selected_close.groupby(GROUPS, sort=False)["return_close"].mean()
    excess = (selected_group - pool_group).dropna() * 10_000.0

    candidate_limit_sum = (
        candidate_close.loc[candidate_close["final_up_limit"]]
        .groupby(GROUPS, sort=False)["return_close"]
        .sum()
    )
    selected_limit_sum = (
        selected_close.loc[selected_close["final_up_limit"]]
        .groupby(GROUPS, sort=False)["return_close"]
        .sum()
    )
    candidate_count = candidate_close.groupby(GROUPS, sort=False)["return_close"].size()
    selected_count = selected_close.groupby(GROUPS, sort=False)["return_close"].size()
    index = excess.index
    limit_contribution = (
        selected_limit_sum.reindex(index).fillna(0.0) / selected_count.reindex(index)
        - candidate_limit_sum.reindex(index).fillna(0.0) / candidate_count.reindex(index)
    ) * 10_000.0
    nonlimit_contribution = excess - limit_contribution

    nonlimit = frame.loc[~frame["final_up_limit"]].copy()
    nonlimit_top = _top(nonlimit).dropna(subset=["return_close"])
    nonlimit_pool = (
        nonlimit.dropna(subset=["return_close"]).groupby(GROUPS, sort=False)["return_close"].mean()
    )
    nonlimit_selected = nonlimit_top.groupby(GROUPS, sort=False)["return_close"].mean()

    result: dict[str, object] = {
        "groups": int(frame.groupby(GROUPS, sort=False).ngroups),
        "candidate_limit_share_pct": float(frame["final_up_limit"].mean() * 100.0),
        "selected_limit_share_pct": float(selected["final_up_limit"].mean() * 100.0),
        "limit_enrichment_x": float(
            selected["final_up_limit"].mean() / frame["final_up_limit"].mean()
        ),
        "own_label_rank_ic": _rank_ic(frame, "own_label"),
        "same_day_close_rank_ic": _rank_ic(frame, "return_close"),
        "same_day_close_excess_bps": float(excess.mean()),
        "same_day_close_limit_contribution_bps": float(limit_contribution.mean()),
        "same_day_close_nonlimit_contribution_bps": float(nonlimit_contribution.mean()),
        "same_day_close_no_limit_reselect_excess_bps": float(
            (nonlimit_selected - nonlimit_pool).dropna().mean() * 10_000.0
        ),
    }
    for name in ("close_to_next_open", "next_open_to_next_close", "close_to_next_close"):
        candidate_valid = frame.dropna(subset=[name])
        selected_valid = selected.dropna(subset=[name])
        candidate_mean = candidate_valid.groupby(GROUPS, sort=False)[name].mean()
        selected_mean = selected_valid.groupby(GROUPS, sort=False)[name].mean()
        result[f"{name}_selected_raw_bps"] = float(selected_mean.mean() * 10_000.0)
        result[f"{name}_excess_bps"] = float(
            (selected_mean - candidate_mean).dropna().mean() * 10_000.0
        )
    return result


def sample_tick_states(
    keys: pd.DataFrame,
    *,
    columns: list[str],
    delay_seconds: int,
    max_workers: int,
    progress_label: str,
) -> pd.DataFrame:
    raw_root = ROOT / "cache/opening_0931_0940_raw_source/year=2026/ticks"
    items = [(date, wanted.copy()) for date, wanted in keys.groupby("date", sort=True)]

    def read_day(date: str, wanted: pd.DataFrame) -> pd.DataFrame:
        path = raw_root / f"date={date}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=[*KEYS, *columns, "raw_state_age_seconds"])
        symbols = sorted(set(wanted["symbol"]))
        ticks = pd.read_parquet(
            path,
            columns=["Symbol", "ExchTimeOffsetUs", *columns],
            filters=[("Symbol", "in", symbols)],
        )
        ticks["Symbol"] = _text(ticks["Symbol"])
        ticks = ticks.loc[ticks["Symbol"].isin(symbols)].copy()
        ticks["ExchTimeOffsetUs"] = pd.to_numeric(ticks["ExchTimeOffsetUs"], errors="coerce")
        ticks = ticks.dropna(subset=["ExchTimeOffsetUs"]).sort_values(
            ["Symbol", "ExchTimeOffsetUs"], kind="mergesort"
        )
        rows: list[pd.DataFrame] = []
        for symbol, part in wanted.groupby("symbol", sort=False):
            state = ticks.loc[ticks["Symbol"].eq(symbol)]
            if state.empty:
                continue
            offsets = state["ExchTimeOffsetUs"].to_numpy(dtype="int64")
            targets = (
                (
                    part["decision_target_timestamp"]
                    - part["decision_target_timestamp"].dt.normalize()
                )
                / pd.Timedelta(microseconds=1)
            ).to_numpy(dtype="int64") + delay_seconds * 1_000_000
            positions = np.searchsorted(offsets, targets, side="right") - 1
            valid = positions >= 0
            if not valid.any():
                continue
            matched = state.iloc[positions[valid]]
            out = part.loc[valid, KEYS].copy()
            for column in columns:
                out[column] = matched[column].to_numpy()
            out["raw_state_age_seconds"] = (
                targets[valid] - offsets[positions[valid]]
            ) / 1_000_000.0
            rows.append(out)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    rows: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_day, date, wanted): date for date, wanted in items}
        for index, future in enumerate(as_completed(futures), start=1):
            part = future.result()
            if not part.empty:
                rows.append(part)
            print(f"{progress_label} progress {index}/{len(items)} {futures[future]}", flush=True)
    return pd.concat(rows, ignore_index=True).drop_duplicates(KEYS, keep="last")


def _raw_entries(keys: pd.DataFrame) -> pd.DataFrame:
    out = sample_tick_states(
        keys,
        columns=["AskPrice1", "AskVolume1", "Status"],
        delay_seconds=6,
        max_workers=4,
        progress_label="raw",
    ).rename(
        columns={
            "AskPrice1": "raw_entry_ask",
            "AskVolume1": "raw_entry_ask_volume",
            "Status": "raw_entry_status",
            "raw_state_age_seconds": "raw_entry_state_age_seconds",
        }
    )
    out["raw_entry_ask"] = pd.to_numeric(out["raw_entry_ask"], errors="coerce")
    out["raw_entry_ask_volume"] = pd.to_numeric(out["raw_entry_ask_volume"], errors="coerce")
    out["raw_entry_status"] = _text(out["raw_entry_status"]).str.upper()
    return out


def _tradeability(selected: pd.DataFrame) -> dict[str, object]:
    final = selected.loc[selected["final_up_limit"]].copy()
    raw_offer = (
        final["raw_entry_ask"].gt(0)
        & final["raw_entry_ask_volume"].gt(0)
        & final["raw_entry_status"].isin(TRADABLE_STATUSES)
    )
    final["raw_offer_valid"] = raw_offer
    final["entry_return_vs_prev_close_pct"] = (
        final["raw_entry_ask"] / final["prev_close"] - 1.0
    ) * 100.0
    final["entry_room_to_limit_pct"] = (
        final["upper_limit_price"] / final["raw_entry_ask"] - 1.0
    ) * 100.0
    final["entry_ask_notional"] = final["raw_entry_ask"] * final["raw_entry_ask_volume"]
    valid = final.loc[raw_offer].copy()

    def q(column: str, quantile: float) -> float:
        return float(valid[column].quantile(quantile))

    reconstructed_entry = final["tick_close"] / (1.0 + final["return_close"])
    label_available = final["return_close"].notna()
    return {
        "selected_final_limit_rows": int(len(final)),
        "close_label_available_pct": float(final["return_close"].notna().mean() * 100.0),
        "raw_entry_found_pct": float(final["raw_entry_ask"].notna().mean() * 100.0),
        "raw_entry_offer_valid_pct": float(raw_offer.mean() * 100.0),
        "raw_offer_valid_given_close_label_pct": float(
            raw_offer.loc[label_available].mean() * 100.0
        ),
        "close_label_available_but_raw_offer_invalid_rows": int(
            (label_available & ~raw_offer).sum()
        ),
        "raw_matches_label_entry_within_half_cent_pct": float(
            final["raw_entry_ask"].sub(reconstructed_entry).abs().le(0.0051).mean() * 100.0
        ),
        "raw_entry_at_limit_pct": float(
            valid["raw_entry_ask"].sub(valid["upper_limit_price"]).abs().le(0.0051).mean() * 100.0
        ),
        "raw_entry_within_1pct_of_limit_pct": float(
            valid["entry_room_to_limit_pct"].between(-0.01, 1.0).mean() * 100.0
        ),
        "entry_return_vs_prev_close_p10_pct": q("entry_return_vs_prev_close_pct", 0.10),
        "entry_return_vs_prev_close_median_pct": q("entry_return_vs_prev_close_pct", 0.50),
        "entry_return_vs_prev_close_p90_pct": q("entry_return_vs_prev_close_pct", 0.90),
        "entry_return_above_5pct_share_pct": float(
            valid["entry_return_vs_prev_close_pct"].gt(5.0).mean() * 100.0
        ),
        "entry_return_above_7pct_share_pct": float(
            valid["entry_return_vs_prev_close_pct"].gt(7.0).mean() * 100.0
        ),
        "entry_return_above_9pct_share_pct": float(
            valid["entry_return_vs_prev_close_pct"].gt(9.0).mean() * 100.0
        ),
        "entry_room_to_limit_p10_pct": q("entry_room_to_limit_pct", 0.10),
        "entry_room_to_limit_median_pct": q("entry_room_to_limit_pct", 0.50),
        "entry_room_to_limit_p90_pct": q("entry_room_to_limit_pct", 0.90),
        "entry_ask_volume_p10": q("raw_entry_ask_volume", 0.10),
        "entry_ask_volume_median": q("raw_entry_ask_volume", 0.50),
        "entry_ask_notional_p10": q("entry_ask_notional", 0.10),
        "entry_ask_notional_median": q("entry_ask_notional", 0.50),
        "entry_state_age_p95_seconds": q("raw_entry_state_age_seconds", 0.95),
    }
