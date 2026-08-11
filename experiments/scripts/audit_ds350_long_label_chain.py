from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.schema import normalize_date_series as normalize_date
from opening_strength_fit.schema import normalize_text_series as text
from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "decision_target_timestamp"]
PREDICTION_COLUMNS = [
    "date",
    "symbol",
    "decision_target_timestamp",
    "prediction",
    "label",
]
TRADABLE_STATUSES = {"T0", "20", "TRADE"}
TOP_N = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=["w0931_0940", "w1001_1010"])
    parser.add_argument("--root", type=Path, default=Path("/mnt/output/opening_strength_fit"))
    parser.add_argument("--sample-dates-per-fold", type=int, default=4)
    parser.add_argument("--skip-pool-metrics", action="store_true")
    return parser.parse_args()


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = text(out["symbol"])
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"], errors="coerce"
    )
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    return out.dropna(subset=["date", "symbol", "decision_target_timestamp", "prediction", "label"])


def prediction_files(root: Path, case: str, horizon: str) -> list[Path]:
    path = root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1" / f"{case}_h{horizon}"
    files = sorted(path.glob("month_*/predictions.parquet"))
    if len(files) != 8:
        raise SystemExit(f"expected 8 prediction files below {path}, got {len(files)}")
    return files


def fold_sample_dates(frame: pd.DataFrame, count: int) -> list[str]:
    dates = sorted(frame["date"].dropna().unique())
    if not dates:
        return []
    positions = np.unique(
        np.linspace(0, len(dates) - 1, num=min(max(1, count), len(dates)), dtype=int)
    )
    return [dates[position] for position in positions]


def load_year_reference(root: Path, case: str, year: int) -> pd.DataFrame:
    raw = root / f"cache/opening_{case.removeprefix('w')}_raw_source/year={year}"
    daily = pd.read_parquet(
        raw / "daily_reference.parquet",
        columns=[
            "TradingDay",
            "Symbol",
            "ClosePrice",
            "PreClosePrice",
            "STStatus",
            "UpdownLimitStatus",
        ],
    ).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "ClosePrice": "daily_close",
            "PreClosePrice": "prev_close",
            "STStatus": "st_status",
            "UpdownLimitStatus": "updown_limit_status",
        }
    )
    close = pd.read_parquet(
        raw / "close_reference.parquet",
        columns=["TradingDay", "Symbol", "ClosePrice", "CloseSourceOffsetUs"],
    ).rename(
        columns={
            "TradingDay": "date",
            "Symbol": "symbol",
            "ClosePrice": "tick_close",
            "CloseSourceOffsetUs": "close_source_offset_us",
        }
    )
    for frame in (daily, close):
        frame["date"] = normalize_date(frame["date"])
        frame["symbol"] = text(frame["symbol"])
    daily = daily.loc[daily["date"].str.startswith(str(year), na=False)].copy()
    close = close.loc[close["date"].str.startswith(str(year), na=False)].copy()
    out = daily.merge(close, on=["date", "symbol"], how="outer", validate="one_to_one")
    for column in (
        "daily_close",
        "prev_close",
        "st_status",
        "updown_limit_status",
        "tick_close",
        "close_source_offset_us",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["daily_closes_up_limit"] = out["updown_limit_status"].eq(1)
    out["close_age_seconds"] = (54_000_000_000 - out["close_source_offset_us"]) / 1_000_000.0
    out["daily_tick_close_delta_bps"] = (out["tick_close"] / out["daily_close"] - 1.0) * 10_000.0
    return out.drop_duplicates(["date", "symbol"], keep="last")


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return np.nan
    return float(left.corr(right, method=method))


def group_metrics(work: pd.DataFrame, *, pool_lag: int) -> pd.DataFrame:
    ordered = work.sort_values(
        [*KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    ordered["score_rank"] = ordered.groupby(KEYS, sort=False).cumcount() + 1
    rows: list[dict[str, object]] = []
    for (date, clock), group in ordered.groupby(KEYS, sort=False):
        top = group.head(TOP_N)
        without_limit = group.loc[~group["daily_closes_up_limit"]]
        reselected = without_limit.head(TOP_N)
        pool_mean = float(group["label"].mean())
        filtered_pool_mean = float(without_limit["label"].mean())
        top_mean = float(top["label"].mean())
        reselected_mean = float(reselected["label"].mean())
        selected_limit = top.loc[top["daily_closes_up_limit"]]
        selected_big_up = top.loc[top["label"].gt(0.05)]
        rows.append(
            {
                "pool_lag": pool_lag,
                "date": date,
                "quarter": str(pd.Timestamp(date).to_period("Q")),
                "clock": clock.strftime("%H:%M"),
                "candidate_rows": len(group),
                "pool_mean_bps": pool_mean * 10_000.0,
                "selected_mean_bps": top_mean * 10_000.0,
                "excess_bps": (top_mean - pool_mean) * 10_000.0,
                "pool_spearman": safe_corr(group["prediction"], group["label"], "spearman"),
                "pool_pearson": safe_corr(group["prediction"], group["label"], "pearson"),
                "candidate_limit_share_pct": float(group["daily_closes_up_limit"].mean() * 100.0),
                "selected_limit_share_pct": float(top["daily_closes_up_limit"].mean() * 100.0),
                "selected_limit_label_bps": float(selected_limit["label"].mean() * 10_000.0)
                if len(selected_limit)
                else np.nan,
                "selected_limit_contribution_bps": float(
                    selected_limit["label"].sum() / TOP_N * 10_000.0
                ),
                "selected_gt5_share_pct": float(top["label"].gt(0.05).mean() * 100.0),
                "selected_gt5_contribution_bps": float(
                    selected_big_up["label"].sum() / TOP_N * 10_000.0
                ),
                "reselected_no_limit_mean_bps": reselected_mean * 10_000.0,
                "reselected_no_limit_excess_vs_original_pool_bps": (reselected_mean - pool_mean)
                * 10_000.0,
                "reselected_no_limit_excess_vs_filtered_pool_bps": (
                    reselected_mean - filtered_pool_mean
                )
                * 10_000.0,
                "pool_close_age_mean_seconds": float(group["close_age_seconds"].mean()),
                "selected_close_age_mean_seconds": float(top["close_age_seconds"].mean()),
                "pool_close_age_gt300_pct": float(
                    group["close_age_seconds"].gt(300).mean() * 100.0
                ),
                "selected_close_age_gt300_pct": float(
                    top["close_age_seconds"].gt(300).mean() * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_group_metrics(metrics: pd.DataFrame) -> dict[str, object]:
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column != "pool_lag"
    ]
    quarter_equal = (
        metrics.groupby("quarter", as_index=False)[numeric].mean().mean(numeric_only=True)
    )
    return {column: float(quarter_equal[column]) for column in numeric}


def sample_rows(work: pd.DataFrame, dates: list[str], *, kind: str) -> pd.DataFrame:
    picked = work.loc[work["date"].isin(dates)].sort_values(
        [*KEYS, "prediction", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    parts = []
    for _keys, group in picked.groupby(KEYS, sort=False):
        top = group.head(100).copy()
        top["sample_scope"] = f"{kind}_top100"
        control_positions = np.unique(
            np.linspace(0, len(group) - 1, num=min(100, len(group)), dtype=int)
        )
        control = group.sort_values("symbol", kind="mergesort").iloc[control_positions].copy()
        control["sample_scope"] = f"{kind}_pool_control"
        parts.extend([top, control])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)[
        ["date", "symbol", "decision_target_timestamp", "prediction", "label", "sample_scope"]
    ]


def latest_positions(offsets: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.searchsorted(offsets, targets, side="right") - 1


def raw_tick_path(root: Path, raw_name: str, date: str) -> Path:
    return root / f"cache/{raw_name}/year={date[:4]}/ticks/date={date}.parquet"


def read_ticks(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    parts = [pd.read_parquet(path, columns=columns) for path in paths]
    out = pd.concat(parts, ignore_index=True)
    out["Symbol"] = text(out["Symbol"])
    out["ExchTimeOffsetUs"] = pd.to_numeric(out["ExchTimeOffsetUs"], errors="coerce")
    out = out.dropna(subset=["Symbol", "ExchTimeOffsetUs"])
    return (
        out.sort_values(["Symbol", "ExchTimeOffsetUs"], kind="mergesort")
        .drop_duplicates(["Symbol", "ExchTimeOffsetUs"], keep="last")
        .reset_index(drop=True)
    )


def reconstruct_close_samples(
    root: Path,
    case: str,
    samples: pd.DataFrame,
    references: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    raw_name = f"opening_{case.removeprefix('w')}_raw_source"
    columns = ["Symbol", "ExchTimeOffsetUs", "AskPrice1", "AskVolume1", "Status"]
    for date, day_samples in samples.groupby("date", sort=False):
        wanted = set(day_samples["symbol"])
        ticks = pd.read_parquet(raw_tick_path(root, raw_name, date), columns=columns)
        ticks["Symbol"] = text(ticks["Symbol"])
        ticks = ticks.loc[ticks["Symbol"].isin(wanted)].copy()
        ticks["ExchTimeOffsetUs"] = pd.to_numeric(ticks["ExchTimeOffsetUs"], errors="coerce")
        reference = references[int(date[:4])]
        close_map = reference.loc[reference["date"].eq(date)].set_index("symbol")["tick_close"]
        for symbol, group in day_samples.groupby("symbol", sort=False):
            state = ticks.loc[ticks["Symbol"].eq(symbol)].sort_values("ExchTimeOffsetUs")
            if state.empty:
                continue
            offsets = state["ExchTimeOffsetUs"].to_numpy(dtype="int64")
            clocks = group["decision_target_timestamp"]
            decision_targets = (
                (clocks - clocks.dt.normalize()) / pd.Timedelta(microseconds=1)
            ).to_numpy(dtype="int64")
            entry_targets = decision_targets + 6_000_000
            decision_index = latest_positions(offsets, decision_targets)
            entry_index = latest_positions(offsets, entry_targets)
            valid = (decision_index >= 0) & (entry_index >= 0)
            if not valid.any():
                continue
            part = group.loc[valid].copy()
            selected_entry = state.iloc[entry_index[valid]]
            part["decision_state_age_seconds"] = (
                decision_targets[valid] - offsets[decision_index[valid]]
            ) / 1_000_000.0
            part["entry_state_age_seconds"] = (
                entry_targets[valid] - offsets[entry_index[valid]]
            ) / 1_000_000.0
            part["entry_ask"] = pd.to_numeric(
                selected_entry["AskPrice1"], errors="coerce"
            ).to_numpy()
            part["entry_ask_volume"] = pd.to_numeric(
                selected_entry["AskVolume1"], errors="coerce"
            ).to_numpy()
            part["entry_status"] = text(selected_entry["Status"]).str.upper().to_numpy()
            part["tick_close"] = float(close_map.get(symbol, np.nan))
            part["reconstructed_label"] = part["tick_close"] / part["entry_ask"] - 1.0
            part["label_delta"] = part["reconstructed_label"] - part["label"]
            part["entry_offer_valid"] = (
                part["entry_ask"].gt(0)
                & part["entry_ask_volume"].gt(0)
                & part["entry_status"].isin(TRADABLE_STATUSES)
            )
            rows.append(part)
        del ticks
        gc.collect()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def reconstruct_h1h_samples(root: Path, case: str, samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_name = f"opening_{case.removeprefix('w')}_raw_source"
    state_names = (
        [base_name, "opening_1001_1010_raw_source", "opening_0931_1010_long_label_raw_source_v1"]
        if case == "w0931_0940"
        else [base_name, "opening_0931_1010_long_label_raw_source_v1"]
    )
    state_names = list(dict.fromkeys(state_names))
    base_columns = [
        "Symbol",
        "ExchTimeOffsetUs",
        "Volume",
        "Turnover",
        "AskPrice1",
        "AskVolume1",
        "Status",
    ]
    state_columns = ["Symbol", "ExchTimeOffsetUs", "Volume", "Turnover"]
    for date, day_samples in samples.groupby("date", sort=False):
        wanted = set(day_samples["symbol"])
        base_ticks = read_ticks([raw_tick_path(root, base_name, date)], base_columns)
        state_ticks = read_ticks(
            [raw_tick_path(root, name, date) for name in state_names], state_columns
        )
        base_ticks = base_ticks.loc[base_ticks["Symbol"].isin(wanted)].copy()
        state_ticks = state_ticks.loc[state_ticks["Symbol"].isin(wanted)].copy()
        for symbol, group in day_samples.groupby("symbol", sort=False):
            entry_state = base_ticks.loc[base_ticks["Symbol"].eq(symbol)].sort_values(
                "ExchTimeOffsetUs"
            )
            exit_state = state_ticks.loc[state_ticks["Symbol"].eq(symbol)].sort_values(
                "ExchTimeOffsetUs"
            )
            if entry_state.empty or exit_state.empty:
                continue
            entry_offsets = entry_state["ExchTimeOffsetUs"].to_numpy(dtype="int64")
            exit_offsets = exit_state["ExchTimeOffsetUs"].to_numpy(dtype="int64")
            volumes = pd.to_numeric(exit_state["Volume"], errors="coerce").to_numpy(dtype="float64")
            turnovers = pd.to_numeric(exit_state["Turnover"], errors="coerce").to_numpy(
                dtype="float64"
            )
            clocks = group["decision_target_timestamp"]
            decision_targets = (
                (clocks - clocks.dt.normalize()) / pd.Timedelta(microseconds=1)
            ).to_numpy(dtype="int64")
            entry_targets = decision_targets + 6_000_000
            start_targets = entry_targets + 3_600_000_000
            end_targets = start_targets + 60_000_000
            decision_index = latest_positions(entry_offsets, decision_targets)
            entry_index = latest_positions(entry_offsets, entry_targets)
            start_index = latest_positions(exit_offsets, start_targets)
            end_index = latest_positions(exit_offsets, end_targets)
            valid = (
                (decision_index >= 0) & (entry_index >= 0) & (start_index >= 0) & (end_index >= 0)
            )
            if not valid.any():
                continue
            part = group.loc[valid].copy()
            entry = entry_state.iloc[entry_index[valid]]
            buy = pd.to_numeric(entry["AskPrice1"], errors="coerce").to_numpy(dtype="float64")
            sell_volume = volumes[end_index[valid]] - volumes[start_index[valid]]
            sell_turnover = turnovers[end_index[valid]] - turnovers[start_index[valid]]
            with np.errstate(divide="ignore", invalid="ignore"):
                sell_vwap = sell_turnover / sell_volume
                reconstructed = sell_vwap / buy - 1.0
            part["decision_state_age_seconds"] = (
                decision_targets[valid] - entry_offsets[decision_index[valid]]
            ) / 1_000_000.0
            part["entry_state_age_seconds"] = (
                entry_targets[valid] - entry_offsets[entry_index[valid]]
            ) / 1_000_000.0
            part["sell_start_state_age_seconds"] = (
                start_targets[valid] - exit_offsets[start_index[valid]]
            ) / 1_000_000.0
            part["sell_end_state_age_seconds"] = (
                end_targets[valid] - exit_offsets[end_index[valid]]
            ) / 1_000_000.0
            part["entry_ask"] = buy
            part["entry_ask_volume"] = pd.to_numeric(
                entry["AskVolume1"], errors="coerce"
            ).to_numpy()
            part["entry_status"] = text(entry["Status"]).str.upper().to_numpy()
            part["sell_volume"] = sell_volume
            part["sell_turnover"] = sell_turnover
            part["reconstructed_label"] = reconstructed
            part["label_delta"] = part["reconstructed_label"] - part["label"]
            part["entry_offer_valid"] = (
                part["entry_ask"].gt(0)
                & part["entry_ask_volume"].gt(0)
                & part["entry_status"].isin(TRADABLE_STATUSES)
            )
            rows.append(part)
        del base_ticks, state_ticks
        gc.collect()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_reconstruction(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for scope, group in frame.groupby("sample_scope", sort=False):
        delta = pd.to_numeric(group["label_delta"], errors="coerce").abs()
        row: dict[str, object] = {
            "sample_scope": scope,
            "rows": len(group),
            "finite_reconstruction_rows": int(delta.notna().sum()),
            "label_abs_delta_mean": float(delta.mean()),
            "label_abs_delta_p99": float(delta.quantile(0.99)),
            "label_abs_delta_max": float(delta.max()),
            "label_delta_gt_1e_6_rows": int(delta.gt(1e-6).sum()),
            "entry_offer_invalid_pct": float((~group["entry_offer_valid"]).mean() * 100.0),
        }
        for column in (
            "decision_state_age_seconds",
            "entry_state_age_seconds",
            "sell_start_state_age_seconds",
            "sell_end_state_age_seconds",
        ):
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_p95"] = float(values.quantile(0.95))
                row[f"{column}_max"] = float(values.max())
                row[f"{column}_gt5_pct"] = float(values.gt(5).mean() * 100.0)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    close_files = prediction_files(args.root, args.case, "close")
    h1h_files = prediction_files(args.root, args.case, "1h")
    references: dict[int, pd.DataFrame] = {}
    metric_parts: dict[int, list[pd.DataFrame]] = {0: [], 1: []}
    close_sample_parts: list[pd.DataFrame] = []
    h1h_sample_parts: list[pd.DataFrame] = []
    sampled_dates: list[str] = []

    for close_path, h1h_path in zip(close_files, h1h_files, strict=True):
        close = normalize_predictions(pd.read_parquet(close_path, columns=PREDICTION_COLUMNS))
        years = sorted({int(value[:4]) for value in close["date"].unique()})
        if len(years) != 1:
            raise SystemExit(f"unexpected years in {close_path}: {years}")
        year = years[0]
        references.setdefault(year, load_year_reference(args.root, args.case, year))
        dates = fold_sample_dates(close, args.sample_dates_per_fold)
        sampled_dates.extend(dates)
        reference = references[year]

        lag0_work: pd.DataFrame | None = None
        if args.skip_pool_metrics:
            mask = stock_pool_membership_mask(close, pool, date_lag_sessions=0)
            lag0_work = close.loc[mask].copy()
        else:
            for lag in (0, 1):
                mask = stock_pool_membership_mask(close, pool, date_lag_sessions=lag)
                work = close.loc[mask].merge(
                    reference[
                        [
                            "date",
                            "symbol",
                            "daily_closes_up_limit",
                            "close_age_seconds",
                        ]
                    ],
                    on=["date", "symbol"],
                    how="left",
                    validate="many_to_one",
                )
                work["daily_closes_up_limit"] = (
                    work["daily_closes_up_limit"].fillna(False).astype(bool)
                )
                metric_parts[lag].append(group_metrics(work, pool_lag=lag))
                if lag == 0:
                    lag0_work = work
                else:
                    del work
                    gc.collect()
        if lag0_work is None:
            raise RuntimeError("lag-0 work was not built")
        close_sample_parts.append(sample_rows(lag0_work, dates, kind="close"))

        h1h = normalize_predictions(pd.read_parquet(h1h_path, columns=PREDICTION_COLUMNS))
        h1h_work = h1h.loc[stock_pool_membership_mask(h1h, pool, date_lag_sessions=0)].copy()
        h1h_sample_parts.append(sample_rows(h1h_work, dates, kind="h1h"))
        del close, lag0_work, h1h, h1h_work
        gc.collect()

    result: dict[str, object] = {
        "case": args.case,
        "prediction_root": str(args.root / "nn/nn_ds350_label15_36m_grouped_gated_v2_mse_max30_v1"),
        "sampled_dates": sorted(set(sampled_dates)),
        "pool_metrics": {},
        "close_reference": {},
    }
    if not args.skip_pool_metrics:
        for lag in (0, 1):
            metrics = pd.concat(metric_parts[lag], ignore_index=True)
            result["pool_metrics"][f"lag_{lag}"] = aggregate_group_metrics(metrics)
            result["pool_metrics"][f"lag_{lag}"]["groups"] = int(len(metrics))

    reference_all = pd.concat(references.values(), ignore_index=True)
    close_delta = pd.to_numeric(reference_all["daily_tick_close_delta_bps"], errors="coerce")
    result["close_reference"] = {
        "rows": int(len(reference_all)),
        "both_prices_rows": int(close_delta.notna().sum()),
        "exact_match_rows": int(close_delta.abs().le(1e-10).sum()),
        "abs_delta_gt_1bp_rows": int(close_delta.abs().gt(1.0).sum()),
        "abs_delta_gt_10bp_rows": int(close_delta.abs().gt(10.0).sum()),
        "abs_delta_bps_mean": float(close_delta.abs().mean()),
        "abs_delta_bps_p99": float(close_delta.abs().quantile(0.99)),
        "abs_delta_bps_max": float(close_delta.abs().max()),
    }

    close_samples = pd.concat(close_sample_parts, ignore_index=True)
    h1h_samples = pd.concat(h1h_sample_parts, ignore_index=True)
    reconstructed_close = reconstruct_close_samples(args.root, args.case, close_samples, references)
    reconstructed_h1h = reconstruct_h1h_samples(args.root, args.case, h1h_samples)
    result["close_reconstruction"] = summarize_reconstruction(reconstructed_close)
    result["h1h_reconstruction"] = summarize_reconstruction(reconstructed_h1h)
    print("LONG_LABEL_CHAIN_AUDIT_BEGIN")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    print("LONG_LABEL_CHAIN_AUDIT_END")


if __name__ == "__main__":
    main()
