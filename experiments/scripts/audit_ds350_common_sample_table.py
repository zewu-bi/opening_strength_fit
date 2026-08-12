from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.stock_pool import (
    DEFAULT_STOCK_POOL_PATHS,
    load_stock_pool,
    stock_pool_membership_mask,
)

KEYS = ["date", "symbol", "decision_target_timestamp"]
GROUPS = ["date", "decision_target_timestamp"]
HORIZONS = ("1m", "3m", "10m", "1h", "close")
TOP_N = 100


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].map(
        lambda value: (
            value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        )
    )
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"], errors="coerce"
    )
    return out


def _rank_ic(frame: pd.DataFrame, score: str, outcome: str) -> pd.Series:
    grouped = frame.groupby(GROUPS, sort=False)
    score_rank = grouped[score].rank(method="average")
    outcome_rank = grouped[outcome].rank(method="average")
    score_centered = score_rank - score_rank.groupby(
        [frame[key] for key in GROUPS], sort=False
    ).transform("mean")
    outcome_centered = outcome_rank - outcome_rank.groupby(
        [frame[key] for key in GROUPS], sort=False
    ).transform("mean")
    numerator = (
        (score_centered * outcome_centered)
        .groupby([frame[key] for key in GROUPS], sort=False)
        .sum()
    )
    score_ss = score_centered.pow(2).groupby([frame[key] for key in GROUPS], sort=False).sum()
    outcome_ss = outcome_centered.pow(2).groupby([frame[key] for key in GROUPS], sort=False).sum()
    return numerator / np.sqrt(score_ss * outcome_ss).replace(0.0, np.nan)


def _fold_metrics(common: pd.DataFrame, fold_name: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        score = f"score_{horizon}"
        own = f"own_{horizon}"
        ordered = common.sort_values(
            [*GROUPS, score, "symbol"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        grouped = ordered.groupby(GROUPS, sort=False)
        selected = grouped.head(TOP_N)
        base = grouped.agg(
            candidate_rows=(score, "size"),
            pool_own_mean=(own, "mean"),
            pool_close_mean=("return_close", "mean"),
            pool_next_mean=("return_next_close", "mean"),
        )
        picked = selected.groupby(GROUPS, sort=False).agg(
            selected_rows=(score, "size"),
            selected_own_mean=(own, "mean"),
            selected_close_mean=("return_close", "mean"),
            selected_next_mean=("return_next_close", "mean"),
        )
        metrics = base.join(picked, how="inner")
        metrics["ic"] = _rank_ic(ordered, score, own).reindex(metrics.index)
        metrics["label_excess_bps"] = (
            metrics["selected_own_mean"] - metrics["pool_own_mean"]
        ) * 10_000.0
        metrics["close_excess_bps"] = (
            metrics["selected_close_mean"] - metrics["pool_close_mean"]
        ) * 10_000.0
        metrics["next_close_excess_bps"] = (
            metrics["selected_next_mean"] - metrics["pool_next_mean"]
        ) * 10_000.0
        limit = ordered["daily_closes_up_limit"]
        selected_limit = selected["daily_closes_up_limit"]
        for prefix, outcome, total_column in (
            ("label", own, "label_excess_bps"),
            ("close", "return_close", "close_excess_bps"),
            ("next_close", "return_next_close", "next_close_excess_bps"),
        ):
            pool_limit_sum = (
                ordered.loc[limit]
                .groupby(GROUPS, sort=False)[outcome]
                .sum()
                .reindex(metrics.index)
                .fillna(0.0)
            )
            selected_limit_sum = (
                selected.loc[selected_limit]
                .groupby(GROUPS, sort=False)[outcome]
                .sum()
                .reindex(metrics.index)
                .fillna(0.0)
            )
            limit_contribution = (
                selected_limit_sum / metrics["selected_rows"]
                - pool_limit_sum / metrics["candidate_rows"]
            ) * 10_000.0
            metrics[f"{prefix}_limit_excess_contribution_bps"] = limit_contribution
            metrics[f"{prefix}_nonlimit_excess_contribution_bps"] = (
                metrics[total_column] - limit_contribution
            )
        metrics = metrics.reset_index()
        metrics["label_horizon"] = horizon
        metrics["fold"] = fold_name
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--raw-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    files = {
        horizon: {
            path.parent.name: path
            for path in sorted(
                (args.model_root / f"w0931_0940_h{horizon}").glob("month_*/predictions.parquet")
            )
        }
        for horizon in HORIZONS
    }
    if any(len(paths) != 8 for paths in files.values()):
        counts = {horizon: len(paths) for horizon, paths in files.items()}
        raise SystemExit(f"expected eight prediction folds per horizon, got {counts}")
    fold_names = sorted(files["close"])
    if any(set(paths) != set(fold_names) for paths in files.values()):
        raise SystemExit("prediction fold names differ across horizons")

    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    all_metrics: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    references: dict[int, pd.DataFrame] = {}
    for fold_index, fold_name in enumerate(fold_names, start=1):
        frames: dict[str, pd.DataFrame] = {}
        for horizon in HORIZONS:
            frame = _normalize(
                pd.read_parquet(
                    files[horizon][fold_name],
                    columns=[*KEYS, "prediction", "label"],
                )
            )
            frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
            frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
            frame = frame.dropna(subset=[*KEYS, "prediction", "label"])
            frame = frame.loc[stock_pool_membership_mask(frame, pool, date_lag_sessions=0)].copy()
            frame = frame.rename(
                columns={"prediction": f"score_{horizon}", "label": f"own_{horizon}"}
            )
            frames[horizon] = frame.drop_duplicates(KEYS, keep="last")

        outcome = _normalize(
            pd.read_parquet(
                files["close"][fold_name],
                columns=[*KEYS, "label_short", "label_next_close"],
            )
        ).rename(
            columns={
                "label_short": "return_close",
                "label_next_close": "return_next_close",
            }
        )
        for column in ("return_close", "return_next_close"):
            outcome[column] = pd.to_numeric(outcome[column], errors="coerce")
        outcome = outcome.dropna(
            subset=[*KEYS, "return_close", "return_next_close"]
        ).drop_duplicates(KEYS, keep="last")

        common = frames[HORIZONS[0]]
        native_rows = {HORIZONS[0]: len(common)}
        for horizon in HORIZONS[1:]:
            native_rows[horizon] = len(frames[horizon])
            common = common.merge(frames[horizon], on=KEYS, how="inner", validate="one_to_one")
        common = common.merge(outcome, on=KEYS, how="inner", validate="one_to_one")
        required = [
            *(f"score_{horizon}" for horizon in HORIZONS),
            *(f"own_{horizon}" for horizon in HORIZONS),
            "return_close",
            "return_next_close",
        ]
        common = common.dropna(subset=required)
        year = int(common["date"].iloc[0][:4])
        if year not in references:
            reference = pd.read_parquet(
                args.raw_source_root / f"year={year}" / "daily_reference.parquet",
                columns=["TradingDay", "Symbol", "UpdownLimitStatus"],
            ).rename(
                columns={
                    "TradingDay": "date",
                    "Symbol": "symbol",
                    "UpdownLimitStatus": "updown_limit_status",
                }
            )
            if pd.api.types.is_numeric_dtype(reference["date"]):
                parsed_date = pd.to_datetime(
                    reference["date"], unit="D", origin="unix", errors="coerce"
                )
            else:
                parsed_date = pd.to_datetime(reference["date"], errors="coerce")
            reference["date"] = parsed_date.dt.strftime("%Y-%m-%d")
            reference["symbol"] = reference["symbol"].map(
                lambda value: (
                    value.decode("utf-8", errors="replace")
                    if isinstance(value, bytes)
                    else str(value)
                )
            )
            reference["daily_closes_up_limit"] = pd.to_numeric(
                reference["updown_limit_status"], errors="coerce"
            ).eq(1)
            references[year] = reference[
                ["date", "symbol", "daily_closes_up_limit"]
            ].drop_duplicates(["date", "symbol"], keep="last")
        common = common.merge(
            references[year], on=["date", "symbol"], how="inner", validate="many_to_one"
        )
        close_delta = (common["own_close"] - common["return_close"]).abs()
        max_close_delta = float(close_delta.max()) if len(close_delta) else np.nan
        if not np.isfinite(max_close_delta) or max_close_delta > 1e-7:
            raise AssertionError(
                f"close label mismatch in {fold_name}: max_abs_delta={max_close_delta}"
            )

        group_counts = common.groupby(GROUPS, sort=False).size()
        if int(group_counts.min()) < TOP_N:
            raise AssertionError(f"common pool has fewer than Top{TOP_N} rows")
        diagnostics.append(
            {
                "fold": fold_name,
                "common_rows": len(common),
                "groups": int(group_counts.size),
                "candidate_rows_mean": float(group_counts.mean()),
                "close_identity_max_abs": max_close_delta,
                "candidate_final_limit_share_pct": float(
                    common["daily_closes_up_limit"].mean() * 100.0
                ),
                **{f"native_rows_{horizon}": rows for horizon, rows in native_rows.items()},
            }
        )
        all_metrics.append(_fold_metrics(common, fold_name))
        print(
            f"progress fold={fold_index}/8 name={fold_name} common_rows={len(common)} "
            f"groups={group_counts.size} candidate_mean={group_counts.mean():.2f}",
            flush=True,
        )

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics["quarter"] = pd.to_datetime(metrics["date"]).dt.to_period("Q").astype(str)
    value_columns = [
        "ic",
        "label_excess_bps",
        "close_excess_bps",
        "next_close_excess_bps",
        "label_limit_excess_contribution_bps",
        "label_nonlimit_excess_contribution_bps",
        "close_limit_excess_contribution_bps",
        "close_nonlimit_excess_contribution_bps",
        "next_close_limit_excess_contribution_bps",
        "next_close_nonlimit_excess_contribution_bps",
        "candidate_rows",
        "selected_rows",
    ]
    quarterly = metrics.groupby(["label_horizon", "quarter"], as_index=False)[value_columns].mean()
    summary = quarterly.groupby("label_horizon", as_index=False)[value_columns].mean()
    summary["groups"] = summary["label_horizon"].map(metrics.groupby("label_horizon").size())
    order = {horizon: index for index, horizon in enumerate(HORIZONS)}
    summary["_order"] = summary["label_horizon"].map(order)
    summary = summary.sort_values("_order").drop(columns="_order")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.output_dir / "common_sample_group_metrics.parquet", index=False)
    quarterly.to_csv(args.output_dir / "common_sample_quarter_summary.csv", index=False)
    summary.to_csv(args.output_dir / "common_sample_table_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "common_sample_diagnostics.csv", index=False)
    (args.output_dir / "_SUCCESS").touch()
    print("COMMON_SAMPLE_TABLE_BEGIN", flush=True)
    print(summary.to_csv(index=False).strip(), flush=True)
    print("COMMON_SAMPLE_TABLE_END", flush=True)


if __name__ == "__main__":
    main()
