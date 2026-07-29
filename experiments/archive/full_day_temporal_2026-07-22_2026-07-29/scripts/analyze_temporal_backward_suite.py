from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN_IDS = (
    "temporal_linear_36m_2022_2025_all_a_abs_backward_1m_relw99_v1",
    "temporal_tcn_36m_2022_2025_all_a_abs_backward_1m_relraw_v1",
    "temporal_tcn_36m_2022_2025_all_a_abs_backward_1m_relw99_v1",
    "temporal_tcn_36m_2022_2025_all_a_relative_backward_1m_relw99_v1",
    "temporal_tcn_36m_2022_2025_all_a_abs_backward_multiscale_relw99_v1",
    "temporal_tcn_36m_2022_2025_all_a_relative_backward_multiscale_relw99_v1",
)


def _rank_ic(score: np.ndarray, target: np.ndarray) -> float:
    if len(score) < 3:
        return float("nan")
    return float(pd.Series(score).rank().corr(pd.Series(target).rank()))


def _day_metrics(
    day: pd.DataFrame,
    *,
    run_id: str,
    universe: str,
    fold: str,
) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    score = day["score"].to_numpy(dtype=np.float64)
    target = day["target"].to_numpy(dtype=np.float64)
    symbols = day["symbol"].astype(str).to_numpy()
    finite = np.isfinite(score) & np.isfinite(target)
    score = score[finite]
    target = target[finite]
    symbols = symbols[finite]
    count = min(100, len(score))
    selected = np.argpartition(score, -count)[-count:]
    base = float(np.mean(target))
    top_return = float(np.mean(target[selected]))
    excess = top_return - base
    cap95 = float(np.quantile(target, 0.95))
    cap99 = float(np.quantile(target, 0.99))
    target_cap95 = np.minimum(target, cap95)
    target_cap99 = np.minimum(target, cap99)
    cap95_excess = float(np.mean(target_cap95[selected]) - np.mean(target_cap95))
    cap99_excess = float(np.mean(target_cap99[selected]) - np.mean(target_cap99))

    order = np.argsort(score, kind="mergesort")
    deciles: list[dict[str, object]] = []
    for index, indices in enumerate(np.array_split(order, 10), start=1):
        deciles.append(
            {
                "run_id": run_id,
                "universe": universe,
                "date": str(day["date"].iloc[0]),
                "fold": fold,
                "decile": index,
                "n": len(indices),
                "return": float(np.mean(target[indices])),
                "excess": float(np.mean(target[indices]) - base),
            }
        )

    return (
        {
            "run_id": run_id,
            "universe": universe,
            "date": str(day["date"].iloc[0]),
            "fold": fold,
            "n": len(target),
            "rank_ic": _rank_ic(score, target),
            "top100_return": top_return,
            "base_return": base,
            "top100_excess": excess,
            "cap95_excess": cap95_excess,
            "cap99_excess": cap99_excess,
            "tail95_contribution": excess - cap95_excess,
            "tail99_contribution": excess - cap99_excess,
            "selected_positive_fraction": float(np.mean(target[selected] > 0)),
            "selected_ge_9p5pct": int(np.sum(target[selected] >= 0.095)),
            "selected_ge_19p5pct": int(np.sum(target[selected] >= 0.195)),
        },
        deciles,
        set(symbols[selected]),
    )


def _summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (run_id, universe), group in daily.groupby(
        ["run_id", "universe"],
        observed=True,
        sort=False,
    ):
        excess = group["top100_excess"].to_numpy(dtype=np.float64)
        std = float(np.std(excess, ddof=1))
        monthly = group.assign(month=group["date"].str[:7]).groupby("month")["top100_excess"].mean()
        halfyear = group.groupby("fold")["top100_excess"].mean()
        rows.append(
            {
                "run_id": run_id,
                "universe": universe,
                "days": len(group),
                "mean_n": float(group["n"].mean()),
                "daily_rank_ic": float(group["rank_ic"].mean()),
                "top100_return_bps": float(group["top100_return"].mean() * 10_000),
                "base_return_bps": float(group["base_return"].mean() * 10_000),
                "top100_excess_bps": float(np.mean(excess) * 10_000),
                "daily_excess_tstat": float(np.mean(excess) / std * np.sqrt(len(excess))),
                "median_daily_excess_bps": float(np.median(excess) * 10_000),
                "p05_daily_excess_bps": float(np.quantile(excess, 0.05) * 10_000),
                "p95_daily_excess_bps": float(np.quantile(excess, 0.95) * 10_000),
                "positive_days": int(np.sum(excess > 0)),
                "positive_months": int(np.sum(monthly > 0)),
                "months": len(monthly),
                "positive_halfyears": int(np.sum(halfyear > 0)),
                "halfyears": len(halfyear),
                "cap95_excess_bps": float(group["cap95_excess"].mean() * 10_000),
                "cap99_excess_bps": float(group["cap99_excess"].mean() * 10_000),
                "tail95_share": float(
                    group["tail95_contribution"].mean() / group["top100_excess"].mean()
                ),
                "tail99_share": float(
                    group["tail99_contribution"].mean() / group["top100_excess"].mean()
                ),
                "selected_positive_fraction": float(group["selected_positive_fraction"].mean()),
                "selected_ge_9p5pct_per_day": float(group["selected_ge_9p5pct"].mean()),
                "selected_ge_19p5pct_per_day": float(group["selected_ge_19p5pct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _period_summary(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    work = daily.copy()
    if period == "year":
        work["period"] = work["date"].str[:4]
    elif period == "fold":
        work["period"] = work["fold"]
    else:
        raise ValueError(period)
    return (
        work.groupby(["run_id", "universe", "period"], observed=True, sort=False)
        .agg(
            days=("date", "nunique"),
            daily_rank_ic=("rank_ic", "mean"),
            top100_excess=("top100_excess", "mean"),
            cap95_excess=("cap95_excess", "mean"),
            cap99_excess=("cap99_excess", "mean"),
        )
        .reset_index()
        .assign(
            top100_excess_bps=lambda frame: frame.pop("top100_excess") * 10_000,
            cap95_excess_bps=lambda frame: frame.pop("cap95_excess") * 10_000,
            cap99_excess_bps=lambda frame: frame.pop("cap99_excess") * 10_000,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    selected_sets: dict[tuple[str, str, str], set[str]] = {}
    trace_rows: list[dict[str, object]] = []

    for run_id in RUN_IDS:
        run_root = root / run_id
        folds = sorted(run_root.glob("fold_*"))
        if len(folds) != 8:
            raise RuntimeError(f"{run_id}: expected 8 folds, found {len(folds)}")
        parts: list[pd.DataFrame] = []
        for fold_path in folds:
            success = fold_path / "_SUCCESS"
            prediction_path = fold_path / "predictions.parquet"
            trace_path = fold_path / "temporal_nn_trace.json"
            if not success.exists() or not prediction_path.exists() or not trace_path.exists():
                raise RuntimeError(f"{fold_path}: incomplete output")
            part = pd.read_parquet(
                prediction_path,
                columns=[
                    "date",
                    "symbol",
                    "score",
                    "target",
                    "stock_pool_member",
                    "evaluation_eligible",
                ],
            )
            part["fold"] = fold_path.name
            parts.append(part)
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace_rows.append(
                {
                    "run_id": run_id,
                    "fold": fold_path.name,
                    "epochs_completed": trace["epochs_completed"],
                    "best_validation_selection_value": trace["best_validation_selection_value"],
                    "target_mode": trace["target_mode"],
                    "target_winsor_lower": (
                        trace["target_winsor_bounds"][0]
                        if trace["target_winsor_bounds"] is not None
                        else np.nan
                    ),
                    "target_winsor_upper": (
                        trace["target_winsor_bounds"][1]
                        if trace["target_winsor_bounds"] is not None
                        else np.nan
                    ),
                }
            )
        predictions = pd.concat(parts, ignore_index=True)
        duplicate_count = int(predictions.duplicated(["date", "symbol"]).sum())
        if duplicate_count:
            raise RuntimeError(f"{run_id}: {duplicate_count} duplicate date-symbol rows")

        for universe in ("all_a", "pool_l"):
            eligible = predictions["evaluation_eligible"].eq(1)
            if universe == "pool_l":
                eligible &= predictions["stock_pool_member"].eq(1)
            work = predictions.loc[eligible]
            for (date, fold), day in work.groupby(
                ["date", "fold"],
                observed=True,
                sort=False,
            ):
                row, deciles, selected = _day_metrics(
                    day,
                    run_id=run_id,
                    universe=universe,
                    fold=str(fold),
                )
                daily_rows.append(row)
                decile_rows.extend(deciles)
                selected_sets[(run_id, universe, str(date))] = selected
        print(f"processed {run_id}: rows={len(predictions)}", flush=True)

    daily = pd.DataFrame(daily_rows)
    deciles = pd.DataFrame(decile_rows)
    summary = _summary(daily)
    annual = _period_summary(daily, "year")
    folds = _period_summary(daily, "fold")
    decile_summary = (
        deciles.groupby(["run_id", "universe", "decile"], observed=True, sort=False)
        .agg(
            days=("date", "nunique"), mean_return=("return", "mean"), mean_excess=("excess", "mean")
        )
        .reset_index()
    )
    decile_summary[["mean_return_bps", "mean_excess_bps"]] = (
        decile_summary[["mean_return", "mean_excess"]] * 10_000
    )
    decile_summary = decile_summary.drop(columns=["mean_return", "mean_excess"])

    overlap_rows: list[dict[str, object]] = []
    for universe in ("all_a", "pool_l"):
        for left, right in itertools.combinations(RUN_IDS, 2):
            dates = sorted(
                {key[2] for key in selected_sets if key[0] == left and key[1] == universe}
                & {key[2] for key in selected_sets if key[0] == right and key[1] == universe}
            )
            overlap_rows.append(
                {
                    "universe": universe,
                    "left_run_id": left,
                    "right_run_id": right,
                    "days": len(dates),
                    "mean_top100_overlap": float(
                        np.mean(
                            [
                                len(
                                    selected_sets[(left, universe, date)]
                                    & selected_sets[(right, universe, date)]
                                )
                                / 100.0
                                for date in dates
                            ]
                        )
                    ),
                }
            )

    turnover_rows: list[dict[str, object]] = []
    for run_id in RUN_IDS:
        for universe in ("all_a", "pool_l"):
            dates = sorted(
                key[2] for key in selected_sets if key[0] == run_id and key[1] == universe
            )
            daily_turnover = [
                1.0
                - len(
                    selected_sets[(run_id, universe, previous)]
                    & selected_sets[(run_id, universe, current)]
                )
                / 100.0
                for previous, current in zip(dates, dates[1:], strict=False)
            ]
            turnover_rows.append(
                {
                    "run_id": run_id,
                    "universe": universe,
                    "days": len(dates),
                    "mean_one_way_turnover": float(np.mean(daily_turnover)),
                    "median_one_way_turnover": float(np.median(daily_turnover)),
                    "p95_one_way_turnover": float(np.quantile(daily_turnover, 0.95)),
                }
            )

    daily.to_parquet(output_dir / "daily_metrics.parquet", index=False)
    summary.to_csv(output_dir / "suite_summary.csv", index=False)
    annual.to_csv(output_dir / "annual_summary.csv", index=False)
    folds.to_csv(output_dir / "fold_summary.csv", index=False)
    decile_summary.to_csv(output_dir / "decile_summary.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output_dir / "top100_overlap.csv", index=False)
    pd.DataFrame(turnover_rows).to_csv(output_dir / "turnover_summary.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(output_dir / "training_trace_summary.csv", index=False)
    (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
