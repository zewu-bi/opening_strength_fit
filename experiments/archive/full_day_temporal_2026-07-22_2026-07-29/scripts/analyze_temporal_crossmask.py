from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = {
    "forward_value__forward_mask": ("temporal_nn_36m_2022_2025_all_a_rank_1m_tcn_mse_v1"),
    "forward_value__backward_mask": (
        "temporal_nn_36m_2022_2025_all_a_rank_forward_1m_backward_mask_tcn_mse_v1"
    ),
    "backward_value__forward_mask": (
        "temporal_nn_36m_2022_2025_all_a_rank_backward_1m_forward_mask_tcn_mse_v1"
    ),
    "backward_value__backward_mask": (
        "temporal_nn_36m_2022_2025_all_a_rank_backward_1m_tcn_mse_v1"
    ),
}


def _read_predictions(root: Path, run_id: str) -> pd.DataFrame:
    fold_roots = sorted((root / run_id).glob("fold_*"))
    if len(fold_roots) != 8:
        raise RuntimeError(f"{run_id}: expected 8 folds, found {len(fold_roots)}")
    parts: list[pd.DataFrame] = []
    for fold_root in fold_roots:
        required = [
            fold_root / "_SUCCESS",
            fold_root / "predictions.parquet",
            fold_root / "temporal_nn_trace.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"{run_id}: incomplete fold {fold_root.name}: {missing}")
        part = pd.read_parquet(fold_root / "predictions.parquet")
        part["fold"] = fold_root.name
        parts.append(part)
    predictions = pd.concat(parts, ignore_index=True)
    duplicates = int(predictions.duplicated(["date", "symbol"]).sum())
    if duplicates:
        raise RuntimeError(f"{run_id}: {duplicates} duplicate date-symbol rows")
    return predictions


def _daily_metrics(
    predictions: pd.DataFrame,
    *,
    label: str,
    universe: str,
) -> pd.DataFrame:
    eligible = np.isfinite(predictions["score"]) & np.isfinite(predictions["target"])
    if "evaluation_eligible" in predictions:
        eligible &= predictions["evaluation_eligible"].eq(1)
    if universe == "pool_l":
        eligible &= predictions["stock_pool_member"].eq(1)
    work = predictions.loc[eligible].copy()

    rows: list[dict[str, object]] = []
    for (date, fold), day in work.groupby(["date", "fold"], observed=True, sort=False):
        count = min(100, len(day))
        selected = day.nlargest(count, "score")
        target = day["target"].to_numpy(dtype=np.float64)
        selected_target = selected["target"].to_numpy(dtype=np.float64)
        cap95 = float(np.quantile(target, 0.95))
        cap99 = float(np.quantile(target, 0.99))
        rows.append(
            {
                "label": label,
                "universe": universe,
                "date": str(date),
                "fold": str(fold),
                "n": len(day),
                "rank_ic": float(day["score"].rank().corr(day["target"].rank())),
                "top100_return": float(np.mean(selected_target)),
                "base_return": float(np.mean(target)),
                "top100_excess": float(np.mean(selected_target) - np.mean(target)),
                "cap95_excess": float(
                    np.mean(np.minimum(selected_target, cap95)) - np.mean(np.minimum(target, cap95))
                ),
                "cap99_excess": float(
                    np.mean(np.minimum(selected_target, cap99)) - np.mean(np.minimum(target, cap99))
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (label, universe), group in daily.groupby(
        ["label", "universe"],
        observed=True,
        sort=False,
    ):
        monthly = group.assign(month=group["date"].str[:7]).groupby("month")["top100_excess"].mean()
        halfyears = group.groupby("fold")["top100_excess"].mean()
        rows.append(
            {
                "label": label,
                "universe": universe,
                "days": len(group),
                "mean_n": float(group["n"].mean()),
                "daily_rank_ic": float(group["rank_ic"].mean()),
                "top100_return_bps": float(group["top100_return"].mean() * 10_000),
                "base_return_bps": float(group["base_return"].mean() * 10_000),
                "top100_excess_bps": float(group["top100_excess"].mean() * 10_000),
                "cap95_excess_bps": float(group["cap95_excess"].mean() * 10_000),
                "cap99_excess_bps": float(group["cap99_excess"].mean() * 10_000),
                "positive_months": int((monthly > 0).sum()),
                "months": len(monthly),
                "positive_halfyears": int((halfyears > 0).sum()),
                "halfyears": len(halfyears),
            }
        )
    return pd.DataFrame(rows)


def _decompose(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for universe, group in summary.groupby("universe", observed=True, sort=False):
        scores = group.set_index("label")["top100_excess_bps"]
        ff = float(scores["forward_value__forward_mask"])
        fb = float(scores["forward_value__backward_mask"])
        bf = float(scores["backward_value__forward_mask"])
        bb = float(scores["backward_value__backward_mask"])
        rows.extend(
            [
                {
                    "universe": universe,
                    "effect": "forward_mask_effect_given_forward_value",
                    "bps": ff - fb,
                },
                {
                    "universe": universe,
                    "effect": "forward_mask_effect_given_backward_value",
                    "bps": bf - bb,
                },
                {
                    "universe": universe,
                    "effect": "forward_value_effect_given_forward_mask",
                    "bps": ff - bf,
                },
                {
                    "universe": universe,
                    "effect": "forward_value_effect_given_backward_mask",
                    "bps": fb - bb,
                },
                {
                    "universe": universe,
                    "effect": "shapley_forward_mask_contribution",
                    "bps": 0.5 * ((ff - fb) + (bf - bb)),
                },
                {
                    "universe": universe,
                    "effect": "shapley_forward_value_contribution",
                    "bps": 0.5 * ((ff - bf) + (fb - bb)),
                },
                {
                    "universe": universe,
                    "effect": "observed_diagonal_gap",
                    "bps": ff - bb,
                },
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_parts: list[pd.DataFrame] = []
    for label, run_id in RUNS.items():
        predictions = _read_predictions(root, run_id)
        for universe in ("all_a", "pool_l"):
            daily_parts.append(
                _daily_metrics(
                    predictions,
                    label=label,
                    universe=universe,
                )
            )
        print(f"loaded {label}: rows={len(predictions)}", flush=True)

    daily = pd.concat(daily_parts, ignore_index=True)
    summary = _summarize(daily)
    decomposition = _decompose(summary)
    daily.to_csv(output_dir / "temporal_crossmask_daily.csv", index=False)
    summary.to_csv(output_dir / "temporal_crossmask_summary.csv", index=False)
    decomposition.to_csv(output_dir / "temporal_crossmask_decomposition.csv", index=False)
    print("\nsummary:")
    print(summary.to_string(index=False))
    print("\ndecomposition:")
    print(decomposition.to_string(index=False))


if __name__ == "__main__":
    main()
