from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
from build_ds350_clip_tables import evaluate_model, limit_states, outcome_labels

from opening_strength_fit.stock_pool import DEFAULT_STOCK_POOL_PATHS, load_stock_pool

CASES = (
    ("09:31-09:40", "w0931_0940", "3m", "w0931_0940_h3m"),
    ("09:31-09:40", "w0931_0940", "10m", "w0931_0940_h10m"),
    ("10:01-10:10", "w1001_1010", "3m", "w1001_1010_h3m"),
    ("10:01-10:10", "w1001_1010", "10m", "w1001_1010_h10m"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two-window 3m/10m baseline and no-limit-training attribution."
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--no-limit-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")

    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    results: dict[str, dict[str, object]] = {}
    table_1_rows = []
    table_2_rows = []
    enrichment_rows = []
    display_names = {"baseline": "Baseline", "no_limit": "无涨跌停"}

    for window, window_key, horizon, case in CASES:
        close_root = args.dataset_root / f"opening_{window_key[1:]}_labels_hclose_v1"
        raw_root = args.cache_root / f"opening_{window_key[1:]}_raw_source"
        labels = outcome_labels(close_root, range(2022, 2026))
        daily = limit_states(raw_root, range(2022, 2026))

        roots = {
            "baseline": args.baseline_root / case,
            "no_limit": args.no_limit_root / case,
        }
        for experiment, root in roots.items():
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
                    "窗口": window,
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
                    "窗口": window,
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
                    "窗口": window,
                    "实验": display_names[experiment],
                    "Label": horizon,
                    "PoolL涨停率": candidate_pct,
                    "Top100涨停率": selected_pct,
                    "富集倍数": selected_pct / candidate_pct if candidate_pct else None,
                }
            )
            gc.collect()
        del labels, daily
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
    enrichment.to_csv(output_dir / "final_limit_enrichment.csv", index=False)
    (output_dir / "_SUCCESS").touch()
    print("TABLE_1", flush=True)
    print(table_1.to_csv(index=False).strip(), flush=True)
    print("TABLE_2", flush=True)
    print(table_2.to_csv(index=False).strip(), flush=True)
    print("FINAL_LIMIT_ENRICHMENT", flush=True)
    print(enrichment.to_csv(index=False).strip(), flush=True)


if __name__ == "__main__":
    main()
