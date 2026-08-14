from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from build_ds350_clip_tables import evaluate_model, limit_states, outcome_labels

from opening_strength_fit.stock_pool import DEFAULT_STOCK_POOL_PATHS, load_stock_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 10:01 1m baseline and no-limit-training Pool-L Top100 attribution."
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--no-limit-root", type=Path, required=True)
    parser.add_argument("--close-label-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be positive")

    years = range(2022, 2026)
    labels = outcome_labels(args.close_label_root, years)
    daily = limit_states(args.raw_root, years)
    pool = load_stock_pool(DEFAULT_STOCK_POOL_PATHS["L"])
    roots = {
        "baseline": args.baseline_root,
        "no_limit_training": args.no_limit_root,
    }
    results = {
        name: evaluate_model(
            root,
            labels=labels,
            daily=daily,
            pool=pool,
            top_n=args.top_n,
        )
        for name, root in roots.items()
    }

    display_names = {
        "baseline": "Baseline",
        "no_limit_training": "无涨跌停",
    }
    table_1_rows = []
    table_2_rows = []
    enrichment_rows = []
    for name in ("baseline", "no_limit_training"):
        result = results[name]
        own = result["outcomes"]["own_label"]
        close = result["outcomes"]["same_day_close"]
        next_close = result["outcomes"]["next_close"]
        table_1_rows.append(
            {
                "实验": display_names[name],
                "Label": "1m",
                "IC": own["rank_ic"],
                "Label对应超额": own["excess_bps"],
                "持有到收盘超额": close["excess_bps"],
                "次日收盘超额": next_close["excess_bps"],
            }
        )
        table_2_rows.append(
            {
                "实验": display_names[name],
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
                "实验": display_names[name],
                "PoolL涨停率": candidate_pct,
                "Top100涨停率": selected_pct,
                "富集倍数": selected_pct / candidate_pct if candidate_pct > 0.0 else float("nan"),
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
