from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opening_strength_fit.pool_internal_plots import write_weekly_pool_internal_cumulative_plot
from opening_strength_fit.pool_internal_weekly import normalize_pools

POOL_CHOICES = ("universe", "S", "M", "L", "pool_S", "pool_M", "pool_L")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render cumulative weekly pool-internal short/next excess from an existing "
            "weekly_pool_internal_summary.csv."
        )
    )
    parser.add_argument("--weekly-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="baseline")
    parser.add_argument(
        "--output-name",
        default="",
        help="Optional exact file stem for flat outputs under --output-dir.",
    )
    parser.add_argument(
        "--plot-variant-label",
        default="",
        help="Display label used in the generated SVG title. Defaults to --output-prefix.",
    )
    parser.add_argument(
        "--pool",
        action="append",
        choices=POOL_CHOICES,
        help="Pools to include. Defaults to universe, S, M, and L.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pools = normalize_pools(args.pool)
    weekly_summary_path = Path(args.weekly_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weekly_summary = pd.read_csv(weekly_summary_path)
    outputs = write_weekly_pool_internal_cumulative_plot(
        weekly_summary,
        output_dir,
        input_path=weekly_summary_path,
        output_prefix=args.output_prefix,
        output_name=args.output_name,
        variant_label=args.plot_variant_label or args.output_prefix,
        pools=pools,
    )

    print("weekly_pool_internal_cumulative_outputs:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
