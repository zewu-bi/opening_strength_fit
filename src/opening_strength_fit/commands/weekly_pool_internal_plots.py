from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opening_strength_fit.pool_internal_plots import slug_label
from opening_strength_fit.pool_internal_weekly import (
    POOL_CHOICES,
    normalize_pools,
    write_weekly_pool_internal_outputs,
)
from opening_strength_fit.reports import print_mapping


def add_weekly_plot_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_name: bool = False,
) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="baseline")
    if output_name:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render weekly pool-internal diagnostics from pool_internal_group_metrics.csv. "
            "The weekly and rolling summaries are equal weighted by trading day."
        )
    )
    parser.add_argument("--group-metrics", required=True)
    add_weekly_plot_arguments(parser)
    parser.add_argument("--rolling-weeks", type=int, default=4)
    parser.add_argument(
        "--top-worst",
        type=int,
        default=5,
        help="Worst single-week and rolling windows to include per pool/horizon.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pools = normalize_pools(args.pool)
    group_metrics_path = Path(args.group_metrics)
    output_dir = Path(args.output_dir)
    group_metrics = pd.read_csv(group_metrics_path)
    outputs, overall = write_weekly_pool_internal_outputs(
        group_metrics,
        output_dir,
        output_prefix=slug_label(args.output_prefix),
        variant_label=args.plot_variant_label or args.output_prefix,
        pools=pools,
        rolling_weeks=args.rolling_weeks,
        top_worst=args.top_worst,
        input_path=group_metrics_path,
    )

    print("weekly_pool_internal_overall_summary:")
    print(overall.to_string(index=False))
    print_mapping("weekly_pool_internal_outputs", outputs)


if __name__ == "__main__":
    main()
