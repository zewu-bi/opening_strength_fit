from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from run_top1000_rank_bucket_diagnostics import (
    TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS,
    plot_score_bucket_histograms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the canonical Top1000 score-bucket return-distribution acceptance plot. "
            "The display contract is fixed at x=[-3000, 3000] bps and y=[1e2, 3e5] on a "
            "log scale."
        )
    )
    parser.add_argument("--histogram-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.histogram_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    histogram = pd.read_csv(args.histogram_csv)
    plot_score_bucket_histograms(
        histogram,
        bin_width_bps=TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS,
        output_dir=output_dir,
        variant=args.variant,
    )
    print(
        output_dir
        / f"top1000_score_bucket_return_{TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS}bps_counts.svg",
        flush=True,
    )


if __name__ == "__main__":
    main()
