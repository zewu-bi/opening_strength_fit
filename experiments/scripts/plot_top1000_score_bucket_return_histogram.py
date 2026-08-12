from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opening_strength_fit.legacy.top1000_rank_data import (
    TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS,
)
from opening_strength_fit.legacy.top1000_return_histograms import (
    plot_score_bucket_histograms,
    plot_score_bucket_histograms_full_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the canonical focused and full-scale Top1000 score-bucket "
            "return-distribution plots. The focused display contract is fixed at "
            "x=[-3000, 3000] bps and y=[1e2, 3e5] on a log scale."
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
    for plotter in (plot_score_bucket_histograms, plot_score_bucket_histograms_full_scale):
        plotter(
            histogram,
            bin_width_bps=TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS,
            output_dir=output_dir,
            variant=args.variant,
        )
    stem = f"top1000_score_bucket_return_{TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS}bps_counts"
    for suffix in ("", "_full_scale"):
        print(output_dir / f"{stem}{suffix}.svg", flush=True)


if __name__ == "__main__":
    main()
