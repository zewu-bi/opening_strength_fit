from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from opening_strength_fit.legacy.multiscale_bucket_diag import (
    MultiscaleBucketDiagConfig,
    run_multiscale_bucket_diagnostics,
)

DEFAULT_VARIANTS = {
    "nn_mlp_base": "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_v1",
    "nn_mlp_base_plus_mse": (
        "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_base_plus_mse_v1"
    ),
    "nn_deep_gelu_mse": (
        "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_mlp_deep_gelu_mse_v1"
    ),
}
DEFAULT_MONTHS = (
    "2022-01",
    "2022-07",
    "2023-01",
    "2023-07",
    "2024-01",
    "2024-07",
    "2025-01",
    "2025-07",
)


def parse_csv_ints(value: str, *, default: Iterable[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str, *, default: Iterable[str]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run multiscale Top1000 bucket diagnostics for the legacy three NN models.")
    )
    parser.add_argument(
        "--prediction-root",
        default="/mnt/output/opening_strength_fit/nn",
        help="PVC directory containing per-run prediction shard directories.",
    )
    parser.add_argument(
        "--next-label-root",
        default=(
            "/mnt/output/opening_strength_fit/cache/"
            "opening_13y_201301_202512_delay2_next_close_labels_v1"
        ),
        help="Directory containing opening_YYYY_next_close_labels_v1.parquet files.",
    )
    parser.add_argument(
        "--pool-path",
        default="lml.bzw@ssd/data/pool_L.parquet",
        help="Stock pool path understood by opening_strength_fit.stock_pool.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/output/opening_strength_fit/runs/legacy/untracked/old_nn_multiscale_bucket_diag_v1",
    )
    parser.add_argument(
        "--months",
        default=",".join(DEFAULT_MONTHS),
        help="Comma-separated half-year month directories to process.",
    )
    parser.add_argument("--bucket-widths", default="50,100,200")
    parser.add_argument("--top-k", default="50,100,150,200,500,1000")
    parser.add_argument("--window-widths", default="50,100,200")
    parser.add_argument("--window-stride", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(DEFAULT_VARIANTS),
        help="Variant to process. May be repeated. Defaults to all legacy variants.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.variant or list(DEFAULT_VARIANTS)
    run_ids = {variant: DEFAULT_VARIANTS[variant] for variant in selected}
    config = MultiscaleBucketDiagConfig(
        prediction_root=Path(args.prediction_root),
        next_label_root=Path(args.next_label_root),
        pool_path=args.pool_path,
        output_dir=Path(args.output_dir),
        run_ids=run_ids,
        months=parse_csv_strings(args.months, default=DEFAULT_MONTHS),
        bucket_widths=parse_csv_ints(args.bucket_widths, default=(50, 100, 200)),
        top_k=parse_csv_ints(args.top_k, default=(50, 100, 150, 200, 500, 1000)),
        window_widths=parse_csv_ints(args.window_widths, default=(50, 100, 200)),
        window_stride=args.window_stride,
        top_n=args.top_n,
    )
    run_multiscale_bucket_diagnostics(config)


if __name__ == "__main__":
    main()
