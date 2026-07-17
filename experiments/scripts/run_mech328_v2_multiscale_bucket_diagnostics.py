from pathlib import Path

from opening_strength_fit.legacy.multiscale_bucket_diag import (
    MultiscaleBucketDiagConfig,
    run_multiscale_bucket_diagnostics,
)

MONTHS = [
    "2022-01",
    "2022-07",
    "2023-01",
    "2023-07",
    "2024-01",
    "2024-07",
    "2025-01",
    "2025-07",
]
RUN_ID = "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse_v1"
PREDICTION_ROOT = Path("/mnt/output/opening_strength_fit/runs/models/nn")
NEXT_LABEL_ROOT = Path(
    "/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1"
)
POOL_PATH = "lml.bzw@ssd/data/pool_L.parquet"
OUTPUT_DIR = Path(
    "/mnt/output/opening_strength_fit/runs/analyses/rank-bucket/"
    "multiscale_bucket_diag_mech328_v2_2022_2025_v1"
)


def main() -> None:
    run_multiscale_bucket_diagnostics(
        MultiscaleBucketDiagConfig(
            prediction_root=PREDICTION_ROOT,
            next_label_root=NEXT_LABEL_ROOT,
            pool_path=POOL_PATH,
            output_dir=OUTPUT_DIR,
            run_ids={"mech328_v2": RUN_ID},
            months=MONTHS,
            bucket_widths=[50, 100, 200],
            top_k=[50, 100, 150, 200, 500, 1000],
            window_widths=[50, 100, 200],
            window_stride=50,
            top_n=1000,
        )
    )
    print(f"multiscale_output={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
