from pathlib import Path

from opening_strength_fit.legacy.multiscale_bucket_diag import (
    MultiscaleBucketDiagConfig,
    run_multiscale_bucket_diagnostics,
)
from opening_strength_fit.legacy.top1000_rank_data import DEFAULT_DIAGNOSTIC_MONTHS

RUN_ID = "nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse_v1"
OUTPUT_DIR = Path(
    "/mnt/output/opening_strength_fit/runs/analyses/rank-bucket/"
    "multiscale_bucket_diag_mech328_v2_2022_2025_v1"
)


def main() -> None:
    run_multiscale_bucket_diagnostics(
        MultiscaleBucketDiagConfig(
            prediction_root=Path("/mnt/output/opening_strength_fit/runs/models/nn"),
            next_label_root=Path(
                "/mnt/output/opening_strength_fit/cache/"
                "opening_13y_201301_202512_delay2_next_close_labels_v1"
            ),
            pool_path="lml.bzw@ssd/data/pool_L.parquet",
            output_dir=OUTPUT_DIR,
            run_ids={"mech328_v2": RUN_ID},
            months=list(DEFAULT_DIAGNOSTIC_MONTHS),
            bucket_widths=[50, 100, 200],
            top_k=[50, 100, 150, 200, 500, 1000],
            window_widths=[50, 100, 200],
        )
    )
    print(f"multiscale_output={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
