from pathlib import Path

from opening_strength_fit.legacy.multiscale_bucket_diag import (
    MultiscaleBucketDiagConfig,
    run_multiscale_bucket_diagnostics,
)
from opening_strength_fit.legacy.rank_bucket_reaudit import (
    RankBucketReauditConfig,
    run_rank_bucket_reaudit,
)
from opening_strength_fit.legacy.top1000_rank_data import DEFAULT_DIAGNOSTIC_MONTHS

RUN_ID = "nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1"
PREDICTION_ROOT = Path("/mnt/output/opening_strength_fit/nn")
NEXT_LABEL_ROOT = Path(
    "/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1"
)
POOL_PATH = "lml.bzw@ssd/data/pool_L.parquet"
OUT_ROOT = Path("/mnt/output/opening_strength_fit/runs/analyses/rank-bucket")


def main() -> None:
    run_ids = {"auction_fresh_pruned": RUN_ID}
    shared = {
        "prediction_root": PREDICTION_ROOT,
        "next_label_root": NEXT_LABEL_ROOT,
        "pool_path": POOL_PATH,
        "run_ids": run_ids,
        "months": list(DEFAULT_DIAGNOSTIC_MONTHS),
    }
    rank_output = OUT_ROOT / "rank_bucket_reaudit_auction_fresh_pruned_2022_2025_v1"
    multiscale_output = OUT_ROOT / "multiscale_bucket_diag_auction_fresh_pruned_2022_2025_v1"
    run_rank_bucket_reaudit(RankBucketReauditConfig(output_dir=rank_output, **shared))
    run_multiscale_bucket_diagnostics(
        MultiscaleBucketDiagConfig(
            output_dir=multiscale_output,
            bucket_widths=[50, 100, 200],
            top_k=[50, 100, 150, 200, 500, 1000],
            window_widths=[50, 100, 200],
            **shared,
        )
    )
    print(f"rank_bucket_output={rank_output}", flush=True)
    print(f"multiscale_output={multiscale_output}", flush=True)


if __name__ == "__main__":
    main()
