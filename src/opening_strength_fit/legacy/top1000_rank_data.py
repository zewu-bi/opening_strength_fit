from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from opening_strength_fit.legacy import multiscale_bucket_diag as ms

DEFAULT_DIAGNOSTIC_MONTHS = tuple(
    "2022-01 2022-07 2023-01 2023-07 2024-01 2024-07 2025-01 2025-07".split()
)
TOP1000_SCORE_BUCKETS = tuple(range(1, 11))
TOP1000_RETURN_HISTOGRAM_BIN_WIDTH_BPS = 100
TOP1000_RETURN_HISTOGRAM_X_LIMIT_BPS = 3000
TOP1000_RETURN_HISTOGRAM_Y_LIMITS = (1e2, 3e5)


load_ranked_pool_shard = ms.load_ranked_pool_shard
ranked_pool_shards = ms.ranked_pool_shards
run_trace = ms.run_trace


def save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    for extension in ("svg", "png"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=140)
    plt.close(fig)
