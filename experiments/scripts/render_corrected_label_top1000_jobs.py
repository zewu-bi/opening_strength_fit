"""Render the historical corrected-label Top1000 job matrix via the shared renderer."""

from itertools import product
from pathlib import Path

from opening_strength_fit.k8s_builder_rendering import render_top1000_job

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "experiments" / "jobs" / "support" / "corrected_label_3x2_grid_2022_2025_v1"
IMAGE = (
    "registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260731-corrected-label-matrix"
)
WINDOWS = ("0931_0940", "1001_1010", "1401_1410")
HORIZONS = ("short1m", "short3m")
RUN_TEMPLATE = "nn_v6_w{window}_{horizon}_corrected_nextclose_36m_grouped_gated_v2_mse"


def _config(run_id: str, window: str, horizon: str, mode: str, node: str) -> dict:
    label_root = (
        "/mnt/output/opening_strength_fit/cache/"
        f"opening_2019_2025_next_close_decision_clock_state_clock6_{window}"
    )
    output_root = (
        f"/mnt/output/opening_strength_fit/nn/{run_id}/analysis/top1000_corrected_acceptance"
    )
    return {
        "run": {"id": run_id},
        "output": {"k8s_dir": output_root if mode == "rank" else f"{output_root}/hist"},
        "analysis": {
            "top1000": {
                "prediction_root": "/mnt/output/opening_strength_fit/nn",
                "next_close_label_input": label_root,
                "next_close_label_filename_template": (
                    f"opening_{{year}}_next_close_decision_clock_state_clock6_{window}.parquet"
                ),
                "years": list(range(2019, 2026)),
                "pool_path": "lml.bzw@ssd/data/pool_L.parquet",
                "variant": f"corrected_{window}_{horizon}",
                "source_run_id": run_id,
                "success_path": f"{output_root}/_{mode.upper()}_SUCCESS",
                **({"histogram_bin_width_bps": 100} if mode == "hist" else {}),
            }
        },
        "k8s": {
            "job_name": f"os-top1000-corrected-{window.replace('_', '-')}-{horizon[5:]}-{mode}",
            "analysis_config_map": "os-top1000-rank-bucket-script-v3",
            "env_secrets": ["opening-strength-clickhouse", "xy-fit-ceph-credentials"],
            "node_selector": {"kubernetes.io/hostname": node},
            "resources": {
                "cpu_request": "8",
                "cpu_limit": "16",
                "tolerate_gpu_nodes": True,
                "memory_request": "256Gi" if mode == "rank" else "192Gi",
                "memory_limit": "384Gi" if mode == "rank" else "320Gi",
            },
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rank_nodes = ("node13", "node16")
    for index, (window, horizon) in enumerate(product(WINDOWS, HORIZONS)):
        run_id = RUN_TEMPLATE.format(window=window, horizon=horizon)
        for mode, node in (("rank", rank_nodes[index % 2]), ("hist", "node14")):
            path = OUTPUT_DIR / f"{run_id}_top1000_{mode}_job.yaml"
            path.write_text(
                render_top1000_job(path, _config(run_id, window, horizon, mode, node), IMAGE),
                encoding="utf-8",
            )
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
