"""Compatibility renderer for the canonical corrected-label 3x2 matrix."""

from pathlib import Path

from opening_strength_fit.commands.k8s_rendering import render_sharded_training_job
from opening_strength_fit.config import load_toml, run_id
from opening_strength_fit.k8s_builder_rendering import render_indexed_builder_job

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
JOBS = ROOT / "experiments" / "jobs"
TARGET_CONFIGS = tuple(sorted(RUNS.glob("build_target_v6_*_corrected_nextclose.toml")))
TRAINING_CONFIGS = tuple(
    sorted(RUNS.glob("nn_v6_*_corrected_nextclose_36m_grouped_gated_v2_mse.toml"))
)


def _render(path: Path, renderer) -> str:
    config = load_toml(path)
    image = str(config.get("k8s", {}).get("helper_image", ""))
    if not image:
        raise SystemExit(f"{path} is missing k8s.helper_image")
    return renderer(path, config, image).rstrip() + "\n"


def main() -> None:
    if len(TARGET_CONFIGS) != 6 or len(TRAINING_CONFIGS) != 6:
        raise SystemExit("corrected-label matrix requires six target and six training configs")
    JOBS.mkdir(parents=True, exist_ok=True)
    for path in TARGET_CONFIGS:
        config = load_toml(path)
        job = JOBS / f"{run_id(config, path)}_sharded_job.yaml"
        job.write_text(_render(path, render_indexed_builder_job), encoding="utf-8")
        print(job.relative_to(ROOT))
    for path in TRAINING_CONFIGS:
        _render(path, render_sharded_training_job)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
