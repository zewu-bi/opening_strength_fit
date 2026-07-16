from __future__ import annotations

from pathlib import Path

from opening_strength_fit.config import config_value

LEGACY_LAYOUT = "legacy"
PVC_LAYOUT_V2 = "v2"
SUPPORTED_LAYOUTS = {LEGACY_LAYOUT, PVC_LAYOUT_V2}
DEFAULT_MOUNT_PATH = "/mnt/output"
PROJECT_DIRNAME = "opening_strength_fit"
MODEL_RUN_KINDS = {"experiment", "exploration"}
DATA_RUN_KINDS = {"cache_transform", "labeled_cache", "next_close_label_cache"}
MODEL_FAMILIES = {
    "clock_segment_lightgbm": "lightgbm",
    "gbm": "gbm",
    "lightgbm": "lightgbm",
    "ridge": "ridge",
    "torch_mlp": "nn",
}


def output_layout(config: dict) -> str:
    output = config.get("output", {})
    output = output if isinstance(output, dict) else {}
    configured = str(output.get("layout", "") or "").strip().lower()
    if configured:
        if configured not in SUPPORTED_LAYOUTS:
            allowed = ", ".join(sorted(SUPPORTED_LAYOUTS))
            raise SystemExit(f"unknown [output].layout={configured!r}; expected one of {allowed}")
        return configured
    return LEGACY_LAYOUT if str(output.get("k8s_dir", "") or "").strip() else PVC_LAYOUT_V2


def run_storage_group(config: dict) -> str:
    run = config.get("run", {})
    run = run if isinstance(run, dict) else {}
    model = config.get("model", {})
    model = model if isinstance(model, dict) else {}
    kind = str(run.get("kind", "") or "").strip().lower()
    model_name = str(model.get("name", "") or "").strip().lower()
    normalized_kind = kind.replace("_", "-")
    if kind in DATA_RUN_KINDS:
        return f"data/{normalized_kind}"
    if kind in MODEL_RUN_KINDS or (not kind and model_name):
        family = MODEL_FAMILIES.get(model_name, model_name.replace("_", "-") or "other")
        return f"models/{family}"
    if kind:
        return f"analyses/{normalized_kind}"
    return "legacy/untracked"


def run_output_dir(
    config: dict,
    run_id_value: str,
    *,
    mount_path: str | None = None,
) -> str:
    explicit = str(config_value(config, "output", "k8s_dir", "") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    mount = str(mount_path or config_value(config, "k8s", "mount_path", DEFAULT_MOUNT_PATH)).rstrip(
        "/"
    )
    return f"{mount}/{PROJECT_DIRNAME}/runs/{run_storage_group(config)}/{run_id_value}"


def rolling_shard_dir_name(start_month: str, end_month: str, layout: str) -> str:
    if layout == PVC_LAYOUT_V2:
        return f"fold_{start_month}_{end_month}"
    return f"month_{start_month}"


def yearly_shard_dir_name(year: int | str, layout: str) -> str:
    if layout == PVC_LAYOUT_V2:
        return f"fold_{year}-01_{year}-12"
    return f"year_{year}"


def rolling_shard_dir_candidates(
    start_month: str,
    end_month: str,
    *,
    preferred_layout: str = LEGACY_LAYOUT,
) -> tuple[str, ...]:
    preferred = rolling_shard_dir_name(start_month, end_month, preferred_layout)
    candidates = (
        preferred,
        rolling_shard_dir_name(start_month, end_month, PVC_LAYOUT_V2),
        rolling_shard_dir_name(start_month, end_month, LEGACY_LAYOUT),
    )
    return tuple(dict.fromkeys(candidates))


def yearly_shard_dir_candidates(
    year: int,
    *,
    preferred_layout: str = LEGACY_LAYOUT,
) -> tuple[str, ...]:
    preferred = yearly_shard_dir_name(year, preferred_layout)
    candidates = (
        preferred,
        yearly_shard_dir_name(year, PVC_LAYOUT_V2),
        yearly_shard_dir_name(year, LEGACY_LAYOUT),
    )
    return tuple(dict.fromkeys(candidates))


def prediction_shard_dirs(path: Path) -> list[Path]:
    prefixes = ("fold_", "month_", "year_")
    return sorted(item for prefix in prefixes for item in path.glob(f"{prefix}*") if item.is_dir())
