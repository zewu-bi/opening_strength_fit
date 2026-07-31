from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "canonical" / "opening.toml"


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_opening_baseline_registry_uses_short_names_and_completed_sources() -> None:
    registry = _load(REGISTRY)

    assert registry["cache"]["base_name"] == "opening_base"
    assert registry["cache"]["name"] == "opening_cache"
    assert registry["model"]["name"] == "opening_model"
    assert registry["baseline"]["name"] == "opening_model"

    for source_key in ("source_base_run_id", "source_target_run_id"):
        source = registry["cache"][source_key]
        config = _load(ROOT / "experiments" / "runs" / f"{source}.toml")
        assert config["run"]["id"] == source
        assert config["run"]["status"] == "completed"

    source = registry["model"]["source_run_id"]
    model = _load(ROOT / "experiments" / "runs" / f"{source}.toml")
    assert model["run"]["id"] == source
    assert model["run"]["status"] == "completed"
    assert model["data"]["labeled_path"] == registry["cache"]["path"]
    assert model["output"]["k8s_dir"] == registry["model"]["path"]


def test_opening_cache_registry_fixes_causal_clock_state_contract() -> None:
    registry = _load(REGISTRY)
    cache = registry["cache"]

    assert cache["decision_alignment"] == "clock_state"
    assert cache["entry_alignment"] == "clock_state"
    assert cache["entry_clock_delay_seconds"] == 6
    assert cache["future_alignment"] == "clock_state"
    assert cache["window"] == "09:31-09:40"
    assert cache["target"] == ("xs_norm(short_return) + 0.30 * xs_norm(next_close_return)")
    assert registry["naming"]["generic_version_suffixes_allowed"] is False
