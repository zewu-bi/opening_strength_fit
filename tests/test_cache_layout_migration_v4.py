from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

migration = SimpleNamespace(
    **runpy.run_path(Path(__file__).parents[1] / "experiments/scripts/migrate_cache_layout_v4.py")
)


def _mapping(*, remove_incomplete_locks: bool = False) -> dict[str, object]:
    mapping: dict[str, object] = {
        "old": "opening_old",
        "new": "opening_new",
        "years": range(2025, 2026),
        "old_file": "opening_{year}_old.parquet",
        "new_file": "opening_{year}_new.parquet",
    }
    if remove_incomplete_locks:
        mapping["remove_incomplete_locks"] = True
    return mapping


def test_migrate_mapping_renames_cache_without_compatibility_aliases(
    tmp_path: Path,
) -> None:
    old_dir = tmp_path / "opening_old"
    old_dir.mkdir()
    old_file = old_dir / "opening_2025_old.parquet"
    old_file.write_bytes(b"cache")
    for suffix in (".manifest.json", ".lock.done"):
        sidecar = old_dir / f"{old_file.name}{suffix}"
        sidecar.write_text(
            json.dumps({"path": str(old_file), "name": old_file.name}),
            encoding="utf-8",
        )

    actions: list[dict[str, object]] = []
    migration._migrate_mapping(tmp_path, _mapping(), actions)

    new_dir = tmp_path / "opening_new"
    new_file = new_dir / "opening_2025_new.parquet"
    assert not old_dir.exists()
    assert new_file.read_bytes() == b"cache"
    assert not (new_dir / old_file.name).exists()
    for suffix in (".manifest.json", ".lock.done"):
        old_sidecar = new_dir / f"{old_file.name}{suffix}"
        new_sidecar = new_dir / f"{new_file.name}{suffix}"
        assert not old_sidecar.exists()
        payload = json.loads(new_sidecar.read_text(encoding="utf-8"))
        assert payload == {"path": str(new_file), "name": new_file.name}

    repeated_actions: list[dict[str, object]] = []
    migration._migrate_mapping(tmp_path, _mapping(), repeated_actions)
    assert repeated_actions == []


def test_remove_stale_directory_accepts_only_heartbeat_locks(tmp_path: Path) -> None:
    stale = tmp_path / migration.STALE_DIRECTORY
    lock = stale / "opening_2025.parquet.lock"
    lock.mkdir(parents=True)
    (lock / "heartbeat").write_text("stale", encoding="utf-8")

    actions: list[dict[str, object]] = []
    migration._remove_stale_directory(tmp_path, actions)

    assert not stale.exists()
    assert actions == [{"action": "remove_stale_directory", "path": str(stale)}]


def test_migrate_mapping_refuses_ambiguous_old_and_new_files(tmp_path: Path) -> None:
    old_dir = tmp_path / "opening_old"
    old_dir.mkdir()
    (old_dir / "opening_2025_old.parquet").write_bytes(b"old")
    (old_dir / "opening_2025_new.parquet").write_bytes(b"new")

    with pytest.raises(SystemExit, match="both old and new cache files exist"):
        migration._migrate_mapping(tmp_path, _mapping(), [])


def test_migrate_mapping_removes_degraded_zero_byte_aliases(tmp_path: Path) -> None:
    old_dir = tmp_path / "opening_old"
    old_dir.touch()
    new_dir = tmp_path / "opening_new"
    new_dir.mkdir()
    new_file = new_dir / "opening_2025_new.parquet"
    new_file.write_bytes(b"cache")
    (new_dir / "opening_2025_old.parquet").touch()

    actions: list[dict[str, object]] = []
    migration._migrate_mapping(tmp_path, _mapping(), actions)

    assert not old_dir.exists()
    assert new_file.read_bytes() == b"cache"
    assert not (new_dir / "opening_2025_old.parquet").exists()
    assert [action["artifact_type"] for action in actions] == [
        "degraded_zero_byte_alias",
        "degraded_zero_byte_alias",
    ]


def test_incomplete_clock_mapping_removes_locks_without_cache_data(
    tmp_path: Path,
) -> None:
    old_dir = tmp_path / "opening_old"
    lock = old_dir / "opening_2025_old.parquet.lock"
    lock.mkdir(parents=True)
    (lock / "heartbeat").write_text("stale", encoding="utf-8")

    actions: list[dict[str, object]] = []
    migration._migrate_mapping(
        tmp_path,
        _mapping(remove_incomplete_locks=True),
        actions,
    )

    new_dir = tmp_path / "opening_new"
    assert not old_dir.exists()
    assert not list(new_dir.iterdir())
    assert {action["action"] for action in actions} == {
        "rename_directory",
        "remove_incomplete_lock",
    }
