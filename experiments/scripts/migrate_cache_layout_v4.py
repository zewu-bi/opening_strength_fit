from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

MAPPINGS = (
    {
        "old": "opening_13y_201301_202512_delay2_base_labeled_v2",
        "new": "opening_2013_2025_label_v1_tick2_physical_base",
        "years": range(2013, 2026),
        "old_file": "opening_{year}_delay2_base_labeled_v2.parquet",
        "new_file": "opening_{year}_label_v1_tick2_physical_base.parquet",
    },
    {
        "old": "opening_13y_201301_202512_delay2_mixed_w030_labeled_v1",
        "new": "opening_2013_2025_label_v1_tick2_physical_mixed_w030",
        "years": range(2013, 2026),
        "old_file": "opening_{year}_delay2_mixed_w030_labeled_v1.parquet",
        "new_file": "opening_{year}_label_v1_tick2_physical_mixed_w030.parquet",
    },
    {
        "old": "opening_13y_201301_202512_delay2_next_close_labels_v1",
        "new": "opening_2013_2025_next_close_labels_v1",
        "years": range(2013, 2026),
        "old_file": "opening_{year}_next_close_labels_v1.parquet",
        "new_file": "opening_{year}_next_close_labels_v1.parquet",
    },
    {
        "old": "opening_2019_2025_delay2_base_labeled_v3_conservative_mcap_lag1_unique_ticks",
        "new": "opening_2019_2025_label_v2_tick2_unique_base_mcap_lag1",
        "years": range(2019, 2026),
        "old_file": (
            "opening_{year}_delay2_base_labeled_v3_conservative_mcap_lag1_unique_ticks.parquet"
        ),
        "new_file": "opening_{year}_label_v2_tick2_unique_base_mcap_lag1.parquet",
    },
    {
        "old": "opening_2019_2025_delay2_base_labeled_v4_auction_fresh_mcap_lag1",
        "new": "opening_2019_2025_label_v3_tick2_gap5_ready_base_mcap_lag1",
        "years": range(2019, 2026),
        "old_file": ("opening_{year}_delay2_base_labeled_v4_auction_fresh_mcap_lag1.parquet"),
        "new_file": ("opening_{year}_label_v3_tick2_gap5_ready_base_mcap_lag1.parquet"),
    },
    {
        "old": "opening_2019_2025_delay2_mixed_w030_labeled_v3_auction_fresh_mcap_lag1",
        "new": "opening_2019_2025_label_v3_tick2_gap5_ready_mixed_w030_mcap_lag1",
        "years": range(2019, 2026),
        "old_file": ("opening_{year}_delay2_mixed_w030_labeled_v3_auction_fresh_mcap_lag1.parquet"),
        "new_file": ("opening_{year}_label_v3_tick2_gap5_ready_mixed_w030_mcap_lag1.parquet"),
    },
    {
        "old": ("opening_2019_2025_delay6_clock_state_base_labeled_v1_mcap_lag1_unique_ticks"),
        "new": "opening_2019_2025_label_v4_clock6_state_unique_base_mcap_lag1",
        "years": range(2019, 2026),
        "old_file": (
            "opening_{year}_delay6_clock_state_base_labeled_v1_mcap_lag1_unique_ticks.parquet"
        ),
        "new_file": ("opening_{year}_label_v4_clock6_state_unique_base_mcap_lag1.parquet"),
        "remove_incomplete_locks": True,
    },
)

STALE_DIRECTORY = "opening_2019_2025_delay2_base_labeled_v3_auction_fresh"
RECORD_NAME = "cache_label_layout_v4_20260717.json"


def _replace_strings(value: object, replacements: tuple[tuple[str, str], ...]) -> object:
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def _rewrite_json(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = _replace_strings(payload, replacements)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _link_alias(alias: Path, target_name: str) -> None:
    if alias.is_symlink():
        if alias.readlink() != Path(target_name):
            raise SystemExit(f"alias points to unexpected target: {alias}")
        return
    if alias.exists():
        raise SystemExit(f"cannot create alias over existing path: {alias}")
    alias.symlink_to(target_name)


def _safe_remove_lock(lock_path: Path) -> None:
    children = list(lock_path.iterdir())
    if any(child.name != "heartbeat" or not child.is_file() for child in children):
        raise SystemExit(f"refusing to remove non-heartbeat lock contents: {lock_path}")
    shutil.rmtree(lock_path)


def _remove_stale_directory(root: Path, actions: list[dict[str, object]]) -> None:
    path = root / STALE_DIRECTORY
    if not path.exists():
        return
    if path.is_symlink():
        raise SystemExit(f"refusing to remove stale-directory symlink: {path}")
    if list(path.glob("*.parquet")) or list(path.glob("*.manifest.json")):
        raise SystemExit(f"refusing to remove stale directory containing cache data: {path}")
    for child in list(path.iterdir()):
        if not child.is_dir() or not child.name.endswith(".parquet.lock"):
            raise SystemExit(f"unexpected stale-directory entry: {child}")
        _safe_remove_lock(child)
    path.rmdir()
    actions.append({"action": "remove_stale_directory", "path": str(path)})


def _migrate_mapping(
    root: Path,
    mapping: dict[str, object],
    actions: list[dict[str, object]],
) -> None:
    old_dir = root / str(mapping["old"])
    new_dir = root / str(mapping["new"])
    if old_dir.is_symlink():
        if old_dir.resolve() != new_dir.resolve():
            raise SystemExit(f"directory alias points to unexpected target: {old_dir}")
    elif old_dir.exists():
        if new_dir.exists():
            raise SystemExit(f"both old and new cache directories exist: {old_dir}, {new_dir}")
        old_dir.rename(new_dir)
        _link_alias(old_dir, new_dir.name)
        actions.append(
            {
                "action": "rename_directory",
                "old": str(old_dir),
                "new": str(new_dir),
                "compatibility_alias": str(old_dir),
            }
        )
    elif not new_dir.exists():
        if mapping.get("remove_incomplete_locks"):
            new_dir.mkdir(parents=True)
            _link_alias(old_dir, new_dir.name)
            actions.append(
                {
                    "action": "create_empty_canonical_directory",
                    "new": str(new_dir),
                    "compatibility_alias": str(old_dir),
                }
            )
        else:
            raise SystemExit(f"missing cache directory: {old_dir}")

    if mapping.get("remove_incomplete_locks"):
        if list(new_dir.glob("*.parquet")) or list(new_dir.glob("*.manifest.json")):
            raise SystemExit(
                f"refusing to clean incomplete locks after cache publication: {new_dir}"
            )
        for lock_path in list(new_dir.glob("*.parquet.lock")):
            _safe_remove_lock(lock_path)
            actions.append({"action": "remove_incomplete_lock", "path": str(lock_path)})

    old_dir_name = str(mapping["old"])
    new_dir_name = str(mapping["new"])
    for year in mapping["years"]:
        old_name = str(mapping["old_file"]).format(year=year)
        new_name = str(mapping["new_file"]).format(year=year)
        old_file = new_dir / old_name
        new_file = new_dir / new_name
        replacements = (
            (old_dir_name, new_dir_name),
            (old_name, new_name),
        )
        if old_name != new_name:
            if old_file.is_symlink():
                if old_file.readlink() != Path(new_name):
                    raise SystemExit(f"file alias points to unexpected target: {old_file}")
            elif old_file.exists():
                if new_file.exists():
                    raise SystemExit(f"both old and new cache files exist: {old_file}, {new_file}")
                old_file.rename(new_file)
                _link_alias(old_file, new_name)
                actions.append(
                    {
                        "action": "rename_file",
                        "old": str(old_file),
                        "new": str(new_file),
                        "compatibility_alias": str(old_file),
                    }
                )
            elif not new_file.exists() and not mapping.get("remove_incomplete_locks"):
                raise SystemExit(f"missing cache file: {old_file}")
        elif old_file.exists():
            new_file = old_file

        for suffix in (".manifest.json", ".lock.done"):
            old_sidecar = new_dir / f"{old_name}{suffix}"
            new_sidecar = new_dir / f"{new_name}{suffix}"
            if old_name != new_name:
                if old_sidecar.is_symlink():
                    if old_sidecar.readlink() != Path(new_sidecar.name):
                        raise SystemExit(
                            f"sidecar alias points to unexpected target: {old_sidecar}"
                        )
                elif old_sidecar.exists():
                    if new_sidecar.exists():
                        raise SystemExit(
                            f"both old and new cache sidecars exist: {old_sidecar}, {new_sidecar}"
                        )
                    old_sidecar.rename(new_sidecar)
                    _link_alias(old_sidecar, new_sidecar.name)
                    actions.append(
                        {
                            "action": "rename_sidecar",
                            "old": str(old_sidecar),
                            "new": str(new_sidecar),
                            "compatibility_alias": str(old_sidecar),
                        }
                    )
            if new_sidecar.exists() and not new_sidecar.is_symlink():
                _rewrite_json(new_sidecar, replacements)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/output/opening_strength_fit/cache"),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("refusing to mutate PVC without --apply")

    root = args.root
    actions: list[dict[str, object]] = []
    _remove_stale_directory(root, actions)
    for mapping in MAPPINGS:
        _migrate_mapping(root, mapping, actions)

    record = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "canonical_pattern": (
            "opening_<range>_label_vN_<entry_semantics>_<base|mixed>_<enrichment>"
        ),
        "actions": actions,
        "mappings": [
            {
                key: list(value) if isinstance(value, range) else value
                for key, value in mapping.items()
            }
            for mapping in MAPPINGS
        ],
        "removed": [STALE_DIRECTORY],
    }
    record_dir = root.parent / ".layout_migrations"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / RECORD_NAME
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"record": str(record_path), "actions": actions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
