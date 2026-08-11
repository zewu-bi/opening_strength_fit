from __future__ import annotations

import argparse
import json
from pathlib import Path

from opening_strength_fit.storage_inventory import TreeInventory, inventory_tree

DEFAULT_ROOTS = ("output", "experiments/results")


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f}{unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _json_record(inventory: TreeInventory) -> dict[str, object]:
    return {
        "path": str(inventory.path),
        "files": inventory.files,
        "bytes": inventory.bytes,
        "stale_files": inventory.stale_files,
        "stale_bytes": inventory.stale_bytes,
        "oldest_mtime": inventory.oldest_mtime.isoformat() if inventory.oldest_mtime else None,
        "newest_mtime": inventory.newest_mtime.isoformat() if inventory.newest_mtime else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory rebuildable local artifacts without deleting them."
    )
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    parser.add_argument("--older-than-days", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        inventories = [
            inventory_tree(Path(root), older_than_days=args.older_than_days) for root in args.roots
        ]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps([_json_record(item) for item in inventories], indent=2, sort_keys=True))
        return

    print(f"storage_inventory (read-only; stale means older than {args.older_than_days} days):")
    for item in inventories:
        print(
            f"  {item.path}: files={item.files}, size={_format_bytes(item.bytes)}, "
            f"stale_files={item.stale_files}, stale_size={_format_bytes(item.stale_bytes)}"
        )
    print("  deleted: no")


if __name__ == "__main__":
    main()
