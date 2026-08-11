from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class TreeInventory:
    path: Path
    files: int
    bytes: int
    stale_files: int
    stale_bytes: int
    oldest_mtime: datetime | None
    newest_mtime: datetime | None


def inventory_tree(
    path: Path,
    *,
    older_than_days: int,
    now: datetime | None = None,
) -> TreeInventory:
    if older_than_days < 0:
        raise ValueError("older_than_days must be non-negative")
    current_time = now or datetime.now(UTC)
    cutoff = current_time - timedelta(days=older_than_days)
    total_files = 0
    total_bytes = 0
    stale_files = 0
    stale_bytes = 0
    oldest: datetime | None = None
    newest: datetime | None = None

    if path.exists():
        for candidate in path.rglob("*"):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            stat = candidate.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            total_files += 1
            total_bytes += stat.st_size
            oldest = modified if oldest is None else min(oldest, modified)
            newest = modified if newest is None else max(newest, modified)
            if modified < cutoff:
                stale_files += 1
                stale_bytes += stat.st_size

    return TreeInventory(
        path=path,
        files=total_files,
        bytes=total_bytes,
        stale_files=stale_files,
        stale_bytes=stale_bytes,
        oldest_mtime=oldest,
        newest_mtime=newest,
    )
