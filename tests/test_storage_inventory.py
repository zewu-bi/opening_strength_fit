from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from opening_strength_fit.storage_inventory import inventory_tree


def test_inventory_tree_reports_total_and_stale_bytes(tmp_path: Path) -> None:
    old_file = tmp_path / "old.bin"
    new_file = tmp_path / "nested" / "new.bin"
    new_file.parent.mkdir()
    old_file.write_bytes(b"old")
    new_file.write_bytes(b"newer")
    now = datetime(2026, 8, 11, tzinfo=UTC)
    old_time = (now - timedelta(days=31)).timestamp()
    new_time = (now - timedelta(days=1)).timestamp()
    old_file.touch()
    new_file.touch()
    old_file.chmod(0o600)
    new_file.chmod(0o600)
    os.utime(old_file, (old_time, old_time))
    os.utime(new_file, (new_time, new_time))

    result = inventory_tree(tmp_path, older_than_days=30, now=now)

    assert result.files == 2
    assert result.bytes == 8
    assert result.stale_files == 1
    assert result.stale_bytes == 3


def test_inventory_tree_rejects_negative_age(tmp_path: Path) -> None:
    try:
        inventory_tree(tmp_path, older_than_days=-1)
    except ValueError as exc:
        assert str(exc) == "older_than_days must be non-negative"
    else:
        raise AssertionError("negative age must fail")
