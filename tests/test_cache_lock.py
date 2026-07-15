from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from opening_strength_fit.cache_lock import (
    acquire_cache_lock,
    release_cache_lock,
    write_cache_lock_heartbeat,
)


class CacheLockTest(unittest.TestCase):
    def test_waiter_continues_past_timeout_when_heartbeat_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "labeled.parquet"
            lock_path = Path(f"{cache_path}.lock")
            lock_path.mkdir()
            write_cache_lock_heartbeat(lock_path)

            def create_cache() -> None:
                time.sleep(0.08)
                cache_path.write_text("ready\n", encoding="utf-8")

            writer = threading.Thread(target=create_cache)
            writer.start()
            try:
                status = acquire_cache_lock(
                    lock_path,
                    timeout_seconds=0.02,
                    cache_path=cache_path,
                    poll_seconds=0.01,
                )
            finally:
                writer.join(timeout=1.0)
                release_cache_lock(lock_path)

            self.assertEqual(status, "cache_ready")

    def test_waiter_requires_companion_ready_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "labeled.parquet"
            manifest_path = root / "labeled.parquet.manifest.json"
            lock_path = Path(f"{cache_path}.lock")
            lock_path.mkdir()
            write_cache_lock_heartbeat(lock_path)

            def create_cache_and_manifest() -> None:
                cache_path.write_text("cache\n", encoding="utf-8")
                time.sleep(0.08)
                manifest_path.write_text("{}\n", encoding="utf-8")

            writer = threading.Thread(target=create_cache_and_manifest)
            writer.start()
            try:
                status = acquire_cache_lock(
                    lock_path,
                    timeout_seconds=0.02,
                    cache_path=cache_path,
                    ready_paths=(manifest_path,),
                    poll_seconds=0.01,
                )
            finally:
                writer.join(timeout=1.0)
                release_cache_lock(lock_path)

            self.assertEqual(status, "cache_ready")

    def test_waiter_times_out_without_fresh_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "labeled.parquet"
            lock_path = Path(f"{cache_path}.lock")
            lock_path.mkdir()

            status = acquire_cache_lock(
                lock_path,
                timeout_seconds=0.02,
                cache_path=cache_path,
                poll_seconds=0.01,
            )
            release_cache_lock(lock_path)

            self.assertEqual(status, "timeout")

    def test_release_cache_lock_removes_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "labeled.parquet.lock"
            lock_path.mkdir()
            write_cache_lock_heartbeat(lock_path)

            release_cache_lock(lock_path)

            self.assertFalse(lock_path.exists())


if __name__ == "__main__":
    unittest.main()
