from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


def cache_lock_done_path(lock_path: Path) -> Path:
    return Path(f"{lock_path}.done")


def cache_lock_heartbeat_path(lock_path: Path) -> Path:
    return lock_path / "heartbeat"


def write_cache_lock_heartbeat(lock_path: Path) -> None:
    try:
        lock_path.mkdir(parents=True, exist_ok=True)
        cache_lock_heartbeat_path(lock_path).write_text(
            json.dumps({"pid": os.getpid(), "time": time.time()}) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def cache_lock_has_fresh_heartbeat(
    lock_path: Path,
    *,
    stale_after_seconds: float,
) -> bool:
    heartbeat_path = cache_lock_heartbeat_path(lock_path)
    try:
        heartbeat_age = time.time() - heartbeat_path.stat().st_mtime
    except OSError:
        return False
    return heartbeat_age <= float(stale_after_seconds)


class CacheLockHeartbeat:
    def __init__(self, lock_path: Path, interval_seconds: float = 60.0) -> None:
        self.lock_path = lock_path
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> CacheLockHeartbeat:
        write_cache_lock_heartbeat(self.lock_path)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self.interval_seconds))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            write_cache_lock_heartbeat(self.lock_path)


def mark_cache_ready(cache_path: Path, lock_path: Path) -> None:
    cache_lock_done_path(lock_path).write_text(
        json.dumps(
            {
                "path": str(cache_path),
                "bytes": cache_path.stat().st_size,
                "time": time.time(),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def clear_cache_ready(lock_path: Path) -> None:
    try:
        cache_lock_done_path(lock_path).unlink()
    except FileNotFoundError:
        return


def acquire_cache_lock(
    lock_path: Path,
    timeout_seconds: float,
    *,
    cache_path: Path | None = None,
    cache_read: bool = True,
    poll_seconds: float = 15.0,
) -> str:
    start = time.monotonic()
    timeout_seconds = float(timeout_seconds)
    heartbeat_stale_after = max(timeout_seconds, float(poll_seconds) * 3.0, 60.0)
    while True:
        if cache_path and cache_read and cache_path.exists():
            return "cache_ready"
        try:
            lock_path.mkdir(parents=True)
            write_cache_lock_heartbeat(lock_path)
            return "acquired"
        except FileExistsError:
            if (
                cache_lock_done_path(lock_path).exists()
                and cache_path
                and cache_read
                and cache_path.exists()
            ):
                return "cache_ready"
            if timeout_seconds > 0.0 and (time.monotonic() - start) >= timeout_seconds:
                if cache_lock_has_fresh_heartbeat(
                    lock_path,
                    stale_after_seconds=heartbeat_stale_after,
                ):
                    start = time.monotonic()
                else:
                    return "timeout"
            time.sleep(float(poll_seconds))


def release_cache_lock(lock_path: Path) -> None:
    try:
        for child in lock_path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
        lock_path.rmdir()
    except FileNotFoundError:
        return
