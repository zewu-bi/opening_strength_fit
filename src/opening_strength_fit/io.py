from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_frame_file(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv" or suffixes == ".csv.gz":
        return pd.read_csv(path, usecols=columns)
    raise SystemExit(f"unsupported input format: {path.suffix}")


def read_frame(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"input path does not exist: {path}")

    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            files = sorted(path.rglob("*.csv")) + sorted(path.rglob("*.csv.gz"))
        if not files:
            raise SystemExit(f"no parquet/csv files found under directory: {path}")
        frames = [_read_frame_file(file, columns=columns) for file in files]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return _read_frame_file(path, columns=columns)


def write_frame(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=index)
        return
    if path.suffix.lower() == ".csv" or suffixes == ".csv.gz":
        df.to_csv(path, index=index)
        return
    raise SystemExit(f"unsupported output format: {path.suffix}")


def resolve_path(root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root) / path
