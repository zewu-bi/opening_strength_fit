from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS
from opening_strength_fit.io import read_frame
from opening_strength_fit.pvc_layout import prediction_shard_dirs
from opening_strength_fit.schema import normalize_decision_keys


def prediction_shard_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for shard_dir in prediction_shard_dirs(path):
        single = shard_dir / "predictions.parquet"
        if single.exists():
            files.append(single)
            continue
        shard_files = sorted(shard_dir.glob("predictions_*.parquet"))
        if shard_files:
            files.extend(shard_files)
    return files


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"prediction input does not exist: {path}")
    raw_dir = path / "raw"
    if raw_dir.exists():
        files = sorted(raw_dir.glob("predictions_*.parquet"))
        if files:
            return files
    combined = path / "predictions_all.parquet"
    if combined.exists():
        return [combined]
    single = path / "predictions.parquet"
    if single.exists():
        return [single]
    sharded_files = prediction_shard_files(path)
    if sharded_files:
        return sharded_files
    files = sorted(path.glob("predictions_*.parquet"))
    if files:
        return files
    raise SystemExit(f"no prediction parquet files found under: {path}")


def next_close_files(path: Path, years: set[str]) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"next-close label input does not exist: {path}")
    files = sorted(path.glob("*.parquet"))
    if years:
        matched = [file for file in files if any(year in file.name for year in sorted(years))]
        if matched:
            return matched
    if files:
        return files
    raise SystemExit(f"no next-close parquet files found under: {path}")


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    return normalize_decision_keys(frame, drop_missing=False)


def clock_label(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    return timestamps.dt.strftime("%H:%M")


def read_prediction_frames(
    paths: list[str],
    *,
    score_col: str,
    label_col: str,
) -> pd.DataFrame:
    required = [*KEY_COLUMNS, score_col, label_col]
    files = [file for raw in paths for file in prediction_files(Path(raw))]
    frames = [read_frame(file, columns=required) for file in files]
    if not frames:
        raise SystemExit("no prediction files supplied")
    return normalize_keys(pd.concat(frames, ignore_index=True))
