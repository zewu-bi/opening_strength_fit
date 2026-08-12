from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

FrameFilters = list[tuple[str, str, object]]


def _unique_directory_files(files: list[Path]) -> list[Path]:
    """Prefer real files and de-duplicate compatibility symlinks by target."""

    unique: list[Path] = []
    seen: set[Path] = set()
    for file in sorted(files, key=lambda item: (item.is_symlink(), str(item))):
        target = file.resolve()
        if target in seen:
            continue
        seen.add(target)
        unique.append(file)
    return unique


def frame_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise SystemExit(f"input path does not exist: {path}")
    files = _unique_directory_files(list(path.rglob("*.parquet")))
    if not files:
        files = _unique_directory_files(list(path.rglob("*.csv")) + list(path.rglob("*.csv.gz")))
    if not files:
        raise SystemExit(f"no parquet/csv files found under directory: {path}")
    return files


def frame_files_many(paths: Iterable[str | Path]) -> list[Path]:
    return [file for path in paths for file in frame_files(path)]


def _filter_frame(frame: pd.DataFrame, filters: FrameFilters | None) -> pd.DataFrame:
    if not filters:
        return frame
    mask = pd.Series(True, index=frame.index)
    for column, op, value in filters:
        if column not in frame.columns:
            raise SystemExit(f"filter column is missing from input frame: {column}")
        values = frame[column]
        if op == "==":
            mask &= values == value
        elif op == "!=":
            mask &= values != value
        elif op == ">=":
            mask &= values >= value
        elif op == ">":
            mask &= values > value
        elif op == "<=":
            mask &= values <= value
        elif op == "<":
            mask &= values < value
        elif op == "in":
            mask &= values.isin(value)
        else:
            raise SystemExit(f"unsupported frame filter operator: {op}")
    return frame.loc[mask].copy()


def _read_frame_file(
    path: Path,
    columns: list[str] | None = None,
    filters: FrameFilters | None = None,
) -> pd.DataFrame:
    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns, filters=filters)
    if path.suffix.lower() == ".csv" or suffixes == ".csv.gz":
        return _filter_frame(pd.read_csv(path, usecols=columns), filters)
    raise SystemExit(f"unsupported input format: {path.suffix}")


def read_frame(
    path: str | Path,
    columns: list[str] | None = None,
    filters: FrameFilters | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"input path does not exist: {path}")

    if path.is_dir():
        files = frame_files(path)
        frames = [_read_frame_file(file, columns=columns, filters=filters) for file in files]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return _read_frame_file(path, columns=columns, filters=filters)


def frame_columns(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"input path does not exist: {path}")
    if path.is_dir():
        files = frame_files(path)
        columns: set[str] = set()
        for file in files:
            columns |= frame_columns(file)
        return columns

    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    if path.suffix.lower() == ".csv" or suffixes == ".csv.gz":
        return set(pd.read_csv(path, nrows=0).columns)
    raise SystemExit(f"unsupported input format: {path.suffix}")


def available_frame_columns(files: Iterable[Path]) -> set[str]:
    columns: set[str] = set()
    for file in files:
        columns |= frame_columns(file)
    return columns


def select_available_columns(
    required: Iterable[str], optional: Iterable[str], available: set[str]
) -> list[str]:
    return list(
        dict.fromkeys((*required, *(item for item in optional if item and item in available)))
    )


def read_frame_files(
    files: list[Path],
    *,
    columns: list[str],
    required: Iterable[str],
) -> pd.DataFrame:
    required_set = set(required)
    frames = []
    for file in files:
        available = frame_columns(file)
        missing = sorted(required_set - available)
        if missing:
            raise SystemExit(f"{file}: missing required columns: {missing}")
        frame = read_frame(file, columns=[column for column in columns if column in available])
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        print(f"  {file}: rows={len(frame)}")
        frames.append(frame[columns])
    if not frames:
        raise SystemExit("no input files supplied")
    return pd.concat(frames, ignore_index=True)


def merge_frame_support(
    frame: pd.DataFrame,
    support: pd.DataFrame,
    *,
    keys: Iterable[str],
) -> pd.DataFrame:
    if support.empty:
        return frame
    key_columns = list(keys)
    keyed = support.drop_duplicates(key_columns, keep="last")
    overlap = [
        column for column in keyed.columns if column not in key_columns and column in frame.columns
    ]
    merged = frame.merge(
        keyed,
        on=key_columns,
        how="left",
        suffixes=("", "_support"),
        validate="many_to_one",
    )
    for column in overlap:
        support_col = f"{column}_support"
        if support_col in merged.columns:
            merged[column] = merged[column].combine_first(merged.pop(support_col))
    return merged


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


def csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.select_dtypes(include=["datetime", "datetimetz"]).columns:
        out[column] = out[column].dt.strftime("%Y-%m-%d")
    return out


def write_frame_atomic(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes[-2:]).lower()
    suffix = ".csv.gz" if suffixes == ".csv.gz" else path.suffix
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp{suffix}")
    try:
        write_frame(df, tmp_path, index=index)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def resolve_path(root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root) / path
