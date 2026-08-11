from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import KEY_COLUMNS, write_json
from opening_strength_fit.feature_utils import finite_numeric
from opening_strength_fit.io import frame_columns, read_frame, write_frame
from opening_strength_fit.prediction_frames import prediction_files
from opening_strength_fit.schema import normalize_decision_keys

KEYS = list(KEY_COLUMNS)
DEFAULT_ASK_LEVELS = tuple(range(1, 11))
CONTEXT_OUTPUT_COLUMNS = (
    "capacity_price",
    "ask_price_1",
    "bid_price_1",
    "mid_price",
    "spread_bps",
    "ask1_to_limit_up_bps",
    "ask_depth_notional",
    "status",
)


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    selected_path: Path
    prediction_root: Path
    output_dir: Path


def diagnostic_cases_from_config(
    config: dict,
    *,
    section_name: str,
    output_root: Path,
) -> list[DiagnosticCase]:
    section = config.get(section_name, {})
    inputs = section.get("inputs", []) if isinstance(section, dict) else []
    if not isinstance(inputs, list):
        raise SystemExit(f"[{section_name}].inputs must be an array of tables")
    cases = []
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise SystemExit(f"each [[{section_name}.inputs]] entry must be a table")
        name = str(item.get("name", f"case_{index}")).strip()
        selected_path = str(item.get("selected_path", "")).strip()
        prediction_root = str(item.get("prediction_root", "")).strip()
        if not selected_path or not prediction_root:
            raise SystemExit(
                f"each [[{section_name}.inputs]] entry requires selected_path and prediction_root"
            )
        case_output_dir = str(item.get("output_dir", "")).strip()
        cases.append(
            DiagnosticCase(
                name=name,
                selected_path=Path(selected_path),
                prediction_root=Path(prediction_root),
                output_dir=Path(case_output_dir) if case_output_dir else output_root / name,
            )
        )
    return cases


def diagnostic_case_from_values(
    *,
    selected_path: str,
    prediction_root: str,
    name: str,
    case_output_dir: str,
    output_root: Path,
) -> list[DiagnosticCase]:
    if not (selected_path or prediction_root):
        return []
    if not selected_path or not prediction_root:
        raise SystemExit("--selected-path and --prediction-root must be supplied together")
    return [
        DiagnosticCase(
            name=name or "case",
            selected_path=Path(selected_path),
            prediction_root=Path(prediction_root),
            output_dir=Path(case_output_dir) if case_output_dir else output_root,
        )
    ]


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    return normalize_decision_keys(frame, key_columns=KEYS)


def _csv_selected_chunks(path: Path, *, columns: list[str], name: str) -> list[pd.DataFrame]:
    chunks = []
    for index, chunk in enumerate(pd.read_csv(path, usecols=columns, chunksize=500_000), 1):
        chunk = normalize_keys(chunk)
        chunks.append(chunk)
        print(f"{name}: selected chunk {index} rows={len(chunk)}", flush=True)
    return chunks


def load_selected(path: Path, *, columns: list[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{name}: selected input does not exist: {path}")
    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".csv" or suffixes == ".csv.gz":
        chunks = _csv_selected_chunks(path, columns=columns, name=name)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
    frame = read_frame(path, columns=columns)
    return normalize_keys(frame)


def load_selected_allocations(path: Path, *, name: str) -> pd.DataFrame:
    print(f"{name}: loading selected allocations from {path}", flush=True)
    selected = load_selected(path, columns=[*KEYS, "allocated_notional"], name=name)
    if "allocated_notional" not in selected:
        raise SystemExit(f"{name}: selected input missing allocated_notional: {path}")
    selected["allocated_notional"] = finite_numeric(selected["allocated_notional"]).fillna(0.0)
    print(f"{name}: selected rows={len(selected)}", flush=True)
    return selected


def load_selected_keys(path: Path, *, name: str) -> pd.DataFrame:
    print(f"{name}: loading selected keys from {path}", flush=True)
    selected = load_selected(path, columns=KEYS, name=name).drop_duplicates(KEYS)
    print(f"{name}: selected unique keys={len(selected)}", flush=True)
    return selected


def read_order_book(
    file: Path, selected_keys: pd.DataFrame, *, levels: tuple[int, ...]
) -> pd.DataFrame:
    available = frame_columns(file)
    missing_keys = sorted(set(KEYS) - available)
    if missing_keys:
        raise SystemExit(f"{file}: missing key columns: {missing_keys}")

    level_columns = []
    for level in levels:
        for prefix in ("ask_price", "ask_volume"):
            column = f"{prefix}_{level}"
            if column in available:
                level_columns.append(column)
    if not level_columns:
        raise SystemExit(f"{file}: no ask_price_i/ask_volume_i columns")

    frame = read_frame(file, columns=[*KEYS, *level_columns])
    frame = normalize_keys(frame)
    return frame.merge(selected_keys, on=KEYS, how="inner")


def ask_level_stats(frame: pd.DataFrame, *, levels: tuple[int, ...]) -> dict[str, float]:
    allocated = finite_numeric(frame["allocated_notional"]).fillna(0.0).clip(lower=0.0)
    remaining = allocated.copy()
    out: dict[str, float] = {
        "rows": float(len(frame)),
        "allocated_notional": float(allocated.sum()),
    }
    cumulative_fill = pd.Series(0.0, index=frame.index)
    for level in levels:
        price_col = f"ask_price_{level}"
        volume_col = f"ask_volume_{level}"
        if price_col not in frame.columns or volume_col not in frame.columns:
            level_notional = pd.Series(0.0, index=frame.index)
        else:
            level_notional = finite_numeric(frame[price_col]).fillna(0.0).clip(
                lower=0.0
            ) * finite_numeric(frame[volume_col]).fillna(0.0).clip(lower=0.0)
        fill = pd.Series(np.minimum(remaining, level_notional), index=frame.index).clip(lower=0.0)
        remaining = (remaining - fill).clip(lower=0.0)
        cumulative_fill = cumulative_fill + fill
        out[f"ask{level}_notional"] = float(level_notional.sum())
        out[f"fill_ask{level}"] = float(fill.sum())
        out[f"full_within_ask{level}_rows"] = float((cumulative_fill >= allocated - 1e-9).sum())
    out[f"beyond_ask{max(levels)}"] = float(remaining.sum())
    return out


def combine_ask_level_stats(
    parts: list[dict[str, float]], *, levels: tuple[int, ...]
) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    totals: dict[str, float] = {}
    for part in parts:
        for key, value in part.items():
            totals[key] = totals.get(key, 0.0) + float(value)

    allocated = totals.get("allocated_notional", 0.0)
    rows = totals.get("rows", 0.0)
    records = []
    cumulative_fill = 0.0
    for level in levels:
        fill = totals.get(f"fill_ask{level}", 0.0)
        cumulative_fill += fill
        full_within_rows = totals.get(f"full_within_ask{level}_rows", 0.0)
        records.append(
            {
                "bucket": f"ask{level}",
                "filled_notional": fill,
                "filled_share": fill / allocated if allocated > 0 else np.nan,
                "cumulative_filled_share": cumulative_fill / allocated if allocated > 0 else np.nan,
                "full_within_rows": full_within_rows,
                "full_within_row_share": full_within_rows / rows if rows > 0 else np.nan,
                "displayed_level_notional": totals.get(f"ask{level}_notional", 0.0),
            }
        )
    beyond_key = f"beyond_ask{max(levels)}"
    records.append(
        {
            "bucket": beyond_key,
            "filled_notional": totals.get(beyond_key, 0.0),
            "filled_share": totals.get(beyond_key, 0.0) / allocated if allocated > 0 else np.nan,
            "cumulative_filled_share": 1.0,
            "full_within_rows": np.nan,
            "full_within_row_share": np.nan,
            "displayed_level_notional": np.nan,
        }
    )
    return pd.DataFrame(records)


def write_success_marker(
    output_dir: Path, *, run_name: str, cases: list[dict[str, object]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "_SUCCESS",
        {
            "run_id": run_name,
            "status": "completed",
            "cases": cases,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "format_version": 1,
        },
        ensure_ascii=True,
    )


def run_ask_level_attribution_case(
    case: DiagnosticCase,
    *,
    levels: tuple[int, ...] = DEFAULT_ASK_LEVELS,
) -> Path:
    selected = load_selected_allocations(case.selected_path, name=case.name)
    selected_keys = selected[KEYS].drop_duplicates(KEYS)
    parts = []
    for file in prediction_files(case.prediction_root):
        print(f"{case.name}: reading {file}", flush=True)
        book = read_order_book(file, selected_keys, levels=levels)
        if book.empty:
            print(f"{case.name}: matched rows=0", flush=True)
            continue
        book = book.merge(selected, on=KEYS, how="inner", validate="one_to_one")
        print(f"{case.name}: matched rows={len(book)}", flush=True)
        parts.append(ask_level_stats(book, levels=levels))

    summary = combine_ask_level_stats(parts, levels=levels)
    output_path = case.output_dir / "ask_level_attribution_summary.csv"
    write_frame(summary, output_path)
    print(summary.to_string(index=False) if not summary.empty else "empty", flush=True)
    print(f"{case.name}: wrote {output_path}", flush=True)
    return output_path


def _prediction_context_columns(available: set[str]) -> list[str]:
    desired = {
        "ask_price_1",
        "bid_price_1",
        "mid_price",
        "spread_bps",
        "ask1_to_limit_up_bps",
        "ask_depth_10",
        "status",
        "limit_up_price",
    }
    desired |= {f"ask_volume_{index}" for index in range(1, 11)}
    desired |= {f"ask_price_{index}" for index in range(1, 11)}
    return [column for column in desired if column in available]


def read_prediction_context(file: Path, selected_keys: pd.DataFrame) -> pd.DataFrame:
    available = frame_columns(file)
    missing_keys = sorted(set(KEYS) - available)
    if missing_keys:
        raise SystemExit(f"{file}: missing key columns: {missing_keys}")

    read_columns = [*KEYS, *_prediction_context_columns(available)]
    frame = read_frame(file, columns=read_columns)
    frame = normalize_keys(frame)
    frame = frame.merge(selected_keys, on=KEYS, how="inner")
    if frame.empty:
        return frame

    if "ask_price_1" in frame:
        frame["capacity_price"] = finite_numeric(frame["ask_price_1"])
    if "spread_bps" not in frame and {"ask_price_1", "bid_price_1"} <= set(frame.columns):
        ask = finite_numeric(frame["ask_price_1"])
        bid = finite_numeric(frame["bid_price_1"])
        mid = (ask + bid) / 2.0
        frame["spread_bps"] = np.where(mid > 0, (ask - bid) / mid * 10_000.0, np.nan)
    if "ask1_to_limit_up_bps" not in frame and {"limit_up_price", "ask_price_1"} <= set(
        frame.columns
    ):
        limit_up = finite_numeric(frame["limit_up_price"])
        ask = finite_numeric(frame["ask_price_1"])
        frame["ask1_to_limit_up_bps"] = np.where(
            ask > 0,
            (limit_up - ask) / ask * 10_000.0,
            np.nan,
        )
    if "ask_depth_10" in frame and "ask_price_1" in frame:
        frame["ask_depth_notional"] = finite_numeric(frame["ask_depth_10"]) * finite_numeric(
            frame["ask_price_1"]
        )
    else:
        depth = None
        for index in range(1, 11):
            volume_col = f"ask_volume_{index}"
            price_col = f"ask_price_{index}"
            if volume_col in frame and price_col in frame:
                part = finite_numeric(frame[volume_col]) * finite_numeric(frame[price_col])
                depth = part if depth is None else depth.add(part, fill_value=0.0)
        if depth is not None:
            frame["ask_depth_notional"] = depth

    keep = [*KEYS, *[column for column in CONTEXT_OUTPUT_COLUMNS if column in frame.columns]]
    return frame[keep]


def run_execution_context_case(case: DiagnosticCase) -> Path:
    selected = load_selected_keys(case.selected_path, name=case.name)
    frames = []
    for file in prediction_files(case.prediction_root):
        print(f"{case.name}: reading {file}", flush=True)
        part = read_prediction_context(file, selected)
        print(f"{case.name}: matched rows={len(part)}", flush=True)
        if not part.empty:
            frames.append(part)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEYS)
    out = out.drop_duplicates(KEYS, keep="last")
    output_path = case.output_dir / "execution_context.parquet"
    write_frame(out, output_path)
    print(f"{case.name}: wrote rows={len(out)} to {output_path}", flush=True)
    return output_path
