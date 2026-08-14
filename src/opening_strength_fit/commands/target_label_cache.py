from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.commands.arguments import CommandArguments
from opening_strength_fit.config import (
    config_float_mapping,
    config_list,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import frame_columns, read_frame, write_frame_atomic, write_json
from opening_strength_fit.labels import normalize_return_label_frame
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns
from opening_strength_fit.targets import (
    DEFAULT_HEAT_NEUTRALIZE_COLUMNS,
    add_cross_sectional_target_label,
    target_label_summary,
)

KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")
SHORT_LABEL_OUTCOME_COLUMNS = (
    "gross_label",
    "sell_start_target_timestamp",
    "sell_start_source_timestamp",
    "sell_start_state_age_seconds",
    "sell_end_target_timestamp",
    "sell_end_source_timestamp",
    "sell_end_state_age_seconds",
    "sell_volume",
    "sell_turnover",
    "sell_vwap",
    "hold_seconds",
    "sell_window_seconds",
    "fee_bps",
)


def _normalize_key_columns(frame):
    out = ensure_timestamp_columns(standardize_columns(frame)).copy()
    if "date" in out.columns:
        out["date"] = out["date"].astype(str)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str)
    if "decision_target_timestamp" in out.columns:
        out["decision_target_timestamp"] = out["decision_target_timestamp"].dt.tz_localize(None)
    return out


def _merge_long_label_input(frame, path: Path, *, label_col: str):
    required = [*KEY_COLUMNS, label_col]
    labels = normalize_return_label_frame(
        read_frame(path, columns=required),
        key_columns=KEY_COLUMNS,
        label_col=label_col,
    )
    out = _normalize_key_columns(frame)
    if label_col in out.columns:
        out = out.drop(columns=[label_col])
    return out.merge(labels, on=list(KEY_COLUMNS), how="left", validate="one_to_one")


def _normalize_sidecar_keys(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    out = standardize_columns(frame).copy()
    missing = [column for column in KEY_COLUMNS if column not in out.columns]
    if missing:
        raise SystemExit(f"{source_name} missing key columns: {missing}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"],
        errors="coerce",
    ).dt.tz_localize(None)
    missing_keys = out[list(KEY_COLUMNS)].isna().any(axis=1)
    if missing_keys.any():
        raise SystemExit(f"{source_name} has {int(missing_keys.sum())} rows with missing keys")
    duplicate_keys = out.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate_keys.any():
        raise SystemExit(
            f"{source_name} keys are not unique: {int(duplicate_keys.sum())} duplicate rows"
        )
    return out


def _merge_short_label_input(
    frame: pd.DataFrame,
    path: Path,
    *,
    label_col: str,
    source_label_col: str,
    source_valid_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace the base cache's short outcome with one strict keyed sidecar."""
    available = frame_columns(path)
    required = {*KEY_COLUMNS, source_label_col, source_valid_col}
    missing = sorted(required - available)
    if missing:
        raise SystemExit(f"short label input missing columns: {missing}")
    optional = [column for column in SHORT_LABEL_OUTCOME_COLUMNS if column in available]
    columns = [*KEY_COLUMNS, source_label_col, source_valid_col, *optional]
    labels = _normalize_sidecar_keys(
        read_frame(path, columns=columns),
        source_name="short label input",
    )
    labels[source_label_col] = pd.to_numeric(
        labels[source_label_col],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    labels[source_valid_col] = (
        labels[source_valid_col].fillna(False).astype(bool) & labels[source_label_col].notna()
    )

    source_to_destination = {
        source_label_col: label_col,
        source_valid_col: "valid_label",
        **{column: column for column in optional},
    }
    renamed = {
        source: f"__short_{destination}" for source, destination in source_to_destination.items()
    }
    labels = labels.rename(columns=renamed)
    labels["__short_sidecar_matched"] = True

    out = _normalize_key_columns(frame)
    destinations = list(dict.fromkeys(source_to_destination.values()))
    existing = [column for column in destinations if column in out.columns]
    if existing:
        out = out.drop(columns=existing)
    out = out.merge(labels, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    for destination in destinations:
        out[destination] = out.pop(f"__short_{destination}")
    out["valid_label"] = out["valid_label"].fillna(False).astype(bool)
    out["valid_label"] &= out[label_col].notna()
    matched = out.pop("__short_sidecar_matched").fillna(False).astype(bool)
    stats = {
        "sidecar_rows": int(len(labels)),
        "matched_rows": int(matched.sum()),
        "valid_rows": int(out["valid_label"].sum()),
        "base_rows_without_sidecar": int((~matched).sum()),
    }
    if stats["matched_rows"] == 0:
        raise SystemExit("short label input did not match any base-cache rows")
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a derived labeled cache with a target-aligned label."
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--mode",
        choices=[
            "raw",
            "demean",
            "zscore",
            "rank_pct",
            "rank_centered",
            "heat_neutral",
            "guard_shrunk",
            "guard_risk_shrunk",
            "mixed",
        ],
        default="",
    )
    parser.add_argument("--group-cols", nargs="*", default=None)
    parser.add_argument("--neutralize-cols", nargs="*", default=None)
    parser.add_argument("--label-col", default="")
    parser.add_argument("--target-col", default="")
    parser.add_argument("--raw-label-col", default="")
    parser.add_argument("--min-group-size", type=int, default=None)
    parser.add_argument("--neutralization-strength", type=float, default=None)
    parser.add_argument("--neutralization-ridge-alpha", type=float, default=None)
    parser.add_argument(
        "--neutralization-transform",
        choices=["rank_centered", "zscore", "center"],
        default="",
    )
    parser.add_argument("--min-neutralize-cols", type=int, default=None)
    parser.add_argument("--guard-shrink-penalty", type=float, default=None)
    parser.add_argument("--guard-pass-col", default="")
    parser.add_argument("--guard-rank-group-cols", nargs="*", default=None)
    parser.add_argument("--guard-rank-method", default="")
    parser.add_argument("--guard-risk-lambda", type=float, default=None)
    parser.add_argument("--guard-risk-normalization", default="")
    parser.add_argument("--short-label-input", default="")
    parser.add_argument("--short-label-col", default="")
    parser.add_argument("--short-valid-col", default="")
    parser.add_argument("--long-label-input", default="")
    parser.add_argument("--long-label-col", default="")
    parser.add_argument("--long-label-weight", type=float, default=None)
    parser.add_argument("--short-label-transform", default="")
    parser.add_argument("--long-label-transform", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if args.config else "target_label_cache"
    arguments = CommandArguments(args, config, "target_cache")

    input_raw = (
        args.input
        or config_str(config, "target_cache", "input_path", "")
        or config_str(config, "data", "labeled_path", "")
    )
    output_raw = args.output or config_str(config, "target_cache", "output_path", "")
    if not input_raw:
        raise SystemExit("missing input path: pass --input or [target_cache].input_path")
    if not output_raw:
        raise SystemExit("missing output path: pass --output or [target_cache].output_path")
    input_path = Path(input_raw)
    output_path = Path(output_raw)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output already exists, pass --overwrite: {output_path}")

    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    mode = arguments.string("mode", "demean")
    group_cols = (
        tuple(args.group_cols)
        if args.group_cols is not None
        else tuple(
            config_list(
                config,
                "target_cache",
                "group_cols",
                ("date", "decision_target_timestamp"),
            )
        )
    )
    neutralize_cols = (
        tuple(args.neutralize_cols)
        if args.neutralize_cols is not None
        else tuple(
            config_list(
                config,
                "target_cache",
                "neutralize_cols",
                DEFAULT_HEAT_NEUTRALIZE_COLUMNS,
            )
        )
    )
    label_col = arguments.string("label_col", "label")
    target_col = arguments.string("target_col", "target_label")
    raw_label_col = arguments.string("raw_label_col", "label_raw")
    min_group_size = arguments.integer("min_group_size", 2)
    neutralization_strength = arguments.float("neutralization_strength", 1.0)
    neutralization_ridge_alpha = arguments.float("neutralization_ridge_alpha", 1.0)
    neutralization_transform = arguments.string("neutralization_transform", "rank_centered")
    min_neutralize_cols = arguments.integer("min_neutralize_cols", 1)
    guard_shrink_penalty = arguments.float("guard_shrink_penalty", 0.5)
    guard_pass_col = arguments.string("guard_pass_col", "next_flip_guard_10t_pass")
    guard_rank_group_cols = (
        tuple(args.guard_rank_group_cols)
        if args.guard_rank_group_cols is not None
        else tuple(
            config_list(
                config,
                "target_cache",
                "guard_rank_group_cols",
                group_cols,
            )
        )
    )
    guard_rank_method = arguments.string("guard_rank_method", "average")
    guard_risk_lambda = arguments.float("guard_risk_lambda", 1.0)
    guard_risk_normalization = arguments.string("guard_risk_normalization", "mean")
    short_label_input = arguments.string("short_label_input")
    short_label_col = arguments.string("short_label_col", "label")
    short_valid_col = arguments.string("short_valid_col", "valid_label")
    long_label_input = arguments.string("long_label_input")
    long_label_col = arguments.string("long_label_col", "alpha_return_next_close")
    long_label_weight = arguments.float("long_label_weight", 0.10)
    short_label_transform = arguments.string("short_label_transform", "zscore")
    long_label_transform = arguments.string("long_label_transform", "zscore")
    guard_min_values = config_float_mapping(config, "target_cache", "guard_min")
    guard_max_values = config_float_mapping(config, "target_cache", "guard_max")
    guard_rank_min_values = config_float_mapping(
        config,
        "target_cache",
        "guard_rank_min",
    )
    guard_rank_max_values = config_float_mapping(
        config,
        "target_cache",
        "guard_rank_max",
    )
    guard_risk_rank_min_values = config_float_mapping(
        config,
        "target_cache",
        "guard_risk_rank_min",
    )
    guard_risk_rank_max_values = config_float_mapping(
        config,
        "target_cache",
        "guard_risk_rank_max",
    )

    print_mapping(
        "target_cache",
        {
            "run_id": run_name,
            "input": str(input_path),
            "output": str(output_path),
            "mode": mode,
            "group_cols": ",".join(group_cols),
            "label_col": label_col,
            "target_col": target_col,
            "raw_label_col": raw_label_col,
            "min_group_size": min_group_size,
            "neutralize_cols": ",".join(neutralize_cols),
            "neutralization_strength": neutralization_strength,
            "neutralization_ridge_alpha": neutralization_ridge_alpha,
            "neutralization_transform": neutralization_transform,
            "min_neutralize_cols": min_neutralize_cols,
            "guard_shrink_penalty": guard_shrink_penalty,
            "guard_pass_col": guard_pass_col,
            "guard_rank_group_cols": ",".join(guard_rank_group_cols),
            "guard_rank_method": guard_rank_method,
            "guard_risk_lambda": guard_risk_lambda,
            "guard_risk_normalization": guard_risk_normalization,
            "short_label_input": short_label_input,
            "short_label_col": short_label_col,
            "short_valid_col": short_valid_col,
            "long_label_input": long_label_input,
            "long_label_col": long_label_col,
            "long_label_weight": long_label_weight,
            "short_label_transform": short_label_transform,
            "long_label_transform": long_label_transform,
            "guard_min": guard_min_values,
            "guard_max": guard_max_values,
            "guard_rank_min": guard_rank_min_values,
            "guard_rank_max": guard_rank_max_values,
            "guard_risk_rank_min": guard_risk_rank_min_values,
            "guard_risk_rank_max": guard_risk_rank_max_values,
        },
    )

    frame = read_frame(input_path)
    short_label_stats: dict[str, int] = {}
    if short_label_input:
        frame, short_label_stats = _merge_short_label_input(
            frame,
            Path(short_label_input),
            label_col=label_col,
            source_label_col=short_label_col,
            source_valid_col=short_valid_col,
        )
    if long_label_input:
        frame = _merge_long_label_input(
            frame,
            Path(long_label_input),
            label_col=long_label_col,
        )
    aligned = add_cross_sectional_target_label(
        frame,
        mode=mode,
        group_cols=group_cols,
        label_col=label_col,
        target_col=target_col,
        raw_label_col=raw_label_col,
        min_group_size=min_group_size,
        neutralize_cols=neutralize_cols,
        neutralization_strength=neutralization_strength,
        neutralization_ridge_alpha=neutralization_ridge_alpha,
        neutralization_transform=neutralization_transform,
        min_neutralize_cols=min_neutralize_cols,
        guard_shrink_penalty=guard_shrink_penalty,
        guard_pass_col=guard_pass_col,
        guard_min_values=guard_min_values,
        guard_max_values=guard_max_values,
        guard_rank_min_values=guard_rank_min_values,
        guard_rank_max_values=guard_rank_max_values,
        guard_rank_group_cols=guard_rank_group_cols,
        guard_rank_method=guard_rank_method,
        guard_risk_lambda=guard_risk_lambda,
        guard_risk_rank_min_values=guard_risk_rank_min_values,
        guard_risk_rank_max_values=guard_risk_rank_max_values,
        guard_risk_normalization=guard_risk_normalization,
        long_label_col=long_label_col,
        long_label_weight=long_label_weight,
        short_label_transform=short_label_transform,
        long_label_transform=long_label_transform,
    )
    write_frame_atomic(aligned, output_path)

    summary = target_label_summary(
        aligned,
        label_col=target_col,
        raw_label_col=raw_label_col,
        group_cols=group_cols,
    )
    trace = {
        "run_id": run_name,
        "input": str(input_path),
        "output": str(output_path),
        "mode": mode,
        "group_cols": list(group_cols),
        "neutralize_cols": list(neutralize_cols),
        "neutralization_strength": neutralization_strength,
        "neutralization_ridge_alpha": neutralization_ridge_alpha,
        "neutralization_transform": neutralization_transform,
        "min_neutralize_cols": min_neutralize_cols,
        "guard_shrink_penalty": guard_shrink_penalty,
        "guard_pass_col": guard_pass_col,
        "guard_rank_group_cols": list(guard_rank_group_cols),
        "guard_rank_method": guard_rank_method,
        "guard_risk_lambda": guard_risk_lambda,
        "guard_risk_normalization": guard_risk_normalization,
        "short_label_input": short_label_input,
        "short_label_col": short_label_col,
        "short_valid_col": short_valid_col,
        "short_label_stats": short_label_stats,
        "long_label_input": long_label_input,
        "long_label_col": long_label_col,
        "long_label_weight": long_label_weight,
        "short_label_transform": short_label_transform,
        "long_label_transform": long_label_transform,
        "guard_min": guard_min_values,
        "guard_max": guard_max_values,
        "guard_rank_min": guard_rank_min_values,
        "guard_rank_max": guard_rank_max_values,
        "guard_risk_rank_min": guard_risk_rank_min_values,
        "guard_risk_rank_max": guard_risk_rank_max_values,
        "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "target_cache_trace.json", trace)
    print_mapping("target_cache_summary", summary)
    print(f"\nwrote: {output_path}")
    print(f"trace: {output_dir / 'target_cache_trace.json'}")


if __name__ == "__main__":
    main()
