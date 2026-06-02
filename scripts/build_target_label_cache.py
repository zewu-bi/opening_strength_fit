from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
from opening_strength_fit.config import (
    config_float,
    config_float_mapping,
    config_int,
    config_str,
    config_value,
    load_toml,
    run_id,
)
from opening_strength_fit.io import read_frame, write_frame
from opening_strength_fit.labels import normalize_return_label_frame
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.schema import ensure_timestamp_columns, standardize_columns
from opening_strength_fit.targets import (
    DEFAULT_HEAT_NEUTRALIZE_COLUMNS,
    add_cross_sectional_target_label,
    target_label_summary,
)


KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")


def _list_config(config: dict, section: str, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = config_value(config, section, key, default)
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = [str(item) for item in value]
    return tuple(part.strip() for part in parts if part and part.strip())


def _arg_or_config(args, config: dict, name: str, default: str = "") -> str:
    value = getattr(args, name)
    if value not in (None, ""):
        return str(value)
    return config_str(config, "target_cache", name, default)


def _write_frame_atomic(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes) or ".parquet"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp{suffix}")
    write_frame(frame, tmp_path)
    os.replace(tmp_path, path)


def _normalize_key_columns(frame):
    out = ensure_timestamp_columns(standardize_columns(frame)).copy()
    if "date" in out.columns:
        out["date"] = out["date"].astype(str)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str)
    if "decision_target_timestamp" in out.columns:
        out["decision_target_timestamp"] = out[
            "decision_target_timestamp"
        ].dt.tz_localize(None)
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
    return out.merge(labels, on=list(KEY_COLUMNS), how="left")


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
    parser.add_argument("--long-label-input", default="")
    parser.add_argument("--long-label-col", default="")
    parser.add_argument("--long-label-weight", type=float, default=None)
    parser.add_argument("--short-label-transform", default="")
    parser.add_argument("--long-label-transform", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if args.config else "target_label_cache"

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
        or config_str(config, "output", "local_dir", f"output/local/{run_name}")
    )
    mode = _arg_or_config(args, config, "mode", "demean")
    group_cols = (
        tuple(args.group_cols)
        if args.group_cols is not None
        else _list_config(
            config,
            "target_cache",
            "group_cols",
            ("date", "decision_target_timestamp"),
        )
    )
    neutralize_cols = (
        tuple(args.neutralize_cols)
        if args.neutralize_cols is not None
        else _list_config(
            config,
            "target_cache",
            "neutralize_cols",
            DEFAULT_HEAT_NEUTRALIZE_COLUMNS,
        )
    )
    label_col = _arg_or_config(args, config, "label_col", "label")
    target_col = _arg_or_config(args, config, "target_col", "target_label")
    raw_label_col = _arg_or_config(args, config, "raw_label_col", "label_raw")
    min_group_size = (
        int(args.min_group_size)
        if args.min_group_size is not None
        else config_int(config, "target_cache", "min_group_size", 2)
    )
    neutralization_strength = (
        float(args.neutralization_strength)
        if args.neutralization_strength is not None
        else config_float(config, "target_cache", "neutralization_strength", 1.0)
    )
    neutralization_ridge_alpha = (
        float(args.neutralization_ridge_alpha)
        if args.neutralization_ridge_alpha is not None
        else config_float(config, "target_cache", "neutralization_ridge_alpha", 1.0)
    )
    neutralization_transform = (
        args.neutralization_transform
        or config_str(
            config,
            "target_cache",
            "neutralization_transform",
            "rank_centered",
        )
    )
    min_neutralize_cols = (
        int(args.min_neutralize_cols)
        if args.min_neutralize_cols is not None
        else config_int(config, "target_cache", "min_neutralize_cols", 1)
    )
    guard_shrink_penalty = (
        float(args.guard_shrink_penalty)
        if args.guard_shrink_penalty is not None
        else config_float(config, "target_cache", "guard_shrink_penalty", 0.5)
    )
    guard_pass_col = _arg_or_config(
        args,
        config,
        "guard_pass_col",
        "next_flip_guard_10t_pass",
    )
    guard_rank_group_cols = (
        tuple(args.guard_rank_group_cols)
        if args.guard_rank_group_cols is not None
        else _list_config(
            config,
            "target_cache",
            "guard_rank_group_cols",
            group_cols,
        )
    )
    guard_rank_method = (
        args.guard_rank_method
        or config_str(config, "target_cache", "guard_rank_method", "average")
    )
    guard_risk_lambda = (
        float(args.guard_risk_lambda)
        if args.guard_risk_lambda is not None
        else config_float(config, "target_cache", "guard_risk_lambda", 1.0)
    )
    guard_risk_normalization = (
        args.guard_risk_normalization
        or config_str(config, "target_cache", "guard_risk_normalization", "mean")
    )
    long_label_input = _arg_or_config(args, config, "long_label_input", "")
    long_label_col = _arg_or_config(
        args,
        config,
        "long_label_col",
        "alpha_return_next_close",
    )
    long_label_weight = (
        float(args.long_label_weight)
        if args.long_label_weight is not None
        else config_float(config, "target_cache", "long_label_weight", 0.10)
    )
    short_label_transform = (
        args.short_label_transform
        or config_str(config, "target_cache", "short_label_transform", "zscore")
    )
    long_label_transform = (
        args.long_label_transform
        or config_str(config, "target_cache", "long_label_transform", "zscore")
    )
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
    _write_frame_atomic(aligned, output_path)

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
    (output_dir / "target_cache_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print_mapping("target_cache_summary", summary)
    print(f"\nwrote: {output_path}")
    print(f"trace: {output_dir / 'target_cache_trace.json'}")


if __name__ == "__main__":
    main()
