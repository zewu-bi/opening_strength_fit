from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.commands.arguments import (
    add_arguments,
    command_context,
    required_io_paths,
)
from opening_strength_fit.config import (
    config_float_mapping,
    prepare_output_dir,
)
from opening_strength_fit.io import frame_columns, read_frame, write_frame_atomic, write_json
from opening_strength_fit.labels import normalize_return_label_frame
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.schema import (
    DECISION_KEY_COLUMNS,
    ensure_timestamp_columns,
    standardize_columns,
)
from opening_strength_fit.targets import (
    DEFAULT_HEAT_NEUTRALIZE_COLUMNS,
    add_cross_sectional_target_label,
    target_label_summary,
)

KEY_COLUMNS = DECISION_KEY_COLUMNS
SHORT_LABEL_OUTCOME_COLUMNS = tuple(
    "gross_label sell_start_target_timestamp sell_start_source_timestamp "
    "sell_start_state_age_seconds sell_end_target_timestamp sell_end_source_timestamp "
    "sell_end_state_age_seconds sell_volume sell_turnover sell_vwap hold_seconds "
    "sell_window_seconds fee_bps".split()
)


def _normalize_key_columns(frame):
    out = ensure_timestamp_columns(standardize_columns(frame)).copy()
    out["date"] = out["date"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    if "decision_target_timestamp" in out.columns:
        out["decision_target_timestamp"] = out["decision_target_timestamp"].dt.tz_localize(None)
    return out


def _merge_long_label_input(frame, path: Path, *, label_col: str):
    labels = normalize_return_label_frame(
        read_frame(path, columns=[*KEY_COLUMNS, label_col]),
        key_columns=KEY_COLUMNS,
        label_col=label_col,
    )
    out = _normalize_key_columns(frame).drop(columns=[label_col], errors="ignore")
    return out.merge(labels, on=list(KEY_COLUMNS), how="left", validate="one_to_one")


def _normalize_sidecar_keys(frame: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    out = standardize_columns(frame).copy()
    if missing := [column for column in KEY_COLUMNS if column not in out.columns]:
        raise SystemExit(f"{source_name} missing key columns: {missing}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"],
        errors="coerce",
    ).dt.tz_localize(None)
    if (missing_keys := out[list(KEY_COLUMNS)].isna().any(axis=1)).any():
        raise SystemExit(f"{source_name} has {int(missing_keys.sum())} rows with missing keys")
    if (duplicate_keys := out.duplicated(list(KEY_COLUMNS), keep=False)).any():
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
    missing = sorted({*KEY_COLUMNS, source_label_col, source_valid_col} - available)
    if missing:
        raise SystemExit(f"short label input missing columns: {missing}")
    optional = [column for column in SHORT_LABEL_OUTCOME_COLUMNS if column in available]
    labels = _normalize_sidecar_keys(
        read_frame(
            path,
            columns=[*KEY_COLUMNS, source_label_col, source_valid_col, *optional],
        ),
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
    labels = labels.rename(
        columns={
            source: f"__short_{destination}"
            for source, destination in source_to_destination.items()
        }
    )
    labels["__short_sidecar_matched"] = True

    destinations = list(dict.fromkeys(source_to_destination.values()))
    out = _normalize_key_columns(frame).drop(columns=destinations, errors="ignore")
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
    add_arguments(parser, "config input output output-dir", default="")
    parser.add_argument(
        "--mode",
        choices="raw demean zscore rank_pct rank_centered heat_neutral guard_shrunk guard_risk_shrunk mixed".split(),
        default="",
    )
    parser.add_argument("--group-cols", nargs="*", default=None)
    parser.add_argument("--neutralize-cols", nargs="*", default=None)
    add_arguments(parser, "label-col target-col raw-label-col", default="")
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
    add_arguments(
        parser,
        "guard-risk-normalization short-label-input short-label-col short-valid-col "
        "long-label-input long-label-col",
        default="",
    )
    parser.add_argument("--long-label-weight", type=float, default=None)
    add_arguments(parser, "short-label-transform long-label-transform", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config, arguments, run_name = command_context(
        args, "target_cache", default_run_name="target_label_cache"
    )

    input_path, output_path = required_io_paths(
        args, config, "target_cache", input_fallback=("data", "labeled_path")
    )
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"output already exists, pass --overwrite: {output_path}")

    group_cols = arguments.optional_tuple("group_cols", ("date", "decision_target_timestamp"))
    settings = {
        "mode": arguments.string("mode", "demean"),
        "group_cols": group_cols,
        "label_col": arguments.string("label_col", "label"),
        "target_col": arguments.string("target_col", "target_label"),
        "raw_label_col": arguments.string("raw_label_col", "label_raw"),
        "min_group_size": arguments.integer("min_group_size", 2),
        "neutralize_cols": arguments.optional_tuple(
            "neutralize_cols", DEFAULT_HEAT_NEUTRALIZE_COLUMNS
        ),
        "neutralization_strength": arguments.float("neutralization_strength", 1.0),
        "neutralization_ridge_alpha": arguments.float("neutralization_ridge_alpha", 1.0),
        "neutralization_transform": arguments.string("neutralization_transform", "rank_centered"),
        "min_neutralize_cols": arguments.integer("min_neutralize_cols", 1),
        "guard_shrink_penalty": arguments.float("guard_shrink_penalty", 0.5),
        "guard_pass_col": arguments.string("guard_pass_col", "next_flip_guard_10t_pass"),
        "guard_rank_group_cols": arguments.optional_tuple("guard_rank_group_cols", group_cols),
        "guard_rank_method": arguments.string("guard_rank_method", "average"),
        "guard_risk_lambda": arguments.float("guard_risk_lambda", 1.0),
        "guard_risk_normalization": arguments.string("guard_risk_normalization", "mean"),
        "short_label_input": arguments.string("short_label_input"),
        "short_label_col": arguments.string("short_label_col", "label"),
        "short_valid_col": arguments.string("short_valid_col", "valid_label"),
        "long_label_input": arguments.string("long_label_input"),
        "long_label_col": arguments.string("long_label_col", "alpha_return_next_close"),
        "long_label_weight": arguments.float("long_label_weight", 0.10),
        "short_label_transform": arguments.string("short_label_transform", "zscore"),
        "long_label_transform": arguments.string("long_label_transform", "zscore"),
        **{
            f"{name}_values": config_float_mapping(config, "target_cache", name)
            for name in (
                "guard_min",
                "guard_max",
                "guard_rank_min",
                "guard_rank_max",
                "guard_risk_rank_min",
                "guard_risk_rank_max",
            )
        },
    }
    group_settings = {"group_cols", "neutralize_cols", "guard_rank_group_cols"}
    sidecar_settings = set(
        "short_label_input short_label_col short_valid_col long_label_input".split()
    )
    target_settings = {key: value for key, value in settings.items() if key not in sidecar_settings}
    display_settings = {
        key.removesuffix("_values"): ",".join(value) if key in group_settings else value
        for key, value in settings.items()
    }

    print_mapping(
        "target_cache",
        {
            "run_id": run_name,
            "input": str(input_path),
            "output": str(output_path),
            **display_settings,
        },
    )

    frame = read_frame(input_path)
    short_label_stats: dict[str, int] = {}
    if settings["short_label_input"]:
        frame, short_label_stats = _merge_short_label_input(
            frame,
            Path(settings["short_label_input"]),
            label_col=settings["label_col"],
            source_label_col=settings["short_label_col"],
            source_valid_col=settings["short_valid_col"],
        )
    if settings["long_label_input"]:
        frame = _merge_long_label_input(
            frame,
            Path(settings["long_label_input"]),
            label_col=settings["long_label_col"],
        )
    aligned = add_cross_sectional_target_label(frame, **target_settings)
    write_frame_atomic(aligned, output_path)

    summary = target_label_summary(
        aligned,
        label_col=settings["target_col"],
        raw_label_col=settings["raw_label_col"],
        group_cols=settings["group_cols"],
    )
    trace_settings = {}
    for key, value in settings.items():
        if key not in {"label_col", "target_col", "raw_label_col", "min_group_size"}:
            trace_settings[key.removesuffix("_values")] = (
                list(value) if key in group_settings else value
            )
        if key == "short_valid_col":
            trace_settings["short_label_stats"] = short_label_stats
    trace = {
        "run_id": run_name,
        "input": str(input_path),
        "output": str(output_path),
        **trace_settings,
        "summary": summary,
    }
    output_dir = prepare_output_dir(config, args.output_dir, run_name)
    write_json(output_dir / "target_cache_trace.json", trace)
    print_mapping("target_cache_summary", summary)
    print(f"\nwrote: {output_path}")
    print(f"trace: {output_dir / 'target_cache_trace.json'}")


if __name__ == "__main__":
    main()
