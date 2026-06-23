from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.commands.feature_dependence_audit import (
    _feature_group_name,
    _feature_groups,
)
from opening_strength_fit.config import (
    config_float,
    config_int,
    config_str,
    config_value,
    run_id,
)
from opening_strength_fit.feature_config import feature_filters_from_config, feature_limit
from opening_strength_fit.feature_hygiene import (
    build_prune_report,
    feature_correlation_pairs,
    load_feature_importance,
    summarize_feature_hygiene,
    write_json_atomic,
)
from opening_strength_fit.io import frame_columns, read_frame, write_frame_atomic
from opening_strength_fit.model import feature_columns
from opening_strength_fit.reports import dataset_summary, print_mapping
from opening_strength_fit.training_args import build_training_parser, load_run_config
from opening_strength_fit.training_data import (
    _labeled_pvc_files,
    _labeled_pvc_path,
    _labeled_pvc_read_columns,
    load_clickhouse_labeled_frame,
    load_training_frame,
    resolve_data_source,
)
from opening_strength_fit.training_labeled import filter_labeled_frame


def _coerce_month_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = [str(item) for item in value]
    return [str(pd.Period(item, freq="M")) for item in raw if str(item).strip()]


def _configured_sample_months(args: argparse.Namespace, config: dict) -> list[str]:
    explicit = _coerce_month_list(
        args.sample_months or config_value(config, "feature_hygiene", "sample_months", [])
    )
    if explicit:
        return explicit

    start = args.test_start_month or config_value(config, "window", "test_start_month", None)
    end = args.test_end_month or config_value(config, "window", "test_end_month", None)
    if not start or not end:
        return []
    stride = (
        args.sample_month_stride
        if args.sample_month_stride is not None
        else config_int(config, "feature_hygiene", "sample_month_stride", 6)
    )
    stride = max(int(stride), 1)
    return [str(period) for period in pd.period_range(str(start), str(end), freq="M")[::stride]]


def _month_bounds(month: str) -> tuple[str, str]:
    period = pd.Period(month, freq="M")
    return str(period.start_time.date()), str(period.end_time.date())


def _normalize_date_values(values: pd.Series) -> list[str]:
    if values.empty:
        return []
    dates = pd.to_datetime(values, errors="coerce").dropna().dt.date.astype(str)
    return sorted(set(dates.tolist()))


def _choose_dates(dates: list[str], days_per_month: int) -> list[str]:
    if days_per_month <= 0 or len(dates) <= days_per_month:
        return dates
    positions = np.linspace(0, len(dates) - 1, days_per_month, dtype=int)
    return [dates[index] for index in sorted(set(int(pos) for pos in positions))]


def _coerce_int_list(value: object, default: tuple[int, ...]) -> list[int]:
    if value in (None, ""):
        return list(default)
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = list(value)
    out = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out or list(default)


def _historical_context_days(config: dict) -> int:
    if not config_value(
        config,
        "features",
        "include_historical_same_minute_surprise",
        False,
    ):
        return 0
    windows = _coerce_int_list(
        config_value(config, "features", "historical_surprise_windows", [20, 60]),
        (20, 60),
    )
    return max(max(windows), 0)


def _context_dates_for_targets(
    target_dates: list[str],
    all_dates: list[str],
    lookback_days: int,
) -> list[str]:
    if lookback_days <= 0:
        return sorted(set(target_dates))
    index_by_date = {date: index for index, date in enumerate(all_dates)}
    context: set[str] = set()
    for date in target_dates:
        index = index_by_date.get(date)
        if index is None:
            context.add(date)
            continue
        start = max(0, index - lookback_days)
        context.update(all_dates[start : index + 1])
    return sorted(context)


def _available_dates(files: list[Path]) -> list[str]:
    date_values = []
    for file in files:
        try:
            date_part = _read_frame_file(file, columns=["date"], filters=None)
        except (KeyError, ValueError):
            continue
        if "date" in date_part.columns and not date_part.empty:
            date_values.append(date_part["date"])
    return _normalize_date_values(pd.concat(date_values, ignore_index=True)) if date_values else []


def _file_year(path: Path) -> int | None:
    match = re.search(r"(?:^|_)((?:19|20)\d{2})(?:_|$)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _file_overlaps_date_range(path: Path, start_date: str, end_date: str) -> bool:
    year = _file_year(path)
    if year is None:
        return True
    return int(start_date[:4]) <= year <= int(end_date[:4])


def _read_frame_file(
    path: Path,
    *,
    columns: list[str] | None,
    filters: list[tuple[str, str, object]] | None,
) -> pd.DataFrame:
    if columns is not None:
        available = frame_columns(path)
        columns = [column for column in columns if column in available]
    return read_frame(path, columns=columns, filters=filters)


def _sample_labeled_pvc_frame(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    path = _labeled_pvc_path(args, config)
    columns = _labeled_pvc_read_columns(path, config)
    files = _labeled_pvc_files(path)
    sample_months = _configured_sample_months(args, config)
    days_per_month = (
        args.days_per_month
        if args.days_per_month is not None
        else config_int(config, "feature_hygiene", "days_per_month", 3)
    )
    max_rows = (
        args.sample_rows
        if args.sample_rows is not None
        else config_int(config, "feature_hygiene", "sample_rows", 500_000)
    )
    random_state = (
        args.random_state
        if args.random_state is not None
        else config_int(config, "feature_hygiene", "random_state", 7)
    )

    if not sample_months:
        frame = read_frame(path, columns=columns)
        labeled = filter_labeled_frame(frame, config)
        if max_rows and len(labeled) > max_rows:
            labeled = labeled.sample(n=max_rows, random_state=random_state).sort_index()
        return labeled.reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    target_dates: list[str] = []
    historical_lookback = _historical_context_days(config)
    for month in sample_months:
        start_date, end_date = _month_bounds(month)
        month_filters = [
            ("date", ">=", start_date),
            ("date", "<=", end_date),
        ]
        date_values = []
        for file in files:
            try:
                date_part = _read_frame_file(file, columns=["date"], filters=month_filters)
            except (KeyError, ValueError):
                continue
            if "date" in date_part.columns and not date_part.empty:
                date_values.append(date_part["date"])
        dates = (
            _normalize_date_values(pd.concat(date_values, ignore_index=True)) if date_values else []
        )
        selected_dates = _choose_dates(dates, int(days_per_month))
        target_dates.extend(selected_dates)
        if not historical_lookback:
            for date in selected_dates:
                date_filters = [("date", "==", date)]
                for file in files:
                    part = _read_frame_file(file, columns=columns, filters=date_filters)
                    if part.empty:
                        continue
                    part = filter_labeled_frame(part, config)
                    if not part.empty:
                        parts.append(part)
        print_mapping(
            "feature_hygiene_sample_month",
            {
                "month": month,
                "available_dates": len(dates),
                "selected_dates": ",".join(selected_dates),
            },
        )

    if historical_lookback:
        context_calendar_days = config_int(
            config,
            "feature_hygiene",
            "historical_context_calendar_days",
            max(historical_lookback * 2 + 10, historical_lookback + 30),
        )
        raw_parts = []
        context_ranges = []
        for date in sorted(set(target_dates)):
            context_start = str(
                (pd.Timestamp(date) - pd.Timedelta(days=int(context_calendar_days))).date()
            )
            context_ranges.append(f"{context_start}:{date}")
            date_filters = [("date", ">=", context_start), ("date", "<=", date)]
            for file in files:
                if not _file_overlaps_date_range(file, context_start, date):
                    continue
                part = _read_frame_file(file, columns=columns, filters=date_filters)
                if not part.empty:
                    raw_parts.append(part)
        if raw_parts:
            context = pd.concat(raw_parts, ignore_index=True)
            dedupe_columns = [
                column
                for column in ("date", "symbol", "timestamp", "decision_target_timestamp")
                if column in context.columns
            ]
            if dedupe_columns:
                context = context.drop_duplicates(subset=dedupe_columns)
            labeled = filter_labeled_frame(context, config)
            target_date_set = set(target_dates)
            date_key = pd.to_datetime(labeled["date"], errors="coerce").dt.date.astype(str)
            labeled = labeled.loc[date_key.isin(target_date_set)].copy()
            if not labeled.empty:
                parts.append(labeled)
        print_mapping(
            "feature_hygiene_historical_context",
            {
                "lookback_days": historical_lookback,
                "target_dates": len(set(target_dates)),
                "context_calendar_days": context_calendar_days,
                "context_ranges": ";".join(context_ranges),
                "raw_context_rows": sum(len(part) for part in raw_parts),
            },
        )

    if not parts:
        raise SystemExit(f"no sampled labeled rows found under {path}")

    labeled = pd.concat(parts, ignore_index=True)
    if max_rows and len(labeled) > max_rows:
        labeled = labeled.sample(n=max_rows, random_state=random_state).sort_index()
    print_mapping(
        "feature_hygiene_sample",
        {
            "source": str(path),
            "months": ",".join(sample_months),
            "rows_before_final_sample": sum(len(part) for part in parts),
            "rows": len(labeled),
            "columns": len(labeled.columns),
        },
    )
    return labeled.reset_index(drop=True)


def _load_labeled_sample(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    tick_path = (
        args.input
        or config_str(config, "data", "tick_path", "")
        or os.environ.get("OPENING_STRENGTH_TICK_PATH", "")
    )
    data_source = resolve_data_source(args, config, tick_path)
    if data_source == "labeled_pvc":
        return _sample_labeled_pvc_frame(args, config)
    if data_source == "clickhouse":
        labeled = load_clickhouse_labeled_frame(args, config)
    else:
        if not tick_path:
            raise SystemExit(
                "No input path supplied. Set [data].tick_path, --input, "
                'or use [data].source = "labeled_pvc"/"clickhouse".'
            )
        labeled = load_training_frame(tick_path, args, config)

    max_rows = (
        args.sample_rows
        if args.sample_rows is not None
        else config_int(config, "feature_hygiene", "sample_rows", 500_000)
    )
    random_state = (
        args.random_state
        if args.random_state is not None
        else config_int(config, "feature_hygiene", "random_state", 7)
    )
    if max_rows and len(labeled) > max_rows:
        labeled = labeled.sample(n=max_rows, random_state=random_state).sort_index()
    return labeled.reset_index(drop=True)


def _default_importance_path(args: argparse.Namespace, config: dict) -> Path | None:
    if args.importance_path:
        return Path(args.importance_path)
    configured = config_value(config, "feature_hygiene", "importance_path", "")
    if configured:
        return Path(str(configured))
    if not args.config:
        return None
    name = run_id(config, args.config)
    candidates = [
        Path("experiments/results/backtests") / name / "feature_importance.csv",
        Path("output/artifacts") / name / "feature_importance.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _add_feature_hygiene_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--sample-months",
        default=None,
        help="Comma/space separated YYYY-MM months. Defaults to every Nth test month.",
    )
    parser.add_argument(
        "--sample-month-stride",
        type=int,
        default=None,
        help="Use every Nth month from the configured test window. Default: 6.",
    )
    parser.add_argument(
        "--days-per-month",
        type=int,
        default=None,
        help="Evenly sample this many trading dates per sampled month. Default: 3.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Final row cap after date sampling. Default: 500000.",
    )
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--corr-method",
        choices=["spearman", "pearson"],
        default=None,
        help="Correlation method. Default: spearman.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=None,
        help="Report/review pairs above this absolute correlation. Default: 0.98.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=None,
        help="Hard-drop non-representatives above this absolute correlation. Default: 0.995.",
    )
    parser.add_argument(
        "--min-corr-periods",
        type=int,
        default=None,
        help="Minimum overlapping non-null observations for correlation. Default: 100.",
    )
    parser.add_argument(
        "--cross-group",
        action="store_true",
        help="Also report high correlations across feature-audit groups.",
    )
    parser.add_argument(
        "--near-constant-top-ratio",
        type=float,
        default=None,
        help="Flag a feature when its most common finite value reaches this ratio. Default: 0.999.",
    )
    parser.add_argument(
        "--importance-path",
        default=None,
        help="Optional feature_importance.csv path used to choose cluster representatives.",
    )
    parser.add_argument(
        "--write-corr-matrix",
        action="store_true",
        help="Also write the full feature correlation matrix. This can be wider/noisier.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_training_parser(
        "Audit feature hygiene and high-correlation duplicate candidates without training."
    )
    _add_feature_hygiene_args(parser)
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    name = run_id(config, args.config) if args.config else "local_feature_hygiene"
    output_dir = Path(
        args.output_dir
        or config_str(
            config,
            "feature_hygiene",
            "output_dir",
            f"output/artifacts/{name}_feature_hygiene",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled = _load_labeled_sample(args, config)
    print_mapping("dataset", dataset_summary(labeled))
    configured_features = feature_columns(
        labeled,
        feature_limit(args, config),
        **feature_filters_from_config(config),
    )
    if not configured_features:
        raise SystemExit("no numeric feature columns found for feature hygiene audit")

    groups = _feature_groups(config)
    group_by_feature = {
        feature: _feature_group_name(feature, groups) for feature in configured_features
    }
    importance_path = _default_importance_path(args, config)
    importance_required = bool(
        args.importance_path or config_value(config, "feature_hygiene", "importance_path", "")
    )
    importance = load_feature_importance(importance_path, required=importance_required)
    near_constant_top_ratio = (
        args.near_constant_top_ratio
        if args.near_constant_top_ratio is not None
        else config_float(config, "feature_hygiene", "near_constant_top_ratio", 0.999)
    )
    hygiene = summarize_feature_hygiene(
        labeled,
        configured_features,
        group_by_feature=group_by_feature,
        importance=importance,
        near_constant_top_ratio=near_constant_top_ratio,
    )

    corr_method = args.corr_method or config_str(
        config,
        "feature_hygiene",
        "corr_method",
        "spearman",
    )
    corr_threshold = (
        args.corr_threshold
        if args.corr_threshold is not None
        else config_float(config, "feature_hygiene", "corr_threshold", 0.98)
    )
    near_duplicate_threshold = (
        args.near_duplicate_threshold
        if args.near_duplicate_threshold is not None
        else config_float(config, "feature_hygiene", "near_duplicate_threshold", 0.995)
    )
    min_corr_periods = (
        args.min_corr_periods
        if args.min_corr_periods is not None
        else config_int(config, "feature_hygiene", "min_corr_periods", 100)
    )
    pairs, corr_matrix = feature_correlation_pairs(
        labeled,
        configured_features,
        group_by_feature=group_by_feature,
        method=corr_method,
        threshold=corr_threshold,
        same_group_only=not args.cross_group,
        min_periods=min_corr_periods,
    )
    candidates, clusters, keep_list, drop_list = build_prune_report(
        configured_features,
        hygiene,
        pairs,
        candidate_threshold=corr_threshold,
        near_duplicate_threshold=near_duplicate_threshold,
    )

    write_frame_atomic(hygiene, output_dir / "feature_hygiene.csv")
    write_frame_atomic(pairs, output_dir / "feature_correlation_pairs.csv")
    write_frame_atomic(clusters, output_dir / "feature_correlation_clusters.csv")
    write_frame_atomic(candidates, output_dir / "feature_prune_candidates.csv")
    if args.write_corr_matrix:
        write_frame_atomic(corr_matrix, output_dir / "feature_correlation_matrix.csv")
    (output_dir / "feature_keep_list.txt").write_text(
        "\n".join(keep_list) + ("\n" if keep_list else ""),
        encoding="utf-8",
    )
    (output_dir / "feature_drop_list.txt").write_text(
        "\n".join(drop_list) + ("\n" if drop_list else ""),
        encoding="utf-8",
    )

    group_counts = hygiene.groupby("group").size().sort_values(ascending=False).to_dict()
    action_counts = (
        candidates.groupby("action").size().sort_values(ascending=False).to_dict()
        if not candidates.empty
        else {}
    )
    trace = {
        "run_id": name,
        "rows": len(labeled),
        "features": len(configured_features),
        "groups": group_counts,
        "corr_method": corr_method,
        "corr_threshold": corr_threshold,
        "near_duplicate_threshold": near_duplicate_threshold,
        "same_group_only": not args.cross_group,
        "min_corr_periods": min_corr_periods,
        "importance_path": str(importance_path) if importance_path else "",
        "outputs": {
            "hygiene": str(output_dir / "feature_hygiene.csv"),
            "correlation_pairs": str(output_dir / "feature_correlation_pairs.csv"),
            "correlation_clusters": str(output_dir / "feature_correlation_clusters.csv"),
            "prune_candidates": str(output_dir / "feature_prune_candidates.csv"),
            "keep_list": str(output_dir / "feature_keep_list.txt"),
            "drop_list": str(output_dir / "feature_drop_list.txt"),
        },
        "summary": {
            "constant_features": int(hygiene["constant"].sum()),
            "near_constant_features": int(hygiene["near_constant"].sum()),
            "high_corr_pairs": len(pairs),
            "corr_clusters": len(clusters),
            "drop_features": len(drop_list),
            "keep_features": len(keep_list),
            "candidate_actions": action_counts,
        },
    }
    write_json_atomic(output_dir / "feature_hygiene_trace.json", trace)

    print_mapping(
        "feature_hygiene_summary",
        {
            "features": len(configured_features),
            "rows": len(labeled),
            "high_corr_pairs": len(pairs),
            "corr_clusters": len(clusters),
            "drop_features": len(drop_list),
            "review_features": int(action_counts.get("review", 0)),
            "output_dir": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()
