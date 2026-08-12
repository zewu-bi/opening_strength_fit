from __future__ import annotations

from pathlib import Path

import pandas as pd

from opening_strength_fit.config import config_bool, config_list, config_str
from opening_strength_fit.io import frame_columns, read_frame
from opening_strength_fit.reports import print_mapping
from opening_strength_fit.schema import standardize_columns


def _normalized_reuse_keys(frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    out = frame.copy(deep=False)
    if "date" in key_columns:
        parsed = pd.to_datetime(out["date"], errors="coerce")
        out = out.copy()
        out["date"] = parsed.dt.strftime("%Y-%m-%d")
    if "symbol" in key_columns:
        out["symbol"] = out["symbol"].astype("string")
    return out


def attach_reused_labeled_features(labeled: pd.DataFrame, config: dict) -> pd.DataFrame:
    source_path_raw = config_str(config, "features", "reuse_labeled_path", "").strip()
    if not source_path_raw:
        return labeled

    source_path = Path(source_path_raw)
    if not source_path.exists():
        raise SystemExit(f"reused labeled feature source does not exist: {source_path}")

    key_columns = config_list(config, "features", "reuse_key_columns", ["date", "symbol"])
    explicit_columns = config_list(config, "features", "reuse_feature_columns", [])
    prefixes = tuple(
        config_list(config, "features", "reuse_feature_prefixes", ["preopen_", "auction_"])
    )
    available = frame_columns(source_path)
    missing_keys = [column for column in key_columns if column not in available]
    if missing_keys:
        raise SystemExit(
            f"reused labeled feature source is missing key columns {missing_keys}: {source_path}"
        )
    missing_explicit = [column for column in explicit_columns if column not in available]
    if missing_explicit:
        raise SystemExit(
            "reused labeled feature source is missing requested columns "
            f"{missing_explicit}: {source_path}"
        )
    reused_columns = list(
        dict.fromkeys(
            explicit_columns
            + sorted(
                column for column in available if prefixes and str(column).startswith(prefixes)
            )
        )
    )
    if not reused_columns:
        raise SystemExit(
            "reused labeled feature source produced no columns for "
            f"prefixes={list(prefixes)}: {source_path}"
        )
    collisions = [
        column
        for column in reused_columns
        if column in labeled.columns and column not in key_columns
    ]
    if collisions:
        raise SystemExit(
            f"reused labeled feature columns already exist in destination frame: {collisions}"
        )

    source = read_frame(source_path, columns=key_columns + reused_columns)
    source = _normalized_reuse_keys(standardize_columns(source), key_columns)
    destination = _normalized_reuse_keys(labeled, key_columns)
    if source[key_columns].isna().any(axis=None):
        raise SystemExit(f"reused labeled feature source has null keys: {source_path}")

    if config_bool(config, "features", "reuse_require_constant", True):
        duplicate_source = source.loc[source.duplicated(key_columns, keep=False)]
        if not duplicate_source.empty:
            nonconstant = (
                duplicate_source.groupby(key_columns, sort=False, dropna=False)[reused_columns]
                .nunique(dropna=False)
                .gt(1)
            )
            bad_columns = nonconstant.columns[nonconstant.any(axis=0)].tolist()
            if bad_columns:
                raise SystemExit(
                    "reused labeled features are not constant within key "
                    f"{key_columns}: {bad_columns}"
                )

    source = source.drop_duplicates(key_columns, keep="first")
    source["_reused_labeled_feature_match"] = True
    join_mode = config_str(config, "features", "reuse_join", "inner").strip().lower()
    if join_mode not in {"inner", "left"}:
        raise SystemExit("[features].reuse_join must be 'inner' or 'left'")
    rows_before = len(destination)
    merged = destination.merge(
        source,
        on=key_columns,
        how=join_mode,
        validate="many_to_one",
    )
    matched_rows = int(merged["_reused_labeled_feature_match"].fillna(False).sum())
    if join_mode == "left" and config_bool(config, "features", "reuse_require_full_match", True):
        unmatched_rows = len(merged) - matched_rows
        if unmatched_rows:
            raise SystemExit(
                f"reused labeled features have {unmatched_rows} unmatched destination rows"
            )
    merged = merged.drop(columns="_reused_labeled_feature_match")
    print_mapping(
        "reused_labeled_features",
        {
            "path": str(source_path),
            "keys": key_columns,
            "columns": len(reused_columns),
            "rows_before": rows_before,
            "rows_after": len(merged),
            "matched_rows": matched_rows,
            "join": join_mode,
        },
    )
    return merged
