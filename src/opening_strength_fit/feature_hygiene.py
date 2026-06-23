from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(
                _json_ready(payload),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_feature_importance(path: Path | None, *, required: bool = False) -> pd.DataFrame:
    if path is not None and required and not path.exists():
        raise SystemExit(f"feature importance file not found: {path}")
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "feature",
                "importance_split_mean",
                "importance_gain_mean",
                "importance_folds",
            ]
        )
    raw = pd.read_csv(path)
    if "feature" not in raw.columns:
        raise SystemExit(f"feature importance file has no feature column: {path}")
    out = raw.copy()
    for column in ("importance_split", "importance_gain", "abs_coefficient"):
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    grouped = (
        out.groupby("feature", dropna=False)
        .agg(
            importance_split_mean=("importance_split", "mean"),
            importance_gain_mean=("importance_gain", "mean"),
            abs_coefficient_mean=("abs_coefficient", "mean"),
            importance_folds=("feature", "size"),
        )
        .reset_index()
    )
    return grouped


def _importance_lookup(importance: pd.DataFrame) -> dict[str, dict[str, float]]:
    if importance.empty:
        return {}
    out = importance.copy()
    for column in (
        "importance_split_mean",
        "importance_gain_mean",
        "abs_coefficient_mean",
        "importance_folds",
    ):
        if column not in out.columns:
            out[column] = 0
    return {
        str(row["feature"]): {
            "importance_split_mean": float(row["importance_split_mean"])
            if pd.notna(row["importance_split_mean"])
            else 0.0,
            "importance_gain_mean": float(row["importance_gain_mean"])
            if pd.notna(row["importance_gain_mean"])
            else 0.0,
            "abs_coefficient_mean": float(row["abs_coefficient_mean"])
            if pd.notna(row["abs_coefficient_mean"])
            else 0.0,
            "importance_folds": int(row["importance_folds"]),
        }
        for _, row in out.iterrows()
    }


def summarize_feature_hygiene(
    frame: pd.DataFrame,
    features: list[str],
    *,
    group_by_feature: dict[str, str],
    importance: pd.DataFrame | None = None,
    near_constant_top_ratio: float = 0.999,
) -> pd.DataFrame:
    importance_by_feature = _importance_lookup(
        importance if importance is not None else pd.DataFrame()
    )
    rows = []
    total_rows = len(frame)
    for index, feature in enumerate(features):
        values = pd.to_numeric(frame[feature], errors="coerce")
        non_null = values.notna()
        finite = np.isfinite(values)
        finite_values = values.loc[finite]
        finite_count = int(finite.sum())
        missing_count = int(total_rows - non_null.sum())
        inf_count = int((non_null & ~finite).sum())
        unique_count = int(finite_values.nunique(dropna=True)) if finite_count else 0
        zero_count = int((finite_values == 0).sum()) if finite_count else 0
        if finite_count:
            top_count = int(finite_values.value_counts(dropna=True).iloc[0])
            top_frequency_ratio = top_count / finite_count
        else:
            top_count = 0
            top_frequency_ratio = np.nan
        stats = importance_by_feature.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "feature_index": index,
                "group": group_by_feature.get(feature, "other"),
                "rows": total_rows,
                "non_null_count": int(non_null.sum()),
                "missing_count": missing_count,
                "missing_rate": missing_count / total_rows if total_rows else np.nan,
                "finite_count": finite_count,
                "finite_rate": finite_count / total_rows if total_rows else np.nan,
                "inf_count": inf_count,
                "zero_count": zero_count,
                "zero_rate": zero_count / finite_count if finite_count else np.nan,
                "unique_count": unique_count,
                "top_count": top_count,
                "top_frequency_ratio": top_frequency_ratio,
                "constant": unique_count <= 1,
                "near_constant": bool(
                    unique_count > 1 and top_frequency_ratio >= near_constant_top_ratio
                ),
                "mean": float(finite_values.mean()) if finite_count else np.nan,
                "std": float(finite_values.std()) if finite_count else np.nan,
                "min": float(finite_values.min()) if finite_count else np.nan,
                "p01": float(finite_values.quantile(0.01)) if finite_count else np.nan,
                "p50": float(finite_values.quantile(0.50)) if finite_count else np.nan,
                "p99": float(finite_values.quantile(0.99)) if finite_count else np.nan,
                "max": float(finite_values.max()) if finite_count else np.nan,
                "importance_split_mean": stats.get("importance_split_mean", 0.0),
                "importance_gain_mean": stats.get("importance_gain_mean", 0.0),
                "abs_coefficient_mean": stats.get("abs_coefficient_mean", 0.0),
                "importance_folds": int(stats.get("importance_folds", 0)),
            }
        )
    return pd.DataFrame(rows)


def feature_correlation_pairs(
    frame: pd.DataFrame,
    features: list[str],
    *,
    group_by_feature: dict[str, str],
    method: str,
    threshold: float,
    same_group_only: bool = True,
    min_periods: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not features:
        empty = pd.DataFrame(columns=["feature_a", "feature_b", "corr", "abs_corr"])
        return empty, pd.DataFrame()
    data = frame[features].replace([np.inf, -np.inf], np.nan)
    corr = data.corr(method=method, min_periods=min_periods)
    rows = []
    for left_index, feature_a in enumerate(features):
        for feature_b in features[left_index + 1 :]:
            group_a = group_by_feature.get(feature_a, "other")
            group_b = group_by_feature.get(feature_b, "other")
            if same_group_only and group_a != group_b:
                continue
            value = corr.at[feature_a, feature_b]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            if abs_value < threshold:
                continue
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "group_a": group_a,
                    "group_b": group_b,
                    "corr": float(value),
                    "abs_corr": abs_value,
                }
            )
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["abs_corr", "feature_a", "feature_b"], ascending=[False, True, True]
        )
    return pairs, corr.reset_index(names="feature")


def _connected_components(features: list[str], pairs: pd.DataFrame) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {feature: set() for feature in features}
    for _, row in pairs.iterrows():
        left = str(row["feature_a"])
        right = str(row["feature_b"])
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    seen: set[str] = set()
    components = []
    for feature in features:
        if feature in seen:
            continue
        stack = [feature]
        component = []
        seen.add(feature)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        if len(component) > 1:
            components.append(sorted(component, key=features.index))
    return components


def _representative_feature(component: list[str], hygiene_by_feature: dict[str, dict]) -> str:
    def score(feature: str) -> tuple[bool, bool, float, float, float, int, int]:
        row = hygiene_by_feature[feature]
        return (
            not bool(row["constant"]),
            not bool(row["near_constant"]),
            float(row["importance_gain_mean"]),
            float(row["importance_split_mean"]),
            -float(row["missing_rate"]),
            int(row["finite_count"]),
            -int(row["feature_index"]),
        )

    return max(component, key=score)


def _pair_abs_corr_lookup(pairs: pd.DataFrame) -> dict[tuple[str, str], float]:
    lookup = {}
    for _, row in pairs.iterrows():
        left = str(row["feature_a"])
        right = str(row["feature_b"])
        value = float(row["abs_corr"])
        lookup[(left, right)] = value
        lookup[(right, left)] = value
    return lookup


def build_prune_report(
    features: list[str],
    hygiene: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    candidate_threshold: float,
    near_duplicate_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    hygiene_by_feature = {str(row["feature"]): row.to_dict() for _, row in hygiene.iterrows()}
    pair_lookup = _pair_abs_corr_lookup(pairs)
    components = _connected_components(features, pairs)
    candidate_rows = []
    cluster_rows = []
    hard_drops: set[str] = set()

    for _, row in hygiene.iterrows():
        feature = str(row["feature"])
        if bool(row["constant"]):
            hard_drops.add(feature)
            candidate_rows.append(
                {
                    "feature": feature,
                    "action": "drop",
                    "reason": "constant",
                    "group": row["group"],
                    "cluster_id": "",
                    "keep_feature": "",
                    "max_abs_corr_to_keep": np.nan,
                    "missing_rate": row["missing_rate"],
                    "importance_gain_mean": row["importance_gain_mean"],
                }
            )
        elif bool(row["near_constant"]):
            hard_drops.add(feature)
            candidate_rows.append(
                {
                    "feature": feature,
                    "action": "drop",
                    "reason": "near_constant",
                    "group": row["group"],
                    "cluster_id": "",
                    "keep_feature": "",
                    "max_abs_corr_to_keep": np.nan,
                    "missing_rate": row["missing_rate"],
                    "importance_gain_mean": row["importance_gain_mean"],
                }
            )

    for cluster_index, component in enumerate(components, start=1):
        cluster_id = f"corr_{cluster_index:04d}"
        keep = _representative_feature(component, hygiene_by_feature)
        corr_values = [
            pair_lookup[(left, right)]
            for left in component
            for right in component
            if left < right and (left, right) in pair_lookup
        ]
        group_values = sorted({str(hygiene_by_feature[feature]["group"]) for feature in component})
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "features": len(component),
                "group": group_values[0] if len(group_values) == 1 else "mixed",
                "representative": keep,
                "max_abs_corr": max(corr_values) if corr_values else np.nan,
                "min_abs_corr_edge": min(corr_values) if corr_values else np.nan,
                "members": " ".join(component),
            }
        )
        for feature in component:
            if feature == keep or feature in hard_drops:
                continue
            max_abs_corr_to_keep = pair_lookup.get((feature, keep), np.nan)
            if pd.isna(max_abs_corr_to_keep):
                max_abs_corr_to_keep = max(
                    (
                        pair_lookup.get((feature, other), np.nan)
                        for other in component
                        if other != feature
                    ),
                    default=np.nan,
                )
            if pd.notna(max_abs_corr_to_keep) and max_abs_corr_to_keep >= near_duplicate_threshold:
                action = "drop"
                reason = "near_duplicate"
                hard_drops.add(feature)
            else:
                action = "review"
                reason = "high_correlation"
            row = hygiene_by_feature[feature]
            candidate_rows.append(
                {
                    "feature": feature,
                    "action": action,
                    "reason": reason,
                    "group": row["group"],
                    "cluster_id": cluster_id,
                    "keep_feature": keep,
                    "max_abs_corr_to_keep": max_abs_corr_to_keep,
                    "missing_rate": row["missing_rate"],
                    "importance_gain_mean": row["importance_gain_mean"],
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(["action", "reason", "group", "feature"])
    clusters = pd.DataFrame(cluster_rows)
    drop_list = [feature for feature in features if feature in hard_drops]
    keep_list = [feature for feature in features if feature not in hard_drops]
    return candidates, clusters, keep_list, drop_list
