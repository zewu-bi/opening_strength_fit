from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from opening_strength_fit.config import config_int, config_str, config_value, load_toml, run_id

KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")
DEFAULT_FEATURES = (
    "spread_bps",
    "turnover_diff_1t",
    "turnover_diff_3t",
    "turnover_diff_10t",
    "turnover_diff_30t",
    "volume_diff_10t",
    "volume_diff_30t",
    "return_10t",
    "return_30t",
    "ask_depth_10",
    "bid_depth_10",
    "ask_volume_1",
    "bid_volume_1",
    "depth_imbalance_1",
    "depth_imbalance_10",
    "preopen_turnover",
    "preopen_volume",
    "buy_price",
    "mid_price",
)
DEFAULT_CONTROLS = (
    "turnover_diff_10t",
    "return_10t",
    "spread_bps",
    "ask_depth_10",
    "depth_imbalance_10",
    "preopen_turnover",
    "buy_price",
)
DEFAULT_VARIANTS = (
    {
        "variant": "gap_penalty_030_p80",
        "risk_model": "gap",
        "penalty": 0.30,
        "candidate_alpha_rank_min": 0.80,
    },
    {
        "variant": "gap_penalty_035_p80",
        "risk_model": "gap",
        "penalty": 0.35,
        "candidate_alpha_rank_min": 0.80,
    },
)
RISK_RANK_BY_MODEL = {
    "gap": "gap_risk_rank",
    "binary": "binary_risk_rank",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute gap-risk Top100 changes: compare baseline Top100, "
            "penalized-out names, replacements, and retained names."
        )
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--labeled-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--months",
        default="",
        help="Comma/space separated months such as 2021-08,2021-09. Overrides config.",
    )
    parser.add_argument("--top-n", type=int, default=None)
    return parser.parse_args()


def string_sequence(value, default: Iterable[str] = ()) -> list[str]:
    if value in (None, ""):
        return list(default)
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    else:
        raw = list(value)
    return [str(item).strip() for item in raw if str(item).strip()]


def month_sequence(config: dict, override: str = "") -> list[str]:
    explicit = string_sequence(override)
    if explicit:
        return explicit
    explicit = string_sequence(config_value(config, "attribution", "months", []))
    if explicit:
        return explicit
    start = config_value(config, "window", "test_start_month", "")
    end = config_value(config, "window", "test_end_month", "")
    if start and end:
        return [str(month) for month in pd.period_range(str(start), str(end), freq="M")]
    raise SystemExit("gap attribution requires --months or [attribution].months")


def variant_specs(config: dict) -> list[dict[str, object]]:
    configured = config.get("attribution", {}).get("variants", [])
    if not configured:
        return [dict(item) for item in DEFAULT_VARIANTS]
    specs = []
    for item in configured:
        specs.append(
            {
                "variant": str(item.get("variant", "")).strip(),
                "risk_model": str(item.get("risk_model", "gap")).strip().lower(),
                "penalty": float(item.get("penalty", 0.0) or 0.0),
                "candidate_alpha_rank_min": float(item.get("candidate_alpha_rank_min", 0.0) or 0.0),
            }
        )
    return [spec for spec in specs if spec["variant"]]


def existing_columns(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        return set(pd.read_csv(path, nrows=0).columns)
    return set(pq.ParquetFile(path).schema.names)


def prediction_path(prediction_dir: Path, month: str) -> Path:
    candidates = (
        prediction_dir / f"predictions_{month}.parquet",
        prediction_dir / f"month_{month}" / "predictions.parquet",
        prediction_dir / f"month_{month}" / f"predictions_{month}.parquet",
    )
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"no prediction shard found for {month} under {prediction_dir}")


def read_predictions(path: Path) -> pd.DataFrame:
    required = [
        *KEY_COLUMNS,
        "label",
        "alpha_return_next_close",
        "candidate_alpha_rank",
        "gap_risk_rank",
    ]
    available = existing_columns(path)
    missing = [column for column in required if column not in available]
    if missing:
        raise SystemExit(f"{path}: prediction shard missing columns: {missing}")
    optional = [column for column in ["binary_risk_rank"] if column in available]
    columns = [*required, *optional]
    frame = pd.read_parquet(path, columns=columns)
    return normalize_frame(frame)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = out["date"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"],
        errors="coerce",
    )
    for column in set(out.columns) - set(KEY_COLUMNS):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=list(KEY_COLUMNS)).copy()


def read_feature_month(
    labeled_path: Path,
    *,
    month: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    available = existing_columns(labeled_path)
    features = [column for column in feature_columns if column in available]
    if not features:
        raise SystemExit(f"{labeled_path}: none of requested feature columns exist")
    columns = [column for column in [*KEY_COLUMNS, *features] if column in available]
    start = pd.Period(month, freq="M").start_time.strftime("%Y-%m-%d")
    end = pd.Period(month, freq="M").end_time.strftime("%Y-%m-%d")
    try:
        frame = pd.read_parquet(
            labeled_path,
            columns=columns,
            filters=[("date", ">=", start), ("date", "<=", end)],
        )
    except Exception as exc:
        print(f"warning: filtered read failed for {labeled_path}: {exc}; reading columns")
        frame = pd.read_parquet(labeled_path, columns=columns)
        frame["date"] = frame["date"].astype(str)
        frame = frame.loc[frame["date"].between(start, end)].copy()
    return normalize_frame(frame)


def finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() == 0:
        return float("nan")
    return float(numeric.mean())


def add_group_feature_scales(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    groupers = [frame["date"], frame["decision_target_timestamp"]]
    for feature in features:
        x = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        rank = x.groupby(groupers).rank(method="average", pct=True)
        mean = x.groupby(groupers).transform("mean")
        std = x.groupby(groupers).transform("std").replace(0.0, np.nan)
        frame[f"{feature}__rank"] = rank
        frame[f"{feature}__z"] = ((x - mean) / std).replace([np.inf, -np.inf], np.nan)
    return frame


def add_group_label_excess(frame: pd.DataFrame) -> pd.DataFrame:
    groupers = [frame["date"], frame["decision_target_timestamp"]]
    for source, target in [
        ("label", "short_excess_bps"),
        ("alpha_return_next_close", "next_excess_bps"),
    ]:
        x = pd.to_numeric(frame[source], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        mean = x.groupby(groupers).transform("mean")
        frame[target] = (x - mean) * 10_000.0
    return frame


def selected_ids(group: pd.DataFrame, variant: dict[str, object], top_n: int) -> set[int]:
    risk_model = str(variant.get("risk_model", "") or "").lower()
    penalty = float(variant.get("penalty", 0.0) or 0.0)
    candidate_min = float(variant.get("candidate_alpha_rank_min", 0.0) or 0.0)
    candidates = group.loc[
        pd.to_numeric(group["candidate_alpha_rank"], errors="coerce").ge(candidate_min)
    ].copy()
    if candidates.empty:
        return set()
    risk_col = RISK_RANK_BY_MODEL.get(risk_model, "")
    risk_values = (
        pd.to_numeric(candidates[risk_col], errors="coerce").fillna(0.0) if risk_col else 0.0
    )
    candidates["final_score"] = (
        pd.to_numeric(candidates["candidate_alpha_rank"], errors="coerce") - penalty * risk_values
    )
    return set(
        candidates.sort_values("final_score", ascending=False)
        .head(top_n)["row_id"]
        .astype(int)
        .tolist()
    )


def build_membership_and_group_metrics(
    frame: pd.DataFrame,
    *,
    month: str,
    variants: list[dict[str, object]],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    group_cols = ["date", "decision_target_timestamp"]
    baseline_spec = {
        "variant": "alpha_rank",
        "risk_model": "",
        "penalty": 0.0,
        "candidate_alpha_rank_min": 0.0,
    }
    for (date, timestamp), group in frame.groupby(group_cols, sort=True):
        baseline_ids = selected_ids(group, baseline_spec, top_n)
        full_short = finite_mean(group["label"])
        full_next = finite_mean(group["alpha_return_next_close"])

        for variant in variants:
            variant_name = str(variant["variant"])
            variant_ids = selected_ids(group, variant, top_n)
            cohorts = {
                "baseline_top100": baseline_ids,
                "variant_top100": variant_ids,
                "baseline_kept": baseline_ids & variant_ids,
                "penalized_out": baseline_ids - variant_ids,
                "replacement_in": variant_ids - baseline_ids,
            }
            for cohort, row_ids in cohorts.items():
                for row_id in row_ids:
                    membership_rows.append(
                        {
                            "test_month": month,
                            "variant": variant_name,
                            "date": str(date),
                            "decision_target_timestamp": pd.Timestamp(timestamp),
                            "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                            "row_id": int(row_id),
                            "cohort": cohort,
                        }
                    )
                selected = group.loc[group["row_id"].isin(row_ids)]
                short_mean = finite_mean(selected["label"]) if len(selected) else float("nan")
                next_mean = (
                    finite_mean(selected["alpha_return_next_close"])
                    if len(selected)
                    else float("nan")
                )
                metric_rows.append(
                    {
                        "test_month": month,
                        "variant": variant_name,
                        "date": str(date),
                        "decision_target_timestamp": pd.Timestamp(timestamp),
                        "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                        "cohort": cohort,
                        "rows": int(len(selected)),
                        "short_mean_bps": short_mean * 10_000.0,
                        "short_excess_bps": (short_mean - full_short) * 10_000.0,
                        "next_mean_bps": next_mean * 10_000.0,
                        "next_excess_bps": (next_mean - full_next) * 10_000.0,
                    }
                )
    return pd.DataFrame(membership_rows), pd.DataFrame(metric_rows)


def summarize_group_metrics(group_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["test_month", "variant", "cohort"]
    month = (
        group_metrics.groupby(keys, dropna=False)
        .agg(
            groups=("rows", "size"),
            mean_rows=("rows", "mean"),
            short_mean_bps=("short_mean_bps", "mean"),
            short_excess_bps=("short_excess_bps", "mean"),
            next_mean_bps=("next_mean_bps", "mean"),
            next_excess_bps=("next_excess_bps", "mean"),
            next_positive_rate=("next_excess_bps", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    overall = (
        group_metrics.groupby(["variant", "cohort"], dropna=False)
        .agg(
            groups=("rows", "size"),
            mean_rows=("rows", "mean"),
            short_mean_bps=("short_mean_bps", "mean"),
            short_excess_bps=("short_excess_bps", "mean"),
            next_mean_bps=("next_mean_bps", "mean"),
            next_excess_bps=("next_excess_bps", "mean"),
            next_positive_rate=("next_excess_bps", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    return month, overall


def summarize_feature_exposure(
    membership: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = membership.merge(
        frame[
            [
                "row_id",
                *features,
                *(f"{feature}__rank" for feature in features),
                *(f"{feature}__z" for feature in features),
            ]
        ],
        on="row_id",
        how="left",
    )
    rows = []
    for feature in features:
        for keys, part in selected.groupby(["test_month", "variant", "cohort"]):
            month, variant, cohort = keys
            rows.append(
                {
                    "test_month": month,
                    "variant": variant,
                    "cohort": cohort,
                    "feature": feature,
                    "rows": int(len(part)),
                    "mean_raw": finite_mean(part[feature]),
                    "median_raw": float(pd.to_numeric(part[feature], errors="coerce").median()),
                    "mean_rank": finite_mean(part[f"{feature}__rank"]),
                    "mean_z": finite_mean(part[f"{feature}__z"]),
                }
            )
    month_summary = pd.DataFrame(rows)
    overall = (
        month_summary.groupby(["variant", "cohort", "feature"], dropna=False)
        .agg(
            months=("test_month", "nunique"),
            rows=("rows", "sum"),
            mean_raw=("mean_raw", "mean"),
            median_raw=("median_raw", "mean"),
            mean_rank=("mean_rank", "mean"),
            mean_z=("mean_z", "mean"),
        )
        .reset_index()
    )
    delta_rows = []
    for (variant, feature), part in overall.groupby(["variant", "feature"]):
        by_cohort = part.set_index("cohort")
        if {"penalized_out", "baseline_kept"} <= set(by_cohort.index):
            delta_rows.append(
                {
                    "variant": variant,
                    "feature": feature,
                    "penalized_minus_kept_rank": float(
                        by_cohort.loc["penalized_out", "mean_rank"]
                        - by_cohort.loc["baseline_kept", "mean_rank"]
                    ),
                    "penalized_minus_kept_z": float(
                        by_cohort.loc["penalized_out", "mean_z"]
                        - by_cohort.loc["baseline_kept", "mean_z"]
                    ),
                }
            )
    return month_summary, overall, pd.DataFrame(delta_rows)


def fit_residual(values: pd.DataFrame, control_columns: list[str]) -> pd.Series:
    y = pd.to_numeric(values["next_excess_bps"], errors="coerce").to_numpy(dtype=float)
    controls = []
    for column in control_columns:
        x = pd.to_numeric(values[f"{column}__z"], errors="coerce").fillna(0.0)
        controls.append(x.to_numpy(dtype=float))
    if controls:
        xmat = np.column_stack([np.ones(len(values)), *controls])
    else:
        xmat = np.ones((len(values), 1))
    mask = np.isfinite(y) & np.isfinite(xmat).all(axis=1)
    residual = np.full(len(values), np.nan, dtype=float)
    if mask.sum() <= xmat.shape[1] + 5:
        return pd.Series(residual, index=values.index)
    beta, *_ = np.linalg.lstsq(xmat[mask], y[mask], rcond=None)
    residual[mask] = y[mask] - xmat[mask] @ beta
    return pd.Series(residual, index=values.index)


def residual_penalized_vs_kept(
    membership: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    controls: list[str],
) -> pd.DataFrame:
    wanted = membership.loc[membership["cohort"].isin(["baseline_kept", "penalized_out"])].copy()
    columns = [
        "row_id",
        "next_excess_bps",
        *(f"{control}__z" for control in controls),
    ]
    values = wanted.merge(frame[columns], on="row_id", how="left")
    rows = []
    for keys, part in values.groupby(["test_month", "variant"], dropna=False):
        month, variant = keys
        part = part.copy()
        part["next_residual_bps"] = fit_residual(part, controls)
        by_cohort = (
            part.groupby("cohort")
            .agg(
                rows=("row_id", "size"),
                next_excess_bps=("next_excess_bps", "mean"),
                next_residual_bps=("next_residual_bps", "mean"),
            )
            .reset_index()
            .set_index("cohort")
        )
        if {"baseline_kept", "penalized_out"} <= set(by_cohort.index):
            rows.append(
                {
                    "test_month": month,
                    "variant": variant,
                    "controls": " ".join(controls),
                    "kept_rows": int(by_cohort.loc["baseline_kept", "rows"]),
                    "penalized_rows": int(by_cohort.loc["penalized_out", "rows"]),
                    "kept_next_excess_bps": float(
                        by_cohort.loc["baseline_kept", "next_excess_bps"]
                    ),
                    "penalized_next_excess_bps": float(
                        by_cohort.loc["penalized_out", "next_excess_bps"]
                    ),
                    "penalized_minus_kept_next_excess_bps": float(
                        by_cohort.loc["penalized_out", "next_excess_bps"]
                        - by_cohort.loc["baseline_kept", "next_excess_bps"]
                    ),
                    "kept_next_residual_bps": float(
                        by_cohort.loc["baseline_kept", "next_residual_bps"]
                    ),
                    "penalized_next_residual_bps": float(
                        by_cohort.loc["penalized_out", "next_residual_bps"]
                    ),
                    "penalized_minus_kept_next_residual_bps": float(
                        by_cohort.loc["penalized_out", "next_residual_bps"]
                        - by_cohort.loc["baseline_kept", "next_residual_bps"]
                    ),
                }
            )
    month_summary = pd.DataFrame(rows)
    if month_summary.empty:
        return month_summary
    overall = (
        month_summary.groupby("variant", dropna=False)
        .agg(
            test_month=("test_month", lambda _: "OVERALL"),
            controls=("controls", "first"),
            kept_rows=("kept_rows", "sum"),
            penalized_rows=("penalized_rows", "sum"),
            kept_next_excess_bps=("kept_next_excess_bps", "mean"),
            penalized_next_excess_bps=("penalized_next_excess_bps", "mean"),
            penalized_minus_kept_next_excess_bps=(
                "penalized_minus_kept_next_excess_bps",
                "mean",
            ),
            kept_next_residual_bps=("kept_next_residual_bps", "mean"),
            penalized_next_residual_bps=("penalized_next_residual_bps", "mean"),
            penalized_minus_kept_next_residual_bps=(
                "penalized_minus_kept_next_residual_bps",
                "mean",
            ),
        )
        .reset_index()
    )
    return pd.concat([month_summary, overall], ignore_index=True)


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if args.config else "gap_risk_attribution"
    prediction_dir = Path(
        args.prediction_dir or config_str(config, "attribution", "prediction_dir", "")
    )
    labeled_path = Path(args.labeled_path or config_str(config, "attribution", "labeled_path", ""))
    if not prediction_dir:
        raise SystemExit(
            "gap attribution requires --prediction-dir or [attribution].prediction_dir"
        )
    if not labeled_path:
        raise SystemExit("gap attribution requires --labeled-path or [attribution].labeled_path")
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    months = month_sequence(config, args.months)
    top_n = args.top_n or config_int(config, "attribution", "top_n", 100)
    variants = variant_specs(config)
    features = string_sequence(
        config_value(config, "attribution", "feature_columns", list(DEFAULT_FEATURES)),
        DEFAULT_FEATURES,
    )
    controls = string_sequence(
        config_value(config, "attribution", "control_columns", list(DEFAULT_CONTROLS)),
        DEFAULT_CONTROLS,
    )
    requested_features = sorted(set(features) | set(controls))
    available = existing_columns(labeled_path)
    active_features = [column for column in requested_features if column in available]
    active_controls = [column for column in controls if column in active_features]
    if not active_features:
        raise SystemExit("no configured feature/control columns exist in labeled_path")

    all_group_metrics = []
    all_membership = []
    all_frames = []
    trace = {
        "run_id": run_name,
        "prediction_dir": str(prediction_dir),
        "labeled_path": str(labeled_path),
        "months": months,
        "top_n": top_n,
        "variants": variants,
        "features": active_features,
        "controls": active_controls,
    }
    for month in months:
        print(f"\nloading month {month}")
        pred = read_predictions(prediction_path(prediction_dir, month))
        feature_frame = read_feature_month(
            labeled_path,
            month=month,
            feature_columns=active_features,
        )
        frame = pred.merge(feature_frame, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
        frame = frame.dropna(
            subset=["label", "alpha_return_next_close", "candidate_alpha_rank"]
        ).copy()
        frame["row_id"] = np.arange(len(frame), dtype=np.int64) + sum(
            len(item) for item in all_frames
        )
        active_for_month = [column for column in active_features if column in frame.columns]
        frame = add_group_feature_scales(frame, active_for_month)
        frame = add_group_label_excess(frame)
        membership, group_metrics = build_membership_and_group_metrics(
            frame,
            month=month,
            variants=variants,
            top_n=top_n,
        )
        all_membership.append(membership)
        all_group_metrics.append(group_metrics)
        keep_columns = [
            "row_id",
            "date",
            "symbol",
            "decision_target_timestamp",
            "label",
            "alpha_return_next_close",
            "candidate_alpha_rank",
            "gap_risk_rank",
            "short_excess_bps",
            "next_excess_bps",
            *active_for_month,
            *(f"{feature}__rank" for feature in active_for_month),
            *(f"{feature}__z" for feature in active_for_month),
        ]
        all_frames.append(frame[keep_columns])
        print(
            json.dumps(
                {
                    "month": month,
                    "prediction_rows": len(pred),
                    "feature_rows": len(feature_frame),
                    "merged_rows": len(frame),
                    "membership_rows": len(membership),
                },
                ensure_ascii=False,
            )
        )

    membership = pd.concat(all_membership, ignore_index=True)
    group_metrics = pd.concat(all_group_metrics, ignore_index=True)
    frames = pd.concat(all_frames, ignore_index=True)
    month_outcome, overall_outcome = summarize_group_metrics(group_metrics)
    feature_month, feature_overall, feature_delta = summarize_feature_exposure(
        membership,
        frames,
        features=[column for column in active_features if column in frames.columns],
    )
    residual = residual_penalized_vs_kept(
        membership,
        frames,
        controls=[column for column in active_controls if column in frames.columns],
    )

    group_metrics.to_csv(output_dir / "gap_attribution_group_metrics.csv", index=False)
    month_outcome.to_csv(output_dir / "gap_attribution_outcomes_by_month.csv", index=False)
    overall_outcome.to_csv(output_dir / "gap_attribution_outcomes_overall.csv", index=False)
    feature_month.to_csv(output_dir / "gap_attribution_feature_exposure_by_month.csv", index=False)
    feature_overall.to_csv(output_dir / "gap_attribution_feature_exposure_overall.csv", index=False)
    feature_delta.to_csv(output_dir / "gap_attribution_penalized_feature_delta.csv", index=False)
    residual.to_csv(output_dir / "gap_attribution_residual_penalized_vs_kept.csv", index=False)
    (output_dir / "gap_attribution_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\ngap_attribution_outcomes_overall")
    print(overall_outcome.to_string(index=False))
    if not residual.empty:
        print("\ngap_attribution_residual_penalized_vs_kept")
        print(residual.loc[residual["test_month"].eq("OVERALL")].to_string(index=False))
    print(f"\noutput_dir={output_dir}")


if __name__ == "__main__":
    main()
