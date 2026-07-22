from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.alpha_conditioning import (
    KEY_COLUMNS,
    add_group_rank,
    alpha_conditioned_reversal_risk,
    fit_lgbm_config_section,
    predict_model_score,
    section_str,
)
from opening_strength_fit.analysis import (
    load_or_fetch_next_close_labels as shared_load_or_fetch_next_close_labels,
)
from opening_strength_fit.analysis import (
    write_json,
)
from opening_strength_fit.clickhouse_ticks import (
    DEFAULT_CLICKHOUSE_TICK_HOST,
    DEFAULT_CLICKHOUSE_TICK_TABLE,
)
from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_float_mapping,
    config_int,
    config_optional_int,
    config_str,
    config_value,
    load_toml,
    run_id,
)
from opening_strength_fit.feature_config import (
    feature_filters_from_config,
    feature_limit,
)
from opening_strength_fit.horizon_clickhouse_labels import compute_clickhouse_close_labels
from opening_strength_fit.horizons import HorizonSpec
from opening_strength_fit.io import read_frame, write_frame
from opening_strength_fit.labels import finite_numeric_series
from opening_strength_fit.model import evaluate_prediction_frame, fit_lightgbm_frame, predict_frame
from opening_strength_fit.reports import (
    dataset_summary,
    metrics_by_year_from_windows,
    print_mapping,
)
from opening_strength_fit.training_args import build_training_parser
from opening_strength_fit.training_data import load_labeled_pvc_frame
from opening_strength_fit.training_windows import date_splits

DEFAULT_CLOSE_OFFSET_US = 54_000_000_000
DEFAULT_CLOSE_LOOKBACK_SECONDS = 1_800
RISK_RANK_MIN = {
    "ask_depth_10": 0.40,
    "depth_imbalance_10": 0.20,
}
RISK_RANK_MAX = {
    "spread_bps": 0.80,
    "turnover_diff_10t": 0.80,
    "return_10t": 0.70,
    "depth_imbalance_10": 0.70,
}


def parse_args() -> argparse.Namespace:
    parser = build_training_parser("Train a learned dirty-risk / next-flip layer.")
    parser.add_argument("--next-close-label-input", default="")
    parser.add_argument("--close-offset-us", type=int, default=DEFAULT_CLOSE_OFFSET_US)
    parser.add_argument(
        "--close-lookback-seconds",
        type=int,
        default=DEFAULT_CLOSE_LOOKBACK_SECONDS,
    )
    parser.add_argument("--calendar-days-after", type=int, default=10)
    return parser.parse_args()


def risk_rank_config(config: dict) -> tuple[dict[str, float], dict[str, float]]:
    rank_min = config_float_mapping(config, "risk_layer", "risk_rank_min") or RISK_RANK_MIN
    rank_max = config_float_mapping(config, "risk_layer", "risk_rank_max") or RISK_RANK_MAX
    return rank_min, rank_max


def manual_dirty_risk(frame: pd.DataFrame, config: dict) -> pd.Series:
    rank_min, rank_max = risk_rank_config(config)
    groupers = [frame["date"], frame["decision_target_timestamp"]]
    components = []
    for column in sorted(set(rank_min) | set(rank_max)):
        if column not in frame.columns:
            raise SystemExit(f"risk teacher missing required column: {column}")
        values = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        rank = values.groupby(groupers).rank(method="average", pct=True)
        risks = []
        if column in rank_min:
            threshold = float(rank_min[column])
            risks.append(((threshold - rank) / threshold).clip(lower=0.0, upper=1.0))
        if column in rank_max:
            threshold = float(rank_max[column])
            risks.append(((rank - threshold) / (1.0 - threshold)).clip(lower=0.0, upper=1.0))
        components.append(pd.concat(risks, axis=1).max(axis=1).fillna(0.0))
    if not components:
        raise SystemExit("risk teacher config produced no risk components")
    return pd.concat(components, axis=1).mean(axis=1).clip(lower=0.0, upper=1.0)


def load_or_fetch_next_close_labels(
    labeled: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    output_dir: Path,
) -> pd.DataFrame:
    configured_input = config_str(config, "risk_layer", "next_close_label_input", "")
    label_input = args.next_close_label_input or configured_input

    username = (
        args.clickhouse_user
        or config_str(config, "clickhouse", "user", "")
        or os.environ.get("CLICKHOUSE_USER", "")
    )
    password = (
        args.clickhouse_password
        or config_str(config, "clickhouse", "password", "")
        or os.environ.get("CLICKHOUSE_PASSWORD", "")
    )

    def _fetch(base: pd.DataFrame) -> pd.DataFrame:
        if not username or not password:
            raise SystemExit(
                "bad-tail risk labels need next-close labels. Pass "
                "--next-close-label-input or set ClickHouse credentials."
            )
        label_base = base[[*KEY_COLUMNS, "buy_price"]].drop_duplicates(list(KEY_COLUMNS))
        return compute_clickhouse_close_labels(
            label_base.copy(),
            [HorizonSpec(name="next_close", label="next close", seconds=None)],
            host=args.clickhouse_host
            or config_str(
                config,
                "clickhouse",
                "host",
                DEFAULT_CLICKHOUSE_TICK_HOST,
            ),
            port=int(
                args.clickhouse_port
                or config_optional_int(config, "clickhouse", "port", None)
                or os.environ.get("CLICKHOUSE_PORT", "8123")
            ),
            username=username,
            password=password,
            table=args.clickhouse_table
            or config_str(
                config,
                "clickhouse",
                "table",
                os.environ.get("CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE),
            ),
            close_offset_us=int(
                config_optional_int(config, "risk_layer", "close_offset_us", None)
                or args.close_offset_us
            ),
            close_lookback_seconds=int(
                config_optional_int(config, "risk_layer", "close_lookback_seconds", None)
                or args.close_lookback_seconds
            ),
            calendar_days_after=int(
                config_optional_int(config, "risk_layer", "calendar_days_after", None)
                or args.calendar_days_after
            ),
            fee_bps=0.0,
        )

    return shared_load_or_fetch_next_close_labels(
        labeled,
        output_dir=output_dir,
        label_input=label_input,
        fetch_labels=_fetch,
        key_columns=KEY_COLUMNS,
    )


def bad_tail_risk(labeled: pd.DataFrame, config: dict) -> pd.Series:
    if "alpha_return_next_close" not in labeled.columns:
        raise SystemExit("bad-tail target requires alpha_return_next_close")
    short_rank_min = config_float(config, "risk_layer", "short_rank_min", 0.50)
    next_rank_max = config_float(config, "risk_layer", "next_rank_max", 0.50)
    groupers = [labeled["date"], labeled["decision_target_timestamp"]]
    short_rank = (
        finite_numeric_series(labeled["label"])
        .groupby(groupers)
        .rank(
            method="average",
            pct=True,
        )
    )
    next_rank = (
        finite_numeric_series(labeled["alpha_return_next_close"])
        .groupby(groupers)
        .rank(method="average", pct=True)
    )
    short_component = ((short_rank - short_rank_min) / (1.0 - short_rank_min)).clip(
        lower=0.0,
        upper=1.0,
    )
    next_component = ((next_rank_max - next_rank) / next_rank_max).clip(
        lower=0.0,
        upper=1.0,
    )
    return (short_component * next_component).clip(lower=0.0, upper=1.0)


def normalize_candidate_alpha_scores(frame: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    required = [*KEY_COLUMNS, score_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"candidate alpha input missing columns: {missing}")
    out = frame[required].copy()
    out["date"] = out["date"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"],
        errors="coerce",
    )
    out["candidate_alpha_score"] = pd.to_numeric(out[score_col], errors="coerce")
    return out.dropna(
        subset=["decision_target_timestamp", "candidate_alpha_score"]
    ).drop_duplicates(list(KEY_COLUMNS))[[*KEY_COLUMNS, "candidate_alpha_score"]]


def add_candidate_alpha_rank(labeled: pd.DataFrame, config: dict) -> pd.DataFrame:
    path_raw = config_str(config, "risk_layer", "candidate_alpha_score_input", "")
    if not path_raw:
        return labeled
    score_col = config_str(config, "risk_layer", "candidate_alpha_score_col", "prediction")
    alpha_scores = normalize_candidate_alpha_scores(read_frame(Path(path_raw)), score_col=score_col)
    out = labeled.merge(alpha_scores, on=list(KEY_COLUMNS), how="left")
    return add_group_rank(out, "candidate_alpha_score", "candidate_alpha_rank")


def add_fit_alpha_conditioning_rank(
    labeled: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    output_dir: Path,
) -> pd.DataFrame:
    splits = date_splits(labeled, args, config)
    if len(splits) != 1:
        raise SystemExit(
            "risk_layer.candidate_alpha_source='fit_alpha_model' currently supports "
            "one chronological split; use a precomputed candidate_alpha_score_input "
            "for rolling windows."
        )
    split = splits[0]
    train = labeled.loc[labeled["date"].isin(split.train_dates)].copy()
    score_dates = set(split.train_dates) | set(split.test_dates)
    score_frame = labeled.loc[labeled["date"].isin(score_dates)].copy()
    alpha_model, alpha_stats = fit_lgbm_config_section(
        train,
        args=args,
        config=config,
        section="alpha_conditioning",
        target_col="label",
        sample_weight_col=section_str(
            config,
            "alpha_conditioning",
            "model",
            "sample_weight_col",
            "",
        ),
        random_state_default=config_int(config, "model", "random_state", 7) + 1000,
    )
    alpha_predictions = score_frame[
        [column for column in [*KEY_COLUMNS, "label"] if column in score_frame.columns]
    ].copy()
    alpha_predictions["alpha_conditioning_prediction"] = pd.to_numeric(
        predict_model_score(alpha_model, score_frame),
        errors="coerce",
    )
    alpha_predictions["candidate_alpha_score"] = alpha_predictions["alpha_conditioning_prediction"]
    keep_columns = [
        column
        for column in [
            *KEY_COLUMNS,
            "label",
            "alpha_conditioning_prediction",
            "candidate_alpha_score",
        ]
        if column in alpha_predictions.columns
    ]
    alpha_predictions = alpha_predictions[keep_columns].copy()
    alpha_predictions.to_parquet(output_dir / "alpha_conditioning_predictions.parquet", index=False)

    out = labeled.merge(
        alpha_predictions[[*KEY_COLUMNS, "candidate_alpha_score"]],
        on=list(KEY_COLUMNS),
        how="left",
    )
    out = add_group_rank(out, "candidate_alpha_score", "candidate_alpha_rank")
    train_candidate_min = config_float(
        config,
        "risk_layer",
        "candidate_alpha_rank_min",
        0.80,
    )
    train_mask = out["date"].isin(split.train_dates)
    candidate_rate = float(
        pd.to_numeric(out.loc[train_mask, "candidate_alpha_rank"], errors="coerce")
        .ge(train_candidate_min)
        .mean()
    )
    trace = {
        "candidate_alpha_source": "fit_alpha_model",
        "train_start_date": split.train_start_date,
        "train_end_date": split.train_end_date,
        "score_start_date": min(score_dates),
        "score_end_date": max(score_dates),
        "alpha_model_name": alpha_model.model_name,
        "alpha_train_stats": alpha_stats,
        "candidate_alpha_rank_min": train_candidate_min,
        "train_candidate_rate": candidate_rate,
        "outputs": {
            "alpha_conditioning_predictions": str(
                output_dir / "alpha_conditioning_predictions.parquet"
            ),
        },
    }
    write_json(output_dir / "alpha_conditioning_trace.json", trace)
    print_mapping("alpha_conditioning", trace)
    return out


def add_alpha_conditioning_rank(
    labeled: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    output_dir: Path,
) -> pd.DataFrame:
    source = config_str(config, "risk_layer", "candidate_alpha_source", "input").strip().lower()
    if source in {"fit_alpha_model", "fit", "model"}:
        return add_fit_alpha_conditioning_rank(
            labeled,
            args=args,
            config=config,
            output_dir=output_dir,
        )
    out = add_candidate_alpha_rank(labeled, config)
    if "candidate_alpha_rank" not in out.columns:
        raise SystemExit(
            "alpha-conditioned risk target requires [risk_layer].candidate_alpha_source="
            "'fit_alpha_model' or candidate_alpha_score_input."
        )
    return out


def conditional_bad_tail_risk(
    labeled: pd.DataFrame,
    config: dict,
) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    if "alpha_return_next_close" not in labeled.columns:
        raise SystemExit("conditional bad-tail target requires alpha_return_next_close")
    groupers = [labeled["date"], labeled["decision_target_timestamp"]]
    short_rank = (
        finite_numeric_series(labeled["label"])
        .groupby(groupers)
        .rank(
            method="average",
            pct=True,
        )
    )
    next_rank = (
        finite_numeric_series(labeled["alpha_return_next_close"])
        .groupby(groupers)
        .rank(method="average", pct=True)
    )

    candidate = pd.Series(False, index=labeled.index)
    if config_bool(config, "risk_layer", "candidate_use_short_rank", True):
        short_rank_min = config_float(
            config,
            "risk_layer",
            "candidate_short_rank_min",
            0.70,
        )
        candidate = candidate | short_rank.ge(short_rank_min)
    alpha_rank_min_raw = config_value(
        config,
        "risk_layer",
        "candidate_alpha_rank_min",
        None,
    )
    if alpha_rank_min_raw not in (None, "") and "candidate_alpha_rank" in labeled.columns:
        candidate = candidate | pd.to_numeric(
            labeled["candidate_alpha_rank"],
            errors="coerce",
        ).ge(float(alpha_rank_min_raw))

    form = config_str(config, "risk_layer", "target_form", "gap").strip().lower()
    if form in {"gap", "rank_gap", "short_minus_next"}:
        risk = (short_rank - next_rank).clip(lower=0.0, upper=1.0)
    elif form in {"binary", "hard"}:
        binary_short_min = config_float(
            config,
            "risk_layer",
            "binary_short_rank_min",
            0.80,
        )
        binary_next_max = config_float(
            config,
            "risk_layer",
            "binary_next_rank_max",
            0.50,
        )
        risk = (short_rank.ge(binary_short_min) & next_rank.le(binary_next_max)).astype("float64")
    else:
        raise SystemExit(f"unknown [risk_layer].target_form={form!r}; expected gap or binary")

    risk = risk.where(candidate, 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)

    sample_weight: pd.Series | None = None
    non_candidate_weight = config_float(
        config,
        "risk_layer",
        "non_candidate_weight",
        1.0,
    )
    candidate_weight = config_float(config, "risk_layer", "candidate_weight", 1.0)
    if non_candidate_weight != 1.0 or candidate_weight != 1.0:
        sample_weight = pd.Series(
            np.where(candidate, candidate_weight, non_candidate_weight),
            index=labeled.index,
            dtype="float64",
        )
    return risk, candidate.astype("bool"), sample_weight


def add_risk_target(
    labeled: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    output_dir: Path,
) -> tuple[pd.DataFrame, str]:
    target_kind = config_str(config, "risk_layer", "target", "guard_teacher").strip().lower()
    if target_kind in {"guard_teacher", "dirty_risk", "manual_dirty_risk"}:
        target_col = config_str(config, "risk_layer", "target_col", "target_dirty_risk")
        labeled[target_col] = manual_dirty_risk(labeled, config)
        return labeled, target_col
    if target_kind in {"bad_tail", "next_flip", "next_underperformance"}:
        target_col = config_str(config, "risk_layer", "target_col", "target_bad_tail_risk")
        labels = load_or_fetch_next_close_labels(
            labeled,
            args=args,
            config=config,
            output_dir=output_dir,
        )
        labeled = labeled.merge(labels, on=list(KEY_COLUMNS), how="left")
        labeled[target_col] = bad_tail_risk(labeled, config)
        return labeled, target_col
    if target_kind in {
        "conditional_bad_tail",
        "conditional_reversal",
        "conditional_next_flip",
    }:
        target_col = config_str(
            config,
            "risk_layer",
            "target_col",
            "target_conditional_bad_tail_risk",
        )
        labels = load_or_fetch_next_close_labels(
            labeled,
            args=args,
            config=config,
            output_dir=output_dir,
        )
        labeled = labeled.merge(labels, on=list(KEY_COLUMNS), how="left")
        labeled = add_candidate_alpha_rank(labeled, config)
        risk, candidate, sample_weight = conditional_bad_tail_risk(labeled, config)
        labeled[target_col] = risk
        labeled["target_conditional_candidate"] = candidate
        if sample_weight is not None:
            weight_col = config_str(
                config,
                "risk_layer",
                "sample_weight_col",
                "risk_sample_weight",
            )
            labeled[weight_col] = sample_weight
            config.setdefault("model", {}).setdefault("sample_weight_col", weight_col)
        return labeled, target_col
    if target_kind in {
        "alpha_conditioned_reversal",
        "alpha_conditioned_bad_tail",
        "alpha_conditioned_next_flip",
    }:
        target_col = config_str(
            config,
            "risk_layer",
            "target_col",
            "target_alpha_conditioned_reversal_risk",
        )
        labeled = add_alpha_conditioning_rank(
            labeled,
            args=args,
            config=config,
            output_dir=output_dir,
        )
        labels = load_or_fetch_next_close_labels(
            labeled,
            args=args,
            config=config,
            output_dir=output_dir,
        )
        labeled = labeled.merge(labels, on=list(KEY_COLUMNS), how="left")
        risk, candidate, sample_weight = alpha_conditioned_reversal_risk(labeled, config)
        labeled[target_col] = risk
        labeled["target_alpha_conditioned_candidate"] = candidate
        if sample_weight is not None:
            weight_col = config_str(
                config,
                "risk_layer",
                "sample_weight_col",
                "risk_sample_weight",
            )
            labeled[weight_col] = sample_weight
            config.setdefault("model", {}).setdefault("sample_weight_col", weight_col)
        return labeled, target_col
    raise SystemExit(
        f"unknown [risk_layer].target={target_kind!r}; expected guard_teacher, "
        "bad_tail, conditional_bad_tail, or alpha_conditioned_reversal"
    )


def fit_predict_split(
    *,
    labeled: pd.DataFrame,
    split,
    run_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    config: dict,
    target_col: str,
    write_period_artifacts: bool = True,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    train = labeled.loc[labeled["date"].isin(split.train_dates)].copy()
    test = labeled.loc[labeled["date"].isin(split.test_dates)].copy()
    model, train_stats = fit_lightgbm_frame(
        train,
        feature_limit=feature_limit(args, config),
        target_col=target_col,
        sample_weight_col=config_str(config, "model", "sample_weight_col", ""),
        feature_filters=feature_filters_from_config(config),
        n_estimators=config_int(config, "model", "n_estimators", 300),
        learning_rate=config_float(config, "model", "learning_rate", 0.03),
        num_leaves=config_int(config, "model", "num_leaves", 63),
        max_depth=config_int(config, "model", "max_depth", -1),
        min_child_samples=config_int(config, "model", "min_child_samples", 200),
        subsample=config_float(config, "model", "subsample", 1.0),
        colsample_bytree=config_float(config, "model", "colsample_bytree", 1.0),
        reg_alpha=config_float(config, "model", "reg_alpha", 0.0),
        reg_lambda=config_float(config, "model", "reg_lambda", 0.0),
        random_state=config_int(config, "model", "random_state", 7),
        n_jobs=config_int(config, "model", "n_jobs", -1),
        device_type=config_str(config, "model", "device_type", "cpu"),
        max_bin=config_optional_int(config, "model", "max_bin", None),
        gpu_use_dp=False,
    )
    predictions = predict_frame(model, test)
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="coerce")
    predictions["risk_prediction"] = predictions["prediction"].clip(lower=0.0, upper=1.0)
    prediction_period = (
        str(pd.Timestamp(split.test_start_date).to_period("M"))
        if split.test_start_date[:7] == split.test_end_date[:7]
        else str(pd.Timestamp(split.test_start_date).year)
    )
    if write_period_artifacts:
        write_frame(predictions, output_dir / f"predictions_{prediction_period}.parquet")
    metrics = evaluate_prediction_frame(
        predictions,
        label_col=target_col,
        score_col="prediction",
        group_cols=("date", "decision_target_timestamp"),
    )
    metrics_row = {
        "run_id": run_name,
        "test_year": int(pd.Timestamp(split.test_start_date).year),
        "test_month": prediction_period,
        "train_start_date": split.train_start_date,
        "train_end_date": split.train_end_date,
        "test_start_date": split.test_start_date,
        "test_end_date": split.test_end_date,
        "train_rows": int(len(train)),
        "test_rows": int(len(predictions)),
        "features": int(train_stats["features"]),
        "model_name": model.model_name,
        "model_target_col": target_col,
        **metrics,
    }
    print_mapping(f"train_stats[{prediction_period}]", train_stats)
    print_mapping(f"risk_prediction_metrics[{prediction_period}]", metrics)
    return predictions, metrics_row, train_stats


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if args.config else "learned_risk_layer"
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled = load_labeled_pvc_frame(args, config)
    labeled, target_col = add_risk_target(
        labeled,
        args=args,
        config=config,
        output_dir=output_dir,
    )
    print_mapping("risk_dataset", {**dataset_summary(labeled), "target_col": target_col})
    splits = date_splits(labeled, args, config)
    print_mapping(
        "split_plan",
        {
            "windows": len(splits),
            "first_test": splits[0].test_start_date,
            "last_test": splits[-1].test_end_date,
        },
    )

    prediction_frames = []
    metric_rows = []
    train_stats_by_window = {}
    write_split_artifacts = config_bool(
        config,
        "output",
        "write_split_artifacts",
        len(splits) > 1,
    )
    for split in splits:
        predictions, metrics_row, train_stats = fit_predict_split(
            labeled=labeled,
            split=split,
            run_name=run_name,
            output_dir=output_dir,
            args=args,
            config=config,
            target_col=target_col,
            write_period_artifacts=write_split_artifacts,
        )
        prediction_frames.append(predictions)
        metric_rows.append(metrics_row)
        train_stats_by_window[str(metrics_row["test_month"])] = train_stats

    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    combined_predictions = combined_predictions.sort_values(
        [
            column
            for column in ("date", "symbol", "decision_target_timestamp")
            if column in combined_predictions.columns
        ]
    )
    write_frame(combined_predictions, output_dir / "predictions.parquet")

    metrics_by_window = pd.DataFrame(metric_rows)
    metrics_by_year = metrics_by_year_from_windows(metrics_by_window)
    metrics_by_year.to_csv(output_dir / "metrics_by_year.csv", index=False)
    metrics_by_year.to_parquet(output_dir / "metrics_by_year.parquet", index=False)
    if len(metrics_by_window) != len(metrics_by_year):
        metrics_by_window.to_csv(output_dir / "metrics_by_month.csv", index=False)
        metrics_by_window.to_parquet(output_dir / "metrics_by_month.parquet", index=False)

    trace = {
        "run_id": run_name,
        "target_col": target_col,
        "risk_layer": config.get("risk_layer", {}),
        "rows": int(len(labeled)),
        "outputs": {
            "predictions": str(output_dir / "predictions.parquet"),
            "metrics_by_year": str(output_dir / "metrics_by_year.csv"),
        },
        "train_stats_by_window": train_stats_by_window,
    }
    write_json(output_dir / "risk_layer_trace.json", trace)
    print(f"\nwrote: {output_dir}")


if __name__ == "__main__":
    main()
