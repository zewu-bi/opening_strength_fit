from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from opening_strength_fit.clickhouse_ticks import DEFAULT_CLICKHOUSE_TICK_TABLE
from opening_strength_fit.config import config_float, config_int, config_str, config_value, load_toml, run_id
from opening_strength_fit.io import read_frame
from run_alpha_horizon_decay import HorizonSpec, compute_clickhouse_close_labels


DEFAULT_CLOSE_OFFSET_US = 54_000_000_000
DEFAULT_CLOSE_LOOKBACK_SECONDS = 1_800
KEY_COLUMNS = ("date", "symbol", "decision_target_timestamp")
RISK_COLUMNS = (
    "spread_bps",
    "turnover_diff_10t",
    "return_10t",
    "ask_depth_10",
    "depth_imbalance_10",
)
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
PENALTIES = (0.25, 0.50, 0.75, 1.00)
RISK_GATES = (0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep alpha-rank minus dirty-risk penalties and hard gates over "
            "existing prediction files."
        )
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--input", action="append", default=[], help="run_id=path")
    parser.add_argument("--next-close-label-input", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--score-col", default="")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--start-clock", default="")
    parser.add_argument("--end-clock", default="")
    parser.add_argument("--clickhouse-host", default=os.environ.get("CLICKHOUSE_HOST", ""))
    parser.add_argument(
        "--clickhouse-port",
        type=int,
        default=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
    )
    parser.add_argument("--clickhouse-user", default=os.environ.get("CLICKHOUSE_USER", ""))
    parser.add_argument("--clickhouse-password", default=os.environ.get("CLICKHOUSE_PASSWORD", ""))
    parser.add_argument(
        "--clickhouse-table",
        default=os.environ.get("CLICKHOUSE_TICK_TABLE", DEFAULT_CLICKHOUSE_TICK_TABLE),
    )
    parser.add_argument("--close-offset-us", type=int, default=DEFAULT_CLOSE_OFFSET_US)
    parser.add_argument("--close-lookback-seconds", type=int, default=DEFAULT_CLOSE_LOOKBACK_SECONDS)
    parser.add_argument("--calendar-days-after", type=int, default=10)
    return parser.parse_args()


def clock_range(start: str, end: str) -> list[str]:
    start_ts = pd.Timestamp(f"2000-01-01 {start}")
    end_ts = pd.Timestamp(f"2000-01-01 {end}")
    if end_ts < start_ts:
        raise SystemExit("--end-clock must be >= --start-clock")
    clocks = []
    current = start_ts
    while current <= end_ts:
        clocks.append(current.strftime("%H:%M"))
        current += pd.Timedelta(minutes=1)
    return clocks


def existing_columns(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        return set(pd.read_csv(path, nrows=0).columns)
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema.names)


def parse_input_specs(args: argparse.Namespace, config: dict) -> list[dict[str, str]]:
    specs = []
    for value in args.input:
        if "=" not in value:
            raise SystemExit("--input must be formatted as run_id=path")
        run_name, path = value.split("=", 1)
        specs.append({"run_id": run_name.strip(), "path": path.strip()})
    if specs:
        return specs
    configured = config.get("risk_sweep", {}).get("inputs", [])
    if not configured:
        raise SystemExit("risk sweep requires --input or [[risk_sweep.inputs]]")
    return [
        {
            "run_id": str(item.get("run_id", "")).strip(),
            "path": str(item.get("prediction_path", item.get("path", ""))).strip(),
        }
        for item in configured
        if str(item.get("run_id", "")).strip()
        and str(item.get("prediction_path", item.get("path", ""))).strip()
    ]


def load_predictions(spec: dict[str, str], *, score_col: str, clocks: list[str]) -> pd.DataFrame:
    path = Path(spec["path"])
    available = existing_columns(path)
    required = [*KEY_COLUMNS, score_col, "label", "buy_price"]
    missing = [column for column in required if column not in available]
    if missing:
        raise SystemExit(f"{spec['run_id']}: prediction input missing columns: {missing}")
    columns = [column for column in [*required, *RISK_COLUMNS] if column in available]
    frame = read_frame(path)[columns] if path.suffix.lower() == ".csv" else pd.read_parquet(path, columns=columns)
    frame = frame.dropna(subset=[*KEY_COLUMNS, score_col, "label", "buy_price"]).copy()
    frame["run_id"] = spec["run_id"]
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["decision_target_timestamp"] = pd.to_datetime(
        frame["decision_target_timestamp"],
        errors="coerce",
    )
    frame["clock"] = frame["decision_target_timestamp"].dt.strftime("%H:%M")
    frame = frame.loc[frame["clock"].isin(clocks)].copy()
    frame["alpha_score"] = pd.to_numeric(frame[score_col], errors="coerce")
    return frame.dropna(subset=["decision_target_timestamp", "alpha_score"])


def normalize_next_close_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = [*KEY_COLUMNS, "alpha_return_next_close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"next-close label input missing columns: {missing}")
    out = frame[required].copy()
    out["date"] = out["date"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["decision_target_timestamp"] = pd.to_datetime(
        out["decision_target_timestamp"],
        errors="coerce",
    )
    return out.dropna(subset=["decision_target_timestamp"]).drop_duplicates(list(KEY_COLUMNS))


def load_or_fetch_next_close_labels(
    predictions: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    output_dir: Path,
) -> pd.DataFrame:
    cached_path = output_dir / "clickhouse_next_close_labels.parquet"
    configured_input = config_str(config, "risk_sweep", "next_close_label_input", "")
    label_input = Path(args.next_close_label_input or configured_input) if (args.next_close_label_input or configured_input) else None
    if label_input and label_input.exists():
        labels = normalize_next_close_labels(read_frame(label_input))
        labels.to_parquet(cached_path, index=False)
        return labels
    if cached_path.exists():
        return normalize_next_close_labels(pd.read_parquet(cached_path))

    username = args.clickhouse_user or config_str(config, "clickhouse", "user", "")
    password = args.clickhouse_password or config_str(config, "clickhouse", "password", "")
    if not username or not password:
        raise SystemExit(
            "next-close labels not found. Pass --next-close-label-input or set "
            "CLICKHOUSE_USER and CLICKHOUSE_PASSWORD."
        )

    label_base = predictions[[*KEY_COLUMNS, "buy_price"]].drop_duplicates(list(KEY_COLUMNS))
    labels = compute_clickhouse_close_labels(
        label_base.copy(),
        [HorizonSpec(name="next_close", label="next close", seconds=None)],
        host=args.clickhouse_host or "ch.db.prod.highfortfunds.com",
        port=int(args.clickhouse_port),
        username=username,
        password=password,
        table=args.clickhouse_table,
        close_offset_us=int(args.close_offset_us),
        close_lookback_seconds=int(args.close_lookback_seconds),
        calendar_days_after=int(args.calendar_days_after),
        fee_bps=0.0,
    )
    labels = normalize_next_close_labels(labels)
    labels.to_parquet(cached_path, index=False)
    return labels


def add_risk_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    groupers = [out["run_id"], out["date"], out["decision_target_timestamp"]]
    out["alpha_rank"] = out["alpha_score"].groupby(groupers).rank(method="average", pct=True)
    components = []
    for column in sorted(set(RISK_RANK_MIN) | set(RISK_RANK_MAX)):
        if column not in out.columns:
            raise SystemExit(f"risk sweep missing required column: {column}")
        values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rank = values.groupby(groupers).rank(method="average", pct=True)
        risks = []
        if column in RISK_RANK_MIN:
            threshold = RISK_RANK_MIN[column]
            risks.append(((threshold - rank) / threshold).clip(lower=0.0, upper=1.0))
        if column in RISK_RANK_MAX:
            threshold = RISK_RANK_MAX[column]
            risks.append(((rank - threshold) / (1.0 - threshold)).clip(lower=0.0, upper=1.0))
        component = pd.concat(risks, axis=1).max(axis=1).fillna(0.0)
        out[f"{column}_risk"] = component.astype("float64")
        components.append(out[f"{column}_risk"])

    risk_frame = pd.concat(components, axis=1)
    out["dirty_risk"] = risk_frame.mean(axis=1).clip(lower=0.0)
    out["dirty_risk_component_count"] = risk_frame.gt(0.0).sum(axis=1)
    out["next_flip_guard_10t_pass"] = (
        out["spread_bps_risk"].eq(0.0)
        & out["turnover_diff_10t_risk"].eq(0.0)
        & out["return_10t_risk"].eq(0.0)
        & out["ask_depth_10_risk"].eq(0.0)
        & out["depth_imbalance_10_risk"].eq(0.0)
    )
    return out


def variant_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"variant": "alpha_rank", "penalty": 0.0, "gate": None},
    ]
    for penalty in PENALTIES:
        specs.append({"variant": f"risk_penalty_{int(penalty * 100):03d}", "penalty": penalty, "gate": None})
    specs.append({"variant": "hard_gate_next_flip_guard_10t", "penalty": 0.0, "gate": "guard"})
    for threshold in RISK_GATES:
        specs.append({"variant": f"hard_gate_risk_le_{int(threshold * 100):03d}", "penalty": 0.0, "gate": threshold})
    return specs


def summarize_selection(frame: pd.DataFrame, *, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for spec in variant_specs():
        variant = str(spec["variant"])
        penalty = float(spec["penalty"])
        gate = spec["gate"]
        work = frame.copy()
        work["final_score"] = work["alpha_rank"] - penalty * work["dirty_risk"]
        if gate == "guard":
            work = work.loc[work["next_flip_guard_10t_pass"]].copy()
        elif gate is not None:
            work = work.loc[work["dirty_risk"].le(float(gate))].copy()

        for (run_name, date, timestamp), group in work.groupby(
            ["run_id", "date", "decision_target_timestamp"],
            sort=True,
        ):
            full_group = frame.loc[
                (frame["run_id"].eq(run_name))
                & (frame["date"].eq(date))
                & (frame["decision_target_timestamp"].eq(timestamp))
            ]
            selected = group.sort_values("final_score", ascending=False).head(top_n)
            all_short = float(full_group["label"].mean())
            all_next = float(full_group["alpha_return_next_close"].mean())
            short_mean = float(selected["label"].mean()) if len(selected) else float("nan")
            next_mean = float(selected["alpha_return_next_close"].mean()) if len(selected) else float("nan")
            rows.append(
                {
                    "run_id": run_name,
                    "variant": variant,
                    "penalty": penalty,
                    "gate": "" if gate is None else str(gate),
                    "date": str(date),
                    "decision_target_timestamp": pd.Timestamp(timestamp),
                    "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                    "rows": int(len(full_group)),
                    "candidate_rows": int(len(group)),
                    "selected_rows": int(len(selected)),
                    "short_top_mean_bps": short_mean * 10_000.0,
                    "short_top_excess_bps": (short_mean - all_short) * 10_000.0,
                    "next_top_mean_bps": next_mean * 10_000.0,
                    "next_top_excess_bps": (next_mean - all_next) * 10_000.0,
                    "selected_guard_pass_count": int(selected["next_flip_guard_10t_pass"].sum()),
                    "selected_dirty_risk": float(selected["dirty_risk"].mean()) if len(selected) else float("nan"),
                }
            )

    group_metrics = pd.DataFrame(rows)
    minute = (
        group_metrics.groupby(["run_id", "variant", "clock"], as_index=False)
        .agg(
            groups=("date", "size"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            short_top_mean_bps=("short_top_mean_bps", "mean"),
            short_top_excess_bps=("short_top_excess_bps", "mean"),
            next_top_mean_bps=("next_top_mean_bps", "mean"),
            next_top_excess_bps=("next_top_excess_bps", "mean"),
            selected_guard_pass_count=("selected_guard_pass_count", "mean"),
            selected_dirty_risk=("selected_dirty_risk", "mean"),
        )
        .sort_values(["run_id", "variant", "clock"])
    )
    summary = (
        group_metrics.groupby(["run_id", "variant"], as_index=False)
        .agg(
            groups=("date", "size"),
            candidate_rows=("candidate_rows", "mean"),
            selected_rows=("selected_rows", "mean"),
            short_top_mean_bps=("short_top_mean_bps", "mean"),
            short_top_excess_bps=("short_top_excess_bps", "mean"),
            next_top_mean_bps=("next_top_mean_bps", "mean"),
            next_top_excess_bps=("next_top_excess_bps", "mean"),
            next_excess_positive_rate=("next_top_excess_bps", lambda s: float((s > 0).mean())),
            selected_guard_pass_count=("selected_guard_pass_count", "mean"),
            selected_dirty_risk=("selected_dirty_risk", "mean"),
        )
        .sort_values(["run_id", "next_top_excess_bps", "short_top_excess_bps"], ascending=[True, False, False])
    )
    next_positive_minutes = (
        minute.groupby(["run_id", "variant"])["next_top_excess_bps"]
        .apply(lambda s: int((s > 0).sum()))
        .reset_index(name="next_positive_minute_count")
    )
    summary = summary.merge(next_positive_minutes, on=["run_id", "variant"], how="left")
    return group_metrics, minute, summary


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if args.config else "score_risk_sweep"
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/local/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    score_col = args.score_col or config_str(config, "risk_sweep", "score_col", "prediction")
    top_n = int(args.top_n if args.top_n is not None else config_int(config, "risk_sweep", "top_n", 100))
    start_clock = args.start_clock or config_str(config, "risk_sweep", "start_clock", "09:31")
    end_clock = args.end_clock or config_str(config, "risk_sweep", "end_clock", "09:40")
    clocks = clock_range(start_clock, end_clock)
    specs = parse_input_specs(args, config)

    frames = [load_predictions(spec, score_col=score_col, clocks=clocks) for spec in specs]
    predictions = pd.concat(frames, ignore_index=True)
    labels = load_or_fetch_next_close_labels(
        predictions,
        args=args,
        config=config,
        output_dir=output_dir,
    )
    frame = predictions.merge(labels, on=list(KEY_COLUMNS), how="inner")
    frame = frame.dropna(subset=["label", "alpha_score", "alpha_return_next_close"]).copy()
    frame = add_risk_scores(frame)

    group_metrics, minute, summary = summarize_selection(frame, top_n=top_n)
    group_metrics.to_csv(output_dir / "score_risk_group_metrics.csv", index=False)
    minute.to_csv(output_dir / "score_risk_minute_summary.csv", index=False)
    summary.to_csv(output_dir / "score_risk_summary.csv", index=False)

    trace = {
        "run_id": run_name,
        "inputs": specs,
        "score_col": score_col,
        "top_n": top_n,
        "clocks": clocks,
        "rows": int(len(frame)),
        "groups": int(
            frame[["run_id", "date", "decision_target_timestamp"]]
            .drop_duplicates()
            .shape[0]
        ),
        "risk_rank_min": RISK_RANK_MIN,
        "risk_rank_max": RISK_RANK_MAX,
        "outputs": {
            "summary": str(output_dir / "score_risk_summary.csv"),
            "minute_summary": str(output_dir / "score_risk_minute_summary.csv"),
            "group_metrics": str(output_dir / "score_risk_group_metrics.csv"),
            "next_close_labels": str(output_dir / "clickhouse_next_close_labels.parquet"),
        },
    }
    (output_dir / "score_risk_trace.json").write_text(
        json.dumps(json_safe(trace), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("score_risk_summary")
    print(
        summary[
            [
                "run_id",
                "variant",
                "short_top_excess_bps",
                "next_top_excess_bps",
                "next_positive_minute_count",
                "selected_guard_pass_count",
                "selected_dirty_risk",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nwrote: {output_dir}")


if __name__ == "__main__":
    main()
