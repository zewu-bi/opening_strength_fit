from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import (
    clock_range as shared_clock_range,
)
from opening_strength_fit.analysis import (
    load_or_fetch_next_close_labels as shared_load_or_fetch_next_close_labels,
)
from opening_strength_fit.analysis import (
    normalize_next_close_labels as shared_normalize_next_close_labels,
)
from opening_strength_fit.analysis import (
    selection_return_stats,
    write_json,
)
from opening_strength_fit.clickhouse_ticks import DEFAULT_CLICKHOUSE_TICK_TABLE
from opening_strength_fit.commands.horizon_clickhouse_labels import compute_clickhouse_close_labels
from opening_strength_fit.commands.horizon_decay import HorizonSpec
from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_float_tuple,
    config_int,
    config_int_tuple,
    config_str,
    load_toml,
    run_id,
)
from opening_strength_fit.io import frame_columns, read_frame

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


def _float_sequence(
    config: dict, section: str, key: str, default: tuple[float, ...]
) -> tuple[float, ...]:
    return config_float_tuple(config, section, key, default)


def _int_sequence(
    config: dict, section: str, key: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    return config_int_tuple(config, section, key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep alpha-rank minus dirty-risk penalties and hard gates over "
            "existing prediction files."
        )
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--input", action="append", default=[], help="run_id=path")
    parser.add_argument(
        "--risk-input",
        action="append",
        default=[],
        help="risk_id=path for learned risk prediction files.",
    )
    parser.add_argument("--next-close-label-input", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--score-col", default="")
    parser.add_argument("--risk-score-col", default="")
    parser.add_argument(
        "--risk-score-transform",
        choices=["", "raw", "rank"],
        default="",
        help="How to scale learned risk scores before applying penalties.",
    )
    parser.add_argument(
        "--wait-for-inputs-seconds",
        type=int,
        default=None,
        help="Wait this long for learned risk prediction inputs to appear.",
    )
    parser.add_argument(
        "--manual-risk",
        choices=["", "true", "false"],
        default="",
        help="Override whether manual dirty-risk variants are included.",
    )
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
    parser.add_argument(
        "--close-lookback-seconds", type=int, default=DEFAULT_CLOSE_LOOKBACK_SECONDS
    )
    parser.add_argument("--calendar-days-after", type=int, default=10)
    return parser.parse_args()


def clock_range(start: str, end: str) -> list[str]:
    return shared_clock_range(start, end)


def existing_columns(path: Path) -> set[str]:
    return frame_columns(path)


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


def parse_risk_input_specs(args: argparse.Namespace, config: dict) -> list[dict[str, str]]:
    specs = []
    for value in args.risk_input:
        if "=" not in value:
            raise SystemExit("--risk-input must be formatted as risk_id=path")
        risk_name, path = value.split("=", 1)
        specs.append(
            {
                "risk_id": risk_name.strip(),
                "path": path.strip(),
                "score_col": args.risk_score_col.strip(),
                "score_transform": args.risk_score_transform.strip(),
            }
        )
    if specs:
        return specs
    configured = config.get("risk_sweep", {}).get("risk_inputs", [])
    return [
        {
            "risk_id": str(item.get("risk_id", item.get("run_id", ""))).strip(),
            "path": str(item.get("prediction_path", item.get("path", ""))).strip(),
            "score_col": str(
                item.get(
                    "score_col",
                    args.risk_score_col
                    or config_str(config, "risk_sweep", "risk_score_col", "prediction"),
                )
            ).strip(),
            "score_transform": str(
                item.get(
                    "score_transform",
                    args.risk_score_transform
                    or config_str(config, "risk_sweep", "risk_score_transform", "rank"),
                )
            ).strip(),
        }
        for item in configured
        if str(item.get("risk_id", item.get("run_id", ""))).strip()
        and str(item.get("prediction_path", item.get("path", ""))).strip()
    ]


def safe_suffix(value: str) -> str:
    suffix = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return suffix or "risk"


def wait_for_path(path: Path, *, timeout_seconds: int, poll_seconds: int = 60) -> None:
    if path.exists():
        return
    if timeout_seconds <= 0:
        raise SystemExit(f"input path does not exist: {path}")
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        print(f"waiting for input: {path}")
        time.sleep(float(poll_seconds))
        if path.exists():
            return
    raise SystemExit(f"timed out waiting for input path: {path}")


def load_predictions(spec: dict[str, str], *, score_col: str, clocks: list[str]) -> pd.DataFrame:
    path = Path(spec["path"])
    available = existing_columns(path)
    required = [*KEY_COLUMNS, score_col, "label", "buy_price"]
    missing = [column for column in required if column not in available]
    if missing:
        raise SystemExit(f"{spec['run_id']}: prediction input missing columns: {missing}")
    columns = [column for column in [*required, *RISK_COLUMNS] if column in available]
    frame = read_frame(path, columns=columns)
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


def load_risk_predictions(
    spec: dict[str, str],
    *,
    default_score_col: str,
    clocks: list[str],
    wait_seconds: int,
) -> tuple[pd.DataFrame, str]:
    risk_name = str(spec["risk_id"]).strip()
    suffix = safe_suffix(risk_name)
    score_col = str(spec.get("score_col") or default_score_col or "prediction").strip()
    path = Path(spec["path"])
    wait_for_path(path, timeout_seconds=int(wait_seconds))
    available = existing_columns(path)
    required = [*KEY_COLUMNS, score_col]
    missing = [column for column in required if column not in available]
    if missing:
        raise SystemExit(f"{risk_name}: risk prediction input missing columns: {missing}")
    frame = read_frame(path, columns=required)
    frame = frame.dropna(subset=[*KEY_COLUMNS, score_col]).copy()
    frame["date"] = frame["date"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["decision_target_timestamp"] = pd.to_datetime(
        frame["decision_target_timestamp"],
        errors="coerce",
    )
    frame["clock"] = frame["decision_target_timestamp"].dt.strftime("%H:%M")
    frame = frame.loc[frame["clock"].isin(clocks)].copy()
    raw_col = f"learned_risk_raw_{suffix}"
    frame[raw_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame.dropna(subset=["decision_target_timestamp", raw_col])
    return frame[[*KEY_COLUMNS, raw_col]].drop_duplicates(list(KEY_COLUMNS)), raw_col


def normalize_next_close_labels(frame: pd.DataFrame) -> pd.DataFrame:
    return shared_normalize_next_close_labels(frame, key_columns=KEY_COLUMNS)


def load_or_fetch_next_close_labels(
    predictions: pd.DataFrame,
    *,
    args: argparse.Namespace,
    config: dict,
    output_dir: Path,
) -> pd.DataFrame:
    configured_input = config_str(config, "risk_sweep", "next_close_label_input", "")
    label_input = args.next_close_label_input or configured_input

    username = args.clickhouse_user or config_str(config, "clickhouse", "user", "")
    password = args.clickhouse_password or config_str(config, "clickhouse", "password", "")

    def _fetch(base: pd.DataFrame) -> pd.DataFrame:
        if not username or not password:
            raise SystemExit(
                "next-close labels not found. Pass --next-close-label-input or set "
                "CLICKHOUSE_USER and CLICKHOUSE_PASSWORD."
            )
        label_base = base[[*KEY_COLUMNS, "buy_price"]].drop_duplicates(list(KEY_COLUMNS))
        return compute_clickhouse_close_labels(
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

    return shared_load_or_fetch_next_close_labels(
        predictions,
        output_dir=output_dir,
        label_input=label_input,
        fetch_labels=_fetch,
        key_columns=KEY_COLUMNS,
    )


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


def add_learned_risk_scores(
    frame: pd.DataFrame,
    risk_specs: list[dict[str, str]],
    *,
    default_score_col: str,
    clocks: list[str],
    wait_seconds: int,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    out = frame.copy()
    learned: list[dict[str, str]] = []
    groupers = [out["run_id"], out["date"], out["decision_target_timestamp"]]
    for spec in risk_specs:
        risk_name = str(spec["risk_id"]).strip()
        suffix = safe_suffix(risk_name)
        risk_frame, raw_col = load_risk_predictions(
            spec,
            default_score_col=default_score_col,
            clocks=clocks,
            wait_seconds=wait_seconds,
        )
        out = out.merge(risk_frame, on=list(KEY_COLUMNS), how="left")
        raw_score = pd.to_numeric(out[raw_col], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        transform = str(spec.get("score_transform") or "rank").strip().lower()
        if transform == "raw":
            score = raw_score.clip(lower=0.0, upper=1.0)
        elif transform in {"", "rank"}:
            score = raw_score.groupby(groupers).rank(method="average", pct=True)
        else:
            raise SystemExit(
                f"{risk_name}: unknown score_transform={transform!r}; expected raw or rank"
            )
        score = score.fillna(score.groupby(groupers).transform("median")).fillna(0.0)
        final_col = f"learned_risk_{suffix}"
        out[final_col] = score.astype("float64")
        learned.append({"risk_id": risk_name, "column": final_col})
    return out, learned


def variant_specs(
    learned_risks: list[dict[str, str]] | None = None,
    *,
    include_alpha_rank: bool = True,
    include_manual_risk: bool = True,
    include_hard_gates: bool = True,
    penalties: tuple[float, ...] = PENALTIES,
    risk_gates: tuple[float, ...] = RISK_GATES,
    learned_risk_gates: tuple[float, ...] = (),
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    if include_alpha_rank:
        specs.append(
            {
                "variant": "alpha_rank",
                "penalty": 0.0,
                "gate": None,
                "risk_col": "dirty_risk",
            }
        )
    if include_manual_risk:
        for penalty in penalties:
            specs.append(
                {
                    "variant": f"risk_penalty_{int(penalty * 100):03d}",
                    "penalty": penalty,
                    "gate": None,
                    "risk_col": "dirty_risk",
                }
            )
    if include_hard_gates:
        specs.append(
            {
                "variant": "hard_gate_next_flip_guard_10t",
                "penalty": 0.0,
                "gate": "guard",
                "risk_col": "dirty_risk",
            }
        )
        for threshold in risk_gates:
            specs.append(
                {
                    "variant": f"hard_gate_risk_le_{int(threshold * 100):03d}",
                    "penalty": 0.0,
                    "gate": threshold,
                    "risk_col": "dirty_risk",
                }
            )
    for learned in learned_risks or []:
        risk_name = safe_suffix(learned["risk_id"])
        for penalty in penalties:
            specs.append(
                {
                    "variant": f"{risk_name}_penalty_{int(penalty * 100):03d}",
                    "penalty": penalty,
                    "gate": None,
                    "risk_col": learned["column"],
                }
            )
        for threshold in learned_risk_gates:
            specs.append(
                {
                    "variant": f"{risk_name}_gate_le_{int(threshold * 100):03d}",
                    "penalty": 0.0,
                    "gate": {
                        "type": "risk_le",
                        "risk_col": learned["column"],
                        "threshold": threshold,
                    },
                    "risk_col": learned["column"],
                }
            )
    return specs


def summarize_selection(
    frame: pd.DataFrame,
    *,
    top_n_list: tuple[int, ...],
    learned_risks: list[dict[str, str]] | None = None,
    include_alpha_rank: bool = True,
    include_manual_risk: bool = True,
    include_hard_gates: bool = True,
    penalties: tuple[float, ...] = PENALTIES,
    risk_gates: tuple[float, ...] = RISK_GATES,
    learned_risk_gates: tuple[float, ...] = (),
    alpha_candidate_rank_min: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    base_frame = frame
    if alpha_candidate_rank_min > 0.0:
        base_frame = frame.loc[frame["alpha_rank"].ge(alpha_candidate_rank_min)].copy()
    for spec in variant_specs(
        learned_risks,
        include_alpha_rank=include_alpha_rank,
        include_manual_risk=include_manual_risk,
        include_hard_gates=include_hard_gates,
        penalties=penalties,
        risk_gates=risk_gates,
        learned_risk_gates=learned_risk_gates,
    ):
        variant = str(spec["variant"])
        penalty = float(spec["penalty"])
        gate = spec["gate"]
        risk_col = str(spec["risk_col"])
        work = base_frame.copy()
        work["final_score"] = work["alpha_rank"] - penalty * work[risk_col]
        if gate == "guard":
            work = work.loc[work["next_flip_guard_10t_pass"]].copy()
        elif gate is not None:
            if isinstance(gate, dict):
                gate_type = str(gate.get("type", ""))
                if gate_type != "risk_le":
                    raise SystemExit(f"unknown risk sweep gate type: {gate_type!r}")
                gate_col = str(gate["risk_col"])
                threshold = float(gate["threshold"])
                work = work.loc[work[gate_col].le(threshold)].copy()
            else:
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
            ranked = group.sort_values("final_score", ascending=False)
            for top_n in top_n_list:
                selected = ranked.head(int(top_n))
                short_stats = selection_return_stats(
                    full_group,
                    selected,
                    label_col="label",
                    prefix="short",
                )
                next_stats = selection_return_stats(
                    full_group,
                    selected,
                    label_col="alpha_return_next_close",
                    prefix="next",
                )
                rows.append(
                    {
                        "run_id": run_name,
                        "variant": variant,
                        "top_n": int(top_n),
                        "penalty": penalty,
                        "gate": "" if gate is None else str(gate),
                        "alpha_candidate_rank_min": alpha_candidate_rank_min,
                        "date": str(date),
                        "decision_target_timestamp": pd.Timestamp(timestamp),
                        "clock": pd.Timestamp(timestamp).strftime("%H:%M"),
                        "rows": int(len(full_group)),
                        "candidate_rows": int(len(group)),
                        "selected_rows": int(len(selected)),
                        "short_top_mean_bps": short_stats["short_top_mean_bps"],
                        "short_top_excess_bps": short_stats["short_top_excess_bps"],
                        "next_top_mean_bps": next_stats["next_top_mean_bps"],
                        "next_top_excess_bps": next_stats["next_top_excess_bps"],
                        "selected_guard_pass_count": int(
                            selected["next_flip_guard_10t_pass"].sum()
                        ),
                        "selected_dirty_risk": (
                            float(selected[risk_col].mean()) if len(selected) else float("nan")
                        ),
                    }
                )

    group_metrics = pd.DataFrame(rows)
    minute = (
        group_metrics.groupby(["run_id", "variant", "top_n", "clock"], as_index=False)
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
        .sort_values(["run_id", "variant", "top_n", "clock"])
    )
    summary = (
        group_metrics.groupby(["run_id", "variant", "top_n"], as_index=False)
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
        .sort_values(
            ["run_id", "top_n", "next_top_excess_bps", "short_top_excess_bps"],
            ascending=[True, True, False, False],
        )
    )
    next_positive_minutes = (
        minute.groupby(["run_id", "variant", "top_n"])["next_top_excess_bps"]
        .apply(lambda s: int((s > 0).sum()))
        .reset_index(name="next_positive_minute_count")
    )
    summary = summary.merge(
        next_positive_minutes,
        on=["run_id", "variant", "top_n"],
        how="left",
    )
    return group_metrics, minute, summary


def main() -> None:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    run_name = run_id(config, args.config) if args.config else "score_risk_sweep"
    output_dir = Path(
        args.output_dir
        or config_str(config, "output", "local_dir", f"output/legacy/analysis/{run_name}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    score_col = args.score_col or config_str(config, "risk_sweep", "score_col", "prediction")
    risk_score_col = args.risk_score_col or config_str(
        config,
        "risk_sweep",
        "risk_score_col",
        "prediction",
    )
    top_n = int(
        args.top_n if args.top_n is not None else config_int(config, "risk_sweep", "top_n", 100)
    )
    top_n_list = (
        (top_n,)
        if args.top_n is not None
        else _int_sequence(config, "risk_sweep", "top_n_list", (top_n,))
    )
    penalties = _float_sequence(config, "risk_sweep", "penalties", PENALTIES)
    risk_gates = _float_sequence(config, "risk_sweep", "risk_gates", RISK_GATES)
    learned_risk_gates = _float_sequence(
        config,
        "risk_sweep",
        "learned_risk_gates",
        (),
    )
    alpha_candidate_rank_min = config_float(
        config,
        "risk_sweep",
        "alpha_candidate_rank_min",
        0.0,
    )
    start_clock = args.start_clock or config_str(config, "risk_sweep", "start_clock", "09:31")
    end_clock = args.end_clock or config_str(config, "risk_sweep", "end_clock", "09:40")
    clocks = clock_range(start_clock, end_clock)
    specs = parse_input_specs(args, config)
    risk_specs = parse_risk_input_specs(args, config)
    wait_seconds = int(
        args.wait_for_inputs_seconds
        if args.wait_for_inputs_seconds is not None
        else config_int(config, "risk_sweep", "wait_for_inputs_seconds", 0)
    )
    manual_override = args.manual_risk.strip().lower()
    include_manual_risk = (
        manual_override == "true"
        if manual_override
        else config_bool(config, "risk_sweep", "include_manual_risk", True)
    )
    include_hard_gates = config_bool(config, "risk_sweep", "include_hard_gates", True)
    include_alpha_rank = config_bool(config, "risk_sweep", "include_alpha_rank", True)

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
    frame, learned_risks = add_learned_risk_scores(
        frame,
        risk_specs,
        default_score_col=risk_score_col,
        clocks=clocks,
        wait_seconds=wait_seconds,
    )

    group_metrics, minute, summary = summarize_selection(
        frame,
        top_n_list=top_n_list,
        learned_risks=learned_risks,
        include_alpha_rank=include_alpha_rank,
        include_manual_risk=include_manual_risk,
        include_hard_gates=include_hard_gates,
        penalties=penalties,
        risk_gates=risk_gates,
        learned_risk_gates=learned_risk_gates,
        alpha_candidate_rank_min=alpha_candidate_rank_min,
    )
    group_metrics.to_csv(output_dir / "score_risk_group_metrics.csv", index=False)
    minute.to_csv(output_dir / "score_risk_minute_summary.csv", index=False)
    summary.to_csv(output_dir / "score_risk_summary.csv", index=False)

    trace = {
        "run_id": run_name,
        "inputs": specs,
        "risk_inputs": risk_specs,
        "learned_risks": learned_risks,
        "score_col": score_col,
        "risk_score_col": risk_score_col,
        "top_n_list": list(top_n_list),
        "penalties": list(penalties),
        "risk_gates": list(risk_gates),
        "learned_risk_gates": list(learned_risk_gates),
        "alpha_candidate_rank_min": alpha_candidate_rank_min,
        "clocks": clocks,
        "rows": int(len(frame)),
        "groups": int(
            frame[["run_id", "date", "decision_target_timestamp"]].drop_duplicates().shape[0]
        ),
        "risk_rank_min": RISK_RANK_MIN,
        "risk_rank_max": RISK_RANK_MAX,
        "include_manual_risk": include_manual_risk,
        "include_hard_gates": include_hard_gates,
        "include_alpha_rank": include_alpha_rank,
        "outputs": {
            "summary": str(output_dir / "score_risk_summary.csv"),
            "minute_summary": str(output_dir / "score_risk_minute_summary.csv"),
            "group_metrics": str(output_dir / "score_risk_group_metrics.csv"),
            "next_close_labels": str(output_dir / "clickhouse_next_close_labels.parquet"),
        },
    }
    write_json(output_dir / "score_risk_trace.json", trace)

    print("score_risk_summary")
    print(
        summary[
            [
                "run_id",
                "variant",
                "top_n",
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
