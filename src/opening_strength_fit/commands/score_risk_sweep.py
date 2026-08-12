"""Score-risk sweep data preparation, evaluation, and reporting workflow."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from opening_strength_fit.analysis import (
    clock_range,
    mean_aggregations,
    selection_group_metrics,
    write_json,
)
from opening_strength_fit.commands.arguments import add_arguments, command_context
from opening_strength_fit.config import (
    config_bool,
    config_float,
    config_float_tuple,
    config_int_tuple,
    config_str,
    prepare_output_dir,
)
from opening_strength_fit.next_close_labels import (
    add_next_close_label_arguments,
)
from opening_strength_fit.prediction_frames import read_clock_predictions
from opening_strength_fit.risk_labels import (
    RISK_RANK_MAX,
    RISK_RANK_MIN,
    load_risk_next_close_labels,
    next_close_label_request,
    rank_risk_components,
)
from opening_strength_fit.schema import DECISION_KEY_COLUMNS

KEY_COLUMNS = DECISION_KEY_COLUMNS
RISK_COLUMNS = tuple(
    "spread_bps turnover_diff_10t return_10t ask_depth_10 depth_imbalance_10".split()
)
PENALTIES = (0.25, 0.50, 0.75, 1.00)
RISK_GATES = (0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep alpha-rank minus dirty-risk penalties and hard gates over existing prediction files."
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--input", action="append", default=[], help="run_id=path")
    parser.add_argument(
        "--risk-input",
        action="append",
        default=[],
        help="risk_id=path for learned risk prediction files.",
    )
    add_arguments(parser, "output-dir score-col risk-score-col", default="")
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
    add_arguments(parser, "start-clock end-clock", default="")
    add_next_close_label_arguments(parser, include_connection=True)
    return parser.parse_args()


def parse_path_specs(values: list[str], *, option: str, id_key: str) -> list[dict[str, str]]:
    specs = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} must be formatted as {id_key}=path")
        name, path = value.split("=", 1)
        specs.append({id_key: name.strip(), "path": path.strip()})
    return specs


def parse_input_specs(args: argparse.Namespace, config: dict) -> list[dict[str, str]]:
    specs = parse_path_specs(args.input, option="--input", id_key="run_id")
    if specs:
        return specs
    configured = config.get("risk_sweep", {}).get("inputs", [])
    if not configured:
        raise SystemExit("risk sweep requires --input or [[risk_sweep.inputs]]")
    return [
        {"run_id": name, "path": path}
        for item in configured
        if (name := str(item.get("run_id", "")).strip())
        and (path := str(item.get("prediction_path", item.get("path", ""))).strip())
    ]


def parse_risk_input_specs(args: argparse.Namespace, config: dict) -> list[dict[str, str]]:
    specs = [
        {
            **spec,
            "score_col": args.risk_score_col.strip(),
            "score_transform": args.risk_score_transform.strip(),
        }
        for spec in parse_path_specs(
            args.risk_input,
            option="--risk-input",
            id_key="risk_id",
        )
    ]
    if specs:
        return specs
    configured = config.get("risk_sweep", {}).get("risk_inputs", [])
    score_col = args.risk_score_col or config_str(
        config, "risk_sweep", "risk_score_col", "prediction"
    )
    score_transform = args.risk_score_transform or config_str(
        config, "risk_sweep", "risk_score_transform", "rank"
    )
    return [
        {
            "risk_id": risk_id,
            "path": path,
            "score_col": str(item.get("score_col", score_col)).strip(),
            "score_transform": str(item.get("score_transform", score_transform)).strip(),
        }
        for item in configured
        if (risk_id := str(item.get("risk_id", item.get("run_id", ""))).strip())
        and (path := str(item.get("prediction_path", item.get("path", ""))).strip())
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
    required = (*KEY_COLUMNS, score_col, "label", "buy_price")
    frame = read_clock_predictions(
        Path(spec["path"]),
        required_columns=required,
        optional_columns=RISK_COLUMNS,
        clocks=clocks,
        context=f"{spec['run_id']}: prediction input",
    )
    frame["run_id"] = spec["run_id"]
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
    raw_col = f"learned_risk_raw_{suffix}"
    frame = read_clock_predictions(
        path,
        required_columns=(*KEY_COLUMNS, score_col),
        clocks=clocks,
        context=f"{risk_name}: risk prediction input",
    )
    frame[raw_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame.dropna(subset=["decision_target_timestamp", raw_col])
    return frame[[*KEY_COLUMNS, raw_col]].drop_duplicates(list(KEY_COLUMNS)), raw_col


def add_risk_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    group_columns = ("run_id", "date", "decision_target_timestamp")
    groupers = [out[column] for column in group_columns]
    out["alpha_rank"] = out["alpha_score"].groupby(groupers).rank(method="average", pct=True)
    components = rank_risk_components(
        out,
        group_columns=group_columns,
        context="risk sweep",
    ).rename(columns=lambda column: f"{column}_risk")
    out[components.columns] = components
    risk_frame = out[components.columns]
    out["dirty_risk"] = risk_frame.mean(axis=1).clip(lower=0.0)
    out["dirty_risk_component_count"] = risk_frame.gt(0.0).sum(axis=1)
    out["next_flip_guard_10t_pass"] = risk_frame.eq(0.0).all(axis=1)
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


def risk_variant(
    variant: str,
    *,
    risk_col: str = "dirty_risk",
    penalty: float = 0.0,
    gate: object = None,
) -> dict[str, object]:
    return {"variant": variant, "penalty": penalty, "gate": gate, "risk_col": risk_col}


def penalty_variants(
    prefix: str, risk_col: str, penalties: tuple[float, ...]
) -> list[dict[str, object]]:
    return [
        risk_variant(f"{prefix}_penalty_{int(value * 100):03d}", penalty=value, risk_col=risk_col)
        for value in penalties
    ]


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
        specs.append(risk_variant("alpha_rank"))
    if include_manual_risk:
        specs.extend(penalty_variants("risk", "dirty_risk", penalties))
    if include_hard_gates:
        specs.append(risk_variant("hard_gate_next_flip_guard_10t", gate="guard"))
        specs.extend(
            risk_variant(
                f"hard_gate_risk_le_{int(threshold * 100):03d}",
                gate=threshold,
            )
            for threshold in risk_gates
        )
    for learned in learned_risks or []:
        risk_name, risk_col = safe_suffix(learned["risk_id"]), learned["column"]
        specs.extend(penalty_variants(risk_name, risk_col, penalties))
        specs.extend(
            risk_variant(
                f"{risk_name}_gate_le_{int(threshold * 100):03d}",
                risk_col=risk_col,
                gate={
                    "type": "risk_le",
                    "risk_col": risk_col,
                    "threshold": threshold,
                },
            )
            for threshold in learned_risk_gates
        )
    return specs


def aggregate_selection_metrics(
    group_metrics: pd.DataFrame,
    keys: list[str],
    *,
    include_positive_rate: bool = False,
) -> pd.DataFrame:
    aggregations = {
        "groups": ("date", "size"),
        **mean_aggregations(
            *"candidate_rows selected_rows short_top_mean_bps short_top_excess_bps "
            "next_top_mean_bps next_top_excess_bps".split()
        ),
    }
    if include_positive_rate:
        aggregations["next_excess_positive_rate"] = (
            "next_top_excess_bps",
            lambda values: float((values > 0).mean()),
        )
    aggregations.update(mean_aggregations("selected_guard_pass_count", "selected_dirty_risk"))
    return group_metrics.groupby(keys, as_index=False).agg(**aggregations)


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
    base_frame = (
        frame.loc[frame["alpha_rank"].ge(alpha_candidate_rank_min)].copy()
        if alpha_candidate_rank_min > 0.0
        else frame
    )
    group_columns = ["run_id", "date", "decision_target_timestamp"]
    full_groups = frame.groupby(group_columns, sort=False)
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

        for (run_name, date, timestamp), group in work.groupby(group_columns, sort=True):
            full_group = full_groups.get_group((run_name, date, timestamp))
            ranked = group.sort_values("final_score", ascending=False)
            for top_n in top_n_list:
                selected = ranked.head(int(top_n))
                rows.append(
                    {
                        "run_id": run_name,
                        "variant": variant,
                        "top_n": int(top_n),
                        "penalty": penalty,
                        "gate": "" if gate is None else str(gate),
                        "alpha_candidate_rank_min": alpha_candidate_rank_min,
                        **selection_group_metrics(
                            full_group,
                            selected,
                            date=date,
                            timestamp=timestamp,
                            candidate_counts={"candidate_rows": int(len(group))},
                            include_all_mean=False,
                            include_win_rate=False,
                        ),
                        "selected_guard_pass_count": int(
                            selected["next_flip_guard_10t_pass"].sum()
                        ),
                        "selected_dirty_risk": (
                            float(selected[risk_col].mean()) if len(selected) else float("nan")
                        ),
                    }
                )

    group_metrics = pd.DataFrame(rows)
    minute = aggregate_selection_metrics(
        group_metrics,
        ["run_id", "variant", "top_n", "clock"],
    ).sort_values(["run_id", "variant", "top_n", "clock"])
    summary = aggregate_selection_metrics(
        group_metrics,
        ["run_id", "variant", "top_n"],
        include_positive_rate=True,
    ).sort_values(
        ["run_id", "top_n", "next_top_excess_bps", "short_top_excess_bps"],
        ascending=[True, True, False, False],
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
    config, settings, run_name = command_context(
        args, "risk_sweep", default_run_name="score_risk_sweep"
    )
    output_dir = prepare_output_dir(config, args.output_dir, run_name)

    score_col = settings.string("score_col", "prediction")
    risk_score_col = settings.string("risk_score_col", "prediction")
    top_n = settings.integer("top_n", 100)
    top_n_list = (
        (top_n,)
        if args.top_n is not None
        else config_int_tuple(config, "risk_sweep", "top_n_list", (top_n,))
    )
    penalties = config_float_tuple(config, "risk_sweep", "penalties", PENALTIES)
    risk_gates = config_float_tuple(config, "risk_sweep", "risk_gates", RISK_GATES)
    learned_risk_gates = config_float_tuple(config, "risk_sweep", "learned_risk_gates", ())
    alpha_candidate_rank_min = config_float(config, "risk_sweep", "alpha_candidate_rank_min", 0.0)
    clocks = clock_range(
        settings.string("start_clock", "09:31"),
        settings.string("end_clock", "09:40"),
    )
    specs = parse_input_specs(args, config)
    risk_specs = parse_risk_input_specs(args, config)
    wait_seconds = settings.integer("wait_for_inputs_seconds", 0)
    manual_override = args.manual_risk.strip().lower()
    include_manual_risk = (
        manual_override == "true"
        if manual_override
        else config_bool(config, "risk_sweep", "include_manual_risk", True)
    )
    include_hard_gates = config_bool(config, "risk_sweep", "include_hard_gates", True)
    include_alpha_rank = config_bool(config, "risk_sweep", "include_alpha_rank", True)

    predictions = pd.concat(
        [load_predictions(spec, score_col=score_col, clocks=clocks) for spec in specs],
        ignore_index=True,
    )
    labels = load_risk_next_close_labels(
        predictions,
        request=next_close_label_request(args, config, section="risk_sweep"),
        output_dir=output_dir,
        context="score-risk labels",
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
            "run_id variant top_n short_top_excess_bps next_top_excess_bps "
            "next_positive_minute_count selected_guard_pass_count selected_dirty_risk".split()
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nwrote: {output_dir}")


if __name__ == "__main__":
    main()
