from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from opening_strength_fit.artifact_catalog import (
    is_non_standard_artifact_run,
)
from opening_strength_fit.commands.artifact_sync_artifacts import (  # noqa: F401
    ARTIFACT_PULLERS,
    combine_rolling_validation_shards,
    local_artifact_dir,
    pull_capacity_acceptance_artifacts,
    pull_capacity_audit_artifacts,
    pull_exposure_audit_artifacts,
    pull_feature_audit_artifacts,
    pull_feature_hygiene_artifacts,
    pull_gap_attribution_artifacts,
    pull_pool_internal_analysis_artifacts,
    pull_rolling_validation_artifacts,
    pull_rolling_validation_shards,
    pull_score_risk_artifacts,
    pull_strategy_acceptance_artifacts,
    record_lightweight_artifacts,
)
from opening_strength_fit.commands.artifact_sync_metrics import (  # noqa: F401
    DEFAULT_NEXT_CLOSE_LABEL_PVC_DIR,
    METRICS_SUFFIX,
    collect_run_statuses,
    combine_metric_frames,
    fetch_predictions,
    pull_metrics,
    pull_next_close_labels,
    record_metrics,
)
from opening_strength_fit.k8s import (
    DEFAULT_IMAGE,
    RunSpec,
    delete_temp_pod,
    ensure_temp_pod,
    load_run_spec,
)

DEFAULT_METRICS_DIR = "experiments/results/metrics"
DEFAULT_RECORDS_DIR = "experiments/evidence"
DEFAULT_PARTIAL_METRICS_DIR = "output/artifacts/_partial_metrics"
DEFAULT_METRIC_SHARDS_ROOT = "output/artifacts"


def parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as label=/pvc/path")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.rstrip("/")
    if not label or not path:
        raise argparse.ArgumentTypeError("--run needs a non-empty label and path")
    return label, path


def ad_hoc_run_spec(label: str, pvc_dir: str, namespace: str) -> RunSpec:
    return RunSpec(
        run_id=label,
        pvc_dir=pvc_dir,
        namespace=namespace,
        pvc="bizewu-private-data",
        mount_path="/mnt/output",
        pull_secret="highfort",
        image=DEFAULT_IMAGE,
        test_start_year=0,
        test_end_year=0,
        kind="ad_hoc",
    )


def validate_specs(specs: list[RunSpec]) -> None:
    if not specs:
        return
    connection_fields = "namespace pvc mount_path pull_secret image".split()
    expected = tuple(getattr(specs[0], field) for field in connection_fields)
    if any(
        tuple(getattr(spec, field) for field in connection_fields) != expected for spec in specs[1:]
    ):
        raise SystemExit(
            "All synced runs must share namespace, pvc, mount_path, "
            "image_pull_secret, and helper image."
        )


@dataclass(frozen=True)
class SyncPlan:
    specs: list[RunSpec]
    fetch_metrics: bool
    fetch_predictions: bool
    fetch_artifacts: bool
    fetch_next_close_labels: bool
    record: bool
    metrics_dir: Path
    metric_shards_root: Path
    predictions_root: Path
    artifacts_root: Path | None
    next_close_labels_root: Path
    records_dir: Path

    @property
    def needs_pod(self) -> bool:
        return any(
            getattr(self, field)
            for field in "fetch_metrics fetch_predictions fetch_artifacts fetch_next_close_labels".split()
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync K8s PVC metrics and lightweight analysis artifacts."
    )
    parser.add_argument("--config", action="append", type=load_run_spec)
    parser.add_argument("--run", action="append", type=parse_run)
    parser.add_argument("--namespace", default="bizewu")
    parser.add_argument("--hfcli", default="hfcli")
    parser.add_argument("--pod", default="")
    parser.add_argument("--timeout", default="300s")
    parser.add_argument(
        "--metrics-dir",
        default=DEFAULT_METRICS_DIR,
        help="Directory for synced metrics CSVs. Defaults to the formal results archive.",
    )
    parser.add_argument(
        "--partial-metrics-dir",
        default=DEFAULT_PARTIAL_METRICS_DIR,
        help="Directory for --allow-partial metrics so incomplete runs do not touch results.",
    )
    parser.add_argument(
        "--metric-shards-root",
        default=DEFAULT_METRIC_SHARDS_ROOT,
        help="Ignored root for raw sharded metrics fetched before local combination.",
    )
    parser.add_argument("--predictions-root", default="output/legacy/predictions")
    parser.add_argument(
        "--artifacts-root",
        default="",
        help=(
            "Override the local artifact mirror root. Defaults to each run config's "
            "[output].local_dir, falling back to output/artifacts/<run_id>."
        ),
    )
    parser.add_argument("--next-close-labels-root", default="output/legacy/labels")
    parser.add_argument("--next-close-label-pvc-dir", default=DEFAULT_NEXT_CLOSE_LABEL_PVC_DIR)
    parser.add_argument(
        "--records-dir",
        default=DEFAULT_RECORDS_DIR,
        help="Tracked destination for compact evidence selected by --record.",
    )
    parser.add_argument("--runs-dir", default="experiments/runs")
    actions = (
        ("--metrics", "Fetch metrics CSVs."),
        (
            "--predictions",
            "Fetch prediction parquet explicitly. Cluster-side analysis artifacts do not need this.",
        ),
        (
            "--artifacts",
            "Fetch lightweight artifact files, including cluster-side pool-internal "
            "analysis outputs and non-standard sweep outputs.",
        ),
        (
            "--analysis-artifacts",
            "Alias for --artifacts when the config declares cluster-side analysis.",
        ),
        (
            "--next-close-labels",
            "Fetch yearly next-close label shards needed by pool-internal analysis.",
        ),
        ("--record", "Archive fetched metrics."),
        (
            "--all",
            "Fetch metrics and lightweight cluster-side artifacts, then archive them. "
            "Prediction parquet is only fetched with --predictions.",
        ),
        (
            "--allow-partial",
            "For sharded runs, sync completed shards without failing on missing "
            "future months/years. Recording to experiments/results is disabled.",
        ),
    )
    for option, help_text in actions:
        parser.add_argument(option, action="store_true", help=help_text)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_sync_plan(args: argparse.Namespace) -> SyncPlan:
    specs: list[RunSpec] = list(args.config or [])
    specs.extend(
        ad_hoc_run_spec(label, pvc_dir, args.namespace) for label, pvc_dir in (args.run or [])
    )
    if not specs:
        raise SystemExit("pass at least one --config or --run")
    validate_specs(specs)

    no_action = not any(
        (
            args.metrics,
            args.predictions,
            args.artifacts,
            args.analysis_artifacts,
            args.next_close_labels,
            args.record,
            args.all,
        )
    )
    metrics_dir = Path(args.metrics_dir)
    if args.allow_partial and args.metrics_dir == DEFAULT_METRICS_DIR:
        metrics_dir = Path(args.partial_metrics_dir)
    return SyncPlan(
        specs=specs,
        fetch_metrics=args.all or args.metrics or no_action,
        fetch_predictions=args.predictions,
        fetch_artifacts=args.all or args.artifacts or args.analysis_artifacts or no_action,
        fetch_next_close_labels=args.next_close_labels,
        record=(args.all or args.record or no_action) and not args.allow_partial,
        metrics_dir=metrics_dir,
        metric_shards_root=Path(args.metric_shards_root),
        predictions_root=Path(args.predictions_root),
        artifacts_root=Path(args.artifacts_root) if args.artifacts_root else None,
        next_close_labels_root=Path(args.next_close_labels_root),
        records_dir=Path(args.records_dir),
    )


def print_sync_plan(args: argparse.Namespace, plan: SyncPlan, pod_name: str) -> None:
    print("sync_plan:")
    print(f"  pod: {pod_name or '<none>'}")
    print(f"  metrics: {plan.fetch_metrics}")
    print(f"  metrics_dir: {plan.metrics_dir}")
    print(f"  predictions: {plan.fetch_predictions}")
    print(f"  artifacts: {plan.fetch_artifacts}")
    if plan.fetch_artifacts or plan.record:
        for spec in plan.specs:
            print(f"  artifact_dir[{spec.run_id}]: {local_artifact_dir(spec, plan.artifacts_root)}")
    print(f"  next_close_labels: {plan.fetch_next_close_labels}")
    print(f"  record: {plan.record}")
    print(f"  allow_partial: {args.allow_partial}")
    for spec in plan.specs:
        print(f"  {spec.run_id}: {spec.pvc_dir}")


def pull_metrics_for_specs(args: argparse.Namespace, plan: SyncPlan, pod_name: str) -> None:
    if not plan.fetch_metrics:
        return
    print("pulled_metrics:")
    for spec in plan.specs:
        if (
            spec.kind == "pool_internal_analysis" or is_non_standard_artifact_run(spec)
        ) and not args.metrics:
            print(f"  {spec.run_id}: skipped metrics for {spec.kind}")
            continue
        paths = pull_metrics(
            args.hfcli,
            spec,
            pod_name,
            plan.metrics_dir,
            allow_partial=args.allow_partial,
            raw_root=plan.metric_shards_root,
        )
        print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")


def pull_artifacts_for_spec(
    args: argparse.Namespace,
    plan: SyncPlan,
    spec: RunSpec,
    pod_name: str,
) -> list[Path] | None:
    common = (args.hfcli, spec, pod_name, plan.artifacts_root)
    if spec.pool_internal_analysis_enabled:
        return pull_pool_internal_analysis_artifacts(*common)
    if not is_non_standard_artifact_run(spec):
        return None
    return ARTIFACT_PULLERS.get(spec.kind, pull_feature_audit_artifacts)(*common)


def record_synced_results(args: argparse.Namespace, plan: SyncPlan) -> None:
    if not plan.record:
        return
    statuses = collect_run_statuses(Path(args.runs_dir))
    print("recorded_metrics:")
    for spec in plan.specs:
        if spec.kind == "pool_internal_analysis" and not args.metrics:
            print(f"  {spec.run_id}: no metrics to record for {spec.kind}")
            continue
        paths = record_metrics(spec.run_id, plan.metrics_dir, plan.records_dir)
        status = statuses.get(spec.run_id, "")
        if paths:
            print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
            if status and status != "completed":
                print(f"  {spec.run_id}: config status is {status!r}; confirm before final archive")
        elif is_non_standard_artifact_run(spec) and not args.metrics:
            print(f"  {spec.run_id}: no metrics to record for {spec.kind}")
        else:
            print(f"  {spec.run_id}: missing {plan.metrics_dir / f'{spec.run_id}{METRICS_SUFFIX}'}")
    print("recorded_artifacts:")
    for spec in plan.specs:
        paths = record_lightweight_artifacts(spec, plan.artifacts_root, plan.records_dir)
        if paths:
            print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
        elif spec.pool_internal_analysis_enabled:
            print(f"  {spec.run_id}: no pool-internal analysis artifacts to record")
        elif is_non_standard_artifact_run(spec):
            print(f"  {spec.run_id}: no lightweight artifacts to record")


def main() -> None:
    args = build_parser().parse_args()
    plan = resolve_sync_plan(args)
    pod_name = args.pod
    created_temp_pod = False
    if plan.needs_pod and not pod_name:
        pod_name = ensure_temp_pod(
            args.hfcli,
            plan.specs[0],
            args.timeout,
            "opening-strength-sync-artifacts",
            dry_run=args.dry_run,
        )
        created_temp_pod = not args.dry_run

    if args.dry_run:
        print_sync_plan(args, plan, pod_name)
        return

    try:
        pull_metrics_for_specs(args, plan, pod_name)
        if plan.fetch_predictions:
            print("pulled_predictions:")
            for spec in plan.specs:
                path = fetch_predictions(
                    args.hfcli,
                    spec,
                    pod_name,
                    plan.predictions_root,
                    allow_partial=args.allow_partial,
                )
                print(f"  {spec.run_id}: {path}")
        if plan.fetch_artifacts:
            print("pulled_artifacts:")
            for spec in plan.specs:
                paths = pull_artifacts_for_spec(args, plan, spec, pod_name)
                if paths is None:
                    if args.artifacts:
                        print(f"  {spec.run_id}: no artifact sync configured")
                    continue
                print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
        if plan.fetch_next_close_labels:
            print("pulled_next_close_labels:")
            for spec in plan.specs:
                paths = pull_next_close_labels(
                    args.hfcli,
                    spec,
                    pod_name,
                    plan.next_close_labels_root,
                    label_pvc_dir=args.next_close_label_pvc_dir,
                )
                print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
    finally:
        if created_temp_pod:
            delete_temp_pod(args.hfcli, plan.specs[0].namespace, pod_name)
    record_synced_results(args, plan)


if __name__ == "__main__":
    main()
