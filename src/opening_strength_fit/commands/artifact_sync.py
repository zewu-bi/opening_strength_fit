from __future__ import annotations

import argparse
from pathlib import Path

from opening_strength_fit.commands.artifact_sync_artifacts import (
    STRATEGY_ACCEPTANCE_ARTIFACTS,
    is_capacity_acceptance,
    is_capacity_audit,
    is_exposure_audit,
    is_feature_hygiene,
    is_gap_attribution,
    is_non_standard_artifact_run,
    is_pool_internal_analysis,
    is_rolling_validation,
    is_score_risk_sweep,
    local_artifact_dir,
    pull_artifact_set,
    pull_capacity_acceptance_artifacts,
    pull_capacity_audit_artifacts,
    pull_exposure_audit_artifacts,
    pull_feature_audit_artifacts,
    pull_feature_hygiene_artifacts,
    pull_gap_attribution_artifacts,
    pull_pool_internal_analysis_artifacts,
    pull_rolling_validation_artifacts,
    pull_score_risk_artifacts,
    record_artifact_fetch,
    record_lightweight_artifacts,
)
from opening_strength_fit.commands.artifact_sync_artifacts import (
    combine_rolling_validation_shards as combine_rolling_validation_shards,
)
from opening_strength_fit.commands.artifact_sync_artifacts import (
    pull_rolling_validation_shards as pull_rolling_validation_shards,
)
from opening_strength_fit.commands.artifact_sync_metrics import (
    DEFAULT_NEXT_CLOSE_LABEL_PVC_DIR,
    METRICS_SUFFIX,
    collect_run_statuses,
    fetch_predictions,
    pull_metrics,
    pull_next_close_labels,
    record_metrics,
)
from opening_strength_fit.commands.artifact_sync_metrics import (
    combine_metric_frames as combine_metric_frames,
)
from opening_strength_fit.k8s import (
    DEFAULT_IMAGE,
    RunSpec,
    delete_temp_pod,
    ensure_temp_pod,
    load_run_spec,
)

DEFAULT_METRICS_DIR = "experiments/results/metrics"
DEFAULT_PARTIAL_METRICS_DIR = "output/artifacts/_partial_metrics"
DEFAULT_METRIC_SHARDS_ROOT = "output/artifacts"


def pull_strategy_acceptance_artifacts(
    hfcli: str,
    spec: RunSpec,
    pod_name: str,
    output_root: Path | None,
) -> list[Path]:
    output_dir, pulled, missing = pull_artifact_set(
        hfcli,
        spec,
        pod_name,
        output_root,
        STRATEGY_ACCEPTANCE_ARTIFACTS,
    )
    if not pulled:
        raise SystemExit(
            f"{spec.run_id}: no strategy-acceptance artifacts found under {spec.pvc_dir}"
        )
    record_artifact_fetch(spec, output_dir, pulled, missing)
    return pulled


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
    first = specs[0]
    for spec in specs[1:]:
        if (
            spec.namespace != first.namespace
            or spec.pvc != first.pvc
            or spec.mount_path != first.mount_path
            or spec.pull_secret != first.pull_secret
            or spec.image != first.image
        ):
            raise SystemExit(
                "All synced runs must share namespace, pvc, mount_path, "
                "image_pull_secret, and helper image."
            )


def main() -> None:
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
    parser.add_argument("--records-dir", default="experiments/results")
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument("--metrics", action="store_true", help="Fetch metrics CSVs.")
    parser.add_argument(
        "--predictions",
        action="store_true",
        help="Fetch prediction parquet explicitly. Cluster-side analysis artifacts do not need this.",
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help=(
            "Fetch lightweight artifact files, including cluster-side pool-internal "
            "analysis outputs and non-standard sweep outputs."
        ),
    )
    parser.add_argument(
        "--analysis-artifacts",
        action="store_true",
        help="Alias for --artifacts when the config declares cluster-side analysis.",
    )
    parser.add_argument(
        "--next-close-labels",
        action="store_true",
        help="Fetch yearly next-close label shards needed by pool-internal analysis.",
    )
    parser.add_argument("--record", action="store_true", help="Archive fetched metrics.")
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Fetch metrics and lightweight cluster-side artifacts, then archive them. "
            "Prediction parquet is only fetched with --predictions."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "For sharded runs, sync completed shards without failing on missing "
            "future months/years. Recording to experiments/results is disabled."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs: list[RunSpec] = []
    if args.config:
        specs.extend(args.config)
    if args.run:
        specs.extend(ad_hoc_run_spec(label, pvc_dir, args.namespace) for label, pvc_dir in args.run)
    if not specs:
        raise SystemExit("pass at least one --config or --run")
    validate_specs(specs)

    no_action = not (
        args.metrics
        or args.predictions
        or args.artifacts
        or args.analysis_artifacts
        or args.next_close_labels
        or args.record
        or args.all
    )
    fetch_metrics_flag = args.all or args.metrics or no_action
    fetch_predictions_flag = args.predictions
    fetch_artifacts_flag = args.all or args.artifacts or args.analysis_artifacts or no_action
    fetch_next_close_labels_flag = args.next_close_labels
    record_flag = args.all or args.record or no_action
    if args.allow_partial:
        record_flag = False

    records_dir = Path(args.records_dir)
    metrics_dir = Path(args.metrics_dir)
    if args.metrics_dir == DEFAULT_METRICS_DIR:
        metrics_dir = records_dir / "metrics"
    if args.allow_partial and args.metrics_dir == DEFAULT_METRICS_DIR:
        metrics_dir = Path(args.partial_metrics_dir)
    metric_shards_root = Path(args.metric_shards_root)
    predictions_root = Path(args.predictions_root)
    artifacts_root = Path(args.artifacts_root) if args.artifacts_root else None
    next_close_labels_root = Path(args.next_close_labels_root)
    needs_pod = (
        fetch_metrics_flag
        or fetch_predictions_flag
        or fetch_artifacts_flag
        or fetch_next_close_labels_flag
    )
    pod_name = args.pod
    created_temp_pod = False
    if needs_pod and not pod_name:
        pod_name = ensure_temp_pod(
            args.hfcli,
            specs[0],
            args.timeout,
            "opening-strength-sync-artifacts",
            dry_run=args.dry_run,
        )
        created_temp_pod = not args.dry_run

    if args.dry_run:
        print("sync_plan:")
        print(f"  pod: {pod_name or '<none>'}")
        print(f"  metrics: {fetch_metrics_flag}")
        print(f"  metrics_dir: {metrics_dir}")
        print(f"  predictions: {fetch_predictions_flag}")
        print(f"  artifacts: {fetch_artifacts_flag}")
        if fetch_artifacts_flag or record_flag:
            for spec in specs:
                print(f"  artifact_dir[{spec.run_id}]: {local_artifact_dir(spec, artifacts_root)}")
        print(f"  next_close_labels: {fetch_next_close_labels_flag}")
        print(f"  record: {record_flag}")
        print(f"  allow_partial: {args.allow_partial}")
        for spec in specs:
            print(f"  {spec.run_id}: {spec.pvc_dir}")
        return

    statuses = collect_run_statuses(Path(args.runs_dir))
    try:
        if fetch_metrics_flag:
            print("pulled_metrics:")
            for spec in specs:
                if spec.kind == "pool_internal_analysis" and not args.metrics:
                    print(f"  {spec.run_id}: skipped metrics for {spec.kind}")
                    continue
                if is_non_standard_artifact_run(spec) and not args.metrics:
                    print(f"  {spec.run_id}: skipped metrics for {spec.kind}")
                    continue
                paths = pull_metrics(
                    args.hfcli,
                    spec,
                    pod_name,
                    metrics_dir,
                    allow_partial=args.allow_partial,
                    raw_root=metric_shards_root,
                )
                print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
        if fetch_predictions_flag:
            print("pulled_predictions:")
            for spec in specs:
                if (is_score_risk_sweep(spec) or is_gap_attribution(spec)) and not args.predictions:
                    print(f"  {spec.run_id}: skipped predictions for {spec.kind}")
                    continue
                path = fetch_predictions(
                    args.hfcli,
                    spec,
                    pod_name,
                    predictions_root,
                    allow_partial=args.allow_partial,
                )
                print(f"  {spec.run_id}: {path}")
        if fetch_artifacts_flag:
            print("pulled_artifacts:")
            for spec in specs:
                if is_pool_internal_analysis(spec):
                    paths = pull_pool_internal_analysis_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                    print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
                    continue
                if not is_non_standard_artifact_run(spec):
                    if args.artifacts:
                        print(f"  {spec.run_id}: no artifact sync configured")
                    continue
                if is_score_risk_sweep(spec):
                    paths = pull_score_risk_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                elif is_rolling_validation(spec):
                    paths = pull_rolling_validation_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                elif is_gap_attribution(spec):
                    paths = pull_gap_attribution_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                elif is_capacity_acceptance(spec):
                    paths = pull_capacity_acceptance_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                elif is_capacity_audit(spec):
                    paths = pull_capacity_audit_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                elif spec.kind == "strategy_acceptance":
                    paths = pull_strategy_acceptance_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                elif is_exposure_audit(spec):
                    paths = pull_exposure_audit_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                elif is_feature_hygiene(spec):
                    paths = pull_feature_hygiene_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                else:
                    paths = pull_feature_audit_artifacts(
                        args.hfcli,
                        spec,
                        pod_name,
                        artifacts_root,
                    )
                print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
        if fetch_next_close_labels_flag:
            print("pulled_next_close_labels:")
            for spec in specs:
                paths = pull_next_close_labels(
                    args.hfcli,
                    spec,
                    pod_name,
                    next_close_labels_root,
                    label_pvc_dir=args.next_close_label_pvc_dir,
                )
                print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
    finally:
        if created_temp_pod:
            delete_temp_pod(args.hfcli, specs[0].namespace, pod_name)

    if record_flag:
        print("recorded_metrics:")
        for spec in specs:
            if spec.kind == "pool_internal_analysis" and not args.metrics:
                print(f"  {spec.run_id}: no metrics to record for {spec.kind}")
                continue
            paths = record_metrics(spec.run_id, metrics_dir, records_dir)
            status = statuses.get(spec.run_id, "")
            if paths:
                print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
                if status and status != "completed":
                    print(
                        f"  {spec.run_id}: config status is {status!r}; confirm before final archive"
                    )
            elif is_non_standard_artifact_run(spec) and not args.metrics:
                print(f"  {spec.run_id}: no metrics to record for {spec.kind}")
            else:
                print(f"  {spec.run_id}: missing {metrics_dir / f'{spec.run_id}{METRICS_SUFFIX}'}")
        print("recorded_artifacts:")
        for spec in specs:
            paths = record_lightweight_artifacts(
                spec,
                artifacts_root,
                records_dir,
            )
            if paths:
                print(f"  {spec.run_id}: {', '.join(str(path) for path in paths)}")
            elif is_pool_internal_analysis(spec):
                print(f"  {spec.run_id}: no pool-internal analysis artifacts to record")
            elif is_non_standard_artifact_run(spec):
                print(f"  {spec.run_id}: no lightweight artifacts to record")


if __name__ == "__main__":
    main()
