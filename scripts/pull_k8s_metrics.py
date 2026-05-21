import argparse
from pathlib import Path
import subprocess

import _bootstrap  # noqa: F401
from opening_strength_fit.k8s import DEFAULT_IMAGE, RunSpec
from opening_strength_fit.k8s import delete_temp_pod, ensure_temp_pod, load_run_spec
from opening_strength_fit.k8s import manifest_job_name, run_command


DEFAULT_RUNS = (
    (
        "gbm_opening_1y_next_month",
        "/mnt/output/opening_strength_fit/gbm_opening_1y_next_month",
    ),
)


def parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be formatted as label=/pvc/path")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.rstrip("/")
    if not label or not path:
        raise argparse.ArgumentTypeError("--run needs a non-empty label and path")
    return label, path


def parse_config(path: str) -> RunSpec:
    return load_run_spec(path)


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
        combine_manifest=None,
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
                "All runs fetched in one call must share namespace, pvc, mount_path, "
                "image_pull_secret, and image."
            )


def run_combine_job(hfcli: str, spec: RunSpec, timeout: str, dry_run: bool) -> None:
    if not spec.combine_manifest:
        return
    job_name = manifest_job_name(spec.combine_manifest)
    delete_command = [
        hfcli,
        "kubectl",
        "delete",
        "job",
        job_name,
        "-n",
        spec.namespace,
        "--ignore-not-found",
    ]
    apply_command = [hfcli, "kubectl", "apply", "-f", spec.combine_manifest]
    wait_command = [
        hfcli,
        "kubectl",
        "wait",
        "--for=condition=complete",
        f"job/{job_name}",
        "-n",
        spec.namespace,
        f"--timeout={timeout}",
    ]
    print("combine_job:")
    print(f"  label: {spec.run_id}")
    print(f"  manifest: {spec.combine_manifest}")
    if dry_run:
        print(f"  delete: {' '.join(delete_command)}")
        print(f"  apply: {' '.join(apply_command)}")
        print(f"  wait: {' '.join(wait_command)}")
        return
    run_command(delete_command)
    run_command(apply_command)
    run_command(wait_command)


def pull_metrics(
    hfcli: str,
    namespace: str,
    pod: str,
    label: str,
    pvc_dir: str,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled = []
    for name in ("metrics_by_year.csv", "metrics_by_month.csv"):
        output_path = output_dir / f"{label}_{name}"
        command = [
            hfcli,
            "kubectl",
            "exec",
            "-n",
            namespace,
            pod,
            "--",
            "sh",
            "-lc",
            f"test -f {pvc_dir}/{name} && cat {pvc_dir}/{name}",
        ]
        try:
            with output_path.open("wb") as file:
                run_command(command, stdout=file)
            pulled.append(output_path)
        except subprocess.CalledProcessError:
            if output_path.exists():
                output_path.unlink()
            if name == "metrics_by_year.csv":
                raise
    return pulled


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch structured opening-strength metrics CSVs from the k8s output PVC. "
            "For sharded runs, automatically runs the shard-combining reader job first."
        )
    )
    parser.add_argument(
        "--pod",
        default="",
        help="Optional existing pod with the output PVC mounted. If omitted, create a temporary pull pod.",
    )
    parser.add_argument("--namespace", default="bizewu")
    parser.add_argument("--hfcli", default="hfcli")
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        help="Metrics source formatted as label=/mnt/output/opening_strength_fit/run_dir.",
    )
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config,
        help="Run config to fetch, using run.id and output.k8s_dir.",
    )
    parser.add_argument("--output-dir", default="output/k8s/metrics")
    parser.add_argument(
        "--timeout",
        default="300s",
        help="kubectl wait timeout for combine jobs and temporary pull pod.",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip auto-running sharded combine jobs even if a sharded reader manifest exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the combine/fetch plan without touching the cluster.",
    )
    args = parser.parse_args()

    runs: list[RunSpec] = []
    if args.config:
        runs.extend(args.config)
    if args.run:
        runs.extend(
            ad_hoc_run_spec(label, pvc_dir, args.namespace) for label, pvc_dir in args.run
        )
    if not runs:
        runs = [
            ad_hoc_run_spec(label, pvc_dir, args.namespace)
            for label, pvc_dir in DEFAULT_RUNS
        ]
    validate_specs(runs)

    if not args.skip_combine:
        for spec in runs:
            run_combine_job(args.hfcli, spec, args.timeout, args.dry_run)

    pod_name = args.pod
    created_temp_pod = False
    if not pod_name and runs:
        pod_name = ensure_temp_pod(
            args.hfcli,
            runs[0],
            args.timeout,
            "opening-strength-fetch-metrics",
            dry_run=args.dry_run,
        )
        created_temp_pod = not args.dry_run

    output_dir = Path(args.output_dir)
    if args.dry_run:
        print("fetch_plan:")
        print(f"  pod: {pod_name or '<provided-at-runtime>'}")
        for spec in runs:
            output_path = output_dir / f"{spec.run_id}_metrics_by_year.csv"
            print(f"  {spec.run_id}: {spec.pvc_dir}/metrics_by_year.csv -> {output_path}")
            month_path = output_dir / f"{spec.run_id}_metrics_by_month.csv"
            print(f"  {spec.run_id}: {spec.pvc_dir}/metrics_by_month.csv -> {month_path} (if present)")
        return

    try:
        print("pulled_metrics:")
        for spec in runs:
            output_paths = pull_metrics(
                args.hfcli,
                spec.namespace,
                pod_name,
                spec.run_id,
                spec.pvc_dir,
                output_dir,
            )
            print(f"  {spec.run_id}: {', '.join(str(path) for path in output_paths)}")
    finally:
        if created_temp_pod:
            delete_temp_pod(args.hfcli, runs[0].namespace, pod_name)


if __name__ == "__main__":
    main()
