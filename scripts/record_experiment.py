from __future__ import annotations

import argparse
from dataclasses import dataclass
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from opening_strength_fit.config import load_toml, run_id


METRICS_SUFFIX = "_metrics_by_year.csv"


@dataclass(frozen=True)
class RunStatus:
    status: str
    config_path: Path


def run_id_from_config(path: Path) -> str:
    return run_id(load_toml(path), path)


def collect_run_statuses(runs_dir: Path) -> dict[str, RunStatus]:
    statuses = {}
    for path in sorted(runs_dir.glob("*.toml")):
        config = load_toml(path)
        run_id_value = run_id(config, path)
        status = str(config.get("run", {}).get("status", "completed"))
        statuses[run_id_value] = RunStatus(status=status, config_path=path)
    return statuses


def discover_run_ids(metrics_source: Path) -> list[str]:
    run_ids = {
        path.name.removesuffix(METRICS_SUFFIX)
        for path in metrics_source.glob(f"*{METRICS_SUFFIX}")
    }
    return sorted(run_ids)


def copy_record(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def record_run(
    run_id_value: str,
    *,
    metrics_source: Path,
    records_dir: Path,
) -> tuple[list[Path], list[str]]:
    written: list[Path] = []
    warnings: list[str] = []

    metrics_path = metrics_source / f"{run_id_value}{METRICS_SUFFIX}"
    metrics_record = records_dir / "metrics" / metrics_path.name
    if copied := copy_record(metrics_path, metrics_record):
        written.append(copied)
    else:
        warnings.append(f"{run_id_value}: metrics not found at {metrics_path}")

    return written, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy lightweight metrics from ignored output/ folders into tracked "
            "experiment records."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="Run config to record. May be repeated.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Run id to record. May be repeated.",
    )
    parser.add_argument("--metrics-source", default="output/k8s/metrics")
    parser.add_argument("--records-dir", default="experiments/results")
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if requested records are missing or status hints are emitted.",
    )
    args = parser.parse_args()

    metrics_source = Path(args.metrics_source)
    records_dir = Path(args.records_dir)
    run_statuses = collect_run_statuses(Path(args.runs_dir))

    run_ids = list(args.run_id)
    if args.config:
        run_ids.extend(run_id_from_config(path) for path in args.config)
    if not run_ids:
        run_ids = discover_run_ids(metrics_source)
    run_ids = sorted(dict.fromkeys(run_ids))

    if not run_ids:
        raise SystemExit("no run ids found to record")

    all_warnings: list[str] = []
    print("recorded_experiments:")
    for run_id_value in run_ids:
        written, warnings = record_run(
            run_id_value,
            metrics_source=metrics_source,
            records_dir=records_dir,
        )
        all_warnings.extend(warnings)
        metrics_copied = any(path.name.endswith(METRICS_SUFFIX) for path in written)
        status_record = run_statuses.get(run_id_value)
        if status_record is not None:
            if metrics_copied and status_record.status != "completed":
                all_warnings.append(
                    f"{run_id_value}: metrics were archived but "
                    f"{status_record.config_path} has status={status_record.status!r}; "
                    "update it to completed after confirming results"
                )
            if not metrics_copied and status_record.status == "completed":
                all_warnings.append(
                    f"{run_id_value}: config status is completed but no metrics were archived"
                )
        print(f"  {run_id_value}:")
        if written:
            for path in written:
                print(f"    - {path}")
        else:
            print("    - no records written")

    if all_warnings:
        print("\nrecord_warnings:")
        for warning in all_warnings:
            print(f"  - {warning}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
