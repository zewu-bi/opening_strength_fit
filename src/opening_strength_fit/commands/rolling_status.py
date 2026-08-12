from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from opening_strength_fit.analysis import month_window_periods
from opening_strength_fit.config import load_toml, run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show K8s Indexed Job status mapped to rolling test months."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-name", default="")
    parser.add_argument("--jobs-dir", default="experiments/jobs")
    parser.add_argument("--namespace", default="bizewu")
    parser.add_argument("--cluster", default="research")
    parser.add_argument("--hfcli", default="hfcli")
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Also print this many log lines for each pod.",
    )
    return parser.parse_args()


def rendered_job_name(config_path: Path, jobs_dir: Path, config: dict) -> str:
    run_id_value = run_id(config, config_path)
    path = jobs_dir / f"{run_id_value}_sharded_job.yaml"
    if not path.exists():
        raise SystemExit(f"rendered sharded job does not exist: {path}")
    in_metadata = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "metadata:":
            in_metadata = True
            continue
        if in_metadata and stripped.startswith("name:"):
            return stripped.split(":", 1)[1].strip()
        if in_metadata and stripped and not line.startswith(" "):
            break
    raise SystemExit(f"could not find metadata.name in {path}")


def run_json(command: list[str]) -> dict:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def container_state(item: dict) -> tuple[str, str, str]:
    statuses = item.get("status", {}).get("containerStatuses") or []
    if not statuses:
        return "", "", ""
    state = statuses[0].get("state") or {}
    for name in ("running", "terminated", "waiting"):
        if name in state:
            detail = state[name]
            reason = detail.get("reason", "")
            finished = detail.get("finishedAt", "")
            return name, reason, finished
    return "", "", ""


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_toml(config_path)
    window = config["window"]
    test_months = int(window.get("test_months", 1) or 1)
    windows = month_window_periods(
        str(window["test_start_month"]),
        str(window["test_end_month"]),
        test_months=test_months,
        stride_months=int(window.get("test_stride_months", test_months) or 1),
    )
    month_labels = [start if start == end else f"{start}..{end}" for start, end in windows]
    job_name = args.job_name or rendered_job_name(config_path, Path(args.jobs_dir), config)

    command = [
        args.hfcli,
        "kubectl",
        "--cluster",
        args.cluster,
        "get",
        "pods",
        "-n",
        args.namespace,
        "-l",
        f"job-name={job_name}",
        "-o",
        "json",
    ]
    payload = run_json(command)
    items = payload.get("items", [])
    print(f"job: {job_name}")
    print(f"namespace: {args.namespace}")
    print(f"months: {month_labels[0]}..{month_labels[-1]} ({len(month_labels)})")
    if not items:
        print("pods: none found")
        print(
            "apply command: "
            f"hfcli kubectl --cluster {args.cluster} apply -f "
            f"{Path(args.jobs_dir) / (run_id(config, config_path) + '_sharded_job.yaml')}"
        )
        return

    rows = []
    for item in sorted(items, key=lambda value: value["metadata"]["name"]):
        metadata = item.get("metadata", {})
        labels = metadata.get("labels") or {}
        index_raw = labels.get("batch.kubernetes.io/job-completion-index", "")
        month = ""
        if index_raw != "":
            index = int(index_raw)
            month = month_labels[index] if 0 <= index < len(month_labels) else ""
        state, reason, finished = container_state(item)
        rows.append(
            {
                "index": index_raw,
                "month": month,
                "pod": metadata.get("name", ""),
                "phase": item.get("status", {}).get("phase", ""),
                "state": state,
                "reason": reason,
                "finished": finished,
            }
        )

    widths = {
        key: max(len(str(row[key])) for row in rows + [{key: key}])
        for key in ("index", "month", "pod", "phase", "state", "reason", "finished")
    }
    header = "  ".join(key.ljust(widths[key]) for key in widths)
    print(header)
    print("  ".join("-" * widths[key] for key in widths))
    for row in rows:
        print("  ".join(str(row[key]).ljust(widths[key]) for key in widths))

    print("\nlog_commands:")
    for row in rows:
        if row["pod"]:
            print(
                f"  {row['month']}: hfcli kubectl --cluster {args.cluster} logs "
                f"-n {args.namespace} {row['pod']} --tail=160"
            )

    if args.tail > 0:
        for row in rows:
            if not row["pod"]:
                continue
            print(f"\nlogs[{row['month']}:{row['pod']}]:")
            print(
                run_text(
                    [
                        args.hfcli,
                        "kubectl",
                        "--cluster",
                        args.cluster,
                        "logs",
                        "-n",
                        args.namespace,
                        row["pod"],
                        f"--tail={args.tail}",
                    ]
                ).rstrip()
            )


if __name__ == "__main__":
    main()
