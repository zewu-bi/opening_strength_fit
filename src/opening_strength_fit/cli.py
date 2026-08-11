from __future__ import annotations

import sys
from importlib.metadata import EntryPoint, entry_points

COMMAND_ALIASES = {
    ("train",): "osf-train",
    ("data", "build-raw"): "osf-build-raw-source-cache",
    ("data", "build-long-raw"): "osf-build-long-label-raw-source",
    ("data", "build-dataset"): "osf-build-training-datasets",
    ("data", "build-short-labels"): "osf-build-short-labels",
    ("data", "build-long-labels"): "osf-build-long-horizon-labels",
    ("data", "build-next-close-labels"): "osf-build-next-close-labels",
    ("data", "build-target-labels"): "osf-build-target-label-cache",
    ("data", "split-labels"): "osf-split-horizon-labels",
    ("data", "split-long-labels"): "osf-split-long-horizon-labels",
    ("experiment", "run"): "osf-run-experiment",
    ("experiment", "audit"): "osf-audit-experiments",
    ("experiment", "compare"): "osf-compare-opening-results",
    ("experiment", "render-job"): "osf-render-k8s-job",
    ("experiment", "status"): "osf-rolling-job-status",
    ("experiment", "sync"): "osf-sync-experiment-artifacts",
    ("audit", "capacity"): "osf-audit-capacity",
    ("audit", "exposure"): "osf-audit-exposure",
    ("audit", "feature-dependence"): "osf-audit-feature-dependence",
    ("audit", "feature-hygiene"): "osf-audit-feature-hygiene",
    ("audit", "strategy"): "osf-audit-strategy-acceptance",
    ("audit", "storage"): "osf-audit-storage",
    ("audit", "contracts"): "osf-check-project-contracts",
    ("analyze", "capacity"): "osf-analyze-capacity-acceptance",
    ("analyze", "pool"): "osf-analyze-pool-internal-top100",
    ("analyze", "realistic"): "osf-analyze-realistic-acceptance",
}


def _console_entrypoints() -> dict[str, EntryPoint]:
    return {
        entrypoint.name: entrypoint
        for entrypoint in entry_points(group="console_scripts")
        if entrypoint.name.startswith("osf-")
    }


def _resolve_command(argv: list[str]) -> tuple[str, list[str]]:
    if len(argv) >= 2 and tuple(argv[:2]) in COMMAND_ALIASES:
        return COMMAND_ALIASES[tuple(argv[:2])], argv[2:]
    if argv and (argv[0],) in COMMAND_ALIASES:
        return COMMAND_ALIASES[(argv[0],)], argv[1:]
    if argv:
        return f"osf-{argv[0]}", argv[1:]
    return "", []


def _help(entrypoints: dict[str, EntryPoint]) -> str:
    grouped: dict[str, list[str]] = {}
    for alias in COMMAND_ALIASES:
        group = alias[0] if len(alias) > 1 else "core"
        grouped.setdefault(group, []).append(alias[-1])
    lines = [
        "usage: osf <group> <command> [options]",
        "       osf <legacy-command-suffix> [options]",
        "",
        "common commands:",
    ]
    for group, commands in sorted(grouped.items()):
        lines.append(f"  {group:10} {', '.join(sorted(commands))}")
    suffixes = sorted(name.removeprefix("osf-") for name in entrypoints)
    lines.extend(("", "all compatibility command suffixes:", f"  {', '.join(suffixes)}"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = _console_entrypoints()
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_help(commands))
        return

    command_name, command_args = _resolve_command(arguments)
    entrypoint = commands.get(command_name)
    if entrypoint is None:
        raise SystemExit(f"unknown osf command: {' '.join(arguments[:2])}\n\n{_help(commands)}")

    previous_argv = sys.argv
    sys.argv = [command_name, *command_args]
    try:
        entrypoint.load()()
    finally:
        sys.argv = previous_argv
