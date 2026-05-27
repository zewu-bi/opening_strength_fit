from __future__ import annotations

import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from opening_strength_fit.config import load_toml


ROOT = Path(__file__).resolve().parents[1]
DIRECT_SCRIPT_EXEMPTIONS = {"_bootstrap.py"}
K8S_JOB_ENTRYPOINTS = (
    "scripts/run_experiment.py",
    "scripts/materialize_labeled_caches.py",
    "scripts/audit_feature_dependence.py",
)
REQUIRED_DIRS = (
    "src/opening_strength_fit",
    "scripts",
    "experiments/runs",
    "experiments/jobs",
    "experiments/results",
    "docs",
)
SKIP_PARTS = {".venv", "__pycache__", "output", ".git", ".pytest_cache", ".mypy_cache"}


def project_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return sorted(
            line for line in result.stdout.splitlines() if line and (ROOT / line).exists()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        files = []
        for path in ROOT.rglob("*"):
            rel = path.relative_to(ROOT)
            if path.is_dir() or set(rel.parts) & SKIP_PARTS:
                continue
            files.append(rel.as_posix())
        return sorted(files)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def mention_count(needle: str, haystacks: dict[str, str]) -> int:
    return sum(1 for text in haystacks.values() if needle in text)


def check_run_configs(files: list[str], errors: list[str]) -> None:
    for path in files:
        if not path.startswith("experiments/runs/") or not path.endswith(".toml"):
            continue
        config = load_toml(ROOT / path)
        run_id = str(config.get("run", {}).get("id", Path(path).stem))
        if Path(path).stem != run_id:
            errors.append(f"{path}: run.id must match filename")
        selection_mode = str(config.get("evaluation", {}).get("selection_mode", "symbol_day"))
        if selection_mode not in {"global", "daily", "symbol_day", "cross_section"}:
            errors.append(f"{path}: unknown evaluation.selection_mode={selection_mode!r}")
        ic_mode = str(config.get("evaluation", {}).get("ic_mode", "daily"))
        if ic_mode not in {"global", "daily", "symbol_day", "cross_section"}:
            errors.append(f"{path}: unknown evaluation.ic_mode={ic_mode!r}")
        window_mode = str(config.get("window", {}).get("mode", "chronological"))
        if window_mode not in {"chronological", "rolling_annual", "rolling_monthly"}:
            errors.append(f"{path}: unknown window.mode={window_mode!r}")


def main() -> None:
    files = project_files()
    docs = {
        "README.md": read("README.md"),
        "docs/runbook.md": read("docs/runbook.md"),
        "docs/project_map.md": read("docs/project_map.md"),
        "docs/project_brief.md": read("docs/project_brief.md"),
        "docs/experiment_log.md": read("docs/experiment_log.md"),
        "experiments/results/README.md": read("experiments/results/README.md"),
    }
    source_text = "\n".join(
        read(path)
        for path in files
        if path.endswith((".py", ".yaml", ".toml"))
        and not path.endswith("scripts/check_workflow_coverage.py")
    )

    errors: list[str] = []
    scripts = [
        path
        for path in files
        if path.startswith("scripts/") and path.endswith(".py")
    ]
    direct_scripts = [
        path for path in scripts if Path(path).name not in DIRECT_SCRIPT_EXEMPTIONS
    ]
    for script in direct_scripts:
        name = Path(script).name
        if mention_count(name, docs) == 0:
            errors.append(f"{script}: not mentioned in docs")

    modules = [
        path
        for path in files
        if path.startswith("src/opening_strength_fit/") and path.endswith(".py")
    ]
    for module in modules:
        name = Path(module).name
        stem = Path(module).stem
        if mention_count(name, docs) == 0:
            errors.append(f"{module}: not described in docs")
        module_import = f"opening_strength_fit.{stem}"
        file_reference = name in source_text
        if stem != "__init__" and module_import not in source_text and not file_reference:
            errors.append(f"{module}: no import or reference found")

    for directory in REQUIRED_DIRS:
        if not any(path.startswith(f"{directory}/") for path in files):
            errors.append(f"{directory}: no files found")

    check_run_configs(files, errors)

    for job in [path for path in files if path.startswith("experiments/jobs/")]:
        if job.endswith("_job.yaml"):
            text = read(job)
            if not any(entrypoint in text for entrypoint in K8S_JOB_ENTRYPOINTS):
                allowed = ", ".join(K8S_JOB_ENTRYPOINTS)
                errors.append(f"{job}: k8s job does not use one of: {allowed}")

    print("workflow_coverage:")
    print(f"  direct_scripts: {len(direct_scripts)}")
    print(f"  library_modules: {len(modules)}")
    print(f"  required_dirs: {len(REQUIRED_DIRS)}")
    print(f"  k8s_job_entrypoints: {', '.join(K8S_JOB_ENTRYPOINTS)}")

    if errors:
        print("\ncoverage_errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("  coverage_ok: yes")


if __name__ == "__main__":
    main()
