from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

from opening_strength_fit.config import load_toml

ROOT = Path(__file__).resolve().parents[3]
MAX_COMMAND_MODULE_LINES = 800
K8S_JOB_ENTRYPOINTS = (
    "osf-audit-exposure",
    "osf-audit-feature-dependence",
    "osf-audit-feature-hygiene",
    "osf-analyze-pool-internal-top100",
    "osf-build-exposure-input",
    "osf-build-labeled-cache",
    "osf-build-next-close-labels",
    "osf-build-target-label-cache",
    "osf-run-alpha-conditioned-rolling-validation",
    "osf-run-gap-risk-attribution",
    "osf-run-learned-risk-layer",
    "osf-run-score-risk-sweep",
    "osf-train",
)
REQUIRED_DIRS = (
    "src/opening_strength_fit",
    "src/opening_strength_fit/cli",
    "src/opening_strength_fit/commands",
    "experiments/runs",
    "experiments/jobs",
    "experiments/results",
    "docs",
    "tests",
)
LOCAL_ARCHIVE_DIRS = {"experiments/results"}
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "output",
}


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


def _module_files(files: list[str], package_dir: str) -> list[str]:
    return [
        path
        for path in files
        if path.startswith(f"{package_dir}/")
        and path.endswith(".py")
        and not path.endswith("/__init__.py")
    ]


def check_project_scripts(files: list[str], errors: list[str]) -> None:
    pyproject = tomllib.loads(read("pyproject.toml"))
    entrypoints = pyproject.get("project", {}).get("scripts", {})
    command_targets = set(entrypoints.values())
    cli_modules = _module_files(files, "src/opening_strength_fit/cli")

    for module in cli_modules:
        module_name = Path(module).stem
        target = f"opening_strength_fit.cli.{module_name}:main"
        text = read(module)
        if target not in command_targets:
            errors.append(f"{module}: missing [project.scripts] entrypoint")
        line_count = len(text.splitlines())
        if line_count > 12:
            errors.append(f"{module}: CLI wrapper should stay thin; found {line_count} lines")
        import_lines = [
            line
            for line in text.splitlines()
            if line.startswith("from opening_strength_fit.") and line.endswith(" import main")
        ]
        if len(import_lines) != 1:
            errors.append(f"{module}: must import exactly one command module main")
            continue
        imported = import_lines[0].removeprefix("from opening_strength_fit.")
        imported = imported.removesuffix(" import main")
        if not imported.startswith("commands."):
            errors.append(f"{module}: CLI wrapper must target opening_strength_fit.commands.*")
            continue
        domain_path = f"src/opening_strength_fit/{imported.replace('.', '/')}.py"
        if domain_path not in files:
            errors.append(f"{module}: imported domain module {domain_path} does not exist")

    for command, target in sorted(entrypoints.items()):
        if not command.startswith("osf-"):
            errors.append(f"pyproject.toml: command {command!r} must use osf- prefix")
        if not target.startswith("opening_strength_fit.cli.") or not target.endswith(":main"):
            errors.append(
                f"pyproject.toml: command {command!r} must target opening_strength_fit.cli.*:main"
            )


def check_required_dirs(files: list[str], errors: list[str]) -> None:
    for directory in REQUIRED_DIRS:
        if any(path.startswith(f"{directory}/") for path in files):
            continue
        if directory in LOCAL_ARCHIVE_DIRS or _has_local_files(directory):
            continue
        errors.append(f"{directory}: no files found")


def _has_local_files(directory: str) -> bool:
    root = ROOT / directory
    if not root.is_dir():
        return False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if set(rel.parts) & SKIP_PARTS:
            continue
        return True
    return False


def check_legacy_script_tree(files: list[str], errors: list[str]) -> None:
    for path in files:
        if path.startswith("scripts/"):
            errors.append(f"{path}: legacy scripts/ tree is not part of the package layout")


def check_command_module_size(files: list[str], errors: list[str]) -> None:
    for module in _module_files(files, "src/opening_strength_fit/commands"):
        line_count = len(read(module).splitlines())
        if line_count > MAX_COMMAND_MODULE_LINES:
            errors.append(
                f"{module}: command module should stay below "
                f"{MAX_COMMAND_MODULE_LINES} lines; found {line_count}"
            )


def check_k8s_jobs(files: list[str], errors: list[str]) -> None:
    for job in [path for path in files if path.startswith("experiments/jobs/")]:
        if not job.endswith("_job.yaml"):
            continue
        text = read(job)
        if "scripts/" in text:
            errors.append(f"{job}: k8s job uses legacy scripts/ entrypoint")
        if not any(entrypoint in text for entrypoint in K8S_JOB_ENTRYPOINTS):
            allowed = ", ".join(K8S_JOB_ENTRYPOINTS)
            errors.append(f"{job}: k8s job does not use one of: {allowed}")


def collect_errors() -> list[str]:
    files = project_files()
    errors: list[str] = []
    check_required_dirs(files, errors)
    check_legacy_script_tree(files, errors)
    check_project_scripts(files, errors)
    check_command_module_size(files, errors)
    check_run_configs(files, errors)
    check_k8s_jobs(files, errors)
    return errors


def main() -> None:
    files = project_files()
    errors = collect_errors()

    cli_modules = [
        path
        for path in files
        if path.startswith("src/opening_strength_fit/cli/")
        and path.endswith(".py")
        and not path.endswith("/__init__.py")
    ]
    library_modules = [
        path
        for path in files
        if path.startswith("src/opening_strength_fit/")
        and path.endswith(".py")
        and not path.startswith("src/opening_strength_fit/cli/")
        and not path.startswith("src/opening_strength_fit/commands/")
    ]
    command_modules = [
        path
        for path in files
        if path.startswith("src/opening_strength_fit/commands/")
        and path.endswith(".py")
        and not path.endswith("/__init__.py")
    ]

    print("project_contracts:")
    print(f"  cli_modules: {len(cli_modules)}")
    print(f"  command_modules: {len(command_modules)}")
    print(f"  library_modules: {len(library_modules)}")
    print(f"  required_dirs: {len(REQUIRED_DIRS)}")
    print(f"  k8s_job_entrypoints: {', '.join(K8S_JOB_ENTRYPOINTS)}")

    if errors:
        print("\ncontract_errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("  contracts_ok: yes")


if __name__ == "__main__":
    main()
