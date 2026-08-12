from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path

from opening_strength_fit.config import load_toml
from opening_strength_fit.k8s import (
    render_config_for_mode,
    rendered_job_image,
    rendered_job_specs,
)

ROOT = Path(__file__).resolve().parents[3]
INCUBATOR_MANIFEST = "experiments/incubator.toml"
MAX_COMMAND_MODULE_LINES = 800
MAX_EVIDENCE_FILE_BYTES = 1_000_000
CANONICAL_EVIDENCE_RUN_ID = (
    "nn_delay6_clock_state_36m_2022_2025_auction_pruned_multi_denominator_"
    "grouped_gated_v2_mech_v3_gelu_mse_v1"
)
CANONICAL_EVIDENCE_DIR = f"experiments/evidence/backtests/{CANONICAL_EVIDENCE_RUN_ID}"
REQUIRED_REPRODUCIBILITY_FILES = tuple(
    ".env.example Dockerfile Dockerfile.overlay Makefile README.md examples/smoke/labeled.csv "
    "examples/smoke/ridge.toml experiments/scripts/build_four_figure_evidence.py pyproject.toml "
    "requirements.lock docs/experiment_log.md docs/project_brief.md docs/runbook.md".split()
) + tuple(
    f"{CANONICAL_EVIDENCE_DIR}/{name}"
    for name in (
        "01_signal_acceptance.svg 01_signal_acceptance.csv 02_top100_cumulative.svg "
        "02_top100_cumulative.csv 03_top1000_bucket_curve.svg 03_top1000_bucket_curve.csv "
        "04_top1000_return_distribution.svg 04_top1000_return_distribution.csv manifest.json "
        "trace_optimization.json trace_top1000_bucket.json trace_top1000_distribution.json"
    ).split()
)
ALLOWED_MARKDOWN_FILES = {path for path in REQUIRED_REPRODUCIBILITY_FILES if path.endswith(".md")}
REQUIRED_DIRS = tuple(
    "src/opening_strength_fit src/opening_strength_fit/commands examples/smoke experiments/runs "
    "experiments/jobs experiments/evidence experiments/results docs tests".split()
)
LOCAL_ARCHIVE_DIRS = {"experiments/results"}
SKIP_PARTS = set(".git .mypy_cache .pytest_cache .ruff_cache .venv __pycache__ output".split())
HEAVY_RUNTIME_DEPENDENCIES = set("boto3 clickhouse-connect lightgbm matplotlib requests".split())
MATRIX_REQUIRED_FIELDS = ("name", "window", "horizon", "feature_path", "label_path")
BINARY_SUFFIXES = set(".parquet .pkl .pickle .pt .pth .joblib".split())


def _git_files(*arguments: str) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return sorted(filter(None, result.stdout.splitlines()))


def project_files() -> list[str]:
    files = _git_files("--cached", "--others", "--exclude-standard")
    if files is not None:
        return [path for path in files if (ROOT / path).exists()]
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and not set(path.relative_to(ROOT).parts) & SKIP_PARTS
    )


def tracked_files() -> list[str]:
    return _git_files() or []


def untracked_project_files() -> set[str]:
    return set(_git_files("--others", "--exclude-standard") or [])


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


PYPROJECT = tomllib.loads(read("pyproject.toml"))
ENTRYPOINTS = dict(PYPROJECT.get("project", {}).get("scripts", {}))
K8S_JOB_ENTRYPOINTS = (*ENTRYPOINTS, "run_top1000_rank_bucket_diagnostics.py")


def check_matrix_cases(path: str, config: dict[str, object], errors: list[str]) -> None:
    matrix = config.get("matrix")
    if matrix is None:
        return
    if not isinstance(matrix, dict):
        errors.append(f"{path}: matrix must be a table")
        return
    cases = matrix.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append(f"{path}: matrix.cases must be a non-empty array of tables")
        return

    seen: set[str] = set()
    for index, case in enumerate(cases, start=1):
        location = f"{path}: matrix case {index}"
        if not isinstance(case, dict):
            errors.append(f"{location} must be a table")
            continue
        missing = [field for field in MATRIX_REQUIRED_FIELDS if not str(case.get(field, ""))]
        if missing:
            errors.append(f"{location} missing fields {', '.join(missing)}")
            continue
        name, window, horizon = (str(case[field]) for field in ("name", "window", "horizon"))
        if name in seen:
            errors.append(f"{path}: duplicate matrix case name {name!r}")
        seen.add(name)
        if not re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2}", window):
            errors.append(f"{location} has invalid window={window!r}")
            continue
        window_slug = window.replace(":", "").replace("-", "_")
        expected_name = f"w{window_slug}_h{horizon}"
        if name != expected_name:
            errors.append(f"{location} name must be {expected_name!r}")
        for field, expected, scope in (
            ("feature_path", f"opening_{window_slug}_features_", f"window {window!r}"),
            ("label_path", f"opening_{window_slug}_labels_h{horizon}", "window/horizon"),
        ):
            value = str(case[field])
            if not value.startswith("/") or expected not in value:
                errors.append(f"{location} {field} does not match {scope}")


def check_run_configs(configs: dict[str, dict], errors: list[str]) -> None:
    valid_modes = {"global", "daily", "symbol_day", "cross_section"}
    for path, config in configs.items():
        if Path(path).stem != str(config.get("run", {}).get("id", Path(path).stem)):
            errors.append(f"{path}: run.id must match filename")
        for field, default in (("selection_mode", "symbol_day"), ("ic_mode", "daily")):
            value = str(config.get("evaluation", {}).get(field, default))
            if value not in valid_modes:
                errors.append(f"{path}: unknown evaluation.{field}={value!r}")
        window_mode = str(config.get("window", {}).get("mode", "chronological"))
        if window_mode not in {"chronological", "rolling_annual", "rolling_monthly"}:
            errors.append(f"{path}: unknown window.mode={window_mode!r}")
        try:
            rendered_job_specs(config)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
        check_matrix_cases(path, config, errors)


def check_rendered_jobs(configs: dict[str, dict], tracked: set[str], errors: list[str]) -> None:
    from opening_strength_fit.commands.k8s_rendering import JOB_RENDERERS

    for path, config in configs.items():
        config_path = Path(path)
        try:
            specs = rendered_job_specs(config)
        except ValueError:
            continue
        run_id = str(config.get("run", {}).get("id", config_path.stem))
        for spec in specs:
            image = rendered_job_image(config, spec.mode)
            if not image:
                errors.append(f"{path}: rendered {spec.mode} job requires helper_image")
                continue
            manifest_path = f"experiments/jobs/{run_id}{spec.suffix}"
            if manifest_path in tracked and (ROOT / manifest_path).exists():
                errors.append(f"{manifest_path}: rendered manifest must not be tracked")
            try:
                render_config = render_config_for_mode(config, spec.mode)
                manifest = (
                    JOB_RENDERERS[spec.mode](config_path, render_config, image).rstrip() + "\n"
                )
            except (KeyError, TypeError, ValueError, SystemExit) as exc:
                errors.append(f"{path}: cannot render {spec.mode} job ({exc})")
                continue
            if (digest := hashlib.sha256(manifest.encode("utf-8")).hexdigest()) != spec.sha256:
                errors.append(
                    f"{path}: rendered {spec.mode} job changed "
                    f"(expected {spec.sha256}, got {digest})"
                )


def check_project_scripts(files: list[str], errors: list[str]) -> None:
    for command, target in sorted(ENTRYPOINTS.items()):
        if not command.startswith("osf-"):
            errors.append(f"pyproject.toml: command {command!r} must use osf- prefix")
        if not target.startswith("opening_strength_fit.commands.") or not target.endswith(":main"):
            errors.append(f"pyproject.toml: invalid command target {target!r}")
            continue
        module_path = (
            "src/opening_strength_fit/"
            + target.removeprefix("opening_strength_fit.").removesuffix(":main").replace(".", "/")
            + ".py"
        )
        if module_path not in files:
            errors.append(f"pyproject.toml: command {command!r} target {module_path} is missing")


def _dependency_name(requirement: object) -> str:
    return re.split(r"[ <>=!~\[]", str(requirement), maxsplit=1)[0].lower()


def check_dependency_profiles(errors: list[str]) -> None:
    project = PYPROJECT.get("project", {})
    base = {_dependency_name(item) for item in project.get("dependencies", [])}
    misplaced = sorted(base & HEAVY_RUNTIME_DEPENDENCIES)
    if misplaced:
        errors.append("pyproject.toml: heavy dependencies in core: " + ", ".join(misplaced))
    optional = project.get("optional-dependencies", {})
    for profile in ("dev", "cluster"):
        dependencies = {_dependency_name(item) for item in optional.get(profile, [])}
        if missing := sorted(HEAVY_RUNTIME_DEPENDENCIES - dependencies):
            errors.append(f"pyproject.toml: {profile} profile is missing: " + ", ".join(missing))
    locked = {
        _dependency_name(line)
        for line in read("requirements.lock").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for profile, requirements in {"core": project.get("dependencies", []), **optional}.items():
        if missing := sorted({_dependency_name(item) for item in requirements} - locked):
            errors.append(
                f"requirements.lock: {profile} dependencies are not pinned: " + ", ".join(missing)
            )


def check_docker_contract(errors: list[str]) -> None:
    for path in ("Dockerfile", "Dockerfile.overlay"):
        text = read(path)
        if re.search(r"^COPY\s+\.\s+\.\s*$", text, flags=re.MULTILINE):
            errors.append(f"{path}: broad COPY . . is forbidden")
        if "org.opencontainers.image.revision" not in text:
            errors.append(f"{path}: source revision OCI label is missing")
    if "ARG DEPENDENCY_PROFILE=cluster" not in read("Dockerfile"):
        errors.append("Dockerfile: dependency profile build argument is missing")


def check_required_assets(files: list[str], errors: list[str]) -> None:
    for directory in REQUIRED_DIRS:
        root = ROOT / directory
        has_local = root.is_dir() and any(
            path.is_file() and not set(path.relative_to(ROOT).parts) & SKIP_PARTS
            for path in root.rglob("*")
        )
        if not any(path.startswith(f"{directory}/") for path in files) and not (
            directory in LOCAL_ARCHIVE_DIRS or has_local
        ):
            errors.append(f"{directory}: no files found")
    for path in REQUIRED_REPRODUCIBILITY_FILES:
        if path not in files:
            errors.append(f"{path}: required reproducibility file is missing")


def check_tracked_files(files: list[str], errors: list[str]) -> None:
    for path in files:
        candidate = Path(path)
        if path.startswith(("output/", "experiments/results/")) or candidate.name in {
            ".env",
            ".coverage",
        }:
            errors.append(f"{path}: generated or private file must not be tracked")
        if candidate.suffix.lower() in BINARY_SUFFIXES:
            errors.append(f"{path}: data or model binary must not be tracked")


def check_source_layout(files: list[str], errors: list[str]) -> None:
    for path in files:
        if path.endswith(".md") and path not in ALLOWED_MARKDOWN_FILES:
            errors.append(f"{path}: duplicate project documentation is forbidden")
        if path.startswith("scripts/"):
            errors.append(f"{path}: legacy scripts/ tree is not part of the package layout")
        if path.startswith("src/opening_strength_fit/commands/") and path.endswith(".py"):
            if len(read(path).splitlines()) > MAX_COMMAND_MODULE_LINES:
                errors.append(f"{path}: command module exceeds {MAX_COMMAND_MODULE_LINES} lines")


def check_evidence(files: list[str], errors: list[str]) -> None:
    prefix = "experiments/evidence/backtests/"
    for path in files:
        if not path.startswith("experiments/evidence/"):
            continue
        evidence_path = ROOT / path
        if evidence_path.suffix.lower() in BINARY_SUFFIXES:
            errors.append(f"{path}: row-level data or model binary does not belong in evidence")
        if evidence_path.stat().st_size > MAX_EVIDENCE_FILE_BYTES:
            errors.append(f"{path}: evidence file exceeds {MAX_EVIDENCE_FILE_BYTES} bytes")
        run_id, separator, _ = (
            path.removeprefix(prefix) if path.startswith(prefix) else ""
        ).partition("/")
        if separator and f"experiments/runs/{run_id}.toml" not in files:
            errors.append(f"{path}: evidence has no matching run config")


def load_incubator_assets(files: list[str], errors: list[str]) -> set[str]:
    if INCUBATOR_MANIFEST not in files:
        return set()
    try:
        entries = tomllib.loads(read(INCUBATOR_MANIFEST)).get("campaign", {}).get("entries", [])
    except tomllib.TOMLDecodeError as exc:
        errors.append(f"{INCUBATOR_MANIFEST}: invalid TOML: {exc}")
        return set()
    assets: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        location = f"{INCUBATOR_MANIFEST}: campaign entry {index}"
        for field in ("id", "owner", "review_by", "purpose", "promote_when"):
            if not str(entry.get(field, "")).strip():
                errors.append(f"{location} must declare {field}")
        try:
            review_by = date.fromisoformat(str(entry.get("review_by", "")))
            if review_by < date.today():
                errors.append(f"{location} review expired on {review_by}")
        except ValueError:
            errors.append(f"{location} has invalid review_by")
        for raw_path in entry.get("assets", []):
            path = str(raw_path)
            if path not in files:
                errors.append(f"{location} asset does not exist: {path}")
            assets.add(path)
    return assets


def check_incubator_coverage(incubator_assets: set[str], errors: list[str]) -> None:
    prefixes = ("experiments/jobs/support/", "experiments/scripts/")
    errors.extend(
        f"{path}: untracked experiment asset is missing from {INCUBATOR_MANIFEST}"
        for path in sorted(untracked_project_files() - incubator_assets)
        if path.startswith(prefixes)
    )


def check_k8s_jobs(files: list[str], errors: list[str], incubator_assets: set[str]) -> None:
    for job in files:
        if (
            not job.startswith("experiments/jobs/")
            or not job.endswith("_job.yaml")
            or job in incubator_assets
        ):
            continue
        text = read(job)
        if "kind: Job" in text and "ttlSecondsAfterFinished:" not in text:
            errors.append(f"{job}: completed Job cleanup TTL is missing")
        if "site-packages/" in text:
            errors.append(
                f"{job}: formal k8s job must use packaged image code, not site-packages overlays"
            )
        if "scripts/" in text:
            errors.append(f"{job}: k8s job uses legacy scripts/ entrypoint")
        if not any(entrypoint in text for entrypoint in K8S_JOB_ENTRYPOINTS):
            errors.append(f"{job}: k8s job does not use a declared project entrypoint")


def collect_errors() -> list[str]:
    files = project_files()
    tracked = tracked_files()
    configs = {
        path: load_toml(ROOT / path)
        for path in files
        if path.startswith("experiments/runs/") and path.endswith(".toml")
    }
    errors: list[str] = []
    check_required_assets(files, errors)
    check_tracked_files(tracked, errors)
    check_source_layout(files, errors)
    check_project_scripts(files, errors)
    check_dependency_profiles(errors)
    check_docker_contract(errors)
    check_evidence(files, errors)
    check_run_configs(configs, errors)
    check_rendered_jobs(configs, set(tracked), errors)
    incubator_assets = load_incubator_assets(files, errors)
    check_incubator_coverage(incubator_assets, errors)
    check_k8s_jobs(files, errors, incubator_assets)
    return errors


def main() -> None:
    files = project_files()
    errors = collect_errors()
    source_modules = [path for path in files if path.endswith(".py")]
    command_modules = [
        path
        for path in source_modules
        if "/commands/" in path and not path.endswith("/__init__.py")
    ]
    library_modules = [
        path
        for path in source_modules
        if path.startswith("src/opening_strength_fit/") and "/commands/" not in path
    ]

    print("project_contracts:")
    print(f"  command_modules: {len(command_modules)}")
    print(f"  library_modules: {len(library_modules)}")
    print(f"  required_dirs: {len(REQUIRED_DIRS)}")
    print(f"  incubator_manifest: {INCUBATOR_MANIFEST if INCUBATOR_MANIFEST in files else 'none'}")
    print(f"  k8s_job_entrypoints: {', '.join(K8S_JOB_ENTRYPOINTS)}")

    if errors:
        print("\ncontract_errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("  contracts_ok: yes")


if __name__ == "__main__":
    main()
