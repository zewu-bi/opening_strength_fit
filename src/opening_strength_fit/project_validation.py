from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from datetime import date
from pathlib import Path

from opening_strength_fit.config import load_toml

ROOT = Path(__file__).resolve().parents[2]
INCUBATOR_MANIFEST = "experiments/incubator.toml"
MAX_COMMAND_MODULE_LINES = 700
MAX_DOMAIN_MODULE_LINES = 900
MAX_EXPERIMENT_SCRIPT_LINES = 800
MAX_EVIDENCE_FILE_BYTES = 1_000_000
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CANONICAL_EVIDENCE_RUN_ID = (
    "nn_delay6_clock_state_36m_2022_2025_auction_pruned_multi_denominator_"
    "grouped_gated_v2_mech_v3_gelu_mse_v1"
)
CANONICAL_EVIDENCE_DIR = f"experiments/evidence/backtests/{CANONICAL_EVIDENCE_RUN_ID}"
REQUIRED_REPRODUCIBILITY_FILES = (
    ".env.example",
    "Dockerfile",
    "Dockerfile.overlay",
    "Makefile",
    "README.md",
    "examples/smoke/labeled.csv",
    "examples/smoke/ridge.toml",
    "experiments/evidence/README.md",
    INCUBATOR_MANIFEST,
    f"{CANONICAL_EVIDENCE_DIR}/README.md",
    f"{CANONICAL_EVIDENCE_DIR}/01_signal_acceptance.svg",
    f"{CANONICAL_EVIDENCE_DIR}/01_signal_acceptance.csv",
    f"{CANONICAL_EVIDENCE_DIR}/02_top100_cumulative.svg",
    f"{CANONICAL_EVIDENCE_DIR}/02_top100_cumulative.csv",
    f"{CANONICAL_EVIDENCE_DIR}/03_top1000_bucket_curve.svg",
    f"{CANONICAL_EVIDENCE_DIR}/03_top1000_bucket_curve.csv",
    f"{CANONICAL_EVIDENCE_DIR}/04_top1000_return_distribution.svg",
    f"{CANONICAL_EVIDENCE_DIR}/04_top1000_return_distribution.csv",
    f"{CANONICAL_EVIDENCE_DIR}/manifest.json",
    f"{CANONICAL_EVIDENCE_DIR}/trace_optimization.json",
    f"{CANONICAL_EVIDENCE_DIR}/trace_top1000_bucket.json",
    f"{CANONICAL_EVIDENCE_DIR}/trace_top1000_distribution.json",
    "experiments/scripts/build_four_figure_evidence.py",
    "pyproject.toml",
    "requirements.lock",
    "docs/experiment_log.md",
    "docs/project_brief.md",
    "docs/runbook.md",
)
MATRIX_REQUIRED_FIELDS = ("name", "window", "horizon", "feature_path", "label_path")
HEAVY_RUNTIME_DEPENDENCIES = {
    "boto3",
    "clickhouse-connect",
    "lightgbm",
    "matplotlib",
    "requests",
}
K8S_JOB_ENTRYPOINTS = (
    "osf-analyze-capacity-acceptance",
    "osf-ask-level-attribution",
    "osf-audit-capacity",
    "osf-audit-exposure",
    "osf-audit-feature-dependence",
    "osf-audit-feature-hygiene",
    "osf-audit-strategy-acceptance",
    "osf-analyze-pool-internal-top100",
    "osf-build-exposure-input",
    "osf-build-labeled-cache",
    "osf-build-long-horizon-labels",
    "osf-build-long-label-raw-source",
    "osf-build-next-close-labels",
    "osf-build-raw-source-cache",
    "osf-build-short-labels",
    "osf-build-target-label-cache",
    "osf-build-training-datasets",
    "osf-split-horizon-labels",
    "osf-split-long-horizon-labels",
    "osf-extract-execution-context",
    "osf-run-alpha-conditioned-rolling-validation",
    "osf-run-gap-risk-attribution",
    "osf-run-learned-risk-layer",
    "osf-run-score-risk-sweep",
    "osf-train",
    "run_top1000_rank_bucket_diagnostics.py",
)
REQUIRED_DIRS = (
    "src/opening_strength_fit",
    "src/opening_strength_fit/commands",
    "examples/smoke",
    "experiments/runs",
    "experiments/jobs",
    "experiments/evidence",
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


def _git_files(*arguments: str) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return sorted(line for line in result.stdout.splitlines() if line)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


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
        check_matrix_cases(path, config, errors)


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

        name = str(case["name"])
        if name in seen:
            errors.append(f"{path}: duplicate matrix case name {name!r}")
        seen.add(name)

        window = str(case["window"])
        horizon = str(case["horizon"])
        if not re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2}", window):
            errors.append(f"{location} has invalid window={window!r}")
            continue
        window_slug = window.replace(":", "").replace("-", "_")
        expected_name = f"w{window_slug}_h{horizon}"
        if name != expected_name:
            errors.append(f"{location} name must be {expected_name!r}")

        feature_path = str(case["feature_path"])
        label_path = str(case["label_path"])
        if (
            not feature_path.startswith("/")
            or f"opening_{window_slug}_features_" not in feature_path
        ):
            errors.append(f"{location} feature_path does not match window {window!r}")
        if (
            not label_path.startswith("/")
            or f"opening_{window_slug}_labels_h{horizon}" not in label_path
        ):
            errors.append(f"{location} label_path does not match window/horizon")


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

    for command, target in sorted(entrypoints.items()):
        if command == "osf":
            if target != "opening_strength_fit.cli:main":
                errors.append(
                    "pyproject.toml: command 'osf' must target opening_strength_fit.cli:main"
                )
            continue
        if not command.startswith("osf-"):
            errors.append(f"pyproject.toml: command {command!r} must use osf- prefix")
        if not target.startswith("opening_strength_fit.commands.") or not target.endswith(":main"):
            errors.append(
                f"pyproject.toml: command {command!r} must target "
                "opening_strength_fit.commands.*:main"
            )
            continue
        module_name = target.removeprefix("opening_strength_fit.").removesuffix(":main")
        module_path = f"src/opening_strength_fit/{module_name.replace('.', '/')}.py"
        if module_path not in files:
            errors.append(f"pyproject.toml: command {command!r} target {module_path} is missing")

    for entrypoint in K8S_JOB_ENTRYPOINTS:
        if entrypoint.endswith(".py"):
            continue
        if entrypoint not in entrypoints:
            errors.append(
                f"project contract: k8s entrypoint {entrypoint!r} is not declared in pyproject.toml"
            )


def _dependency_name(requirement: object) -> str:
    return re.split(r"[ <>=!~\[]", str(requirement), maxsplit=1)[0].lower()


def check_dependency_profiles(errors: list[str]) -> None:
    pyproject = tomllib.loads(read("pyproject.toml"))
    project = pyproject.get("project", {})
    base = {_dependency_name(item) for item in project.get("dependencies", [])}
    misplaced = sorted(base & HEAVY_RUNTIME_DEPENDENCIES)
    if misplaced:
        errors.append(
            "pyproject.toml: adapter/model/plot dependencies must stay out of the core install: "
            + ", ".join(misplaced)
        )

    optional = project.get("optional-dependencies", {})
    for profile in ("dev", "cluster"):
        dependencies = {_dependency_name(item) for item in optional.get(profile, [])}
        missing = sorted(HEAVY_RUNTIME_DEPENDENCIES - dependencies)
        if missing:
            errors.append(
                f"pyproject.toml: {profile} profile is missing runtime dependencies "
                + ", ".join(missing)
            )

    locked = {
        _dependency_name(line)
        for line in read("requirements.lock").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    profiles = {"core": project.get("dependencies", []), **optional}
    for profile, requirements in sorted(profiles.items()):
        missing = sorted({_dependency_name(item) for item in requirements} - locked)
        if missing:
            errors.append(
                f"requirements.lock: {profile} dependencies are not pinned: " + ", ".join(missing)
            )


def check_docker_contract(errors: list[str]) -> None:
    for path in ("Dockerfile", "Dockerfile.overlay"):
        text = read(path)
        if re.search(r"^COPY\s+\.\s+\.\s*$", text, flags=re.MULTILINE):
            errors.append(f"{path}: broad COPY . . is forbidden; copy runtime assets explicitly")
        if "org.opencontainers.image.revision" not in text:
            errors.append(f"{path}: source revision OCI label is missing")
    if "ARG DEPENDENCY_PROFILE=cluster" not in read("Dockerfile"):
        errors.append("Dockerfile: dependency profile build argument is missing")


def check_command_entrypoint_dependencies(files: list[str], errors: list[str]) -> None:
    pyproject = tomllib.loads(read("pyproject.toml"))
    entrypoint_modules = {
        str(target).partition(":")[0]
        for target in pyproject.get("project", {}).get("scripts", {}).values()
    }
    for module in _module_files(files, "src/opening_strength_fit/commands"):
        module_name = module.removeprefix("src/").removesuffix(".py").replace("/", ".")
        tree = ast.parse(read(module), filename=module)
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
        for imported in sorted((imported_modules & entrypoint_modules) - {module_name}):
            errors.append(
                f"{module}: command entrypoint must not import sibling entrypoint {imported}"
            )


def check_required_dirs(files: list[str], errors: list[str]) -> None:
    for directory in REQUIRED_DIRS:
        if any(path.startswith(f"{directory}/") for path in files):
            continue
        if directory in LOCAL_ARCHIVE_DIRS or _has_local_files(directory):
            continue
        errors.append(f"{directory}: no files found")


def check_reproducibility_files(files: list[str], errors: list[str]) -> None:
    for path in REQUIRED_REPRODUCIBILITY_FILES:
        if path not in files:
            errors.append(f"{path}: required reproducibility file is missing")

    forbidden_prefixes = ("output/", "experiments/results/")
    forbidden_names = {".env", ".coverage"}
    forbidden_suffixes = {".parquet", ".pkl", ".pickle", ".pt", ".pth", ".joblib"}
    for path in tracked_files():
        candidate = Path(path)
        if path.startswith(forbidden_prefixes) or candidate.name in forbidden_names:
            errors.append(f"{path}: generated or private file must not be tracked")
        if candidate.suffix.lower() in forbidden_suffixes:
            errors.append(f"{path}: data or model binary must not be tracked")


def check_document_links(files: list[str], errors: list[str]) -> None:
    current_docs = [
        path
        for path in files
        if path.endswith(".md")
        and not path.startswith(("docs/archive/", "experiments/archive/"))
        and (
            "/" not in path or path.startswith("docs/") or path == "experiments/evidence/README.md"
        )
    ]
    for path in current_docs:
        for match in MARKDOWN_LINK_RE.finditer(read(path)):
            target = match.group(1).strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            target = target.split(maxsplit=1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_path = target.partition("#")[0]
            if not relative_path:
                continue
            resolved = (ROOT / path).parent.joinpath(relative_path).resolve()
            if not resolved.exists():
                errors.append(f"{path}: broken local documentation link {target!r}")


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


def check_module_sizes(files: list[str], errors: list[str]) -> None:
    profiles = (
        ("src/opening_strength_fit/commands/", MAX_COMMAND_MODULE_LINES, "command module"),
        ("src/opening_strength_fit/", MAX_DOMAIN_MODULE_LINES, "domain module"),
        ("experiments/scripts/", MAX_EXPERIMENT_SCRIPT_LINES, "experiment script"),
    )
    for path in files:
        if not path.endswith(".py") or path.endswith("/__init__.py"):
            continue
        for prefix, maximum, label in profiles:
            if not path.startswith(prefix):
                continue
            if label == "domain module" and path.startswith("src/opening_strength_fit/commands/"):
                continue
            line_count = len(read(path).splitlines())
            if line_count > maximum:
                errors.append(
                    f"{path}: {label} should stay below {maximum} lines; found {line_count}"
                )
            break


def check_shared_schema_constants(files: list[str], errors: list[str]) -> None:
    for module in _module_files(files, "src/opening_strength_fit"):
        if module == "src/opening_strength_fit/schema.py":
            continue
        tree = ast.parse(read(module), filename=module)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if not any(name.endswith("KEY_COLUMNS") for name in names):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                continue
            if tuple(value) == ("date", "symbol", "decision_target_timestamp"):
                errors.append(
                    f"{module}: import schema.DECISION_KEY_COLUMNS instead of duplicating the key"
                )


def check_evidence(files: list[str], errors: list[str]) -> None:
    forbidden_suffixes = {".parquet", ".pkl", ".pickle", ".pt", ".pth", ".joblib"}
    prefix = "experiments/evidence/backtests/"
    for path in files:
        if not path.startswith("experiments/evidence/"):
            continue
        evidence_path = ROOT / path
        if evidence_path.suffix.lower() in forbidden_suffixes:
            errors.append(f"{path}: row-level data or model binaries do not belong in evidence")
        if evidence_path.stat().st_size > MAX_EVIDENCE_FILE_BYTES:
            errors.append(
                f"{path}: evidence file exceeds {MAX_EVIDENCE_FILE_BYTES} bytes; "
                "record an aggregate instead"
            )
        if not path.startswith(prefix):
            continue
        relative = path.removeprefix(prefix)
        run_id_value, separator, _ = relative.partition("/")
        if separator and f"experiments/runs/{run_id_value}.toml" not in files:
            errors.append(f"{path}: evidence has no matching run config")


def load_incubator_assets(files: list[str], errors: list[str]) -> set[str]:
    if INCUBATOR_MANIFEST not in files:
        return set()
    try:
        manifest = tomllib.loads(read(INCUBATOR_MANIFEST))
    except tomllib.TOMLDecodeError as exc:
        errors.append(f"{INCUBATOR_MANIFEST}: invalid TOML: {exc}")
        return set()

    campaign = manifest.get("campaign", {})
    entries = campaign.get("entries", []) if isinstance(campaign, dict) else []
    if not isinstance(entries, list) or not entries:
        errors.append(f"{INCUBATOR_MANIFEST}: campaign.entries must not be empty")
        return set()

    assets: set[str] = set()
    allowed_prefixes = ("experiments/jobs/support/", "experiments/scripts/")
    for index, entry in enumerate(entries, start=1):
        location = f"{INCUBATOR_MANIFEST}: campaign entry {index}"
        if not isinstance(entry, dict):
            errors.append(f"{location} must be a table")
            continue
        for field in ("id", "owner", "review_by", "purpose", "promote_when"):
            if not str(entry.get(field, "")).strip():
                errors.append(f"{location} must declare {field}")
        review_by = str(entry.get("review_by", ""))
        try:
            review_date = date.fromisoformat(review_by)
        except ValueError:
            errors.append(f"{location} has invalid review_by={review_by!r}")
        else:
            if review_date < date.today():
                errors.append(f"{location} review expired on {review_by}")

        entry_assets = entry.get("assets", [])
        if not isinstance(entry_assets, list) or not entry_assets:
            errors.append(f"{location} must list assets")
            continue
        for raw_path in entry_assets:
            path = str(raw_path)
            if not path.startswith(allowed_prefixes):
                errors.append(f"{location} asset is outside supported incubator paths: {path}")
            if path not in files:
                errors.append(f"{location} asset does not exist: {path}")
            if path in assets:
                errors.append(f"{location} repeats incubator asset: {path}")
            assets.add(path)
    return assets


def check_incubator_coverage(incubator_assets: set[str], errors: list[str]) -> None:
    prefixes = ("experiments/jobs/support/", "experiments/scripts/")
    missing = sorted(
        path
        for path in untracked_project_files()
        if path.startswith(prefixes) and path not in incubator_assets
    )
    for path in missing:
        errors.append(f"{path}: untracked experiment asset is missing from {INCUBATOR_MANIFEST}")


def check_k8s_jobs(files: list[str], errors: list[str], incubator_assets: set[str]) -> None:
    for job in [path for path in files if path.startswith("experiments/jobs/")]:
        if not job.endswith("_job.yaml"):
            continue
        if job in incubator_assets:
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
            allowed = ", ".join(K8S_JOB_ENTRYPOINTS)
            errors.append(f"{job}: k8s job does not use one of: {allowed}")


def check_canonical_registry(files: list[str], errors: list[str]) -> None:
    path = "experiments/canonical/opening.toml"
    config = load_toml(ROOT / path)
    matrix = config.get("research_matrix", {})
    if not isinstance(matrix, dict):
        errors.append(f"{path}: research_matrix must be a table")
        return
    run_id = str(matrix.get("source_run_id", ""))
    run_path = f"experiments/runs/{run_id}.toml"
    evidence = str(matrix.get("evidence", ""))
    if not run_id or run_path not in files:
        errors.append(f"{path}: research_matrix source run is missing")
        return
    if not evidence or f"{evidence}/README.md" not in files:
        errors.append(f"{path}: research_matrix evidence README is missing")
    run_config = load_toml(ROOT / run_path)
    cases = run_config.get("matrix", {}).get("cases", [])
    if int(matrix.get("case_count", -1)) != len(cases):
        errors.append(f"{path}: research_matrix case_count does not match source run")
    expected_folds = len(cases) * 8
    if int(matrix.get("fold_count", -1)) != expected_folds:
        errors.append(f"{path}: research_matrix fold_count must be {expected_folds}")


def collect_errors() -> list[str]:
    files = project_files()
    errors: list[str] = []
    check_required_dirs(files, errors)
    check_reproducibility_files(files, errors)
    check_document_links(files, errors)
    check_legacy_script_tree(files, errors)
    check_project_scripts(files, errors)
    check_dependency_profiles(errors)
    check_docker_contract(errors)
    check_command_entrypoint_dependencies(files, errors)
    check_module_sizes(files, errors)
    check_shared_schema_constants(files, errors)
    check_evidence(files, errors)
    check_run_configs(files, errors)
    incubator_assets = load_incubator_assets(files, errors)
    check_incubator_coverage(incubator_assets, errors)
    check_k8s_jobs(files, errors, incubator_assets)
    check_canonical_registry(files, errors)
    return errors
