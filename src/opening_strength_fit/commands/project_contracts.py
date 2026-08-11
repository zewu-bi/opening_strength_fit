from __future__ import annotations

from opening_strength_fit.project_validation import (
    INCUBATOR_MANIFEST,
    K8S_JOB_ENTRYPOINTS,
    REQUIRED_DIRS,
    check_matrix_cases,
    collect_errors,
    project_files,
)

__all__ = ["check_matrix_cases", "collect_errors", "main"]


def main() -> None:
    files = project_files()
    errors = collect_errors()
    library_modules = [
        path
        for path in files
        if path.startswith("src/opening_strength_fit/")
        and path.endswith(".py")
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
    print(f"  command_modules: {len(command_modules)}")
    print(f"  library_modules: {len(library_modules)}")
    print(f"  required_dirs: {len(REQUIRED_DIRS)}")
    print(f"  incubator_manifest: {INCUBATOR_MANIFEST}")
    print(f"  k8s_job_entrypoints: {', '.join(K8S_JOB_ENTRYPOINTS)}")

    if errors:
        print("\ncontract_errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("  contracts_ok: yes")


if __name__ == "__main__":
    main()
