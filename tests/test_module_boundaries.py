from __future__ import annotations

import ast
import inspect
from pathlib import Path

import opening_strength_fit.features as features
import opening_strength_fit.features_relative as relative_features
import opening_strength_fit.model as model
import opening_strength_fit.model_torch as torch_facade
import opening_strength_fit.multiscale_bucket_diag as multiscale_facade
import opening_strength_fit.optimization_acceptance_plots as acceptance_plots
from opening_strength_fit.commands import old_nn_multiscale_bucket_diag as old_command_facade
from opening_strength_fit.legacy import multiscale_bucket_diag as multiscale_legacy
from opening_strength_fit.legacy import old_nn_multiscale_bucket_diag as old_command_legacy
from opening_strength_fit.torch_model import architectures as torch_architectures
from opening_strength_fit.torch_model import prediction as torch_prediction
from opening_strength_fit.torch_model import preprocessing as torch_preprocessing
from opening_strength_fit.torch_model import training as torch_training

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "opening_strength_fit"


def _line_count(path: str) -> int:
    return len((ROOT / path).read_text(encoding="utf-8").splitlines())


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT / "src").with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _internal_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name for alias in node.names if alias.name.startswith("opening_strength_fit")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("opening_strength_fit"):
                imports.add(node.module)
    return imports


def _is_command_or_cli_module(module: str) -> bool:
    return module in {
        "opening_strength_fit.commands",
        "opening_strength_fit.cli",
    } or module.startswith(("opening_strength_fit.commands.", "opening_strength_fit.cli."))


def _module_graph() -> dict[str, set[str]]:
    paths = sorted(PACKAGE_ROOT.rglob("*.py"))
    modules = {_module_name(path): path for path in paths}
    graph: dict[str, set[str]] = {}
    for module, path in modules.items():
        dependencies = set()
        for imported in _internal_imports(path):
            candidate = imported
            while candidate and candidate not in modules:
                candidate = candidate.rpartition(".")[0]
            if candidate and candidate != module:
                dependencies.add(candidate)
        graph[module] = dependencies
    return graph


def _dependency_cycle(graph: dict[str, set[str]]) -> list[str]:
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(module: str) -> list[str]:
        if module in active_set:
            start = active.index(module)
            return [*active[start:], module]
        if module in visited:
            return []
        active.append(module)
        active_set.add(module)
        for dependency in sorted(graph.get(module, ())):
            cycle = visit(dependency)
            if cycle:
                return cycle
        active.pop()
        active_set.remove(module)
        visited.add(module)
        return []

    for module in sorted(graph):
        cycle = visit(module)
        if cycle:
            return cycle
    return []


def test_compatibility_modules_stay_thin() -> None:
    assert _line_count("src/opening_strength_fit/features.py") <= 120
    assert _line_count("src/opening_strength_fit/features_relative.py") <= 40
    assert _line_count("src/opening_strength_fit/model.py") <= 120
    assert _line_count("src/opening_strength_fit/model_torch.py") <= 120
    assert _line_count("src/opening_strength_fit/multiscale_bucket_diag.py") <= 60


def test_archived_diagnostic_import_paths_remain_compatible() -> None:
    assert (
        multiscale_facade.run_multiscale_bucket_diagnostics
        is multiscale_legacy.run_multiscale_bucket_diagnostics
    )
    assert old_command_facade.main is old_command_legacy.main
    assert callable(acceptance_plots.write_optimization_direction_plots)


def test_feature_public_api_is_exported_from_compatibility_module() -> None:
    assert features.add_postopen_v2_decision_features.__module__.endswith("features_postopen")
    assert features.add_historical_same_minute_surprise_features.__module__.endswith(
        "features_history"
    )
    assert features.add_cross_sectional_relative_features.__module__.endswith(
        "feature_transforms.cross_sectional"
    )
    assert features.transform_cross_sectional_feature_values.__module__.endswith(
        "feature_transforms.cross_sectional"
    )
    assert features.add_price_scale_features.__module__.endswith(
        "feature_transforms.cross_sectional"
    )
    mechanism_exports = (
        features.mechanismized_feature_value_reference_columns,
        features.transform_mechanismized_feature_values,
        features.transform_mechanismized_v2_feature_values,
        features.transform_mechanismized_v3_feature_values,
    )
    assert all(
        export.__module__.endswith("feature_transforms.mechanism") for export in mechanism_exports
    )
    assert features.build_feature_frame.__module__.endswith("features_base")


def test_relative_feature_facade_exports_only_public_transforms() -> None:
    assert relative_features.__all__ == [
        "add_cross_sectional_relative_features",
        "transform_cross_sectional_feature_values",
        "add_price_scale_features",
        "mechanismized_feature_value_reference_columns",
        "transform_mechanismized_feature_values",
        "transform_mechanismized_v2_feature_values",
        "transform_mechanismized_v3_feature_values",
    ]


def test_model_public_api_is_exported_from_compatibility_module() -> None:
    assert model.feature_columns.__module__.endswith("model_features")
    assert model.fit_lightgbm_frame.__module__.endswith("model_sklearn")
    assert model.fit_torch_mlp_frame is torch_training.fit_torch_mlp_frame
    assert model.predict_frame.__module__.endswith("model_prediction")
    assert model.evaluate_prediction_frame.__module__.endswith("model_metrics")


def test_torch_model_facade_preserves_api_across_implementation_layers() -> None:
    assert torch_facade.__all__ == [
        "fit_torch_mlp_frame",
        "_torch_mlp_score",
        "_torch_feature_value_frame",
        "_fit_symbol_train_standardization",
        "_standardized_float_matrix",
    ]
    assert inspect.signature(torch_facade.fit_torch_mlp_frame) == inspect.signature(
        torch_training.fit_torch_mlp_frame
    )
    assert torch_facade.fit_torch_mlp_frame is torch_training.fit_torch_mlp_frame
    assert torch_facade._torch_mlp_score is torch_prediction._torch_mlp_score
    assert torch_facade._torch_feature_value_frame is torch_preprocessing._torch_feature_value_frame
    assert (
        torch_facade._fit_symbol_train_standardization
        is torch_preprocessing._fit_symbol_train_standardization
    )
    assert torch_facade._standardized_float_matrix is torch_preprocessing._standardized_float_matrix


def test_torch_model_implementation_ownership_is_explicit() -> None:
    assert torch_training.fit_torch_mlp_frame.__module__.endswith("torch_model.training")
    assert torch_prediction._torch_mlp_score.__module__.endswith("torch_model.prediction")
    assert torch_preprocessing._torch_feature_value_frame.__module__.endswith(
        "torch_model.preprocessing"
    )
    assert torch_architectures._TorchMLPModule.__module__.endswith("torch_model.architectures")
    assert "torch" not in vars(torch_architectures)
    assert "torch" not in vars(torch_training)
    assert "torch" not in vars(torch_prediction)


def test_internal_module_graph_is_acyclic() -> None:
    cycle = _dependency_cycle(_module_graph())

    assert cycle == [], " -> ".join(cycle)


def test_domain_modules_do_not_import_command_or_cli_layers() -> None:
    violations = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] in {"cli", "commands"}:
            continue
        forbidden = {
            imported for imported in _internal_imports(path) if _is_command_or_cli_module(imported)
        }
        if forbidden:
            violations.append((relative.as_posix(), sorted(forbidden)))

    assert violations == []
