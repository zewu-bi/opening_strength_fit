# Experiments

This file describes experiment directory conventions and run-kind entrypoints.
Run results and decisions live in `docs/experiment_log.md`.

| path | content |
| --- | --- |
| `runs/*.toml` | Run configs. `run.id` matches the filename. |
| `jobs/*.yaml` | Rendered K8s Job manifests. |
| `config_templates/` | Reusable TOML snippets for run configs. |
| `results/` | Lightweight tracked evidence. |

Run kind entrypoints:

| run kind | entrypoint |
| --- | --- |
| standard training / exploration | `osf-train` |
| feature audit | `osf-audit-feature-dependence` |
| labeled cache | `osf-build-labeled-cache` |
| cache transform / target cache | `osf-build-target-label-cache` |
| next-close label cache | `osf-build-next-close-labels` |
| learned risk layer | `osf-run-learned-risk-layer` |
| alpha-conditioned rolling | `osf-run-alpha-conditioned-rolling-validation` |
| gap-risk attribution | `osf-run-gap-risk-attribution` |
| score-risk sweep | `osf-run-score-risk-sweep` |
