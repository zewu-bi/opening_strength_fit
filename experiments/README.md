# Experiments

This file describes experiment directory conventions and run-kind entrypoints.
It does not record experiment results or research decisions; those live in
`docs/experiment_log.md` and `docs/project_brief.md`.

| path | content |
| --- | --- |
| `runs/*.toml` | Run configs. `run.id` matches the filename. |
| `jobs/*.yaml` | Rendered K8s Job manifests. |
| `config_templates/` | Reusable TOML snippets for run configs. |
| `results/` | Local compact evidence archive; normally ignored and indexed from `docs/experiment_log.md`. |

Run kind entrypoints:

| run kind | entrypoint |
| --- | --- |
| standard training / exploration | `osf-train` |
| capacity audit | `osf-audit-capacity` |
| capacity acceptance | `osf-analyze-capacity-acceptance` |
| ask-level attribution | `osf-ask-level-attribution` |
| execution context extraction | `osf-extract-execution-context` |
| exposure input | `osf-build-exposure-input` |
| exposure audit | `osf-audit-exposure` |
| feature audit | `osf-audit-feature-dependence` |
| feature hygiene | `osf-audit-feature-hygiene` |
| labeled cache | `osf-build-labeled-cache` |
| cache transform / target cache | `osf-build-target-label-cache` |
| next-close label cache | `osf-build-next-close-labels` |
| learned risk layer | `osf-run-learned-risk-layer` |
| alpha-conditioned rolling | `osf-run-alpha-conditioned-rolling-validation` |
| gap-risk attribution | `osf-run-gap-risk-attribution` |
| score-risk sweep | `osf-run-score-risk-sweep` |
