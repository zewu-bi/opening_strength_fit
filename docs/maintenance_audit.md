# Maintenance Audit

Date: 2026-06-03

This audit records the safe maintenance boundary for local refactors while research
jobs may be running in the cluster.

## Current Safety Boundary

- Do not edit `experiments/runs/*.toml` during unrelated maintenance. These files
  are the source of future rendered jobs and may describe running or pending work.
- Do not edit `experiments/jobs/*.yaml` unless the goal is explicitly to render or
  update a job manifest.
- Do not run `hfcli kubectl`, `kubectl`, artifact sync, Docker build, Docker push,
  PVC cleanup, or cache rebuild commands during local maintenance.
- Do not change cache paths, label schema, feature column names, model output paths,
  or `run.id` values as incidental cleanup.
- Treat `docs/experiment_log.md` as experiment history. Update it only for
  experiment bookkeeping, not for code-structure maintenance.

The current working tree already had local experiment changes before this audit:

- `docs/experiment_log.md`
- `experiments/runs/lgbm_delay2_18m_postopen_mixed_w030_reg_mid_v1.toml`
- `experiments/runs/lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_mid_v1.toml`
- new `experiments/runs/*no_preopen_reg_mid_v1.toml`
- new `experiments/runs/*drop_raw_reg_mid_v1.toml`
- matching new rendered sharded job YAML files

Those files are considered pre-existing user work and are outside this maintenance
commit.

## Entrypoints And Risk

Cluster jobs are rendered from `scripts/render_k8s_job.py` and then applied from
`experiments/jobs/*.yaml`. The rendered commands call these scripts inside a built
Docker image:

| run kind | entrypoint |
| --- | --- |
| standard training / exploration | `scripts/run_experiment.py` |
| feature audit | `scripts/audit_feature_dependence.py` |
| labeled cache | `scripts/build_labeled_cache.py` |
| cache transform / target cache | `scripts/build_target_label_cache.py` |
| learned risk layer | `scripts/run_learned_risk_layer.py` |
| alpha-conditioned rolling | `scripts/run_alpha_conditioned_rolling_validation.py` |
| gap-risk attribution | `scripts/run_gap_risk_attribution.py` |
| score-risk sweep | `scripts/run_score_risk_sweep.py` |

Running pods should continue using the image and manifest they were launched with.
Local maintenance becomes cluster-affecting only when it is built, pushed, rendered,
applied, synced, or pointed at shared PVC state.

## Safe Maintenance Started

The first local maintenance change is intentionally narrow:

- Extract labeled-cache lock helpers from `src/opening_strength_fit/training.py`
  into `src/opening_strength_fit/cache_lock.py`.
- Keep the old private names imported in `training.py` so existing tests and scripts
  that import them remain compatible.
- Leave experiment configs, rendered jobs, PVC paths, and remote commands untouched.

## Recommended Next Steps

- Continue extracting isolated helper groups from `training.py` only when they have
  focused tests.
- Prefer adding tests around any helper before changing behavior.
- Keep orchestration scripts thin by moving stable pure logic into `src/`, but avoid
  moving script entrypoints while jobs may depend on their paths.
- Archive or remove stale experiment traces only after checking references in
  `docs/experiment_log.md`, `experiments/results/**`, and rendered job manifests.
