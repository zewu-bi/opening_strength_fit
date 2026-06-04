# Experiments

| path | content |
| --- | --- |
| `runs/*.toml` | Run configs. `run.id` matches the filename. |
| `jobs/*.yaml` | Rendered K8s Job manifests. |
| `config_templates/` | Reusable TOML snippets for run configs. |
| `results/` | Lightweight tracked evidence. |

Run kind entrypoints:

| run kind | script |
| --- | --- |
| standard training / exploration | `scripts/run_experiment.py` |
| feature audit | `scripts/audit_feature_dependence.py` |
| labeled cache | `scripts/build_labeled_cache.py` |
| cache transform / target cache | `scripts/build_target_label_cache.py` |
| next-close label cache | `scripts/build_next_close_labels.py` |
| learned risk layer | `scripts/run_learned_risk_layer.py` |
| alpha-conditioned rolling | `scripts/run_alpha_conditioned_rolling_validation.py` |
| gap-risk attribution | `scripts/run_gap_risk_attribution.py` |
| score-risk sweep | `scripts/run_score_risk_sweep.py` |
