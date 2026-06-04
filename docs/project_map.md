# Project Map

这个文件只回答“代码和脚本在哪里”。研究逻辑看 [project_brief.md](project_brief.md)，实验顺序看
[experiment_log.md](experiment_log.md)，操作命令看 [runbook.md](runbook.md)。

## Root

- `README.md`: project entrypoint and current spine.
- `Dockerfile`: CPU training image; default command is `scripts/run_experiment.py --help`.
- `pyproject.toml`: package metadata, runtime dependencies, and test settings.
- `.env.example`: local ClickHouse, tick path, and Ceph S3 stock-pool variables.

## Data And Labels

Library:

- `src/opening_strength_fit/clickhouse_ticks.py`: ClickHouse `stock.tick` query and normalization.
- `src/opening_strength_fit/schema.py`: canonical column names, depth levels, time filters, and standardization.
- `src/opening_strength_fit/io.py`: parquet/csv read/write helpers.
- `src/opening_strength_fit/dataset.py`: raw tick loading and labeled feature frame construction.
- `src/opening_strength_fit/sampling.py`: decision-point sampling helpers.
- `src/opening_strength_fit/labels.py`: entry-delay buy price, sell VWAP, label, and replay context columns.
- `src/opening_strength_fit/label_audit.py`: label validity and distribution summaries.

Scripts:

- `scripts/build_labeled_cache.py`: build a single labeled cache from ClickHouse without training a model.
- `scripts/inspect_labeled_cache.py`: inspect a labeled cache schema/summary and write a compact manifest.
- `scripts/build_target_label_cache.py`: derive target-aligned labeled caches while preserving raw labels.
- `scripts/build_labels.py`: build labeled rows from existing tick files.
- `scripts/audit_labels.py`: audit labeled dataset quality.
- `scripts/fetch_clickhouse_ticks.py`: fetch one symbol/day tick window.
- `scripts/inspect_dataset.py`: local source inspection and small labeled sample generation.
- `scripts/prepare_research_dataset.py`: partition labeled research datasets by date.
- `scripts/concat_frames.py`: concatenate parquet/csv frames.
- `scripts/probe_clickhouse_data.py`: inspect ClickHouse schema and opening-window coverage.

## Features And Pools

Library:

- `src/opening_strength_fit/universe.py`: A-share symbol filtering and symbol-list loading.
- `src/opening_strength_fit/stock_pool.py`: Ceph/local `date x symbol` bool stock-pool loading, masks, and summaries.
- `src/opening_strength_fit/features.py`: preopen/auction summaries, orderbook, `postopen_*` trajectory/response features, trade-flow, and momentum features.
- `src/opening_strength_fit/feature_config.py`: feature include/drop filter and feature-limit config helpers.
- `src/opening_strength_fit/candidates.py`: visible-information opening candidate filters.
- `src/opening_strength_fit/targets.py`: cross-sectional demean/zscore/rank, heat-neutral, guard-shrunk, and risk-shrunk transforms.

## Training And Evaluation

Library:

- `src/opening_strength_fit/config.py`: TOML loading, typed config lookup, run id, and slug helpers.
- `src/opening_strength_fit/analysis.py`: shared research-script helpers for clock/month ranges, next-close labels, finite TopN summaries, and JSON artifact traces.
- `src/opening_strength_fit/model.py`: Ridge, sklearn GBM, LightGBM, prediction, feature filtering, and IC helpers.
- `src/opening_strength_fit/training.py`: unified training pipeline, configured feature transforms/filters, and output writer.
- `src/opening_strength_fit/evaluation.py`: score buckets and top-score summaries.
- `src/opening_strength_fit/reports.py`: compact dataset summaries, metrics reporting, and yearly aggregation.
- `src/opening_strength_fit/rolling.py`: chronological, annual, and monthly split helpers.
- `src/opening_strength_fit/rules.py`: non-ML baseline scores.
- `src/opening_strength_fit/alpha_conditioning.py`: alpha-conditioned risk target, section-scoped LightGBM fit, scoring, and group-rank helpers.
- `src/opening_strength_fit/__init__.py`: package marker.

Scripts:

- `scripts/run_experiment.py`: train/evaluate a configured experiment.
- `scripts/evaluate_predictions.py`: evaluate an existing prediction file.
- `scripts/summarize_opening_results.py`: summarize metrics CSVs.
- `scripts/compare_opening_results.py`: compare archived model metrics.
- `scripts/run_rule_baselines.py`: evaluate simple visible-information rules.

## Diagnostics And Research Loops

- `scripts/audit_feature_dependence.py`: grouped feature importance, permutation, and drop-retrain ablation audits.
- `scripts/run_opening_intraday_backtest.py`: constrained opening TopN replay.
- `scripts/run_lgbm_delay_replays.py`: standard delay0/1/2 replay grid.
- `scripts/plot_lgbm_delay_decay.py`: plot delay replay decay summaries.
- `scripts/run_alpha_horizon_decay.py`: evaluate opening scores on intraday/close horizons.
- `scripts/plot_signal_baseline_panels.py`: render delay2 short/next-close baseline panels.
- `scripts/run_score_tail_guards.py`: sweep visible-information TopN guard rules over an existing score file.
- `scripts/run_score_risk_sweep.py`: historical/superseded alpha-minus-risk score sweeps over existing score files.
- `scripts/run_learned_risk_layer.py`: historical learned dirty-risk / next-flip risk layers for score-sweep evidence.
- `scripts/run_alpha_conditioned_rolling_validation.py`: historical monthly validation for alpha-conditioned Top100 risk-penalty scores.
- `scripts/plot_rolling_validation_tradeoff.py`: render the archived rolling short-vs-next Top100 tradeoff chart from `experiments/results/backtests/*_month_summary.csv`.
- `scripts/run_gap_risk_attribution.py`: historical attribution of risk-penalized Top100 replacements.

## Infrastructure

Library:

- `src/opening_strength_fit/k8s.py`: RunSpec, PVC helper pod, and kubectl wrappers.
- `src/opening_strength_fit/cache_manifest.py`: JSON-safe labeled-cache schema and summary manifest helpers.
- `src/opening_strength_fit/cache_lock.py`: labeled-cache lock acquisition, heartbeat, ready marker, and release helpers.

Scripts:

- `scripts/render_k8s_job.py`: render training, feature-audit, cache-transform, and sharded K8s manifests.
- `scripts/sync_experiment_artifacts.py`: pull metrics/predictions from PVC, combine shard metrics, and archive lightweight metrics.
- `scripts/audit_experiments.py`: check config/job/metrics alignment.
- `scripts/check_workflow_coverage.py`: check script/module/doc/job coverage.
- `scripts/_bootstrap.py`: makes `python scripts/<name>.py` import `src/`.

## Experiments

- `experiments/runs/*.toml`: run configs. `run.id` must match the filename.
- `experiments/jobs/*.yaml`: rendered K8s jobs.
- `experiments/results/metrics/`: tracked metrics CSV evidence.
- `experiments/results/backtests/`: tracked replay and horizon-decay summaries.

Experiment layers:

- Current active core: current docs spine, shared library modules, main training/cache/sync/audit scripts, and the
  `build_delay2_2015_cache_v2` through `build_delay2_2024_cache_v2` labeled-cache run/job pairs.
- Historical evidence: every run/job/config recorded in `docs/experiment_log.md`, represented in `experiments/results/**`,
  or explicitly referenced by docs. This includes guard, clean-target, learned-risk, score-risk, rolling-risk, and
  attribution experiments that are now superseded by the single mixed-label mainline.
- Generated retained trace: rendered K8s YAML and lightweight artifact sync traces that connect a historical result back to
  its executable job. Keep these even when the research route is no longer active.
- Stale unrecorded: run/job/config files with no docs/results reference and no current running value. These can be removed
  after a reference check; ignored local caches such as `__pycache__`, `.pytest_cache`, and `*.egg-info` can be cleaned
  without preserving trace.

Job entrypoints:

| run kind | script |
| --- | --- |
| standard training / exploration | `scripts/run_experiment.py` |
| feature audit | `scripts/audit_feature_dependence.py` |
| labeled cache | `scripts/build_labeled_cache.py` |
| cache transform / target cache | `scripts/build_target_label_cache.py` |
| learned risk layer | `scripts/run_learned_risk_layer.py` |
| alpha-conditioned rolling | `scripts/run_alpha_conditioned_rolling_validation.py` |
| gap-risk attribution | `scripts/run_gap_risk_attribution.py` |
| score-risk sweep | `scripts/run_score_risk_sweep.py` |

`*_sharded_job.yaml` runs monthly/yearly shards.

## Ignored Outputs

- `output/predictions/<run_id>/`: pulled parquet predictions.
- `output/k8s/metrics/`: raw pulled metrics before archive.
- `output/reports/`: local PNGs, markdown reports, and heavy diagnostics.
- `output/local/`: ignored artifact-sync and scratch buffer; lightweight evidence is archived under `experiments/results/`.
