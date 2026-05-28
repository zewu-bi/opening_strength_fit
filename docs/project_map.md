# Project Map

## Root

- `README.md`: project entrypoint and current spine.
- `Dockerfile`: CPU training image; default command is `scripts/run_experiment.py --help`.
- `pyproject.toml`: package metadata, runtime dependencies, and test settings.
- `.env.example`: local ClickHouse/tick path variables.

## Library: `src/opening_strength_fit/`

- `config.py`: TOML loading, typed config lookup, run id, and slug helpers.
- `schema.py`: canonical column names, depth levels, time filters, and standardization.
- `clickhouse_ticks.py`: ClickHouse `stock.tick` query and normalization.
- `io.py`: parquet/csv read/write helpers.
- `dataset.py`: raw tick loading and labeled feature frame construction.
- `universe.py`: A-share symbol filtering and symbol-list loading.
- `sampling.py`: decision-point sampling helpers.
- `features.py`: preopen, orderbook, post-open v1/v2 decision, trade-flow, and momentum features.
- `labels.py`: entry-delay buy price, sell VWAP, label, and replay context columns.
- `candidates.py`: visible-information opening candidate filters.
- `label_audit.py`: label validity and distribution summaries.
- `targets.py`: cross-sectional demean/zscore/rank, heat-neutral, guard-shrunk, and risk-shrunk target-label transforms for derived training caches.
- `rules.py`: non-ML baseline scores.
- `model.py`: Ridge, sklearn GBM, LightGBM, prediction, configurable feature selection, and IC helpers.
- `evaluation.py`: score buckets and top-score summaries.
- `rolling.py`: chronological, annual, and monthly split helpers.
- `training.py`: unified training pipeline, configured feature transforms/filters, and output writer.
- `reports.py`: compact dataset summaries, metrics reporting, and yearly aggregation.
- `k8s.py`: RunSpec, PVC helper pod, and kubectl wrappers.
- `__init__.py`: package marker.

## Commands: `scripts/`

- `run_experiment.py`: train/evaluate a configured experiment.
- `audit_feature_dependence.py`: run grouped feature importance, permutation, and drop-retrain ablation audits.
- `build_target_label_cache.py`: derive target-aligned labeled caches, including heat-neutral, guard-shrunk, and risk-shrunk targets, while preserving raw labels.
- `render_k8s_job.py`: render training, feature-audit, cache-transform, and sharded K8s manifests.
- `sync_experiment_artifacts.py`: pull metrics/predictions from PVC, combine shard metrics, and archive lightweight metrics.
- `audit_experiments.py`: check config/job/metrics alignment.
- `check_workflow_coverage.py`: check script/module/doc/job coverage.
- `probe_clickhouse_data.py`: inspect ClickHouse schema and opening-window coverage.
- `inspect_dataset.py`: local source inspection and small labeled sample generation.
- `prepare_research_dataset.py`: partition labeled research datasets by date.
- `fetch_clickhouse_ticks.py`: fetch one symbol/day tick window.
- `concat_frames.py`: concatenate parquet/csv frames.
- `build_labels.py`: build labeled rows from existing tick files.
- `audit_labels.py`: audit labeled dataset quality.
- `run_rule_baselines.py`: evaluate simple visible-information rules.
- `evaluate_predictions.py`: evaluate an existing prediction file.
- `summarize_opening_results.py`: summarize metrics CSVs.
- `compare_opening_results.py`: compare archived model metrics.
- `run_opening_intraday_backtest.py`: constrained opening TopN replay.
- `run_lgbm_delay_replays.py`: standard delay0/1/2 replay grid.
- `plot_lgbm_delay_decay.py`: plot delay replay decay summaries.
- `run_alpha_horizon_decay.py`: evaluate opening scores on intraday/close horizons.
- `plot_signal_baseline_panels.py`: render delay2 short/next-close baseline panels.
- `run_score_tail_guards.py`: sweep visible-information TopN guard rules over an existing score file.
- `run_score_risk_sweep.py`: sweep alpha-rank minus dirty-risk penalties and hard gates over existing score files.
- `run_learned_risk_layer.py`: train learned dirty-risk / next-flip risk layers for later score sweeps.
- `_bootstrap.py`: makes `python scripts/<name>.py` import `src/`.

## Experiments

- `experiments/runs/*.toml`: run configs. `run.id` must match the filename.
- `experiments/jobs/*.yaml`: rendered K8s jobs. Training jobs use
  `scripts/run_experiment.py`; feature-audit jobs use
  `scripts/audit_feature_dependence.py`; cache-transform jobs use
  `scripts/build_target_label_cache.py`; learned-risk jobs use
  `scripts/run_learned_risk_layer.py`; score-risk sweep jobs use
  `scripts/run_score_risk_sweep.py`.
- `experiments/results/metrics/`: tracked metrics CSV evidence.
- `experiments/results/backtests/`: tracked replay and horizon-decay summaries.

## Ignored Outputs

- `output/predictions/<run_id>/`: pulled parquet predictions.
- `output/k8s/metrics/`: raw pulled metrics before archive.
- `output/reports/`: local PNGs, markdown reports, and heavy diagnostics.
- `output/local/`: smoke and scratch outputs.
