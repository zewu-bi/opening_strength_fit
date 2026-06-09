# Project Map

代码、CLI 和项目目录索引。研究逻辑看 [project_brief.md](project_brief.md)，实验顺序看
[experiment_log.md](experiment_log.md)，运行命令看 [runbook.md](runbook.md)。

## Root

- `README.md`: project entrypoint and current spine.
- `Dockerfile`: CPU training image; default command is `osf-train --help`.
- `pyproject.toml`: package metadata, runtime/dev dependencies, CLI entrypoints, pytest, coverage, and ruff settings.
- `requirements.lock`: frozen Python environment versions used as a reproducibility baseline.
- `Makefile`: standard local entrypoints for install, test, lint, format, contracts, and CI.
- `.github/workflows/ci.yml`: GitHub Actions lint and pytest workflow.
- `.pre-commit-config.yaml`: ruff check/format hooks.
- `.env.example`: local ClickHouse, tick path, and Ceph S3 stock-pool variables.

## Python Package Layout

- `src/opening_strength_fit/cli/`: thin `osf-*` command wrappers; each wrapper imports and calls one command `main`.
- `src/opening_strength_fit/commands/`: command implementations for audits, K8s rendering, artifact sync, plots, replay loops, and diagnostics.
- `src/opening_strength_fit/*.py`: reusable primitives and project contracts shared by commands and tests.

## Data And Labels

Library:

- `src/opening_strength_fit/clickhouse_ticks.py`: ClickHouse `stock.tick` query and normalization.
- `src/opening_strength_fit/schema.py`: canonical column names, depth levels, time filters, and standardization.
- `src/opening_strength_fit/io.py`: parquet/csv read/write helpers.
- `src/opening_strength_fit/dataset.py`: raw tick loading and labeled feature frame construction.
- `src/opening_strength_fit/sampling.py`: decision-point sampling helpers.
- `src/opening_strength_fit/labels.py`: entry-delay buy price, sell VWAP, label, and replay context columns.
- `src/opening_strength_fit/label_audit.py`: label validity and distribution summaries.

CLI:

- `osf-build-labeled-cache`: build a single labeled cache from ClickHouse without training a model.
- `osf-inspect-labeled-cache`: inspect a labeled cache schema/summary and write a compact manifest.
- `osf-build-target-label-cache`: derive target-aligned labeled caches while preserving raw labels.
- `osf-build-labels`: build labeled rows from existing tick files.
- `osf-audit-labels`: audit labeled dataset quality.
- `osf-fetch-clickhouse-ticks`: fetch one symbol/day tick window.
- `osf-inspect-dataset`: local source inspection and small labeled sample generation.
- `osf-prepare-research-dataset`: partition labeled research datasets by date.
- `osf-concat-frames`: concatenate parquet/csv frames.
- `osf-probe-clickhouse-data`: inspect ClickHouse schema and opening-window coverage.
- `osf-build-next-close-labels`: fetch/cache ClickHouse next-close labels for decision rows.

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

CLI:

- `osf-train` / `osf-run-experiment`: train/evaluate a configured experiment.
- `osf-evaluate-predictions`: evaluate an existing prediction file.
- `osf-analyze-pool-internal-top100`: join predictions with next-close labels and selection masks to produce Top100 pool-internal validation panels.
- `osf-plot-weekly-pool-internal`: render optional trading-day-equal weekly / 4w rolling pool-internal stability diagnostics from `pool_internal_group_metrics.csv`.
- `osf-plot-weekly-pool-internal-cumulative`: render cumulative short/next pool-internal excess from pre-aggregated daily or weekly summary rows.
- `osf-summarize-opening-results`: summarize metrics CSVs.
- `osf-compare-opening-results`: compare archived model metrics.
- `osf-run-rule-baselines`: evaluate simple visible-information rules.

## Diagnostics And Research Loops

- `osf-audit-feature-dependence`: grouped feature importance, permutation, and drop-retrain ablation audits.
- `osf-run-opening-intraday-backtest`: constrained opening TopN replay.
- `osf-run-lgbm-delay-replays`: standard delay0/1/2 replay grid.
- `osf-plot-lgbm-delay-decay`: plot delay replay decay summaries.
- `osf-run-alpha-horizon-decay`: evaluate opening scores on intraday/close horizons.
- `osf-plot-signal-baseline-panels`: render delay2 short/next-close baseline panels using cached next-close labels.
- `osf-run-score-tail-guards`: sweep visible-information TopN guard rules over an existing score file.
- `osf-run-score-risk-sweep`: alpha-minus-risk score sweeps over existing score files.
- `osf-run-learned-risk-layer`: learned dirty-risk / next-flip risk layers for score-sweep evidence.
- `osf-run-alpha-conditioned-rolling-validation`: monthly validation for alpha-conditioned Top100 risk-penalty scores.
- `osf-plot-rolling-validation-tradeoff`: render the archived rolling short-vs-next Top100 tradeoff chart from `experiments/results/backtests/*_month_summary.csv`.
- `osf-run-gap-risk-attribution`: gap-risk Top100 replacement attribution.

## Infrastructure

Library:

- `src/opening_strength_fit/k8s.py`: RunSpec, PVC helper pod, and kubectl wrappers.
- `src/opening_strength_fit/cache_manifest.py`: JSON-safe labeled-cache schema and summary manifest helpers.
- `src/opening_strength_fit/cache_lock.py`: labeled-cache lock acquisition, heartbeat, ready marker, and release helpers.

CLI:

- `osf-render-k8s-job`: render training, feature-audit, cache-transform, sharded training, and cluster-side pool-internal analysis K8s manifests.
- `osf-sync-experiment-artifacts`: pull metrics and lightweight cluster-side analysis artifacts from PVC, combine shard metrics, and archive compact evidence. Prediction parquet sync is explicit debug/legacy behavior.
- `osf-rolling-job-status`: map K8s Indexed Job pods back to rolling months and print per-month log commands.
- `osf-audit-experiments`: check config/job/metrics alignment.
- `osf-check-project-contracts`: check CLI, config, directory, and K8s entrypoint contracts.

## Experiments

- `experiments/runs/*.toml`: run configs. `run.id` must match the filename.
- `experiments/jobs/*.yaml`: rendered K8s jobs.
- `experiments/config_templates/`: reusable TOML snippets for run configs.
- `experiments/results/metrics/`: tracked metrics CSV evidence.
- `experiments/results/backtests/`: tracked replay, horizon-decay, sweep, and rolling summaries. Multi-file pool-internal archives use `backtests/<record_prefix>/`.
- `*_sharded_job.yaml`: monthly/yearly sharded K8s jobs.

## Ignored Outputs

- `output/artifacts/<run_id>/` and `output/artifacts/_partial_metrics/`: current local mirrors and partial metrics; ignored.
- `output/legacy/{artifacts,predictions,analysis,labels,reports}/`: old pulls, debug prediction/label/report state, smoke output, and heavy diagnostics; ignored.
