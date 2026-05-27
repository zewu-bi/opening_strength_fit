# Runbook

This file is the operating manual. Research interpretation belongs in
`docs/project_brief.md`; historical numbers belong in `docs/experiment_log.md`.

## 1. Preflight

```bash
cd /home/hefu/projects/opening_strength_fit
source .venv/bin/activate
set -a; . ./.env; set +a

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

## 2. Local Smoke

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet

python scripts/run_experiment.py \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --input output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --input-kind labeled \
  --split-mode chronological \
  --test-start-date 2021-09-23 \
  --test-end-date 2021-09-23 \
  --feature-limit 80 \
  --top-n 2 \
  --output-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke

python scripts/summarize_opening_results.py \
  --input-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke
```

Do not build multi-month labeled datasets locally; use existing PVC labeled caches
or dedicated training runs.

## 3. Experiment Config

Every formal experiment has:

```text
experiments/runs/<run_id>.toml
experiments/jobs/<run_id>_job.yaml
experiments/results/metrics/<run_id>_metrics_by_year.csv
```

Exceptions:

- `[run].kind = "feature_audit"` runs grouped importance, permutation, and
  drop-retrain ablations and writes audit CSVs under the run output dir.
- `[run].kind = "exploration"` may be active/running without metrics until it
  graduates into a formal archived experiment.
- Post-open signal experiments use the normal `scripts/run_experiment.py` path with
  `[features].include_postopen_decision = true`; richer v2 feature experiments add
  `[features].include_postopen_v2 = true`.

PVC convention:

```text
cache:      /mnt/output/opening_strength_fit/cache/*.parquet
run output: /mnt/output/opening_strength_fit/<run_id>/
local pull: output/predictions/<run_id>/predictions_all.parquet
```

`*.tmp.parquet`, `*.parquet.lock`, and heartbeat files are in-progress state, not
training inputs.

## 4. Build and Run on K8s

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-lgbm-cpu-v1
docker build --build-arg CACHE_BUST=${TAG} -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

python scripts/render_k8s_job.py \
  --config experiments/runs/<run_id>.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

hfcli kubectl --cluster research apply --dry-run=client -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research delete job opening-strength-<run-slug> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/opening-strength-<run-slug> -n bizewu --timeout=24h
```

CPU LightGBM + PVC labeled cache is the default path. GPU is used only when
`[model].device_type = "gpu"` and `[k8s.resources].gpu_limit` are explicitly set.

## 5. Sync Artifacts

Metrics pull, predictions pull, shard metric combination, and lightweight archive
use one interface:

```bash
python scripts/sync_experiment_artifacts.py \
  --config experiments/runs/<run_id>.toml \
  --all

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

Default outputs:

```text
output/k8s/metrics/<run_id>_metrics_by_year.csv
output/predictions/<run_id>/predictions_all.parquet
experiments/results/metrics/<run_id>_metrics_by_year.csv
```

## 6. Analysis

Metrics:

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/<run_id>_metrics_by_year.csv

python scripts/compare_opening_results.py
```

Standard LightGBM delay replay:

```bash
python scripts/run_lgbm_delay_replays.py --check-interface-only
python scripts/run_lgbm_delay_replays.py
python scripts/plot_lgbm_delay_decay.py
```

Horizon decay:

```bash
python scripts/run_alpha_horizon_decay.py \
  --decision-time 09:30:00 \
  --horizon 1m --horizon 2m --horizon 5m --horizon 10m \
  --horizon close --horizon next_close \
  --no-sampled-intraday \
  --clickhouse-intraday-labels \
  --clickhouse-close-labels \
  --allow-missing-horizons \
  --output-root output/reports/opening_alpha_horizon_decay_delay2_clickhouse_point_0930_selected
```

Feature dependence audit:

```bash
python scripts/audit_feature_dependence.py \
  --config experiments/runs/lgbm_delay2_feature_dependence_v1.toml \
  --output-dir output/local/lgbm_delay2_feature_dependence_v1
```

## 7. Troubleshooting

- `field is immutable`: delete the same-name Job, then apply again.
- missing config in K8s: rebuild/push image and rerender the Job.
- missing PVC cache: wait for final `*.parquet`; do not train from `.tmp` or lock files.
- replay missing context columns: pass `--context-input` or run only interface checks.
- completed config with no metrics: run `sync_experiment_artifacts.py --all`, then audit.
