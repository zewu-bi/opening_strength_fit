# 运行手册

这个文件只放操作命令。研究解释放在 [docs/project_brief.md](project_brief.md)，历史数值放在
[docs/experiment_log.md](experiment_log.md)。

## 1. 预检

```bash
cd /home/hefu/projects/opening_strength_fit
source .venv/bin/activate
set -a; . ./.env; set +a

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

## 2. 本地 Smoke

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

不要在本地构造多月或一年级别 labeled dataset；正式长窗口使用已有 PVC cache 或专门训练任务。

## 3. 实验配置

每个正式实验至少对应：

```text
experiments/runs/<run_id>.toml
experiments/jobs/<run_id>_job.yaml
experiments/results/metrics/<run_id>_metrics_by_year.csv
```

例外：

- `[run].kind = "feature_audit"`：运行 grouped importance、permutation 和 drop-retrain ablation，
  audit CSV 写到 run output dir。
- `[run].kind = "cache_transform"` 或 `"target_cache"`：运行 target-label cache 构建，
  output 通常是 `/mnt/output/opening_strength_fit/cache/*.parquet`。
- `[run].kind = "score_risk_sweep"`：对已有 prediction 做 score/risk penalty 或 hard gate 扫描，
  不训练新模型。
- `[run].kind = "exploration"`：可以先保持 active/running，不要求立刻有 metrics；确认后再归档成正式实验。
- Post-open 实验仍走 `scripts/run_experiment.py`；`[features].include_postopen_decision = true`
  打开 post-open decision 特征，v2 特征再加 `[features].include_postopen_v2 = true`。

PVC 约定：

```text
cache:      /mnt/output/opening_strength_fit/cache/*.parquet
run output: /mnt/output/opening_strength_fit/<run_id>/
local pull: output/predictions/<run_id>/predictions_all.parquet
```

`*.tmp.parquet`、`*.parquet.lock` 和 heartbeat 文件都是运行中状态，不能当训练输入。

## 4. 构建和 K8s

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

默认正式路径是 CPU LightGBM + PVC labeled cache。GPU 只在显式设置
`[model].device_type = "gpu"` 和 `[k8s.resources].gpu_limit` 时使用。

## 5. 同步产物

metrics 拉回、predictions 拉回、shard metrics 合并和轻量归档统一使用：

```bash
python scripts/sync_experiment_artifacts.py \
  --config experiments/runs/<run_id>.toml \
  --all

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

默认输出：

```text
output/k8s/metrics/<run_id>_metrics_by_year.csv
output/predictions/<run_id>/predictions_all.parquet
experiments/results/metrics/<run_id>_metrics_by_year.csv
```

## 6. 分析命令

Metrics：

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/<run_id>_metrics_by_year.csv

python scripts/compare_opening_results.py
```

标准 LightGBM delay replay：

```bash
python scripts/run_lgbm_delay_replays.py --check-interface-only
python scripts/run_lgbm_delay_replays.py
python scripts/plot_lgbm_delay_decay.py
```

Horizon decay：

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

Target-label cache build:

```bash
python scripts/build_target_label_cache.py \
  --config experiments/runs/<build_target_run_id>.toml
```

Existing-score TopN guard sweep:

```bash
python scripts/run_score_tail_guards.py \
  --input output/predictions/lgbm_delay2_postopen_0931_0940_baseline_v1/predictions_all.parquet \
  --next-close-label-input output/reports/lgbm_delay2_postopen_0931_0940_baseline_v1_four_panel/clickhouse_next_close_labels.parquet \
  --output-dir output/reports/lgbm_delay2_postopen_tail_guards_v1
```

Existing-score risk penalty sweep:

```bash
python scripts/run_score_risk_sweep.py \
  --config experiments/runs/score_risk_sweep_guard_shrunk_v1.toml \
  --output-dir output/local/score_risk_sweep_guard_shrunk_v1
```

Feature dependence audit：

```bash
python scripts/audit_feature_dependence.py \
  --config experiments/runs/lgbm_delay2_feature_dependence_v1.toml \
  --output-dir output/local/lgbm_delay2_feature_dependence_v1
```

## 7. 排查

- `field is immutable`：删除同名 Job 后重新 apply。
- K8s 内找不到新 config：重新 build/push 镜像，并重新 render Job。
- PVC cache 缺失：等待最终 `*.parquet`，不要使用 `.tmp` 或 lock 文件。
- replay 缺少上下文字段：传 `--context-input`，或先运行 interface check。
- completed config 没有 metrics：运行 `sync_experiment_artifacts.py --all`，然后 audit。
