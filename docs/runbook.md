# 运行手册

Scope: runnable configs, commands, paths, sync steps, and troubleshooting.

标准闭环：

```text
precheck -> render job -> apply/wait -> sync artifacts -> audit/coverage -> analysis
```

## 当前可运行配置

```text
36m smoke:
experiments/runs/lgbm_delay2_36m_visible_mixed_w030_2024_smoke_v1.toml

18m source feature/model config:
experiments/runs/lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_light_v1.toml
```

36m rolling 使用的 `2021-2024` cache：

```text
base cache:
/mnt/output/opening_strength_fit/cache/opening_10y_201501_202412_delay2_base_labeled_v2/

next-close labels:
/mnt/output/opening_strength_fit/cache/opening_10y_201501_202412_delay2_next_close_labels_v1/

mixed w030 cache:
/mnt/output/opening_strength_fit/cache/opening_10y_201501_202412_delay2_mixed_w030_labeled_v1/
```

创建 2024 全年 12-shard rolling run 时，复用 `soft_core_reg_light` 的 feature include/drop 规则和
LightGBM 参数。

## 1. 预检

```bash
cd /home/hefu/projects/opening_strength_fit
source .venv/bin/activate
set -a; . ./.env; set +a

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

## 2. 外部股池

mentor 发来的股池和隔壁 `xy_fit` 项目的 `X.parquet` / `Y.parquet` 在同一个 Ceph S3 目录：

```text
bucket:   lml.bzw@ssd
endpoint: http://ceph-s3-ssd.prod.highfortfunds.com
prefix:   data/

data/pool_L.parquet
data/pool_M.parquet
data/pool_S.parquet
```

本地读取使用司令部 LDAP 凭据。把真实值写在项目根目录的 `.env`；`.env` 已在 `.gitignore` 中。

```bash
CEPH_LDAP_ID='your_headquarter_username'
CEPH_LDAP_KEY='your_headquarter_password'
```

加载环境变量：

```bash
cd /home/hefu/projects/opening_strength_fit
set -a; . ./.env; set +a
```

项目原生支持 `bucket@ssd/path.parquet` 形式的股池路径。需要快速核对 Ceph 文件时，可以复用
`xy_fit` 的 venv：

```bash
cd /home/hefu/projects/xy_fit
set -a; . /home/hefu/projects/opening_strength_fit/.env; set +a

.venv/bin/python - <<'PY'
from xyfit.io import build_client

client = build_client()
resp = client.list_objects_v2(Bucket="lml.bzw", Prefix="data/")
for item in sorted(resp.get("Contents", []), key=lambda x: x["LastModified"], reverse=True):
    print(item["LastModified"], item["Size"], item["Key"])
PY
```

CLI 快速试验：

```bash
python scripts/run_experiment.py \
  --config experiments/runs/lgbm_delay2_postopen_0931_0940_baseline_v1.toml \
  --pool S \
  --output-dir output/local/lgbm_delay2_postopen_pool_s_selection
```

`--pool S|M|L` 映射到：

```text
S -> lml.bzw@ssd/data/pool_S.parquet
M -> lml.bzw@ssd/data/pool_M.parquet
L -> lml.bzw@ssd/data/pool_L.parquet
```

默认语义是 `filter_train=false`、`filter_selection=true`：模型在 full universe 上训练和打分，
TopN 从池内候选里选。保守日期口径加：

```bash
--pool S --pool-date-lag-sessions 1
```

TOML 模板见
[experiments/config_templates/stock_pool_selection.toml](../experiments/config_templates/stock_pool_selection.toml)。

开启 `filter_selection=true` 后，`metrics_by_year.csv` 的 TopN 汇总使用池内候选行；
`predictions*.parquet` 保留全 universe 打分并额外写出 `stock_pool_member`。同时输出：

```text
score_buckets_<period>_stock_pool.csv
score_buckets_stock_pool.csv
```

## 3. 本地 Smoke

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

长窗口 labeled dataset 走 PVC cache 或专门的 cache build Job。

## 4. 实验配置

正式实验通常对应三类文件：

```text
experiments/runs/<run_id>.toml
experiments/jobs/<run_id>_job.yaml
experiments/results/metrics/<run_id>_metrics_by_year.csv
```

Run kind 映射见 [experiments/README.md](../experiments/README.md)。TOML 模板见
[experiments/config_templates/](../experiments/config_templates/)。

PVC 约定：

```text
cache:      /mnt/output/opening_strength_fit/cache/*.parquet
run output: /mnt/output/opening_strength_fit/<run_id>/
local pull: output/predictions/<run_id>/predictions_all.parquet
```

`*.tmp.parquet`、`*.parquet.lock` 和 heartbeat 文件表示 cache 正在写入。

## 5. 构建和 K8s

集群命令统一使用 `hfcli kubectl --cluster research ...`，namespace 使用 `bizewu`。

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

默认正式路径是 CPU LightGBM + PVC labeled cache。GPU TOML 模板见
[experiments/config_templates/gpu_lightgbm.toml](../experiments/config_templates/gpu_lightgbm.toml)。

monthly rolling 或长窗口任务使用 sharded Job：

```bash
python scripts/render_k8s_job.py \
  --config experiments/runs/<run_id>.toml \
  --sharded \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

hfcli kubectl --cluster research delete job opening-strength-<run-slug>-sharded --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_sharded_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/<rendered-sharded-job-name> -n bizewu --timeout=24h
```

Indexed Job 的每个 shard 写独立子目录，例如：

```text
month_YYYY-MM/
year_YYYY/
```

并行度由 `[k8s].shard_parallelism` 控制。

## 6. 同步产物

metrics、predictions、shard metrics 合并和轻量归档统一使用：

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

`score_risk_sweep` 的轻量 artifact：

```text
output/local/<run_id>/score_risk_summary.csv
output/local/<run_id>/score_risk_minute_summary.csv
output/local/<run_id>/score_risk_group_metrics.csv
output/local/<run_id>/score_risk_trace.json
experiments/results/backtests/<run_id>_summary.csv
```

`alpha_conditioned_rolling_validation` 会同步 root-level
`rolling_summary.csv` / `rolling_month_summary.csv` / `rolling_group_metrics.csv`；root summary 缺失时，
sync 会拉取 `month_YYYY-MM/` shards 并本地合并。

`gap_risk_attribution` 同步 outcome / exposure / residual-control 轻量 CSV。

## 7. 分析命令

Metrics：

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/<run_id>_metrics_by_year.csv

python scripts/compare_opening_results.py
```

Rolling short-vs-next tradeoff chart：

```bash
python scripts/plot_rolling_validation_tradeoff.py \
  --input experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1_month_summary.csv \
  --output-dir output/reports/rolling_alpha_conditioned_top100_validation_v1
```

Feature dependence audit：

```bash
python scripts/audit_feature_dependence.py \
  --config experiments/runs/lgbm_delay2_feature_dependence_v1.toml \
  --output-dir output/local/lgbm_delay2_feature_dependence_v1
```

Replay / horizon diagnostics：

```bash
python scripts/run_lgbm_delay_replays.py --check-interface-only
python scripts/run_lgbm_delay_replays.py
python scripts/plot_lgbm_delay_decay.py

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

## 8. 排查

| symptom | action |
| --- | --- |
| `kubectl` 没有 current-context | 使用 `hfcli kubectl --cluster research ...`。 |
| PVC API 被 RBAC 拒绝 | 用 Pod/Job yaml 的 `claimName` 和容器内 `/mnt/output` 检查文件状态。 |
| `field is immutable` | 重新创建同名 Job 后 apply。 |
| K8s 内找不到新 config | 重新 build/push 镜像，并重新 render Job。 |
| cache 只有 `.tmp` / lock / heartbeat | 等待最终 `.parquet` 和 manifest。 |
| replay 缺少上下文字段 | 传 `--context-input`，或先运行 interface check。 |
| completed config 没有 metrics | 运行 `sync_experiment_artifacts.py --all`，然后 audit。 |
