# 运行手册

Scope: runnable configs, commands, paths, sync steps, and troubleshooting. 研究判断见
[project_brief.md](project_brief.md)，实验记录见 [experiment_log.md](experiment_log.md)。

标准闭环：

```text
precheck -> render training job -> apply/wait -> render analysis job -> apply/wait -> sync compact artifacts -> audit
```

## 0. 环境和预检

```bash
cd ~/projects/opening_strength_fit
source .venv/bin/activate
set -a; . ./.env; set +a

osf-audit-experiments
osf-check-project-contracts
osf-probe-clickhouse-data --schema --field-notes
```

## 1. 配置和路径

实验配置放在：

```text
experiments/runs/<run_id>.toml
experiments/jobs/<run_id>_job.yaml
experiments/results/metrics/<run_id>_metrics_by_year.csv
```

常用 PVC cache：

```text
base cache:
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_base_labeled_v2/

next-close labels:
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1/

mixed w030 cache:
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_mixed_w030_labeled_v1/
```

PVC 输出约定：

```text
run output: /mnt/output/opening_strength_fit/<run_id>/
analysis:   /mnt/output/opening_strength_fit/<run_id>/analysis/pool_internal_top100/
local pull: output/artifacts/<run_id>/
```

Tracked evidence 写入 `experiments/results/{metrics,backtests}/`。旧 pulls、prediction parquet、
本地分析、label shards 和重报告放 `output/legacy/`。

Run kind 映射见 [experiments/README.md](../experiments/README.md)。TOML 模板见
[experiments/config_templates/](../experiments/config_templates/)。

`*.tmp.parquet`、`*.parquet.lock` 和 heartbeat 文件表示 cache 正在写入。

## 2. 外部股池

Ceph S3：

```text
bucket:   lml.bzw@ssd
endpoint: http://ceph-s3-ssd.prod.highfortfunds.com
prefix:   data/

data/pool_L.parquet
data/pool_M.parquet
data/pool_S.parquet
```

`.env` 中放司令部 LDAP 凭据：

```bash
CEPH_LDAP_ID='your_headquarter_username'
CEPH_LDAP_KEY='your_headquarter_password'
```

需要快速核对 Ceph 文件时，可以复用 `xy_fit` 的 venv 和 `xyfit.io.build_client()` 列
`Bucket="lml.bzw", Prefix="data/"`。

CLI 映射：

```text
S -> lml.bzw@ssd/data/pool_S.parquet
M -> lml.bzw@ssd/data/pool_M.parquet
L -> lml.bzw@ssd/data/pool_L.parquet
```

默认语义：`filter_train=false`、`filter_selection=true`。模型在 full universe 上训练和打分，
TopN 从池内候选里选。保守日期口径加：

```bash
--pool L --pool-date-lag-sessions 1
```

TOML 模板见
[experiments/config_templates/stock_pool_selection.toml](../experiments/config_templates/stock_pool_selection.toml)。

开启 `filter_selection=true` 后，TopN 汇总使用池内候选行；`predictions*.parquet` 保留全
universe 打分并额外写出 `stock_pool_member` 和 stock-pool score buckets。

## 3. 构建镜像

集群命令统一使用 `hfcli kubectl --cluster research ...`，namespace 是 `bizewu`。

```bash
IMAGE_REPO=registry.corp.highfortfunds.com/bizewu/opening-strength-fit
VERSION=$(date +%Y%m%d)-lgbm-cpu-v1

docker build --build-arg CACHE_BUST=${VERSION} -t ${IMAGE_REPO}:${VERSION} .
docker push ${IMAGE_REPO}:${VERSION}
```

GPU TOML 模板见 [experiments/config_templates/gpu_lightgbm.toml](../experiments/config_templates/gpu_lightgbm.toml)。

## 4. 训练 Job

普通 Job：

```bash
osf-render-k8s-job \
  --config experiments/runs/<run_id>.toml \
  --image ${IMAGE_REPO}:${VERSION}

hfcli kubectl --cluster research apply --dry-run=client -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research delete job opening-strength-<run-slug> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/opening-strength-<run-slug> -n bizewu --timeout=24h
```

Rolling / sharded Job：

```bash
osf-render-k8s-job \
  --config experiments/runs/<run_id>.toml \
  --sharded \
  --image ${IMAGE_REPO}:${VERSION}

hfcli kubectl --cluster research delete job opening-strength-<run-slug>-sharded --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_sharded_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/<rendered-sharded-job-name> -n bizewu --timeout=24h
```

Indexed Job 的每个 shard 写独立子目录，例如 `month_YYYY-MM/` 或 `year_YYYY/`。

TOML 中手动设置短 job name：

```toml
[k8s]
job_name = "os-lgbm-36m-2225-mainline"
shard_parallelism = 1
```

命名格式使用 `os-<model>-<window>-<period>-<target>-<display>`，例如
`os-lgbm-36m-2225-mainline`。

调整并行度：

```bash
hfcli kubectl --cluster research patch job <job-name> \
  -n bizewu \
  -p '{"spec":{"parallelism":<parallelism>}}'
```

观察 rolling：

```bash
osf-rolling-job-status --config experiments/runs/<rolling_run_id>.toml
osf-rolling-job-status --config experiments/runs/<rolling_run_id>.toml --tail 160
osf-rolling-job-status --config experiments/runs/<rolling_run_id>.toml --job-name <job-name>
hfcli kubectl --cluster research logs -n bizewu <pod-name> -f
```

## 5. Pool-Internal 分析 Job

正式 pool-internal Top100 / Rank IC / plot data / SVG 由独立 analysis Job 在集群侧完成。

TOML：

```toml
[analysis.pool_internal]
enabled = true
job_name = "os-analyze-36m-2225-<variant>"
variant = "<variant>"
plot_prefix = "<variant>"
plot_variant_label = "2022-2025 <variant>"
plot_period = "quarter"
pools = ["universe", "L"]
top_n = 100
next_close_label_input = "/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1"
env_secrets = ["opening-strength-clickhouse", "xy-fit-ceph-credentials"]

[analysis.pool_internal.resources]
cpu_request = "8"
cpu_limit = "16"
memory_request = "256Gi"
memory_limit = "384Gi"
```

渲染和提交：

```bash
osf-render-k8s-job \
  --config experiments/runs/<run_id>.toml \
  --analysis \
  --image ${IMAGE_REPO}:${VERSION}

hfcli kubectl --cluster research apply --dry-run=client -f experiments/jobs/<run_id>_pool_internal_analysis_job.yaml
hfcli kubectl --cluster research delete job <analysis-job-name> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_pool_internal_analysis_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/<analysis-job-name> -n bizewu --timeout=24h
```

输入和输出：

```text
predictions:       /mnt/output/opening_strength_fit/<run_id>/
next-close labels: /mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1/
stock pools:       lml.bzw@ssd/data/pool_{S,M,L}.parquet
output:            /mnt/output/opening_strength_fit/<run_id>/analysis/pool_internal_top100/
```

analysis Job 会根据 config 的 rolling window 等待每个 `month_YYYY-MM/predictions.parquet`
出现，再开始分析。

2020 年以前没有 S/M/L 股池日期；早期 shard 分析使用：

```toml
[analysis.pool_internal]
pools = ["universe"]
```

## 6. 同步和归档

同步 compact artifacts：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<run_id>.toml \
  --all

osf-audit-experiments
osf-check-project-contracts
```

主要输出：`experiments/results/metrics/`、`output/artifacts/<run_id>/`、
`experiments/results/backtests/<record-prefix>/`。

常见文件：

```text
experiments/results/metrics/<run_id>_metrics_by_year.csv
experiments/results/metrics/<run_id>_metrics_by_month.csv
output/artifacts/<run_id>/pool_internal_{summary,month_summary,clock_summary,halfyear_summary,year_summary,group_metrics,trace}.*
output/artifacts/<run_id>/reports/**/*.{csv,svg}
experiments/results/backtests/<record-prefix>/{pool_internal_*.csv,*_with_mean.svg}
```

调试单个 shard 才拉 prediction parquet：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<rolling_run_id>.toml \
  --predictions --allow-partial
```

`--allow-partial` 写入 `output/artifacts/_partial_metrics/`。

非标准轻量 artifact：`score_risk_sweep`、`alpha_conditioned_rolling_validation`、
`gap_risk_attribution` 会同步各自 summary/trace CSV。

## 7. 汇总和作图

Metrics：

```bash
osf-summarize-opening-results \
  --metrics-csv experiments/results/metrics/<run_id>_metrics_by_year.csv

osf-compare-opening-results
```

2022-2025 主展示使用 universe + `pool_L`，`plot_period = "quarter"`。核心输出：

```text
pool_internal_summary.csv
pool_internal_month_summary.csv
pool_internal_clock_summary.csv
pool_internal_halfyear_summary.csv
pool_internal_year_summary.csv
pool_internal_group_metrics.csv
pool_internal_trace.json
<plot-prefix>_universe_pool_l_short_excess_rank_ic_with_mean/*.svg
<plot-prefix>_universe_pool_l_next_excess_rank_ic_with_mean/*.svg
reports/cumulative/<plot-prefix>_universe_pool_l_daily_cumulative.svg
```

周度 4w rolling 稳定性补充：

```bash
osf-plot-weekly-pool-internal \
  --group-metrics experiments/results/backtests/<run_id>_pool_internal_group_metrics.csv \
  --output-dir output/legacy/reports/<run_id>_weekly_trading_day_equal \
  --output-prefix <variant_label> \
  --plot-variant-label "<display label>" \
  --rolling-weeks 4
```

Legacy diagnostics entrypoints: `osf-plot-rolling-validation-tradeoff`、
`osf-audit-feature-dependence`、`osf-run-lgbm-delay-replays`、`osf-plot-lgbm-delay-decay`、
`osf-run-alpha-horizon-decay`。

## 8. 排查

| symptom | action |
| --- | --- |
| `kubectl` 没有 current-context | 使用 `hfcli kubectl --cluster research ...`。 |
| PVC API 被 RBAC 拒绝 | 用 Pod/Job yaml 的 `claimName` 和容器内 `/mnt/output` 检查文件状态。 |
| `field is immutable` | 删除同名 Job 后重新 apply。 |
| K8s 内找不到新 config | 重新 build/push 镜像，并重新 render Job。 |
| cache 只有 `.tmp` / lock / heartbeat | 等待最终 `.parquet` 和 manifest。 |
| replay 缺少上下文字段 | 传 `--context-input`，或先运行 interface check。 |
| completed config 没有 metrics | 运行 `osf-sync-experiment-artifacts --all`，然后 audit。 |
