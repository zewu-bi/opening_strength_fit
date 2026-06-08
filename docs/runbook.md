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

36m formal 2024 rolling baseline:
experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1.toml
experiments/jobs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1_sharded_job.yaml

36m halfyear rolling job:
experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1.toml
experiments/jobs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_sharded_job.yaml

18m source feature/model config:
experiments/runs/lgbm_delay2_18m_postopen_mixed_w030_soft_core_reg_light_v1.toml
```

36m rolling 使用的 `2015-2024` cache：

```text
base cache:
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_base_labeled_v2/

next-close labels:
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1/

mixed w030 cache:
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_mixed_w030_labeled_v1/
```

当前判断、运行状态和实验结果见 [project_brief.md](project_brief.md) 和
[experiment_log.md](experiment_log.md)。本 runbook 只保留可执行配置、命令和路径。

## 1. 预检

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
set -a; . ./.env; set +a

osf-audit-experiments
osf-check-project-contracts
osf-probe-clickhouse-data --schema --field-notes
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
cd "$(git rev-parse --show-toplevel)"
set -a; . ./.env; set +a
```

项目原生支持 `bucket@ssd/path.parquet` 形式的股池路径。需要快速核对 Ceph 文件时，可以复用
`xy_fit` 的 venv：

```bash
cd /home/hefu/projects/xy_fit
set -a; . "$(git rev-parse --show-toplevel)/.env"; set +a

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
osf-train \
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
osf-inspect-dataset \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet

osf-train \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --input output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --input-kind labeled \
  --split-mode chronological \
  --test-start-date 2021-09-23 \
  --test-end-date 2021-09-23 \
  --feature-limit 80 \
  --top-n 2 \
  --output-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke

osf-summarize-opening-results \
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

osf-render-k8s-job \
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
osf-render-k8s-job \
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

K8s Job 命名约定：

- sharded rolling 必须在 TOML 的 `[k8s]` 里显式设置短 `job_name`。
- 格式使用 `os-<model>-<window>-<year>-<target>-<display>`，例如
  `os-lgbm-36m-2023-w030-baseline`。
- 展示名沿用文档里的模型别名，例如当前正式模型叫 `baseline`。
- 不要依赖 renderer 自动生成的 `opening-strength-...-<hash>` 名字；这类 hash 名只作为旧运行的追溯信息。

```toml
[k8s]
job_name = "os-lgbm-36m-2023-w030-baseline"
shard_parallelism = 1
```

### 5.1 2024 月度 rolling baseline

正式 baseline run 的执行入口：

```text
display: baseline
run_id: lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1
job:    os-lgbm-36m-2024-w030-baseline
image:  registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260604-36m-soft-core-v1
```

渲染、检查和启动：

```bash
osf-render-k8s-job \
  --config experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1.toml \
  --sharded \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260604-36m-soft-core-v1
```

```bash
hfcli kubectl --cluster research apply --dry-run=client \
  -f experiments/jobs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1_sharded_job.yaml

hfcli kubectl --cluster research delete job os-lgbm-36m-2024-w030-baseline \
  --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply \
  -f experiments/jobs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1_sharded_job.yaml
```

重新启动或补跑前先 build/push 当前 clean worktree 到 manifest 里的 image tag，或重新 render manifest 使用新 tag。

### 5.2 2018-2024 半年 rolling job

正式 baseline 的半年 rolling 执行入口：

```text
display: baseline halfyear
run_id: lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1
job:    os-lgbm-36m-2018-2024-w030-halfyear
image:  registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260605-halfyear-v1
```

该 run 使用 `36m train -> next 6m test`，`2018-01` 至 `2024-12` 共 14 个半年 Indexed Job
shards。每个 shard 只训练一次，并预测对应的 6 个月 OOS 窗口：

```text
2018-01..2018-06
2018-07..2018-12
...
2024-01..2024-06
2024-07..2024-12
```

渲染和提交前检查：

```bash
osf-render-k8s-job \
  --config experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1.toml \
  --sharded \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:opening-strength-fit-20260605-halfyear-v1

hfcli kubectl --cluster research apply --dry-run=client \
  -f experiments/jobs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_sharded_job.yaml
```

启动或重启：

```bash
hfcli kubectl --cluster research delete job os-lgbm-36m-2018-2024-w030-halfyear \
  --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply \
  -f experiments/jobs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1_sharded_job.yaml
```

调整正在运行的 Indexed Job 并行度：

```bash
hfcli kubectl --cluster research patch job os-lgbm-36m-2018-2024-w030-halfyear \
  -n bizewu \
  -p '{"spec":{"parallelism":<parallelism>}}'
```

### 5.3 Rolling 过程观测

Indexed Job 的 shard index 对应月份或半年窗口，使用：

```bash
osf-rolling-job-status \
  --config experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1.toml
```

输出会列出 `index -> month -> pod -> phase`，并打印每个月对应的 log 命令。看最近日志：

```bash
osf-rolling-job-status \
  --config experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1.toml \
  --tail 160
```

也可以直接跟随当前 pod：

```bash
hfcli kubectl --cluster research logs -n bizewu <pod-name> -f
```

半年 rolling 观测：

```bash
osf-rolling-job-status \
  --config experiments/runs/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2018_2024_halfyear_rolling_v1.toml \
  --job-name os-lgbm-36m-2018-2024-w030-halfyear
```

## 6. 同步产物

metrics、predictions、shard metrics 合并和轻量归档统一使用：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<run_id>.toml \
  --all

osf-audit-experiments
osf-check-project-contracts
```

Rolling 仍在运行时，只同步已完成月份：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<rolling_run_id>.toml \
  --metrics --predictions --allow-partial
```

`--allow-partial` 不写入 `experiments/results/`，只更新 `output/k8s/metrics/` 和
`output/predictions/<run_id>/raw/predictions_YYYY-MM.parquet`，方便每个 shard/window 完成后立即分析。
对半年 rolling，raw predictions 文件名使用窗口标签，例如
`predictions_2018-01_2018-06.parquet`，shard 目录仍以窗口起点命名为 `month_YYYY-MM/`。

默认输出：

```text
output/k8s/metrics/<run_id>_metrics_by_year.csv
output/k8s/metrics/<run_id>_metrics_by_month.csv
output/predictions/<run_id>/predictions_all.parquet
experiments/results/metrics/<run_id>_metrics_by_year.csv
experiments/results/metrics/<run_id>_metrics_by_month.csv
```

pool-internal 分析需要 next-close 年度 label 时，可以复用同一个 sync helper pod 拉取：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<run_id>.toml \
  --next-close-labels
```

该命令按 config 的 test window 自动推断年份，输出到：

```text
output/local/next_close_labels_<year-or-range>/
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
osf-summarize-opening-results \
  --metrics-csv output/k8s/metrics/<run_id>_metrics_by_year.csv

osf-compare-opening-results
```

正式 36m rolling 的 universe/S/M/L pool-internal 验收面板：

```bash
osf-analyze-pool-internal-top100 \
  --predictions output/predictions/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1 \
  --next-close-label-input output/local/next_close_labels_2021_2024 \
  --run-id lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1 \
  --variant baseline \
  --output-dir output/local/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1_pool_internal \
  --report-dir output/reports/lgbm_delay2_36m_visible_mixed_w030_soft_core_reg_light_2024_rolling_v1_pool_internal \
  --plot-prefix baseline \
  --plot-variant-label baseline
```

该脚本输出：

```text
pool_internal_summary.csv
pool_internal_month_summary.csv
pool_internal_clock_summary.csv
pool_internal_halfyear_summary.csv
pool_internal_year_summary.csv
pool_internal_group_metrics.csv
pool_internal_trace.json
<plot-prefix>_universe_sml_pool_internal_with_mean/*.svg
<plot-prefix>_universe_sml_rank_ic_with_mean/*.svg
```

S/M/L 股池文件当前覆盖 `2020-01-02` 至 `2025-12-31`。2020 年之前没有 S/M/L 股池文件；
分析 halfyear 任务的 2018/2019 shard 时只跑 universe 口径：

```bash
osf-analyze-pool-internal-top100 \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2018-01_2018-06.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2018-07_2018-12.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2019-01_2019-06.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2019-07_2019-12.parquet \
  --next-close-label-input output/local/next_close_labels_2018_2019 \
  --pool universe \
  --variant baseline_pre2020_universe \
  --output-dir output/local/<halfyear_run_id>_pre2020_universe_pool_internal \
  --report-dir output/reports/<halfyear_run_id>_pre2020_universe_pool_internal
```

该模式会输出 `<plot-prefix>_universe_*` SVG / plot data，不要求 S/M/L 股池存在。

2020-2024 halfyear shard 可以跑完整 universe / S / M / L：

```bash
osf-analyze-pool-internal-top100 \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2020-01_2020-06.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2020-07_2020-12.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2021-01_2021-06.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2021-07_2021-12.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2022-01_2022-06.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2022-07_2022-12.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2023-01_2023-06.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2023-07_2023-12.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2024-01_2024-06.parquet \
  --predictions output/predictions/<halfyear_run_id>/raw/predictions_2024-07_2024-12.parquet \
  --next-close-label-input output/local/next_close_labels_2020_2024 \
  --variant baseline_halfyear_2020_2024 \
  --output-dir output/local/<halfyear_run_id>_pool_internal_2020_2024 \
  --report-dir output/reports/<halfyear_run_id>_pool_internal_2020_2024 \
  --weekly-report-dir output/reports/<halfyear_run_id>_weekly_2020_2024_trading_day_equal \
  --weekly-output-prefix baseline_halfyear_2020_2024 \
  --weekly-rolling-weeks 4 \
  --records-dir experiments/results \
  --record-prefix <halfyear_run_id>_2020_2024
```

传入 `--weekly-report-dir` 时，命令会顺手生成 4 周滚动周度诊断：先把同一 `pool x date`
的多个决策点聚成日度均值，再按交易日等权生成周度和滚动窗口，避免 1 个交易日的节假日周与
5 个交易日的正常周等权。传入 `--records-dir` / `--record-prefix` 时，会同步归档轻量
pool-internal CSV 和 SVG 到 `experiments/results/backtests/`。

仍然可以单独从已存在的 `pool_internal_group_metrics.csv` 重画周度图：

```bash
osf-plot-weekly-pool-internal \
  --group-metrics experiments/results/backtests/<halfyear_run_id>_2020_2024_pool_internal_group_metrics.csv \
  --output-dir output/reports/<halfyear_run_id>_weekly_2020_2024_trading_day_equal \
  --output-prefix baseline_halfyear_2020_2024 \
  --plot-variant-label "baseline halfyear 2020-2024" \
  --rolling-weeks 4
```

该脚本输出 `daily_pool_internal_summary.csv`、`weekly_pool_internal_summary.csv`、
`weekly_pool_internal_overall_summary.csv`、`weekly_worst_windows.csv` 和
`<plot-prefix>_universe_sml_weekly_rolling_4w/*.svg`。

`predictions` 里不保留 `alpha_return_next_close`，这是训练防泄漏设计；分析前需要把 PVC 上的 next-close 年度
label 小文件拉到本地；优先使用第 6 节的 `osf-sync-experiment-artifacts --next-close-labels`。

Rolling short-vs-next tradeoff chart：

```bash
osf-plot-rolling-validation-tradeoff \
  --input experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1_month_summary.csv \
  --output-dir output/reports/rolling_alpha_conditioned_top100_validation_v1
```

Feature dependence audit：

```bash
osf-audit-feature-dependence \
  --config experiments/runs/lgbm_delay2_feature_dependence_v1.toml \
  --output-dir output/local/lgbm_delay2_feature_dependence_v1
```

Replay / horizon diagnostics：

```bash
osf-run-lgbm-delay-replays --check-interface-only
osf-run-lgbm-delay-replays
osf-plot-lgbm-delay-decay

osf-run-alpha-horizon-decay \
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
| completed config 没有 metrics | 运行 `osf-sync-experiment-artifacts --all`，然后 audit。 |
