# 运行手册

Scope: runnable configs, commands, paths, sync steps, and troubleshooting.

标准闭环：

```text
precheck -> render training job -> apply/wait -> render analysis job -> apply/wait -> sync compact artifacts -> audit/coverage
```

## 配置和输入

实验配置放在 `experiments/runs/*.toml`，K8s manifest 由 `osf-render-k8s-job` 写到
`experiments/jobs/`。历史 run、数字和归档路径见 [experiment_log.md](experiment_log.md)。

常用 PVC cache：

```text
base cache:
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_base_labeled_v2/

next-close labels:
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1/

mixed w030 cache:
/mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_mixed_w030_labeled_v1/
```

当前判断见 [project_brief.md](project_brief.md)。本 runbook 只保留操作步骤、命令模板和排查口径。

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
  --config experiments/runs/<run_id>.toml \
  --pool S \
  --output-dir output/legacy/analysis/<run_id>_pool_s_selection
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
  --labeled-output output/legacy/analysis/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet

osf-train \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --input output/legacy/analysis/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --input-kind labeled \
  --split-mode chronological \
  --test-start-date 2021-09-23 \
  --test-end-date 2021-09-23 \
  --feature-limit 80 \
  --top-n 2 \
  --output-dir output/legacy/analysis/gbm_opening_1y_next_month_multi_symbol_smoke

osf-summarize-opening-results \
  --input-dir output/legacy/analysis/gbm_opening_1y_next_month_multi_symbol_smoke
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
analysis:   /mnt/output/opening_strength_fit/<run_id>/analysis/pool_internal_top100/
local artifact pull: output/artifacts/<run_id>/  # compact analysis artifacts only
```

`*.tmp.parquet`、`*.parquet.lock` 和 heartbeat 文件表示 cache 正在写入。

当前 workflow 的正式落点：

```text
experiments/results/metrics/             # tracked training/evaluation metrics evidence
experiments/results/backtests/           # tracked compact analysis evidence
output/artifacts/<run_id>/               # current 2022-2025 cluster-side analysis local mirror
output/artifacts/_partial_metrics/       # ignored partial metrics from --allow-partial
```

`osf-sync-experiment-artifacts` 默认使用 run config 里的 `[output].local_dir` 作为
artifact 本地镜像；没有配置时才回落到 `output/artifacts/<run_id>/`。`output/artifacts/`
保持窄视图：只放当前 `2022-2025` baseline 和正在比较的 pool_L 优化实验。历史 artifact
拉取和 raw shard metrics 放到 `output/legacy/artifacts/`。

旧本地分析和 debug 输出统一收进 `output/legacy/`，不再占用顶层目录：

```text
output/legacy/predictions/<run_id>/       # optional prediction parquet pulls; delete/re-sync as needed
output/legacy/analysis/<name>/            # local smoke, scratch, and one-off analysis
output/legacy/labels/next_close_labels_*/ # local next-close label shards for legacy/debug
output/legacy/reports/<name>/             # standalone local reports and heavy diagnostics
```

## 5. 构建和 K8s

集群命令统一使用 `hfcli kubectl --cluster research ...`，namespace 使用 `bizewu`。

```bash
IMAGE_REPO=registry.corp.highfortfunds.com/bizewu/opening-strength-fit
VERSION=$(date +%Y%m%d)-lgbm-cpu-v1
docker build --build-arg CACHE_BUST=${VERSION} -t ${IMAGE_REPO}:${VERSION} .
docker push ${IMAGE_REPO}:${VERSION}

osf-render-k8s-job \
  --config experiments/runs/<run_id>.toml \
  --image ${IMAGE_REPO}:${VERSION}

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
  --image ${IMAGE_REPO}:${VERSION}

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
- 格式使用 `os-<model>-<window>-<period>-<target>-<display>`，例如
  `os-lgbm-36m-2026-w030-mainline`。
- 展示名使用短标签，方便在 K8s 和图表中追溯同一 run family。
- 不要依赖 renderer 自动生成的 `opening-strength-...-<hash>` 名字；这类 hash 名只作为旧运行的追溯信息。

```toml
[k8s]
job_name = "os-lgbm-36m-2026-w030-mainline"
shard_parallelism = 1
```

调整正在运行的 Indexed Job 并行度：

```bash
hfcli kubectl --cluster research patch job <job-name> \
  -n bizewu \
  -p '{"spec":{"parallelism":<parallelism>}}'
```

### 5.1 集群侧分析 Job

从 `2022-2025` 这一轮新实验开始，正式预测结果分析改为集群侧完成。训练 Job 只负责在
PVC 上写 metrics 和 raw predictions；pool-internal Top100 / Rank IC / plot data / SVG
由独立 analysis Job 在 PVC 和 S3 附近完成，本地只同步压缩后的 CSV / JSON / SVG。

TOML 中使用 `[analysis.pool_internal]` 声明分析契约：

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

analysis Job 会根据 config 的 rolling window 自动等待每个 `month_YYYY-MM/predictions.parquet`
出现，再读取：

```text
predictions:       /mnt/output/opening_strength_fit/<run_id>/
next-close labels: /mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1/
stock pools:       lml.bzw@ssd/data/pool_{S,M,L}.parquet
```

产物写在：

```text
/mnt/output/opening_strength_fit/<run_id>/analysis/pool_internal_top100/
```

### 5.2 Rolling 过程观测

Indexed Job 的 shard index 对应月份或半年窗口，使用：

```bash
osf-rolling-job-status \
  --config experiments/runs/<rolling_run_id>.toml
```

输出会列出 `index -> month -> pod -> phase`，并打印每个月对应的 log 命令。看最近日志：

```bash
osf-rolling-job-status \
  --config experiments/runs/<rolling_run_id>.toml \
  --tail 160
```

也可以直接跟随当前 pod：

```bash
hfcli kubectl --cluster research logs -n bizewu <pod-name> -f
```

指定 Job 名观测：

```bash
osf-rolling-job-status \
  --config experiments/runs/<rolling_run_id>.toml \
  --job-name <job-name>
```

## 6. 同步产物

metrics、cluster-side analysis artifacts 和轻量归档统一使用：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<run_id>.toml \
  --all

osf-audit-experiments
osf-check-project-contracts
```

`--all` 和无显式动作时不再拉取 prediction parquet；prediction 是训练/分析之间的 PVC 内部大文件，
本地 parquet 用后可以删除。
只有需要本地复盘旧流程或排查单个 shard 时，才显式加 `--predictions`：

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<rolling_run_id>.toml \
  --predictions --allow-partial
```

默认 metrics 直接写入 `experiments/results/metrics/`。`--allow-partial` 不写入
`experiments/results/`，metrics 自动落到 `output/artifacts/_partial_metrics/`，方便排查已完成 shard。
对半年 rolling，raw predictions 文件名使用窗口标签，例如
`predictions_2018-01_2018-06.parquet`，shard 目录仍以窗口起点命名为 `month_YYYY-MM/`。

默认输出：

```text
experiments/results/metrics/<run_id>_metrics_by_year.csv
experiments/results/metrics/<run_id>_metrics_by_month.csv
output/artifacts/<run_id>/pool_internal_summary.csv
output/artifacts/<run_id>/pool_internal_month_summary.csv
output/artifacts/<run_id>/pool_internal_clock_summary.csv
output/artifacts/<run_id>/pool_internal_halfyear_summary.csv
output/artifacts/<run_id>/pool_internal_year_summary.csv
output/artifacts/<run_id>/pool_internal_group_metrics.csv
output/artifacts/<run_id>/pool_internal_trace.json
output/artifacts/<run_id>/reports/**/*.csv
output/artifacts/<run_id>/reports/**/*.svg
experiments/results/backtests/<record-prefix>/pool_internal_*.csv
experiments/results/backtests/<record-prefix>/*_with_mean.svg
```

pool-internal 分析不再把 next-close label 拉到本地。正式 analysis Job 直接读取 PVC 上的
`next_close_label_input`。`--next-close-labels` 仅保留给旧本地分析和排查使用。

历史 `score_risk_sweep` 的轻量 artifact：

```text
output/legacy/artifacts/<run_id>/score_risk_summary.csv
output/legacy/artifacts/<run_id>/score_risk_minute_summary.csv
output/legacy/artifacts/<run_id>/score_risk_group_metrics.csv
output/legacy/artifacts/<run_id>/score_risk_trace.json
experiments/results/backtests/<run_id>_summary.csv
```

`alpha_conditioned_rolling_validation` 会同步 root-level
`rolling_summary.csv` / `rolling_month_summary.csv` / `rolling_group_metrics.csv`；root summary 缺失时，
sync 会拉取 `month_YYYY-MM/` shards 并本地合并。正式归档写到
`experiments/results/backtests/<run_id>/`。

`gap_risk_attribution` 同步 outcome / exposure / residual-control 轻量 CSV，并归档到
`experiments/results/backtests/<run_id>/`。

## 7. 分析命令

Metrics：

```bash
osf-summarize-opening-results \
  --metrics-csv experiments/results/metrics/<run_id>_metrics_by_year.csv

osf-compare-opening-results
```

2022-2025 universe + pool_L pool-internal 验收面板按第 5.1 节运行 analysis Job，再按第 6 节
`osf-sync-experiment-artifacts --all` 同步。核心输出：

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

2022-2025 主展示只使用 `universe` 和 `pool_L`，`plot_period = "quarter"`；short / next
excess + Rank IC 使用季度聚合，累计超额曲线优先用日度路径且横轴只标年份。不要把
`pool_S/M`、月度图或周度图作为主图。2020 年以前没有 S/M/L 股池文件；分析早期 shard 时
在 TOML 里把池限制到 universe：

```toml
[analysis.pool_internal]
pools = ["universe"]
```

该模式会输出 `<plot-prefix>_universe_*` SVG / plot data，不要求 S/M/L 股池存在。

本地 `osf-analyze-pool-internal-top100` 只作为 smoke / legacy fallback；正式结果不要再先拉
prediction parquet 到本地。需要小样本调试时，它仍可直接读取本地路径或 PVC 风格的
`month_*` / `year_*` 目录：

```bash
osf-analyze-pool-internal-top100 \
  --predictions /mnt/output/opening_strength_fit/<run_id> \
  --next-close-label-input /mnt/output/opening_strength_fit/cache/opening_13y_201301_202512_delay2_next_close_labels_v1 \
  --variant <variant_label> \
  --output-dir /mnt/output/opening_strength_fit/<run_id>/analysis/pool_internal_top100 \
  --report-dir /mnt/output/opening_strength_fit/<run_id>/analysis/pool_internal_top100/reports
```

本地归档由 `osf-sync-experiment-artifacts --all` 完成。

周度 4w rolling 只作为稳定性补充诊断，不是 baseline summary 默认口径。需要检查周度稳定性时，
可以单独从已存在的 `pool_internal_group_metrics.csv` 重画：

```bash
osf-plot-weekly-pool-internal \
  --group-metrics experiments/results/backtests/<run_id>_pool_internal_group_metrics.csv \
  --output-dir output/legacy/reports/<run_id>_weekly_trading_day_equal \
  --output-prefix <variant_label> \
  --plot-variant-label "<display label>" \
  --rolling-weeks 4
```

该脚本输出 `daily_pool_internal_summary.csv`、`weekly_pool_internal_summary.csv`、
`weekly_pool_internal_overall_summary.csv`、`weekly_worst_windows.csv` 和
`<plot-prefix>_universe_pool_l_weekly_rolling_4w/*.svg`。

周度图可作为附录；应先做日度聚合再按周求和后累加，避免把周均值直接累加导致尺度失真。

`predictions` 里不保留 `alpha_return_next_close`，这是训练防泄漏设计；cluster-side analysis
直接在 PVC 上 join next-close 年度 label，不再通过本地 label cache 完成正式分析。

Rolling short-vs-next tradeoff chart：

```bash
osf-plot-rolling-validation-tradeoff \
  --input experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1/month_summary.csv \
  --output-dir output/legacy/reports/rolling_alpha_conditioned_top100_validation_v1
```

Feature dependence audit：

```bash
osf-audit-feature-dependence \
  --config experiments/runs/lgbm_delay2_feature_dependence_v1.toml \
  --output-dir output/legacy/analysis/lgbm_delay2_feature_dependence_v1
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
  --output-root output/legacy/reports/opening_alpha_horizon_decay_delay2_clickhouse_point_0930_selected
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
