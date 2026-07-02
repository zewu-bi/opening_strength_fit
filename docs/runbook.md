# 运行手册

本文件只放可执行流程：配置路径、K8s Job、artifact sync、analysis 命令和排查。不记录研究判断或实验复盘；
研究判断见 [project_brief.md](project_brief.md)，实验事实源见 [experiment_log.md](experiment_log.md)。

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

Compact artifacts 同步到 `experiments/results/{metrics,backtests}/`，该目录默认被 Git 忽略；
正式事实索引写入 [experiment_log.md](experiment_log.md)。旧 pulls、prediction parquet、本地分析、
label shards 和重报告放 `output/legacy/`。

大规模 analysis / attribution / audit 默认在集群侧读取 PVC 上的 prediction parquet 和 cache，
只把 summary、group metrics、trace、报告图等 compact artifacts 同步回本地。只有调试单个
shard 或排查 schema/坏行时，才把 prediction parquet 拉到本地 `output/`。

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
`experiments/results/backtests/<record-prefix>/`。其中 `experiments/results/` 是本地 compact
归档根目录，默认被 Git 忽略；正式事实索引同步写入 [experiment_log.md](experiment_log.md)。

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
`gap_risk_attribution`、`feature_hygiene` 会同步各自 summary/trace CSV 或审计报告。

## 7. 汇总和作图

Metrics：

```bash
osf-summarize-opening-results \
  --metrics-csv experiments/results/metrics/<run_id>_metrics_by_year.csv

osf-compare-opening-results
```

2022-2025 单个 run 的 pool-internal 分析仍使用 universe + `pool_L`，`plot_period = "quarter"`。
这些文件用于 drilldown 和验收图的数据源：

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

固定研究流程：尝试新的特征工程或模型优化，render K8s Job，在集群上重新训练和
pool-internal analysis，同步轻量 artifacts 后，用 `osf-plot-optimization-direction-comparison`
生成两张验收图。

默认输出目录：

```text
experiments/results/backtests/optimization_overlay_acceptance_2022_2025/
```

验收图：

```text
short rank IC和next pool_L 超额: optimization_directions_overlay_acceptance.svg
池内Top100隔夜收益累和: optimization_directions_net_alpha_cumulative.svg
```

第一张图是主验收：上 panel 用 `universe short Rank IC` 检查开盘短期模型本身；下 panel 用
`pool_L Top100 next internal excess` 检查叠加 mentor 股池后的 overnight overlay 效果。
默认图上画 baseline、hist_surprise 和 path_shape，也可以通过重复 `--direction key=label=run_id`
选择 1-3 个新的 comparison models；baseline 始终由 `--baseline-run-id` 提供，不需要作为
`--direction` 传入。柱顶标数值。
不再主看 short excess、`pool_L` short IC、universe next excess 或 next IC。

第二张 cumulative 图只画 next：上 panel 是全 A 股市场平均、扣费后的 `pool_L`
background、baseline Top100 和 comparison models Top100 的累计收益；下 panel 是扣费后的
`pool_L` background、baseline 和 comparison models 相对全 A 股市场平均的累计 alpha。
本仓库目前没有公司回测 API 封装；未来若接入公司回测 API，可替换 background 数据源。
默认扣费口径为 A 股 all-in round-trip 估计 `8 bps`；如需 stress 旧口径可传
`--realized-fee-bps 5`。底层来自 daily pool-internal summary，图上保留日频累计点。

选择新模型的例子：

```bash
osf-plot-optimization-direction-comparison \
  --output-dir experiments/results/backtests/<comparison_dir> \
  --direction scale_norm=scale_norm=<scale_norm_run_id>
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
`osf-audit-feature-dependence`、`osf-run-alpha-horizon-decay`。

## 8. Capacity Audit

生产化容量验收使用 `osf-audit-capacity`。它按每个 `date x decision_time`
在候选池内按 score 由高到低分配目标资金，显式限制单票权重、可见成交额参与率、盘口深度参与率、
行业权重，并报告未填满资金而不是强行满仓。
`target_notional` 是单个 `date x decision_time` group 的目标资金；若策略总资金需要分成
`N` 个执行切片，应先用总资金除以 `N`，例如 `10 亿 / 20 = 5000 万`。

正式大样本 audit 走第 1 节的集群侧原则；下面的本地命令只用于小样本 debug：

```bash
osf-audit-capacity \
  --predictions output/legacy/predictions/<run_id>/predictions_all.parquet \
  --output-dir output/legacy/analysis/<run_id>_capacity_audit \
  --pool L \
  --target-notional 50000000 \
  --capacity-notional-col turnover_diff_30t \
  --max-participation-rate 0.10 \
  --max-symbol-weight 0.01
```

常用输出：

```text
capacity_audit_summary.csv          # pool 级 fill / return / concentration 摘要
capacity_audit_month_summary.csv    # 月度稳定性
capacity_audit_daily_summary.csv    # 日度容量和收益摘要
capacity_audit_group_metrics.csv    # date x clock 组合构造结果
capacity_audit_selected.csv         # 逐笔入选 notional / weight / participation
capacity_audit_trace.json
```

容量主读字段：

```text
fill_success_rate          # 截面塞满目标资金的比例
mean_top_depth_to_target   # 塞满目标资金平均要吃到 score 排名第几
p95_top_depth_to_target    # 95% 截面塞满所需的 top depth
max_top_depth_to_target    # 最深一次塞满所需的 top depth
```

K8s run config 可设置：

```toml
[run]
kind = "capacity_audit"

[capacity_audit]
predictions = ["/mnt/output/opening_strength_fit/<source_run_id>"]
pool = ["L"]
target_notional = 50000000
capacity_notional_col = "turnover_diff_30t"
max_participation_rate = 0.10
max_symbol_weight = 0.01
ask_depth_levels = 0
```

正式 split20 容量验收的当前事实数字放在 [project_brief.md](project_brief.md) 和
[experiment_log.md](experiment_log.md)。本手册只保留标准配置口径和字段读法：

```text
total capital example: 1,000,000,000 split into 20 execution slices
target_notional: 50,000,000 per date x clock and slice
capacity_notional_col: turnover_diff_30t
max_participation_rate: 0.10
max_symbol_weight: 0.01
ask_depth_levels: 0
```

读法：`fill_success_rate` 看截面是否能塞满目标资金；`mean/p95/max_top_depth_to_target`
看需要沿 score 排名往后拿多深。若 Top100 覆盖不足但较深排名可以塞满，结论应写成
“容量组合可行但不能硬卡固定 Top100”。

如果 next-close label 不在 prediction parquet 中，可用 `label_input` 传 keyed label frame，
并设置 `label_col`。若 ADV、市值、行业或额外容量列不在 prediction parquet 中，可用
`capacity_input` 传 `date,symbol,decision_target_timestamp` 对齐的 keyed frame。

## 9. Exposure Audit

Top100 生产化前的暴露验收使用 `osf-audit-exposure`。默认审 `pool_L`，会从 prediction
parquet 自动检测常见可见暴露列：price、spread、depth、成交/换手代理、短窗 return、
市值/ADV/波动等；也可以用 `--exposure-col` 显式指定列。若暴露列不在 prediction parquet
中，可用 `--exposure-input` 传 keyed exposure frame。`--exposure-input` 支持日频
`date,symbol` key，也支持 intraday `date,symbol,decision_target_timestamp` key。

市值/行业外部 exposure input 可从 ClickHouse 日频表构建：

```bash
osf-build-exposure-input \
  --predictions output/legacy/predictions/<run_id>/raw \
  --pool L \
  --output output/legacy/exposures/<run_id>_pool_l_size_industry_daily.parquet
```

默认从 `stock.daily_bar_jy` 拉 `market_cap` / `float_market_cap` / log cap / 日频成交额，
从 `stock.industry` 拉申万一二三级行业。输出是日频 keyed parquet，可被
`osf-audit-exposure` 直接 join。

正式大样本 exposure audit / attribution 走第 1 节的集群侧原则；下面的本地命令只用于
小样本 debug 或复现已同步的 compact 输入：

```bash
osf-audit-exposure \
  --predictions output/legacy/predictions/<run_id>/predictions_all.parquet \
  --exposure-input output/legacy/exposures/<run_id>_pool_l_size_industry_daily.parquet \
  --output-dir output/legacy/analysis/<run_id>_exposure_audit \
  --pool L \
  --top-n 100 \
  --exposure-col log_market_cap \
  --exposure-col log_float_market_cap \
  --industry-col industry_sw1
```

常用输出：

```text
exposure_audit_summary.csv                 # pool x category x exposure 总体暴露
exposure_audit_month_summary.csv           # 月度稳定性
exposure_audit_group_metrics.csv           # date x clock 明细
exposure_audit_category_summary.csv         # 类别级最大/平均暴露
exposure_audit_industry_group_metrics.csv   # date x clock x industry active share 明细
exposure_audit_industry_month_summary.csv   # 月度行业 active share 稳定性
exposure_audit_industry_summary.csv         # 总体行业超/低配
exposure_audit_daily_concentration.csv      # 日内重复选股和行业集中
exposure_audit_concentration_summary.csv    # 集中度总体摘要
exposure_audit_trace.json
```

K8s run config 可设置：

```toml
[run]
kind = "exposure_audit"

[exposure_audit]
predictions = ["/mnt/output/opening_strength_fit/<source_run_id>"]
pool = ["L"]
top_n = 100
```

该工具只审计 TopN / 已给定组合的暴露，不构造容量组合；后续 `10 亿` capacity portfolio
可以用 `selection_col` 或 `weight_col` 复用同一套暴露计算。

## 10. 排查

| symptom | action |
| --- | --- |
| `kubectl` 没有 current-context | 使用 `hfcli kubectl --cluster research ...`。 |
| PVC API 被 RBAC 拒绝 | 用 Pod/Job yaml 的 `claimName` 和容器内 `/mnt/output` 检查文件状态。 |
| `field is immutable` | 删除同名 Job 后重新 apply。 |
| K8s 内找不到新 config | 重新 build/push 镜像，并重新 render Job。 |
| cache 只有 `.tmp` / lock / heartbeat | 等待最终 `.parquet` 和 manifest。 |
| replay 缺少上下文字段 | 传 `--context-input`，或先运行 interface check。 |
| completed config 没有 metrics | 运行 `osf-sync-experiment-artifacts --all`，然后 audit。 |
