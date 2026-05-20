# opening_strength_fit Runbook

日常实验闭环：检查环境和 ClickHouse 数据，做极小本地 smoke，提交 K8s 训练，拉回 metrics/predictions，回测，分析，归档。

## 1. 环境准备

```bash
cd /home/hefu/projects/opening_strength_fit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

日常进入项目：

```bash
cd /home/hefu/projects/opening_strength_fit
source .venv/bin/activate

set -a
. ./.env
set +a
```

只读预检：

```bash
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

预检用于发现 config、Job YAML、metrics/backtest 和文档索引是否不同步。

## 2. 数据检查

查看 ClickHouse `stock.tick` 源表规模、schema 和字段说明。`probe_clickhouse_data.py`
按 `xy-fit` 的 probe 风格输出：连接状态、行列规模、日期/时间范围和可选 layout；
窗口检查通过时只显示 `PASS`，只有异常时才展开原因：

```bash
python scripts/probe_clickhouse_data.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

检查目标实验窗口的原始 tick 覆盖。这个命令只读 ClickHouse 聚合信息，不会把全年原始 tick 或 labeled dataset 拉到本地：

```bash
python scripts/probe_clickhouse_data.py \
  --start-date 2021-01-01 \
  --end-date 2022-01-31 \
  --opening-window \
  --a-share-only \
  --year-layout
```

检查单票小窗口的 tick schema、X/label 对齐和标签覆盖。`inspect_dataset.py`
会从 ClickHouse 抓少量真实 tick，在本地临时计算 feature/label；这一步只用于确认代码链路，不用于准备正式训练集。输出里的 `source_quality_checks`
和 `sample_quality_checks` 通过时只显示 `PASS`；若原始 09:15 quote 有 ask/bid 非正，`tick_dataset_check.rows`
会显示成 `正常行+异常行 (raw quote ask/bid<=0)`，决策样本是否可交易另看 `sample_quality_checks`：

```bash
python scripts/inspect_dataset.py \
  --symbol 000925.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3
```

需要留小样本用于后续本地 smoke 时，在同一命令后加：

```bash
--output output/local/inspect_smoke/000925_SZ_2021-09-22_2021-09-23.parquet \
--labeled-output output/local/inspect_smoke/000925_SZ_2021-09-22_2021-09-23_labeled.parquet
```

多股票横截面小样本可以在同一个命令里传多只股票，确认 `cross_section` 评估链路能跑通：

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet
```

默认不要在本地为一年或多月窗口运行 `prepare_research_dataset.py`。ClickHouse 里是原始 tick，feature/label 需要 Python 计算，长窗口本地准备会很慢且占空间。正式长窗口训练直接用 `[data].source = "clickhouse"` 的 K8s Job；长窗口 label audit / rule baseline 也应放到集群或专门的小结果 Job，只把 CSV/metrics 拉回本地。

如果只是临时调试 `audit_labels.py` 或 `run_rule_baselines.py` 的输出格式，可以只对上面的小 labeled parquet 跑：

```bash
python scripts/audit_labels.py \
  --input output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --output output/local/inspect_smoke/multi_symbol_label_audit.csv

python scripts/run_rule_baselines.py \
  --input output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --output-dir output/local/rule_baselines_multi_symbol_smoke
```

小样本 audit/rule 只能检查程序链路和字段口径，不能当作研究结论。研究结论以 K8s 长窗口输出的 metrics、prediction 和后续回测为准。

## 3. 本地 Smoke

本地只跑真实数据的小样本 smoke，用 full config 加输入、split、feature 和输出目录覆盖。不要在本地准备全年或多月 opening labeled dataset。

### 3.1 单票代码 Smoke

单票 smoke 只确认 ClickHouse 抓取、feature/label 计算、训练、预测和指标写出能跑通：

```bash
python scripts/inspect_dataset.py \
  --symbol 000925.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --output output/local/inspect_smoke/000925_SZ_2021-09-22_2021-09-23.parquet \
  --labeled-output output/local/inspect_smoke/000925_SZ_2021-09-22_2021-09-23_labeled.parquet

python scripts/run_experiment.py \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --input output/local/inspect_smoke/000925_SZ_2021-09-22_2021-09-23_labeled.parquet \
  --input-kind labeled \
  --split-mode chronological \
  --test-start-date 2021-09-23 \
  --test-end-date 2021-09-23 \
  --feature-limit 80 \
  --output-dir output/local/ridge_opening_1y_next_month_000925_2d_smoke
```

查看 smoke metrics：

```bash
python scripts/summarize_opening_results.py \
  --input-dir output/local/ridge_opening_1y_next_month_000925_2d_smoke

python scripts/evaluate_predictions.py \
  --input output/local/ridge_opening_1y_next_month_000925_2d_smoke/predictions.parquet \
  --bucket-mode symbol_day \
  --ic-mode symbol_day \
  --selection-mode symbol_day
```

单票 smoke 只有一只股票，不能验证 `cross_section` 排序；这里看
`symbol_day` 指标，只确认同一只股票当天多个开盘决策点之间的择时链路。
`cross_section` 要到 3.2 的多股票小样本 smoke 或正式 K8s 实验里看。

### 3.2 多股票横截面 Smoke

多股票 smoke 用几只股票、两天数据确认 `cross_section` 评估链路。它不是正式研究集，只检查同一时刻股票间排序的代码路径：

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet

python scripts/run_experiment.py \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --input output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --input-kind labeled \
  --split-mode chronological \
  --test-start-date 2021-09-23 \
  --test-end-date 2021-09-23 \
  --feature-limit 80 \
  --output-dir output/local/ridge_opening_1y_next_month_multi_symbol_smoke
```

查看横截面 smoke：

```bash
python scripts/summarize_opening_results.py \
  --input-dir output/local/ridge_opening_1y_next_month_multi_symbol_smoke

python scripts/evaluate_predictions.py \
  --input output/local/ridge_opening_1y_next_month_multi_symbol_smoke/predictions.parquet \
  --bucket-mode cross_section \
  --ic-mode cross_section \
  --selection-mode cross_section \
  --top-n 2
```

小样本横截面 smoke 的股票太少、日期太少，`group_rank_ic_mean`、bucket 和 top-score 只用于确认程序不报错，不用于判断模型是否有效。

### 3.3 正式研究口径

正式 one-year / full-window opening 实验不要在本地准备 labeled dataset。使用 K8s Job 直接读 ClickHouse 原始 tick，在集群内计算 feature/label、训练和评估，然后只拉回 `metrics_by_year.csv`、`metrics_by_month.csv`、`predictions.parquet` 或回测所需结果。

## 4. 新建实验

复制最接近的 config：

```text
ridge: experiments/runs/ridge_opening_full.toml
```

至少修改：

```text
[run].id
[run].description
[run].status
[data].source
[data].tick_path
[window].test_start_date / test_end_date 或 test_start_month / test_end_month
[model]...
[evaluation].selection_mode
[output].local_dir
[output].k8s_dir
[k8s.resources]...
```

约定：

- `run.id` 必须等于 config 文件名。
- `status` 提交前写 `queued` 或 `running`，拉回 metrics 并确认后写 `completed`。
- 集群训练默认用 `[data].source = "clickhouse"`，通过 ClickHouse 读取训练窗口原始 tick，只把结果写到 PVC。
- 本地 smoke 可传 `--input` 或设置 `[data].tick_path` 使用小的 prepared parquet/cache。
- `[output].k8s_dir` 必须在 `/mnt/output/opening_strength_fit/` 下，且一个实验一个目录。
- `evaluation.selection_mode` 第一版用 `cross_section`。

## 5. Build 镜像

修改训练代码、config 或依赖后：

```text
改代码/config -> build 新 TAG -> push -> render Job YAML -> 确认 image -> apply
```

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-v1
docker build -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}
```

换了 `TAG` 后必须重新 render Job YAML。

## 6. 生成 Job YAML

单个 opening Job：

```bash
python scripts/render_k8s_job.py \
  --config experiments/runs/ridge_opening_full.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}
```

full-window sharded Job：

```bash
python scripts/render_k8s_job.py \
  --config experiments/runs/ridge_opening_full.toml \
  --sharded \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}
```

输出路径：

```text
experiments/jobs/<run_id>_job.yaml
experiments/jobs/<run_id>_reader_job.yaml
experiments/jobs/<run_id>_sharded_job.yaml
experiments/jobs/<run_id>_sharded_reader_job.yaml
```

确认镜像：

```bash
rg -n "image:" experiments/jobs/ridge_opening_full_job.yaml
```

## 7. 提交和查看 Job

```bash
hfcli kubectl delete job opening-strength-ridge-opening-full --ignore-not-found -n bizewu
hfcli kubectl apply -f experiments/jobs/ridge_opening_full_job.yaml
hfcli kubectl get jobs,pods -n bizewu -o wide
hfcli kubectl logs -f job/opening-strength-ridge-opening-full -n bizewu
```

sharded 实验完成后，用 reader Job 合并 monthly/yearly metrics 和 predictions：

```bash
hfcli kubectl delete job opening-strength-read-ridge-opening-full-sharded --ignore-not-found -n bizewu
hfcli kubectl apply -f experiments/jobs/ridge_opening_full_sharded_reader_job.yaml
hfcli kubectl wait --for=condition=complete job/opening-strength-read-ridge-opening-full-sharded -n bizewu --timeout=300s
```

排查要点：

- `field is immutable`: 删除同名 Job 后重新 apply。
- `FileNotFoundError: experiments/runs/*.toml`: 镜像里没有新 config，重新 build/push。
- `python: can't open file scripts/run_experiment.py`: Job 使用的镜像不是当前代码镜像，重新 build/push/render。
- `No tick data path supplied`: 本地/path 模式没填 `[data].tick_path`，或本地 smoke 忘了传 `--input`。
- `missing ClickHouse credentials`: 集群 Job 没有 `CLICKHOUSE_USER` / `CLICKHOUSE_PASSWORD` 环境变量；配置 `[k8s].clickhouse_secret` 或检查平台注入。
- `input path does not exist`: 容器看不到该路径，检查 PVC、`mount_path` 和 `[data].tick_path`。

## 8. 拉回 Metrics

重复传 `--config` 可一次拉一个或多个实验：

```bash
python scripts/pull_k8s_metrics.py \
  --config experiments/runs/ridge_opening_full.toml
```

拉回文件写到：

```text
output/k8s/metrics/<run_id>_metrics_by_year.csv
```

`output/k8s/metrics/` 是本地原始产物；`experiments/results/metrics/` 是收尾时归档的轻量证据。

## 9. 拉回 Predictions 和回测

拉回预测并合并：

```bash
python scripts/fetch_k8s_predictions.py \
  --config experiments/runs/ridge_opening_full.toml \
  --output-dir output/backtest/ridge_opening_full
```

调用回测 API。tick-level 预测会先按 `date x symbol` 聚合，默认取开盘窗口内最大预测分：

```bash
python scripts/run_backtest_api.py \
  --predictions output/backtest/ridge_opening_full/predictions_all.parquet \
  --output-dir output/backtest/ridge_opening_full \
  --aggregate max \
  --tar I500
```

画单个 run 曲线：

```bash
python scripts/plot_backtest_curves.py \
  --input-dir output/backtest/ridge_opening_full \
  --output output/backtest/ridge_opening_full/cumulative_curves.png \
  --summary output/backtest/ridge_opening_full/curve_summary.json \
  --baseline-output output/backtest/ridge_opening_full/profit_vs_baseline.png
```

## 10. 分析 Metrics

单个实验：

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/ridge_opening_full_metrics_by_year.csv
```

比较已归档 full-window 实验：

```bash
python scripts/compare_opening_results.py
```

指定实验：

```bash
python scripts/compare_opening_results.py \
  --run ridge=experiments/results/metrics/ridge_opening_full_metrics_by_year.csv
```

重点看：

```text
cross_section IC: group_rank_ic_mean
cross_section IC IR: group_rank_ic_ir
daily_rank_ic_mean
model_test_r2
top_score_mean_return
```

当前主配置的 `ic_mode` 是 `cross_section`。

## 11. 分析 Backtest

默认比较 full-window 回测：

```bash
python scripts/compare_backtest_runs.py
```

重点看：

```text
alpha cumulative_end
profit cumulative_end
max_drawdown
turnover_mean
solve_rate_mean
```

## 12. 收尾：记录和审计

一轮实验包括：训练完成、reader 合并完成、metrics 拉回、需要的 predictions/backtest/分析完成，并且相关 config 的 `status` 已更新为 `completed`。

```bash
python scripts/record_experiment.py \
  --config experiments/runs/ridge_opening_full.toml
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

提交前确认：

- 新实验 config、Job YAML、metrics 记录和 backtest 记录都落在对应目录。
- `docs/experiment_log.md` 写入实验状态或结论。
- `output/` 只保留本地运行产物，不提交大 parquet / pkl。
- 新 Job YAML 使用刚 build/push 的镜像 tag。
