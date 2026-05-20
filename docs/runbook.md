# opening_strength_fit Runbook

## 1. 环境准备

```bash
cd /home/hefu/projects/opening_strength_fit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```bash
cd /home/hefu/projects/opening_strength_fit
source .venv/bin/activate

set -a
. ./.env
set +a
```

```bash
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

## 2. 数据检查

ClickHouse 表：

```bash
python scripts/probe_clickhouse_data.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

实验窗口：

```bash
python scripts/probe_clickhouse_data.py \
  --start-date 2021-01-01 \
  --end-date 2022-01-31 \
  --opening-window \
  --a-share-only \
  --year-layout
```

单票小窗口：

```bash
python scripts/inspect_dataset.py \
  --symbol 000925.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3
```

保存小样本：

```bash
--output output/local/inspect_smoke/000925_SZ_2021-09-22_2021-09-23.parquet \
--labeled-output output/local/inspect_smoke/000925_SZ_2021-09-22_2021-09-23_labeled.parquet
```

多股票横截面：

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet
```

不要在本地为一年或多月窗口运行 `prepare_research_dataset.py`。正式长窗口训练用 `[data].source = "clickhouse"` 的 K8s Job。

小样本 audit / rule：

```bash
python scripts/audit_labels.py \
  --input output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --output output/local/inspect_smoke/multi_symbol_label_audit.csv

python scripts/run_rule_baselines.py \
  --input output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --config experiments/runs/ridge_opening_1y_next_month.toml \
  --output-dir output/local/rule_baselines_multi_symbol_smoke
```

小样本结果不作研究结论。

## 3. 本地 Smoke

本地只跑小样本 smoke。不要本地准备全年或多月 opening labeled dataset。

### 3.1 单票代码 Smoke

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

```bash
python scripts/summarize_opening_results.py \
  --input-dir output/local/ridge_opening_1y_next_month_000925_2d_smoke

python scripts/evaluate_predictions.py \
  --input output/local/ridge_opening_1y_next_month_000925_2d_smoke/predictions.parquet \
  --bucket-mode symbol_day \
  --ic-mode symbol_day \
  --selection-mode symbol_day
```

单票 smoke 看 `symbol_day`；`cross_section` 看多股票 smoke 或 K8s 实验。

### 3.2 多股票横截面 Smoke

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

小样本横截面指标不作研究结论。

### 3.3 正式研究口径

正式 one-year / full-window opening 实验走 K8s。只拉回 `metrics_by_year.csv`、`metrics_by_month.csv`、`predictions.parquet` 或回测所需结果。

## 4. 新建实验

Config 模板：

```text
ridge: experiments/runs/ridge_opening_full.toml
```

必改：

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
- 集群训练用 `[data].source = "clickhouse"`。
- 本地 smoke 可传 `--input` 或设置 `[data].tick_path` 使用小的 prepared parquet/cache。
- `[output].k8s_dir` 必须在 `/mnt/output/opening_strength_fit/` 下，且一个实验一个目录。
- `evaluation.selection_mode` 第一版用 `cross_section`。

## 5. Build 镜像

Build：

```text
改代码/config -> build 新 TAG -> push -> render Job YAML -> 确认 image -> apply
```

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-v1
docker build -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}
```

换 `TAG` 后重新 render Job YAML。

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

sharded reader：

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

```bash
python scripts/pull_k8s_metrics.py \
  --config experiments/runs/ridge_opening_full.toml
```

输出：

```text
output/k8s/metrics/<run_id>_metrics_by_year.csv
```

归档到 `experiments/results/metrics/`。

## 9. 拉回 Predictions 和回测

```bash
python scripts/fetch_k8s_predictions.py \
  --config experiments/runs/ridge_opening_full.toml \
  --output-dir output/backtest/ridge_opening_full
```

```bash
python scripts/run_backtest_api.py \
  --predictions output/backtest/ridge_opening_full/predictions_all.parquet \
  --output-dir output/backtest/ridge_opening_full \
  --aggregate max \
  --tar I500
```

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

```bash
python scripts/compare_opening_results.py
```

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

`ic_mode`: `cross_section`

## 11. 分析 Backtest

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

完成条件：训练完成、reader 合并完成、metrics 拉回、需要的 predictions/backtest/分析完成、config `status=completed`。

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
