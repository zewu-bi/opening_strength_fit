# opening_strength_fit

`opening_strength_fit` 是一个开盘强势股日内短周期 alpha 项目，用 ClickHouse `stock.tick` 里的集合竞价和开盘盘口/成交数据生成 tick-level label，训练模型预测“当前主动买入并持有约一分钟”的期望收益，并把 Kubernetes 运行、metrics 拉回、预测拉回、回测结果和轻量实验档案串成一条可复查的流程。

第一版核心问题是：在开盘后 `09:30-09:40` 的整分钟决策时刻，能否只用当时及此前可见的竞价、盘口和成交信息，挑出更值得买入并短持有的 `date x symbol x timestamp` 机会。复杂仓位、复杂执行模拟和更强模型暂不作为第一版主线。

```text
ClickHouse stock.tick / prepared tick parquet
-> tick schema 检查，date-symbol-timestamp 对齐
-> A 股 universe 过滤 + opening feature + tradable label
-> label audit + rule baseline
-> config-driven rolling training
-> predictions + metrics_by_month/year.csv
-> cross_section / symbol_day IC + bucket/top-score analysis
-> 简单 threshold/top-N 交易转化 + backtest + experiment records
```

## 项目结构

```text
src/opening_strength_fit/  可复用库代码
scripts/run_experiment.py  统一训练入口，按 config 训练 opening Ridge baseline
scripts/                  数据检查、K8s、回测、分析、记录、审计命令
experiments/runs/          每个实验一个 TOML，run.id 必须等于文件名
experiments/jobs/          已渲染的 Kubernetes Job YAML
experiments/results/       可提交的轻量实验记录
docs/runbook.md            日常实验操作手册
docs/project_map.md        文件地图和代码职责说明
output/                    本地运行产物，被 git 忽略
```

核心库职责：

- `clickhouse_ticks.py`: ClickHouse tick 查询、表名校验、字段说明和标准化。
- `schema.py`: 标准列名、盘口层级、时间窗口和列标准化 helper。
- `dataset.py`: 读取 tick、构造 feature/label 表。
- `universe.py`, `sampling.py`: A 股股票池过滤和整分钟开盘决策点抽样。
- `features.py`: 盘口、成交、动量和集合竞价基础特征。
- `labels.py`: 可成交收益 label 计算。
- `label_audit.py`, `rules.py`: label 可交易性审计和最弱规则分数 baseline。
- `model.py`: Ridge baseline、预测和 IC metrics。
- `evaluation.py`: score 分组和 top-score 交易评估。
- `rolling.py`: 日期切分 helper。
- `training.py`: 统一训练编排，可从 ClickHouse 或 parquet/cache 读取，写出 `predictions.parquet` 和 `metrics_by_year.csv`。
- `io.py`, `config.py`: parquet/csv I/O、TOML 配置、run id 和 slug helper。
- `k8s.py`, `reports.py`, `backtest.py`: 集群辅助、报告表格、回测序列工具。

## 快速开始

```bash
cd /home/hefu/projects/opening_strength_fit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

访问 ClickHouse 前配置凭证：

```bash
cp .env.example .env
set -a
. ./.env
set +a
```

本地基础检查：

```bash
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
python scripts/probe_clickhouse_data.py
```

按日期分区准备研究集，并先做 label/rule 检查：

```bash
python scripts/prepare_research_dataset.py \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --start-date 2021-09-01 \
  --end-date 2021-09-30 \
  --output-root output/local/opening_labeled

python scripts/audit_labels.py \
  --input output/local/opening_labeled \
  --output output/local/opening_labeled/label_audit.csv

python scripts/run_rule_baselines.py \
  --input output/local/opening_labeled \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --output-dir output/local/rule_baselines_2021_09
```

按 config 训练 opening baseline：

```bash
python scripts/inspect_dataset.py \
  --symbol 000925.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --labeled-output output/local/inspect_smoke/000925_SZ_2021-09-22_2021-09-23_labeled.parquet

python scripts/run_experiment.py \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --input output/local/inspect_smoke/000925_SZ_2021-09-22_2021-09-23_labeled.parquet \
  --input-kind labeled \
  --split-mode chronological \
  --test-start-date 2021-09-23 \
  --test-end-date 2021-09-23 \
  --feature-limit 80 \
  --output-dir output/local/gbm_opening_1y_next_month_000925_2d_smoke
```

查看结果：

```bash
python scripts/summarize_opening_results.py \
  --input-dir output/local/gbm_opening_1y_next_month_000925_2d_smoke
python scripts/evaluate_predictions.py \
  --input output/local/gbm_opening_1y_next_month_000925_2d_smoke/predictions.parquet
```

所有 smoke 和实验都使用真实 ClickHouse tick 或由真实 tick 生成的 parquet。
低层调试入口包括 `fetch_clickhouse_ticks.py`、`concat_frames.py`、`build_labels.py` 和 `inspect_dataset.py`。

## 实验记录

每个实验的最小闭环：

1. `experiments/runs/<run_id>.toml` 保存配置和状态。
2. `scripts/render_k8s_job.py` 生成 training Job 和 reader Job YAML。
3. 训练输出写到 config 的 `[output].k8s_dir` 或 `[output].local_dir`。
4. `scripts/pull_k8s_metrics.py` 拉回 `metrics_by_month.csv` / `metrics_by_year.csv`。
5. `scripts/fetch_k8s_predictions.py` 拉回 `predictions_all.parquet`。
6. `scripts/run_backtest_api.py`、`plot_backtest_curves.py` 和 `compare_backtest_runs.py` 生成回测记录。
7. `scripts/record_experiment.py` 把轻量 metrics/backtest JSON 复制到 `experiments/results/`。
8. `scripts/audit_experiments.py` 检查 config、Job、metrics、backtest 是否对齐。

`output/` 保存本地 parquet、pkl、图表和临时报告，默认不进 git；`experiments/results/` 保存 `record_experiment.py` 归档后的轻量 CSV/JSON 证据，可用于审计、分析和汇报。跨实验 metrics 对比用 `compare_opening_results.py`。

## 当前基线

当前默认配置是 `gbm_opening_1y_next_month`。集群训练从 ClickHouse 读取所需日期窗口；本地 smoke 可用小 parquet/cache。第一版口径先做 A 股 universe 过滤、`09:30-09:40` 整分钟决策点、规则分数 baseline、`cross_section` / `symbol_day` IC、score bucket/top-score 分析，以及 12 个月训练到 1 个月测试的 rolling baseline。label 采用：

```text
buy_price = current ask1
sell_vwap = turnover[t+120s] - turnover[t+60s]
            -----------------------------------
             volume[t+120s] - volume[t+60s]
label = sell_vwap / buy_price - 1 - fee_bps / 10000
```

`Volume` 是累计成交量，`Turnover` 是累计成交额。实验索引和当前结论见 [docs/experiment_log.md](docs/experiment_log.md)。完整操作流程见 [docs/runbook.md](docs/runbook.md)。

测试/交易阶段不能按未来 label 排序，只能用模型 `prediction` 决策。第一版交易转化先使用最简单规则：每个决策时刻选 prediction 最高的 top N，或选择 prediction 超过阈值的样本，等权买入并按 label 里的 `sell_vwap` 退出，用事后 label 检查是否真的挑出了更赚钱的一分钟机会。
