# Project Map

## 根目录

- `README.md`: 项目目标、高层结构、快速开始、实验记录和当前基线。
- `requirements.txt`: Python 运行依赖。
- `Dockerfile`: 训练镜像定义，工作目录为 `/app/opening_strength_fit`。
- `.env.example`: 本地 tick path 和 ClickHouse 凭证示例。
- `.gitignore`: 忽略虚拟环境、缓存和本地输出。

## `src/opening_strength_fit/`

- `__init__.py`: 包版本和说明。
- `config.py`: TOML 加载、配置取值、run id 和 slug helper。
- `schema.py`: 标准列名、盘口层级、时间窗口和列标准化 helper。
- `clickhouse_ticks.py`: ClickHouse `stock.tick` 查询、表名校验、字段说明和原始 tick 标准化 helper。
- `io.py`: parquet/csv 读取、写出和路径解析。
- `dataset.py`: tick 数据加载、时间过滤、特征和 label 表构造。
- `universe.py`: A 股股票池正则过滤和正式股票池文件读取 helper。
- `sampling.py`: `09:30` 到 `09:40` 等整分钟决策点抽样 helper。
- `features.py`: 盘口、成交、动量和集合竞价基础特征。
- `labels.py`: 可成交收益 label 计算。
- `candidates.py`: 基于当前及过去可见特征的开盘强势候选池过滤。
- `label_audit.py`: 按年份、月份、分钟段和 fee_bps 统计 label 有效性与分布。
- `rules.py`: 不训练模型的规则分数 baseline。
- `model.py`: Ridge baseline、预测和 IC metrics。
- `evaluation.py`: score 分组和 top-score 交易评估。
- `rolling.py`: 日期切分 helper。
- `training.py`: 统一训练编排，支持 ClickHouse 或 parquet/cache 数据源、月度/年度 rolling，写出 `predictions.parquet`、`predictions_<period>.parquet`、`score_buckets.csv`、`metrics.json`、`metrics_by_month.csv` 和 `metrics_by_year.csv`。
- `reports.py`: 数据摘要、yearly metrics 表格和打印 helper。
- `backtest.py`: top-score 摘要、回测序列读取和累计曲线摘要 helper。
- `k8s.py`: K8s run spec、临时 PVC pull pod、manifest job name 和 kubectl command helper。

## `scripts/`

- `_bootstrap.py`: 让 `python scripts/<name>.py` 可以 import `src/opening_strength_fit`。
- `audit_experiments.py`: 检查 run config、Job YAML、metrics 归档和 backtest 归档是否一致。
- `check_workflow_coverage.py`: 检查 README、runbook、项目地图、脚本、模块和 run config 的覆盖关系。
- `probe_clickhouse_data.py`: 连接 ClickHouse，查看源表 schema、字段说明、A 股过滤后的开盘窗口样本。
- `prepare_research_dataset.py`: 按交易日读取 ClickHouse 或 raw parquet，过滤 A 股 universe，生成分区 labeled research dataset。
- `inspect_dataset.py`: 从 ClickHouse 或已有 parquet 构建本地样本，检查 tick schema、label 覆盖和训练特征口径。
- `audit_labels.py`: 读取 labeled research dataset，输出 fee 前后 label 有效性和分布审计 CSV。
- `run_rule_baselines.py`: 用动量、成交速度、盘口不平衡、spread 等规则分数做 bucket/top-score baseline。
- `fetch_clickhouse_ticks.py`: 按 symbol/date 从 ClickHouse 抓取 tick 窗口并写出 parquet/csv。
- `concat_frames.py`: 合并多个 parquet/csv tick 文件，生成训练输入。
- `build_labels.py`: 从 tick 表生成带特征和 label 的 parquet/csv。
- `run_experiment.py`: 按 TOML 配置运行 baseline 训练和预测。
- `evaluate_predictions.py`: 读取 prediction parquet/csv 并按 `selection_mode` 输出 IC、分桶和 top-score 摘要。
- `summarize_opening_results.py`: 读取 `metrics_by_year.csv`，输出 yearly table 和稳定性摘要。
- `compare_opening_results.py`: 比较多个 opening metrics CSV，生成 CSV、Markdown 和 PNG 报告。
- `render_k8s_job.py`: 从 TOML config 渲染 training Job 和 reader Job YAML，支持 `--sharded`。
- `pull_k8s_metrics.py`: 从 K8s PVC 拉回 `metrics_by_year.csv`。
- `fetch_k8s_predictions.py`: 从 K8s PVC 拉回 `predictions.parquet` / `predictions_all.parquet`。
- `run_backtest_api.py`: 将 tick predictions 汇总为 date-symbol score matrix，调用回测 API。
- `plot_backtest_curves.py`: 画单个 run 的 cumulative alpha/profit 曲线。
- `compare_backtest_runs.py`: 比较多个 backtest 输出目录的 cumulative alpha/profit 曲线。
- `record_experiment.py`: 把 `output/` 里的轻量 metrics/backtest 证据归档到 `experiments/results/`。

## `experiments/`

- `runs/*.toml`: 实验配置，`run.id` 必须等于文件名。
- `jobs/`: 已渲染 Kubernetes training/reader Job YAML。
- `results/metrics/`: 可提交的轻量 metrics CSV。
- `results/backtests/`: 可提交的轻量 backtest JSON。
