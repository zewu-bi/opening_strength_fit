# opening_strength_fit

`opening_strength_fit` 是一个 A 股开盘阶段 short-horizon alpha 研究项目。项目从
ClickHouse `stock.tick` 或本地 tick parquet 读取集合竞价与开盘盘口/成交数据，在
`09:30-09:40` 的整分钟 decision point 上构造
`trading day x symbol x opening timestamp` 样本，预测“当前主动买入并短持有约一分钟”的
future return proxy，并检验模型分数是否有稳定的横截面排序价值。

它当前解决的是高频信号发现闭环，不是完整实盘策略。A 股 T+1 约束下，当前 60 秒 label
只能作为 opening microstructure 的 proxy label；如果该信号稳定，后续还需要接
close、next open、next close 等 longer-horizon label，验证能否沉淀成日频选股或组合特征。

```text
ClickHouse stock.tick / local tick parquet
-> schema 标准化 + A 股 universe 过滤
-> 集合竞价、盘口、成交、动量特征
-> short-horizon tradable label
-> 09:30-09:40 整分钟 decision point 抽样
-> 可选 opening-strength candidate filter
-> Ridge / sklearn GBM rolling training
-> predictions + IC / bucket / TopN metrics
-> 日频 sanity-check 回测 + 开盘短周期 TopN 回测
-> 轻量实验记录归档到 experiments/results/
```

## 研究问题

核心问题是：只使用 decision point 当时及以前可见的信息，能否在同一开盘时刻识别更强的股票，
也能否在同一股票当天的多个开盘 tick 之间识别更好的入场时刻。

当前 baseline 覆盖：

- 数据 probe：检查 `stock.tick` schema、A 股过滤、开盘窗口覆盖和字段口径。
- labeled research dataset：用真实 tick 构造 feature/label 表，只保留当前及过去可见的 X。
- Ridge / GBM baseline：按 config 做 chronological 或 rolling train/test。
- 评估：`cross_section` / `symbol_day` IC、score bucket、TopN label replay。
- 回测：把 tick prediction 转成日频 score 做 sanity check；另外做更贴近 label 的开盘短周期 TopN replay。
- 实验审计：配置、K8s Job、metrics、backtest 轻量结果均可复查。

## 文档分工

- `README.md`: 项目入口。
- `docs/project_brief.md`: 研究目标、数据、label、特征和评估口径。
- `docs/project_map.md`: 逐文件职责索引。
- `docs/runbook.md`: 日常实验操作步骤。
- `docs/experiment_log.md`: 实验记录和阶段性结论。
- `experiments/results/README.md`: 轻量证据目录约定。

## 项目结构

```text
.
├── src/opening_strength_fit/      可复用库代码
├── scripts/                       数据检查、训练、K8s、评估、回测、归档命令
├── experiments/runs/              每个实验一个 TOML，run.id 必须等于文件名
├── experiments/jobs/              已渲染的 Kubernetes Job YAML
├── experiments/results/           可提交的轻量 metrics/backtest 证据
├── docs/project_brief.md          研究目标、数据、label、评估口径
├── docs/project_map.md            逐文件职责索引
├── docs/runbook.md                日常实验操作手册
├── docs/experiment_log.md         已跑实验和阶段性结论
├── output/                        本地运行产物，默认不进 git
├── Dockerfile                     集群训练镜像
└── requirements.txt               Python 依赖
```

目录级结构放在 README；逐文件职责和脚本索引见 [docs/project_map.md](docs/project_map.md)。

## 数据口径

默认数据源是 ClickHouse：

```text
host: ch.db.prod.highfortfunds.com
table: stock.tick
default window: 09:15:00 - 09:45:00
```

时间窗口设计：

- `09:15-09:30`: 集合竞价特征。
- `09:30-09:40`: decision point 抽样和预测。
- `09:40` 后：覆盖尾部 decision point 的 future label。

关键原始字段包括：

```text
TradingDay
Symbol
ExchTimeOffsetUs
Volume
Turnover
AskPrice1..10 / BidPrice1..10
AskVolume1..10 / BidVolume1..10
Status
```

项目内部统一为 `date`、`symbol`、`timestamp`、`volume`、`turnover`、
`ask_price_1`、`bid_price_1` 等 snake_case 列。默认 universe 正则覆盖
`00/30.SZ` 和 `60/68.SH` A 股 symbol，也可以通过 `[universe].symbols_file`
指定股票池。

## Label 和特征

当前 label 是“当前 tick 决策、延迟 N 个 tick 成交、持有 60 秒、再用后续 60 秒 VWAP 退出”的
short-horizon proxy。当前配置默认 `entry_tick_delay = 1`、`fee_bps = 0`：

```text
decision_t = 当前样本 tick
entry_t = decision_t 之后第 entry_tick_delay 个 tick

buy_price = ask1[entry_t]

sell_vwap =
    (turnover[entry_t + 120s] - turnover[entry_t + 60s])
    / (volume[entry_t + 120s] - volume[entry_t + 60s])

label = sell_vwap / buy_price - 1 - fee_bps / 10000
```

`volume` 是累计成交量，`turnover` 是累计成交额。`valid_label` 要求买入价有效、entry tick 有效、
退出窗口内有正成交量/成交额，并可按 `[filters].tradable_statuses` 约束交易状态。

X 特征只允许使用 decision point 当时及此前可见的信息。当前重点包括：

- 盘口结构：`mid_price`、`spread_bps`、一档/十档深度、买卖盘不平衡、挂单档位 gap。
- 成交活跃度：`volume_diff_*t`、`turnover_diff_*t`、短窗口成交 VWAP。
- 动量：相对昨收/开盘收益、`1/3/10/30` tick return。
- 集合竞价：竞价累计量额、竞价末价、竞价价格区间、竞价不平衡。
- 交易约束：涨停距离、A 股 universe、交易状态、候选池阈值和横截面 rank filter。

`model.feature_columns()` 会排除 label、entry/sell future 字段、timestamp future 字段等泄漏列。

## 训练和评估

实验由 `experiments/runs/*.toml` 驱动，常见 section 如下：

```text
[data]          clickhouse/path、本地输入、feature limit
[clickhouse]    host/table/09:15-09:45 offset
[universe]      A 股 regex 或 symbols_file
[sample]        09:30-09:40 decision times 和 max lag
[labels]        buy price、holding window、sell VWAP window、entry delay、fee
[features]      是否加入 preopen 特征
[filters]       tradable status
[candidate_filter]  strong candidate 阈值与横截面 rank filter
[window]        chronological / rolling_annual / rolling_monthly
[model]         ridge 或 gbm 参数
[evaluation]    IC/bucket/selection group、score bins、TopN
[output]        local_dir 和 k8s_dir
[k8s]           namespace、PVC、镜像拉取和 ClickHouse secret
```

模型：

- Ridge regression：`SimpleImputer -> StandardScaler -> Ridge`。
- GBM：`SimpleImputer -> HistGradientBoostingRegressor`。

评估 group mode：

- `cross_section`: `date x decision_target_timestamp`，回答“同一时刻哪只股票更强”。
- `symbol_day`: `date x symbol`，回答“这只股票当天哪个开盘时刻更强”。
- `daily`: 按交易日整体评估。
- `global`: 全局分组。

训练输出通常包括：

```text
predictions.parquet
predictions_<period>.parquet
score_buckets.csv
score_buckets_<period>.csv
metrics.json
metrics_by_year.csv / metrics_by_year.parquet
metrics_by_month.csv / metrics_by_month.parquet  # monthly rolling 时生成
```

重点指标：

- `group_rank_ic_mean` / `group_rank_ic_ir`: 当前 group mode 下的 rank IC 和稳定性。
- `daily_rank_ic_mean`: 按日聚合的排序能力。
- `model_test_r2`: 模型对 label 的 out-of-sample 拟合度。
- `top_score_mean_return` / `top_score_win_rate`: 每个 group 只按 prediction 选 TopN 后的 label replay。
- `score_buckets.csv`: 观察收益是否随模型分数单调变化。

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

只读预检：

```bash
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

检查一个多股票小窗口，并保存 labeled parquet 供 smoke 使用：

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet
```

本地训练 smoke：

```bash
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
```

查看 smoke 结果：

```bash
python scripts/summarize_opening_results.py \
  --input-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke

python scripts/evaluate_predictions.py \
  --input output/local/gbm_opening_1y_next_month_multi_symbol_smoke/predictions.parquet
```

不要在本地为一年或多月窗口准备完整 labeled dataset。正式长窗口实验应通过
`[data].source = "clickhouse"` 的 K8s Job 在集群内读取原始 tick、构造 feature/label、训练和评估，
本地只拉回 metrics、predictions 和轻量回测结果。

## K8s 实验闭环

常规长窗口实验流程：

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-v1
docker build -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

python scripts/render_k8s_job.py \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

hfcli kubectl delete job opening-strength-gbm-opening-1y-next-month --ignore-not-found -n bizewu
hfcli kubectl apply -f experiments/jobs/gbm_opening_1y_next_month_job.yaml
hfcli kubectl logs -f job/opening-strength-gbm-opening-1y-next-month -n bizewu
```

reader、拉回和归档：

```bash
hfcli kubectl delete job opening-strength-read-gbm-opening-1y-next-month --ignore-not-found -n bizewu
hfcli kubectl apply -f experiments/jobs/gbm_opening_1y_next_month_reader_job.yaml
hfcli kubectl wait --for=condition=complete job/opening-strength-read-gbm-opening-1y-next-month -n bizewu --timeout=300s

python scripts/pull_k8s_metrics.py \
  --config experiments/runs/gbm_opening_1y_next_month.toml

python scripts/fetch_k8s_predictions.py \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --output-dir output/backtest/gbm_opening_1y_next_month

python scripts/record_experiment.py \
  --config experiments/runs/gbm_opening_1y_next_month.toml

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

## 回测和结果分析

单个实验 metrics：

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/gbm_opening_1y_next_month_metrics_by_year.csv
```

多实验 metrics 对比：

```bash
python scripts/compare_opening_results.py
```

日频 API sanity check 会把 tick-level prediction 聚合为 `date x symbol` score matrix：

```bash
python scripts/run_backtest_api.py \
  --predictions output/backtest/gbm_opening_1y_next_month/predictions_all.parquet \
  --output-dir output/backtest/gbm_opening_1y_next_month \
  --aggregate max \
  --tar I500
```

更贴近当前 label 的开盘短周期回测直接在 tick predictions 上运行。默认每天本金从 1.0 开始，
在 `09:30/09:32/09:34/09:36/09:38` 五个非重叠两分钟 cycle 上，按 prediction 选 Top20、
等权满仓买入，用 label 均值复利滚动：

```bash
python scripts/run_opening_intraday_backtest.py \
  --run gbm=output/backtest/gbm_opening_1y_next_month/predictions_all.parquet \
  --run gbm_strong=output/backtest/gbm_opening_1y_next_month_strong/predictions_all.parquet \
  --run ridge=output/backtest/ridge_opening_1y_next_month/predictions_all.parquet \
  --run ridge_strong=output/backtest/ridge_opening_1y_next_month_strong/predictions_all.parquet \
  --output-dir output/reports/opening_intraday_top20_1y_next_month
```

`output/` 保存本地 parquet、模型对象、图表和临时报告，默认不提交；
`experiments/results/` 保存 `record_experiment.py` 归档后的轻量 CSV/JSON 证据，适合审计和汇报。

## 当前实验状态

已有配置覆盖 Ridge/GBM、普通 universe/strong candidate、小窗 smoke、1y next-month 和 H1 rolling。
截至 `2026-05-21` 的实验索引与口径说明见 [docs/experiment_log.md](docs/experiment_log.md)。

归档的 2021 训练、2022-01 测试结果显示，修正到
`date x decision_target_timestamp` 的横截面口径后，普通 GBM 暂时最强：

| run | decision rank IC | rank IC IR | Top20 mean bps | Top20 win rate |
| --- | ---: | ---: | ---: | ---: |
| `gbm_opening_1y_next_month` | 0.1831 | 2.7548 | +34.33 | 60.7% |
| `gbm_opening_1y_next_month_strong` | 0.1454 | 1.9402 | +18.78 | 53.8% |
| `ridge_opening_1y_next_month_strong` | 0.1156 | 1.5987 | +9.63 | 51.2% |
| `ridge_opening_1y_next_month` | 0.0799 | 1.4788 | +18.96 | 54.4% |

日频 I500 sanity-check 回测为负，主要说明 tick-level 开盘信号被压成日频 score 后口径不匹配。
与 label 更一致的开盘短周期 Top20 replay 在 2022-01 单月上为正，普通 GBM 的
mean cycle return 约 `+42.21 bps`、19 个测试日均为正。这个结果只能说明短周期方向性值得继续验证，
还需要加入成本、滑点、容量、重复持仓、成交约束以及更长 rolling out-of-sample。

注意：部分历史归档结果与当前 config 的 `entry_tick_delay` 设置可能不同；做横向比较时以
[docs/experiment_log.md](docs/experiment_log.md) 中标注的口径为准。

## 开发约定

- 每个实验一个 `experiments/runs/<run_id>.toml`，`[run].id` 必须等于文件名。
- 拉回并确认结果后，把 config 的 `[run].status` 更新为 `completed`。
- 新代码、config 或依赖进入集群前，需要 build/push 新镜像并重新 render Job YAML。
- 大体积 predictions、模型文件、图表和临时输出放在 `output/`，不要提交。
- 可提交证据放在 `experiments/results/metrics/` 和 `experiments/results/backtests/`。
- 提交前运行 `python scripts/audit_experiments.py` 和 `python scripts/check_workflow_coverage.py`。
- 详细操作步骤以 [docs/runbook.md](docs/runbook.md) 为准；逐文件职责以
  [docs/project_map.md](docs/project_map.md) 为准。
