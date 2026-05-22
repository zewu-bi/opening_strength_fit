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
-> short-horizon proxy label
-> 09:30-09:40 整分钟 decision point 抽样
-> 可选 opening-strength candidate filter
-> Ridge / sklearn GBM / LightGBM rolling training
-> predictions + IC / bucket / TopN metrics
-> 开盘短周期 TopN replay
-> 轻量实验记录归档到 experiments/results/
```

## 研究问题

核心问题是：只使用 decision point 当时及以前可见的信息，能否在同一开盘时刻识别更强的股票，
也能否在同一股票当天的多个开盘 tick 之间识别更好的入场时刻。

当前 baseline 覆盖：

- 数据 probe：检查 `stock.tick` schema、A 股过滤、开盘窗口覆盖和字段口径。
- labeled research dataset：用真实 tick 构造 feature/label 表，只保留当前及过去可见的 X。
- Ridge / GBM / LightGBM baseline：按 config 做 chronological 或 rolling train/test。
- labeled feature cache：长窗口 ClickHouse 数据先缓存成 base labeled parquet，再让普通/strong 分支复用。
- 评估：`cross_section` / `symbol_day` IC、score bucket、TopN label replay。
- 开盘短周期 replay：直接在 tick predictions 上按开盘 decision point 选 TopN，用 label 回放方向性。
- 实验审计：配置、K8s Job、metrics 和 opening replay 轻量结果均可复查。

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
├── experiments/results/           可提交的轻量 metrics/replay 证据
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
short-horizon proxy。已归档 Ridge/GBM baseline 使用 `entry_tick_delay = 0`；
代码和 replay 工具支持后续新实验按 `entry_tick_delay = 0/1/2`、`fee_bps = 0` 拆分比较：

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
delay 和新鲜度分开记录：`entry_delay_seconds` 是从 decision tick 到 entry tick 的总等待时间；
`entry_max_tick_gap_seconds` 是这段路径里相邻 tick 的最大间隔，用于判断 entry 行情是否过旧。

训练/replay 边界：改变 label 或样本域的口径进训练；只改变执行、成本、容量或选股后的约束先放 replay。
后续新实验可按 delay0/1/2 拆分；也支持用瘦 prediction + replay context 复算 delay realized label。
fee/slippage/spread/容量/状态/同股一次等约束用同一批 predictions 压测。
fee 和滑点对固定入选交易是确定性 haircut，当前不需要为 fee 立刻重训；只有当研究目标变成
net label 排序、成本改变样本有效性，或 fee/slippage 与成交容量一起改变训练样本域时，才另开训练分支。

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
[k8s]           namespace、PVC 和镜像拉取配置
```

模型：

- Ridge regression：`SimpleImputer -> StandardScaler -> Ridge`。
- sklearn GBM：`SimpleImputer -> HistGradientBoostingRegressor`。
- LightGBM：`SimpleImputer -> LGBMRegressor`，支持 `device_type = "cpu"` / `"gpu"`；PVC labeled cache + CPU LightGBM workflow 已实现，当前本地实验注册表只保留已完成的 Ridge/GBM baseline。

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

## Replay 约束

开盘短周期 replay 只用于压力测试当前 proxy signal，不代表 A 股 T+1 可交易收益。当前真实约束按以下边界处理：

| 约束 | 处理方式 |
| --- | --- |
| `entry_tick_delay` | 改变 entry price 和 label，进训练配置；后续可按 delay0/1/2 新开 run。 |
| fee / slippage | 不改变 prediction 排序，先在 replay 用 `--fee-bps` / `--slippage-bps` 扣减。 |
| 交易状态 | 训练用 `[filters].tradable_statuses` 控制 label 有效性；replay 再检查 `status` / `entry_status`。 |
| tick 新鲜度 | replay 用 `--max-decision-lag-seconds` 控制 decision tick；用 `--max-entry-tick-gap-seconds` 控制 entry 路径相邻 tick 最大间隔。 |
| spread / 一档深度 | replay 用 `--max-spread-bps`、`--min-ask-volume-1`、`--min-bid-volume-1`。 |
| entry 卖盘容量 | replay 用 prediction 自带或 `--context-input` 补齐的 `entry_ask_price_1..10` / `entry_ask_volume_1..10` 检查目标金额能否在 entry tick 的卖盘中成交；十档 sweep 场景会用 sweep VWAP 修正收益。 |
| 涨停距离 | 若 prediction 或 `--context-input` 有 `ask1_to_limit_up_bps`，replay 用 `--min-limit-up-room-bps`。 |
| 容量/参与率 | replay 用 `turnover_diff_30t` 等可见 notional proxy、`--capital-per-cycle` 和 `--max-participation-rate`。 |
| TopN/现金/单票 | replay 控制 `--top-n`、`--max-symbol-weight`，未选满资金留现金。 |
| 同股重复/冷却 | replay 控制 `--max-symbol-trades-per-day`、`--symbol-cooldown-minutes`。 |
| T+1 | 当前 60s label 只是 microstructure proxy；真实选股要接 close/next open/next close 等 longer-horizon label。 |

旧 prediction 如果没有 `entry_max_tick_gap_seconds`，需要重建 prediction 或通过 `--context-input`
补齐后再做 entry 新鲜度过滤。

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

本地训练 smoke 仍用 CPU config；正式长窗口任务在 research 集群直接读取 PVC 上的 labeled cache。

查看 smoke 结果：

```bash
python scripts/summarize_opening_results.py \
  --input-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke

python scripts/evaluate_predictions.py \
  --input output/local/gbm_opening_1y_next_month_multi_symbol_smoke/predictions.parquet
```

不要在本地为一年或多月窗口准备完整 labeled dataset。正式长窗口实验应通过
`[data].source = "path"` / `[data].input_kind = "labeled"` 的 K8s Job 直接读取 PVC cache，
训练和评估后只拉回 metrics、predictions 和轻量 opening replay 结果。

提交训练前先确认 PVC cache 是新口径完整文件：计划使用的 delay cache 应存在于
`/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay*_labeled.parquet`，
且 schema 包含 `entry_delay_seconds` 和 `entry_max_tick_gap_seconds`。只有
`entry_lag_seconds` 的旧 cache 不用于新鲜度/延迟拆分后的正式训练。cache 由
`scripts/materialize_labeled_caches.py` 生成；如果需要在 K8s 上跑 materialize，使用专用
Job 入口，不要用通用训练 renderer 改成 `scripts/run_experiment.py`。

## K8s 实验闭环

常规长窗口实验流程。当前 repo 不保留活跃 LightGBM run/job YAML；新开实验时先创建
`experiments/runs/<new_run_id>.toml`，再渲染对应 Job：

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-lgbm-cpu-v1
docker build --build-arg CACHE_BUST=${TAG} -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

python scripts/render_k8s_job.py \
  --config experiments/runs/<new_run_id>.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

rg -n "nvidia|gpu|nodeSelector|envFrom|image:" experiments/jobs/<new_run_id>_job.yaml
hfcli kubectl --cluster research delete job opening-strength-<new-run-slug> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<new_run_id>_job.yaml
hfcli kubectl --cluster research logs -f job/opening-strength-<new-run-slug> -n bizewu
```

reader、拉回和归档：

```bash
hfcli kubectl --cluster research delete job opening-strength-read-<new-run-slug> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<new_run_id>_reader_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/opening-strength-read-<new-run-slug> -n bizewu --timeout=300s

python scripts/pull_k8s_metrics.py \
  --config experiments/runs/<new_run_id>.toml

python scripts/fetch_k8s_predictions.py \
  --config experiments/runs/<new_run_id>.toml \
  --output-dir output/predictions/<new_run_id>

python scripts/record_experiment.py \
  --config experiments/runs/<new_run_id>.toml

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

当前 LightGBM 正式任务使用 `[model].device_type = "cpu"`，Job YAML 不应包含
`nvidia.com/gpu`、GPU toleration 或 GPU nodeSelector。

## 结果分析

单个实验 metrics：

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/gbm_opening_1y_next_month_metrics_by_year.csv
```

多实验 metrics 对比：

```bash
python scripts/compare_opening_results.py
```

更贴近当前 label 的开盘短周期回测直接在 tick predictions 上运行。默认每天本金从 1.0 开始，
在 `09:30/09:32/09:34/09:36/09:38` 五个非重叠两分钟 cycle 上，按 prediction 选 Top20、
等权满仓买入，用 label 均值复利滚动：

```bash
python scripts/run_opening_intraday_backtest.py \
  --run gbm=output/predictions/gbm_opening_1y_next_month/predictions_all.parquet \
  --run gbm_strong=output/predictions/gbm_opening_1y_next_month_strong/predictions_all.parquet \
  --fee-bps 5 \
  --slippage-bps 5 \
  --max-symbol-trades-per-day 1 \
  --output-dir output/reports/opening_intraday_top20_1y_next_month_constrained
```

新预测产物可能额外带上 `status`、`entry_status`、`spread_bps`、`turnover_diff_30t` 等上下文列；
有这些列，或用 `--context-input` 补齐这些列后，可以继续加 `--tradable-status`、`--max-spread-bps`、
`--min-capacity-notional` 和 `--max-participation-rate` 做更严格的成交约束。旧归档 prediction 文件
如果不传 context，只能复算成本和同股重复交易约束。
prediction 如果已经带 `entry_ask_price_1..10` 和 `entry_ask_volume_1..10`，replay 会直接使用；
旧的瘦 prediction 也可以通过 `--context-input` 指向 raw tick 或 labeled research dataset 来补齐这些
执行上下文；`--context-label-mode replace` 会用 context label 作为 replay PnL，同时保留
`prediction_label` 便于审计，用于区分“时间漂移后的 entry price”和“entry tick 上的真实卖盘容量”。

如果后续重新生成 LightGBM delay0/1/2 predictions，可以直接跑标准 replay 网格：

```bash
python scripts/run_lgbm_delay_replays.py
```

默认会对 delay0/1/2 的普通与 strong predictions 跑：

```text
proxy_top20
cost_10bps
tradable_cost
liquidity_cost
capacity_10m_5pct
strict_10m_2pct
```

总表写到：

```text
output/reports/opening_intraday_lgbm_delay_replays/scenario_summary.csv
```

`output/` 保存本地 parquet、模型对象、图表和临时报告，默认不提交；
`experiments/results/` 保存 `record_experiment.py` 归档后的轻量 CSV/JSON 证据，适合审计和汇报。

## 当前实验状态

当前本地实验注册表只保留两组已完成归档：`1m3d` 小窗口 Ridge/GBM 对比，以及
`1y_next_month` Ridge/GBM/strong 对比。后续若继续 LightGBM，需要新建 run/job，再按
`entry_tick_delay = 0/1/2` 比较延迟衰减；普通 universe 与 strong candidate 分支可共享同一个
delay 对应的 PVC labeled feature cache。实验索引与口径说明见 [docs/experiment_log.md](docs/experiment_log.md)。

归档的 2021 训练、2022-01 测试结果显示，修正到
`date x decision_target_timestamp` 的横截面口径后，旧 sklearn GBM 暂时最强：

| run | decision rank IC | rank IC IR | Top20 mean bps | Top20 win rate |
| --- | ---: | ---: | ---: | ---: |
| `gbm_opening_1y_next_month` | 0.1831 | 2.7548 | +34.33 | 60.7% |
| `gbm_opening_1y_next_month_strong` | 0.1454 | 1.9402 | +18.78 | 53.8% |
| `ridge_opening_1y_next_month_strong` | 0.1156 | 1.5987 | +9.63 | 51.2% |
| `ridge_opening_1y_next_month` | 0.0799 | 1.4788 | +18.96 | 54.4% |

与 label 一致的开盘短周期 Top20 replay 在 2022-01 单月上为正，旧普通 GBM 的
mean cycle return 约 `+42.21 bps`、19 个测试日均为正。这个结果只能说明短周期方向性值得继续验证，
后续 LightGBM 主要比较 IC、bucket 单调性、Top/Bottom spread 和延迟衰减；opening replay
只作为 proxy 压力测试，不作为 A 股 T+1 可交易回测。

注意：旧 Ridge/GBM 归档结果使用无成交延迟口径（`entry_tick_delay = 0`），新 LightGBM delay0
只用于同模型延迟基准；不要和旧模型归档直接横向混比。

## 开发约定

- 每个实验一个 `experiments/runs/<run_id>.toml`，`[run].id` 必须等于文件名。
- 拉回并确认结果后，把 config 的 `[run].status` 更新为 `completed`。
- 新代码、config 或依赖进入集群前，需要 build/push 新镜像并重新 render Job YAML。
- 大体积 predictions、模型文件、图表和临时输出放在 `output/`，不要提交。
- 可提交证据放在 `experiments/results/metrics/` 和 `experiments/results/backtests/`。
- 提交前运行 `python scripts/audit_experiments.py` 和 `python scripts/check_workflow_coverage.py`。
- 详细操作步骤以 [docs/runbook.md](docs/runbook.md) 为准；逐文件职责以
  [docs/project_map.md](docs/project_map.md) 为准。
