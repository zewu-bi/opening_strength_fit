# opening_strength_fit

`opening_strength_fit` 是一个 A 股开盘阶段 short-horizon alpha 研究项目。它从
ClickHouse `stock.tick` 或本地 tick parquet 读取集合竞价与开盘盘口/成交数据，在
`09:30-09:40` 的整分钟 decision point 上构造
`trading day x symbol x opening timestamp` 样本，预测“当前主动买入并短持有约一分钟”的
future return proxy，并检查模型分数是否有稳定横截面排序价值。

当前结论很明确：opening proxy signal 存在，但它不是完整实盘策略。A 股 T+1 约束下，
60s label 只能作为 microstructure discovery label；已经完成 close / next close 衰减检查。
后续目标已调整为先把开盘后横截面信号做强，再考虑交易约束和日频 overlay。

```text
ClickHouse stock.tick / local tick parquet
-> schema 标准化 + A 股 universe
-> opening features + short-horizon label
-> Ridge / GBM / LightGBM rolling training
-> IC / bucket / TopN metrics
-> constrained opening replay
-> lightweight evidence in experiments/results/
```

## 文档分工

| 文档 | 职责 |
| --- | --- |
| [docs/project_brief.md](docs/project_brief.md) | 研究目标、阶段结论和下一步路线。 |
| [docs/runbook.md](docs/runbook.md) | 日常实验命令、K8s 闭环、replay 和归档步骤。 |
| [docs/experiment_log.md](docs/experiment_log.md) | 已跑实验、关键数值和阶段性证据。 |
| [docs/project_map.md](docs/project_map.md) | 逐文件代码和脚本索引。 |
| [experiments/results/README.md](experiments/results/README.md) | 可提交轻量结果目录约定。 |

## 核心口径

默认数据源：

```text
host: ch.db.prod.highfortfunds.com
table: stock.tick
window: 09:15:00 - 09:45:00
sample: 09:30:00 - 09:40:00 integer-minute decision points
universe: 00/30.SZ and 60/68.SH A-share symbols, or [universe].symbols_file
```

当前 short-horizon label：

```text
decision_t = 当前样本 tick
entry_t = decision_t 之后第 entry_tick_delay 个 tick
buy_price = ask1[entry_t]
sell_vwap = VWAP(entry_t + 60s, entry_t + 120s)
label = sell_vwap / buy_price - 1 - fee_bps / 10000
```

已归档 Ridge/GBM baseline 使用 `entry_tick_delay = 0`；LightGBM 主线使用
`entry_tick_delay = 1`，并补充 delay0/2 敏感性。`entry_delay_seconds` 记录总等待时间，
`entry_max_tick_gap_seconds` 记录 decision-to-entry 路径最大相邻 tick gap。

X 特征只允许使用 decision point 当时及以前可见的信息。新主线优先强化开盘后的盘口结构、
ask/bid 档位、深度变化、成交活跃度和短期动量；集合竞价特征只作为对照组或诊断项。
`model.feature_columns()` 会排除 label、entry/sell future 字段和 future timestamp 字段。

## 训练和 Replay

实验由 `experiments/runs/*.toml` 驱动。长窗口正式路径优先使用 K8s CPU LightGBM Job 读取 PVC
labeled cache：

```toml
[data]
source = "labeled_pvc"
labeled_path = "/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay<0|1|2>_labeled.parquet"

[model]
name = "lightgbm"
device_type = "cpu"
max_bin = 63
```

训练/评估/replay 边界：

| 变化 | 放哪里 |
| --- | --- |
| label、horizon、entry delay、feature set、universe/candidate、固定部署硬过滤 | 新 run / 重训 |
| Rank IC、Top100 选股收益、分分钟诊断 | 主评估 |
| 容量 | 暂只看 ask1，先不做多档 sweep 主线 |
| fee、slippage、spread、状态、tick 新鲜度、同股冷却 | 后续 replay |
| close / next close / T+1 价值 | 信号增强后再做新 horizon label 或日频候选 overlay |

已归档的 LightGBM delay replay 默认 6 个场景：

```text
proxy_top20
cost_10bps
tradable_cost
liquidity_cost
capacity_l3_1m
capacity_l5_2m
```

这些 replay 是历史压力测试证据。新主线暂不把多档 sweep 作为优化目标，容量只保留 ask1
可买量口径，避免过早把问题转成执行建模。

## 快速开始

```bash
cd /home/hefu/projects/opening_strength_fit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

set -a
. ./.env
set +a
```

预检：

```bash
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

本地 smoke：

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet

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

不要在本地为一年或多月窗口准备完整 labeled dataset；长窗口数据先在集群 materialize 到 PVC cache。

## K8s 闭环

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-lgbm-cpu-v1
docker build --build-arg CACHE_BUST=${TAG} -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

python scripts/render_k8s_job.py \
  --config experiments/runs/<run_id>.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

rg -n "nvidia|gpu|nodeSelector|envFrom|image:" experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research logs -f job/opening-strength-<run-slug> -n bizewu
```

reader、拉回和归档：

```bash
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_reader_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/opening-strength-read-<run-slug> -n bizewu --timeout=300s

python scripts/pull_k8s_metrics.py --config experiments/runs/<run_id>.toml
python scripts/fetch_k8s_predictions.py --config experiments/runs/<run_id>.toml --output-dir output/predictions/<run_id>
python scripts/record_experiment.py --config experiments/runs/<run_id>.toml
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

## 分析命令

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/<run_id>_metrics_by_year.csv

python scripts/compare_opening_results.py

python scripts/run_lgbm_delay_replays.py --check-interface-only
python scripts/run_lgbm_delay_replays.py
python scripts/plot_lgbm_delay_decay.py

python scripts/run_alpha_horizon_decay.py \
  --decision-time 09:30:00 \
  --horizon 1m --horizon 2m --horizon 5m --horizon 10m \
  --horizon close --horizon next_close \
  --no-sampled-intraday \
  --clickhouse-intraday-labels \
  --clickhouse-close-labels \
  --allow-missing-horizons \
  --output-root output/reports/opening_alpha_horizon_decay_delay2_clickhouse_point_0930_selected
```

`output/` 保存本地 parquet、模型、图表和临时报告，默认不提交；
`experiments/results/` 保存可提交的轻量 CSV/JSON 证据。

## 当前状态

当前本地实验注册表保留：

- `1m3d` 小窗口 Ridge/GBM 对比。
- `1y_next_month` Ridge/GBM/strong 对比。
- `1y_next_month` CPU LightGBM delay0/1/2 普通 universe 与 strong 分支。

CPU LightGBM delay 结果已完成，metrics 已归档，predictions 已拉回。关键结果见
[docs/experiment_log.md](docs/experiment_log.md)；摘要如下：

| 分支 | delay0 Top20 bps | delay1 Top20 bps | delay2 Top20 bps |
| --- | ---: | ---: | ---: |
| universe | +57.04 | +45.37 | +39.00 |
| strong | +42.04 | +26.41 | +18.97 |

delay2 universe 在基础 liquidity 约束下约 `+28.74 bps`，容量场景
`capacity_l3_1m` / `capacity_l5_2m` 约 `+21.88` / `+18.38 bps`。Alpha horizon decay 显示：
固定 `09:30` opening score 到 close / next close 仍有弱正 Rank IC，但 next close Top20 收益不稳定；
`09:30-09:39` 简单平均后长周期排序效果基本消失。

下一步 active work：先做信号增强。评估标准改为 Rank IC 和 Top100 选股收益；重点检查
`09:30` 是否受集合竞价特征或跨竞价边界累计成交字段影响，并加强开盘后 ask/bid 档位、
盘口深度和队列变化特征。交易约束、日频 overlay 和多档容量放到信号变强之后。

## 开发约定

- 每个实验一个 `experiments/runs/<run_id>.toml`，`[run].id` 必须等于文件名。
- 拉回并确认结果后，把 config 的 `[run].status` 更新为 `completed`。
- 新代码、config 或依赖进入集群前，需要 build/push 新镜像并重新 render Job YAML。
- 大体积 predictions、模型文件、图表和临时输出放在 `output/`，不要提交。
- 可提交证据放在 `experiments/results/metrics/` 和 `experiments/results/backtests/`。
- 提交前运行 `python scripts/audit_experiments.py` 和 `python scripts/check_workflow_coverage.py`。
