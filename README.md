# opening_strength_fit

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon alpha。项目从
ClickHouse `stock.tick` 或本地 tick parquet 构造
`date x symbol x opening decision time` 样本，只使用 decision point 当时及以前可见的信息，
预测“当前主动买入并短持有约一分钟”的 buy-at-ask / sell-VWAP return proxy。

项目级采样窗口是 `09:30:00-09:40:00` 的整分钟 decision points。后续实验发现 `09:30`
更像特殊 opening snapshot，所以当前优化主线聚焦 `09:31-09:40` post-open decision points；
这只是当前建模子域，不是改变项目定义。

当前结论：opening proxy signal 真实存在，但 `09:31-09:40` raw short-alpha 里混有明显
“短正长负”的 dirty tail。下一阶段主线是两层建模：raw-label baseline 继续把短周期信号做强，
另建 learned risk layer 或 reranker，在只使用当时可见信息的前提下扣掉拥挤追涨、低深度和失衡 tail。

```text
alpha_model = raw short-label post-open model
risk_model  = learned dirty-risk / next-flip layer
final_score = alpha_score - lambda * risk_score
```

## 文档分工

| 文件 | 职责 |
| --- | --- |
| [docs/project_brief.md](docs/project_brief.md) | 研究目标、核心口径、当前结论和下一步路线。 |
| [docs/experiment_log.md](docs/experiment_log.md) | 实验事实源：run、数字、K8s 输出、配置索引。 |
| [docs/runbook.md](docs/runbook.md) | 本地 smoke、K8s Job、artifact sync、replay 和归档命令。 |
| [docs/project_map.md](docs/project_map.md) | 文件、模块和脚本索引。 |
| [experiments/results/README.md](experiments/results/README.md) | 可提交轻量结果目录约定。 |

## 核心口径

- 样本粒度：`date x symbol x decision_time`。
- 项目窗口：`09:30:00-09:40:00` 整分钟 decision points。
- 当前优化子域：`09:31:00-09:40:00`。
- 默认数据窗口：`09:15:00-09:45:00`。
- 股票池：A 股 `00/30.SZ` 和 `60/68.SH`，除非 config 显式指定 symbols file。
- 主评估：short-horizon Rank IC 和 Top100 excess。
- sanity check：next close；不直接混进 alpha model 的训练目标。

Label 定义：

```text
decision_t = sampled decision tick
entry_t    = decision_t + entry_tick_delay ticks
buy_price  = ask_price_1[entry_t]
sell_vwap  = VWAP(entry_t + 60s, entry_t + 120s)
label      = sell_vwap / buy_price - 1 - fee_bps / 10000
```

## 常用命令

```bash
cd /home/hefu/projects/opening_strength_fit
source .venv/bin/activate

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
python -m pytest -q
```

K8s 标准闭环见 [docs/runbook.md](docs/runbook.md)。大体积 predictions、models、PNGs 和临时报告保留在
ignored `output/`；可提交轻量证据放在 `experiments/results/`。
