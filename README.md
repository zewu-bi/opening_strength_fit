# opening_strength_fit

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon alpha。项目从 ClickHouse
`stock.tick` 或本地 tick parquet 构造 `date x symbol x opening decision time` 样本，
使用 decision point 当时及以前可见的信息，预测“当前主动买入并短持有约一分钟”的
buy-at-ask / sell-VWAP return proxy。

Current snapshot:

```text
slice: 09:31:00-09:40:00 decision points
label: mixed short + next-close, w_long = 0.30
baseline: soft_core_reg_light
overlay final candidate: mech328 v2 dimensionless robust-zscore
next scope: independent causal strategy; longer time range; minute-frequency label path features
acceptance: strategy PnL/capacity/exposure first; overlay gates retained as diagnostics
```

## 文档职责

| 文件 | 内容 |
| --- | --- |
| [docs/project_brief.md](docs/project_brief.md) | 当前判断、固定口径、验收 gate、晋级证据包和下一步目标。 |
| [docs/experiment_log.md](docs/experiment_log.md) | 实验事实源：run id、数字、状态、归档路径。 |
| [docs/runbook.md](docs/runbook.md) | 可执行流程和命令。 |
| [docs/project_map.md](docs/project_map.md) | 代码、CLI、目录索引。 |
| [experiments/README.md](experiments/README.md) | 实验目录和 run kind 入口。 |

更新文档时按职责放置：当前判断和晋级标准写 brief，实验事实写 log，命令写 runbook，
代码索引写 map。不要在多个文档重复同一段操作规则或实验结论。

## 常用命令

```bash
cd ~/projects/opening_strength_fit
source .venv/bin/activate
python -m pip install -e ".[dev]"

make ci
make contracts
```

K8s 标准闭环见 [docs/runbook.md](docs/runbook.md)。可提交轻量证据放在
`experiments/results/`；大体积 predictions 只按需拉到 ignored `output/` 做 debug，
用后可删。
