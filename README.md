# opening_strength_fit

`opening_strength_fit` 研究 A 股开盘阶段的分钟级短周期信号。项目从 ClickHouse
`stock.tick` 或本地 tick parquet 构造 `date × symbol × decision_time` 样本，只使用决策时点
及以前可见的信息，为 `pool_L` 内部的选股与调仓提供 overlay score。

截至 2026-07-10：

```text
研究切片       09:31:00-09:40:00，每分钟一个 decision point
训练目标       short label + 0.30 × next-close label
历史基线       soft_core_reg_light
当前 incumbent grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse
当前阶段       原始特征无量纲化、全天分钟决策序列、可交易 pool-overlay 回测
在途实验       mech328 v3 histavg-activity；完成前不改变 incumbent
```

项目不以脱离股池的独立 universe 策略为目标。下一阶段聚焦三件事：尝试一个原始特征的无量纲化表达，
可使用多个参考值并借鉴 hist/path 口径；扩展到全天分钟频决策序列；把开盘 Top100 诊断补成完整、因果、
可交易的 `pool_L` overlay 组合回测，覆盖候选 refill、持仓与退出、全日资金预算、成本、容量和市场冲击，
并以成本后 PnL 验收。

## 文档

| 文件 | 唯一职责 |
| --- | --- |
| [docs/project_brief.md](docs/project_brief.md) | 当前目标、固定口径、incumbent、验收标准与下一步。 |
| [docs/experiment_log.md](docs/experiment_log.md) | 严格按时间排序的实验事实、数字、状态与结论。 |
| [docs/runbook.md](docs/runbook.md) | 从本地检查到 K8s、同步、验收的操作步骤。 |
| [docs/project_map.md](docs/project_map.md) | 目录、模块边界、数据流与 CLI 所有权。 |
| [experiments/README.md](experiments/README.md) | run/job/result 的目录契约和生命周期。 |

历史叙述、命令和代码索引不要复制到其他文档；需要引用实验数字时，链接到 experiment log。

## 本地检查

```bash
cd ~/projects/opening_strength_fit
source .venv/bin/activate
python -m pip install -c requirements.lock -e ".[dev]"
make ci
make contracts
```

集群实验闭环见 [docs/runbook.md](docs/runbook.md)。`experiments/results/` 与 `output/` 默认忽略；
run config、K8s manifest、trace 和实验日志共同承担可追溯性。
