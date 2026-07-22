# opening_strength_fit

`opening_strength_fit` 研究 A 股开盘阶段的分钟级短周期信号。项目从 ClickHouse
`stock.tick` 或本地 tick parquet 构造 `date × symbol × decision_time` 样本，只使用决策时点
及以前可见的信息，为 `pool_L` 内部的选股与调仓提供 overlay score。

截至 2026-07-22：

```text
研究切片       09:31:00-09:40:00，每分钟一个 decision point
训练目标       short label + 0.30 × next-close label
历史基线       soft_core_reg_light
当前 incumbent grouped_gated_v2_mech328_v2_robust_zscore_gelu_mse
已完成挑战者 普通 328 mech v3：pool_L next excess 16.3318 bps
旧 cache 最强  auction-fresh pruned mech v3：pool_L next excess 16.9692 bps
canonical v4   auction-pruned control / 多分母：pool_L next excess 16.8024 / 17.1714 bps
```

项目不以脱离股池的独立 universe 策略为目标。canonical cache v4 的 control 与只新增 25 个
无量纲比例的 challenger 已完成；多分母只带来小幅均值增量，分期胜率不足以支持保留第二条复杂分支。
control 的统一下游验收也已完成：realistic no-refill 的 fill/累计资金净收益为
`80.7803%/7183.6 bps`，因果可见 refill 提高到 `99.9969%/8431.1 bps`，但 P95 winsor 后仍为
`-9.33 bps`，且 overlap 与月块下界略有恶化，因此 refill 保留为标准策略机制和验收项、当前策略不晋级。
下一阶段扩展到全天分钟级、因果可见的时序 label/score，并构建完整持仓、退出、现金复用、成本、容量和
市场冲击账本，以成本后 PnL 与尾部稳健性验收。

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
