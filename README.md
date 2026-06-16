# opening_strength_fit

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon alpha。项目从 ClickHouse
`stock.tick` 或本地 tick parquet 构造 `date x symbol x opening decision time` 样本，
使用 decision point 当时及以前可见的信息，预测“当前主动买入并短持有约一分钟”的
buy-at-ask / sell-VWAP return proxy。

Current snapshot:

```text
sample window: 09:30:00-09:40:00, current slice 09:31:00-09:40:00
main label: mixed short + next-close label, w_long = 0.30
current baseline: soft_core_reg_light
current focus: 继续做强开盘短期模型；用 pool_L overnight overlay 验收增量
acceptance plots: experiments/results/backtests/optimization_overlay_acceptance_2022_2025/
current brief: docs/project_brief.md
run facts and artifact index: docs/experiment_log.md
```

## 文档入口

| 文件 | 内容 |
| --- | --- |
| [docs/project_brief.md](docs/project_brief.md) | 当前研究判断、固定口径、验收口径和关键证据摘要。 |
| [docs/runbook.md](docs/runbook.md) | 可执行命令：K8s Job、artifact sync、analysis 和排查。 |
| [docs/experiment_log.md](docs/experiment_log.md) | 实验事实源：run、数字、K8s 输出、归档路径和配置索引。 |
| [docs/project_map.md](docs/project_map.md) | 文件、模块和 CLI 索引。 |
| [experiments/README.md](experiments/README.md) | Run config 目录约定、run kind 映射和 TOML 模板入口。 |
| [experiments/results/README.md](experiments/results/README.md) | 可提交轻量结果目录约定。 |

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
