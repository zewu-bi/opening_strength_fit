# opening_strength_fit

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon alpha。项目从 ClickHouse
`stock.tick` 或本地 tick parquet 构造 `date x symbol x opening decision time` 样本，
使用 decision point 当时及以前可见的信息，预测“当前主动买入并短持有约一分钟”的
buy-at-ask / sell-VWAP return proxy。

Current snapshot:

```text
sample window: 09:30:00-09:40:00, current slice 09:31:00-09:40:00
main label: mixed short + next-close label, w_long = 0.30
archived feature/model: soft_core_reg_light
completed archive: 2024 monthly validation, 2018H1..2025H2 halfyear/OOS, and 2020-2025 rolling-window summary
current baseline slice: 2022-2025 universe + pool_L cluster-side quarterly excess/IC and daily cumulative archived
next work: 2022-2025 pool_L feature engineering and model optimization
```

## 文档入口

| 文件 | 内容 |
| --- | --- |
| [docs/project_brief.md](docs/project_brief.md) | 研究目标、核心口径、当前主线和关键里程碑。 |
| [docs/runbook.md](docs/runbook.md) | K8s Job、artifact sync、analysis 和排查命令。 |
| [docs/experiment_log.md](docs/experiment_log.md) | 实验事实源：run、数字、K8s 输出、配置索引。 |
| [docs/project_map.md](docs/project_map.md) | 文件、模块和 CLI 索引。 |
| [experiments/README.md](experiments/README.md) | Run config 目录约定、run kind 映射和 TOML 模板入口。 |
| [experiments/results/README.md](experiments/results/README.md) | 可提交轻量结果目录约定。 |

## 常用命令

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pip install -e ".[dev]"

make ci
make contracts
```

K8s 标准闭环见 [docs/runbook.md](docs/runbook.md)。可提交轻量证据放在
`experiments/results/`；大体积 predictions 只按需拉到 ignored `output/` 做 debug，
用后可删。
