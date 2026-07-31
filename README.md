# opening_strength_fit

`opening_strength_fit` 是 A 股开盘阶段分钟级 overlay 信号研究工具。它把 tick 数据变成
`date × symbol × decision_time` 样本，只使用决策时点及以前的信息，在既定股票池内部完成训练、排序、
容量分配、执行约束和尾部/集中度诊断。

这不是可直接交易的完整策略。当前实现覆盖开盘 `09:31-09:40` 的研究链路；完整日内持仓、退出、现金复用、
滑点和市场冲击账本仍是下一阶段工作。

## 研究逻辑

```text
tick / labeled input
  -> 因果采样、entry/exit label
  -> 盘口/成交/竞价/历史特征
  -> rolling OOS 训练与预测
  -> pool_L 内 TopN 排序
  -> capacity allocation
  -> execution / refill / overlap / tail diagnostics
```

当前信号基准是 `opening_model`，训练输入是 `opening_cache`。两者采用严格 clock-state 语义：每个决策
时钟读取当时已经可见的最后状态，entry 固定为 decision clock `+6s`。`opening_model` 的 `pool_L`
next excess 为 `17.7934 bps`，Top100 fee8 累和为 `10193.0 bps`，略高于旧 v4 基准的
`17.1714/9891.7 bps`。旧 v4 的 capacity/refill 结果保留为策略层历史参考，等待在 `opening_model`
上重跑。完整数字、边界和决策见 [project brief](docs/project_brief.md)、
[opening_model evidence](experiments/evidence/baselines/opening_model/)、
[canonical registry](experiments/canonical/opening.toml) 与 [experiment log](docs/experiment_log.md)。

2026-07-22 至 2026-07-29 的全天时序/隔夜 TCN 路线源于需求理解偏差，现已
[封存](experiments/archive/full_day_temporal_2026-07-22_2026-07-29/README.md)，不再属于当前研究主线。
下一步保持既有实验口径不变，只把 `09:31-09:40` 替换为另外 2–3 个固定日内窗口，比较模型选股能力
随时段的衰减。

## 可复现范围

| 层级 | 是否需要私有数据 | 入口 | Git 中保留 |
| --- | --- | --- | --- |
| 本地 smoke | 否 | `make smoke` | 小型 CSV、TOML、代码、预期结构 |
| 软件回归 | 否 | `make ci && make contracts` | 测试、依赖锁、项目/实验契约 |
| 研究复跑 | 是 | `osf-train --config ...` | run TOML、Job manifest、代码、compact evidence |
| 大型数据层 | 是 | cache/build/sync 命令 | 只保留 schema、配置、manifest 语义；Parquet/prediction/model 留在 PVC |

因此，干净 checkout 可以验证软件和跑通一个确定性 Ridge 示例；复跑正式 2022-2025 研究还需要原始 tick、
股票池和对应 PVC cache。仓库不会伪装成包含这些外部数据。

## 快速开始

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -c requirements.lock -e ".[dev]"

make smoke
make ci
make contracts
```

`make smoke` 的输出写到已忽略的 `output/smoke/`。CPU/Ridge/LightGBM 依赖由
`requirements.lock` 固定；GPU Torch 由集群镜像显式安装，正式 NN 的镜像 tag 保存在对应 run/job 中。

## 目录

```text
src/opening_strength_fit/  算法、数据适配器和 commands
tests/                     单元、回归、边界与 smoke 测试
examples/smoke/            无私有数据的最小复现样例
experiments/runs/          人工维护的实验定义
experiments/jobs/          渲染后的实际执行 trace
experiments/canonical/     当前短名、cache/model 标准与不可变来源映射
experiments/evidence/      Git 跟踪的摘要、稳健性结果和 trace
experiments/results/       本地结果镜像，忽略
docs/                      当前研究口径、操作与代码地图
output/                    cache、prediction、debug 和同步产物，忽略
```

继续阅读：

- [项目目标与当前结论](docs/project_brief.md)
- [复现和集群操作](docs/runbook.md)
- [代码与数据流](docs/project_map.md)
- [实验目录契约](experiments/README.md)
