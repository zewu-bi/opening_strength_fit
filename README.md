# opening_strength_fit

`opening_strength_fit` 是 A 股开盘阶段分钟级 overlay 信号研究工具：从因果可见的 tick 状态生成样本，
在既定股票池内完成训练、排序、容量和执行诊断。它不是完整交易系统。

## 研究逻辑

```text
tick / labeled input
  -> 因果采样、feature、label
  -> rolling OOS 训练与预测
  -> pool_L 内排序
  -> capacity / execution / risk diagnostics
```

当前信号基准为 `opening_model`；当前结论、数据合同和下一步见
[project brief](docs/project_brief.md)，不可变来源见
[canonical registry](experiments/canonical/opening.toml)，历史事实见
[experiment log](docs/experiment_log.md)。

## 可复现范围

| 层级 | 私有数据 | 入口 | Git 中保留 |
| --- | --- | --- | --- |
| 本地 smoke | 否 | `make smoke` | 小型输入、配置、代码 |
| 软件回归 | 否 | `make ci && make contracts` | 测试、依赖锁、契约 |
| 正式研究 | 是 | `osf-train --config ...` | run、Job、代码、compact evidence |
| 大型数据层 | 是 | cache/build/sync 命令 | schema、配置和 lineage；数据留在 PVC |

干净 checkout 可验证软件和最小 Ridge 流程；正式复跑还需要 tick、股票池和 PVC 数据。

## 快速开始

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -c requirements.lock -e ".[dev]"

make smoke
make ci
make contracts
```

## 目录

```text
src/opening_strength_fit/  算法、数据适配器和 commands
tests/                     单元、回归、边界与 smoke 测试
examples/smoke/            无私有数据的最小样例
experiments/runs/          实验定义
experiments/jobs/          K8s manifest
experiments/canonical/     当前短名与不可变来源
experiments/evidence/      compact 证据
docs/                      研究口径、操作和代码地图
```

继续阅读：

- [项目目标与当前结论](docs/project_brief.md)
- [复现和集群操作](docs/runbook.md)
- [代码与数据流](docs/project_map.md)
- [实验目录契约](experiments/README.md)
