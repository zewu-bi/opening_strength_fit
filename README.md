# opening_strength_fit

> **项目状态：已于 2026-08-14 冻结归档。** 当前仓库是最终研究快照，不再安排新实验；PVC 上的大型数据不随 Git 归档。

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

当前信号基准为 `opening_model`；当前结论、数据合同和封存边界见
[project brief](docs/project_brief.md)，不可变来源见
[canonical registry](experiments/canonical/opening.toml)，历史事实见
[experiment log](docs/experiment_log.md)，未来信息边界见
[leakage audit](docs/leakage_audit.md)。

`opening_label_matrix` 是当前权威 15-label 研究矩阵，但只代表诊断结论；在完成同口径策略验收前，
不会替换 `opening_model`。

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
.venv/bin/osf-train --help
```

最小本地库只安装 NumPy/Pandas/PyArrow/sklearn；按用途选择 `.[data]`、`.[models]`、`.[plots]`，开发和
集群环境分别使用 `.[dev]`、`.[cluster]`。Torch/CUDA 继续按节点驱动单独安装，避免污染 CPU 环境。

## 目录

```text
src/opening_strength_fit/  算法、数据适配器和 commands
tests/                     单元、回归、边界与 smoke 测试
examples/smoke/            无私有数据的最小样例
experiments/runs/          实验定义
experiments/jobs/          K8s manifest
experiments/canonical/     当前短名与不可变来源
experiments/evidence/      compact 证据
docs/project_brief.md      当前项目报告
docs/runbook.md            开发、数据、训练和集群操作
docs/experiment_log.md     历史实验事实与决策
```

## 工程结构与边界

```text
CLI / Kubernetes
  -> commands：参数、用例编排、输出摘要
    -> data / feature / label / training / evaluation domain
      -> schema / config / atomic I/O
    -> ClickHouse / S3 / HTTP / PVC / Kubernetes adapters
```

- domain 不得导入 `commands`；兼容 command 只做薄转发，业务实现保持单一来源。
- 样本三键统一使用 `schema.DECISION_KEY_COLUMNS`；DataFrame join 必须显式声明 key 与 `validate=`。
- 发布物先原子写数据和 manifest，最后写 `_SUCCESS`；它不单独证明内容正确。
- run、Job、image revision、compact evidence 是不可变 trace；cache、prediction 和模型留在 PVC。
- command 模块限制为 800 行，McCabe 上限 20，分支覆盖率不得低于 55%。
- 凭证只进入环境变量、`.env` 或 Kubernetes Secret；`.env`、私有数据、模型和带签名 URL 不进 Git。
- 本地/PVC/Git 历史清理必须先给出 dry-run、精确目标和恢复方案，不能作为普通重构的附带动作。

这些规则由 `tests/test_module_boundaries.py` 和 `make contracts` 执行。包采用 PyPA `src` layout 和
`pyproject.toml`，运行配置遵循“代码/构建、release config、运行”分离；不为了治理额外引入平台。

## 四份主文档

- [项目目标与当前结论](docs/project_brief.md)
- [未来信息专项审计](docs/leakage_audit.md)
- [复现和集群操作](docs/runbook.md)
- [实验历史与决策](docs/experiment_log.md)
