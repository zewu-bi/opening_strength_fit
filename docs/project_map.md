# Project Map

本文只说明所有权与依赖方向；研究判断见 [project brief](project_brief.md)，操作见
[runbook](runbook.md)。

## 顶层结构

| 路径 | 责任 | Git 策略 |
| --- | --- | --- |
| `src/opening_strength_fit/` | 可安装 Python 包 | tracked |
| `tests/` | 单元、回归、契约、smoke | tracked |
| `examples/smoke/` | 无私有数据的最小训练例 | tracked |
| `experiments/runs/` | 实验意图与完整参数 | tracked |
| `experiments/jobs/` | 实际执行 manifest | tracked |
| `experiments/evidence/` | compact 结果与 trace | tracked |
| `experiments/results/` | 本地结果 mirror | ignored |
| `output/` | cache、prediction、模型和 debug | ignored |

## 依赖方向

```text
installed command -> commands/workflow -> domain algorithms -> schema/types/io
                                      -> ClickHouse/K8s/PVC/S3 adapters
```

`pyproject.toml` 的 `osf-*` 入口直接指向 `commands.*:main`，不再保留一层只做 import 转发的 CLI
模块。domain 模块不得反向依赖 `commands`；项目契约和测试会检查循环依赖及层级边界。

## 核心数据流

```text
raw tick or labeled frame
  -> dataset / sampling / labels
  -> features_* / feature_transforms
  -> training_* / model_* / torch_model
  -> prediction frame
  -> pool_internal / capacity / execution / strategy_acceptance
  -> evidence summary + trace
```

### 数据与 schema

- `schema.py`：列名、盘口层级、时钟和 timestamp 规范。
- `clickhouse_ticks.py`、`clickhouse_daily_reference.py`：外部数据适配器。
- `dataset.py`、`sampling.py`、`labels.py`：raw tick 到因果样本和短周期 label。
- `horizon_*`：本地/ClickHouse horizon label 与报告。
- `cache_lock.py`、`cache_manifest.py`：原子发布、并发和 fingerprint。
- `universe.py`、`stock_pool.py`：股票 universe 与 S/M/L membership。

### 特征与训练

- `features_base.py`：盘口、成交、动量和竞价基础特征。
- `features_postopen.py`、`features_history.py`：决策时点轨迹与历史参照。
- `feature_transforms/`：截面和机制化 value transform。
- `features.py`：稳定的公共 feature API；实现仍归属上述模块。
- `training_data.py`、`training_labeled.py`、`training_windows.py`：输入、变换和时间切分。
- `training_modeling.py`、`training.py`：fit/predict/metric 编排。
- `model_*`、`torch_model/`：模型类型、sklearn、NN、预测和指标。
- `model.py`：稳定的公共 model API。

### 分析与验收

- `pool_internal_*`：股池内 TopN、稳定性与图表。
- `capacity_audit.py`、`capacity_acceptance.py`：容量组合和收益。
- `execution_diagnostics.py`、`realistic_acceptance.py`：执行上下文与重放。
- `strategy_acceptance.py`：capacity/no-refill/refill、overlap 和 tail 的统一验收。
- `exposure_audit.py`：规模、行业和集中度。
- `optimization_*`：固定信号比较和验收图。
- `legacy/`：仅用于复现封存诊断；新代码不得依赖其兼容路径。

### 命令与基础设施

- `commands/experiment_run.py`：训练入口。
- `commands/*_build.py`、`*_audit.py`、`*_analysis.py`：用例编排。
- `commands/k8s_*.py`：Job 渲染。
- `commands/artifact_sync*.py`：远端同步、合并和 compact evidence 选择。
- `pvc_layout.py`、`k8s.py`：PVC 布局与 K8s helper。

## 边界规则

- run id、TOML、Job、入口和输出目录必须对齐；
- 代码 revision、配置、镜像和 compact trace 进 Git，cache/prediction/model 不进 Git；
- 新实验不能覆盖旧 run；状态只用 `queued/running/completed/canceled/superseded`；
- `experiments/evidence/` 单文件不超过 1MB，禁止 Parquet、pickle 和模型二进制；
- `tests/test_module_boundaries.py` 与 `osf-check-project-contracts` 是结构门禁。
