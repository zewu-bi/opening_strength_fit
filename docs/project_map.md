# Project Map

本文件回答“代码在哪里、谁负责什么、依赖应往哪一层流动”。研究判断见
[project_brief.md](project_brief.md)，实验事实见 [experiment_log.md](experiment_log.md)，操作见
[runbook.md](runbook.md)。

## 顶层目录

```text
opening_strength_fit/
├── src/opening_strength_fit/   可安装 Python 包
├── tests/                      单元、回归和项目契约测试
├── experiments/
│   ├── runs/                   人工维护的 TOML 实验定义
│   ├── jobs/                   由 config 渲染的 K8s manifest
│   ├── scripts/                大样本研究诊断与固定验收复画脚本
│   ├── config_templates/       可复用配置片段
│   └── results/                ignored compact result mirror
├── docs/                       当前维护文档
│   └── archive/                只读历史叙述
└── output/                     ignored 本地同步、debug 与 legacy 产物
```

根文件：

| 文件 | 职责 |
| --- | --- |
| `pyproject.toml` | 包元数据、依赖、34 个 `osf-*` console scripts、pytest/coverage/Ruff 配置 |
| `requirements.lock` | 本地/集群 Python 环境约束；GPU Torch 仍由镜像构建显式安装 |
| `Makefile` | install-dev、install-cluster、test、lint、format、contracts、CI 的标准入口 |
| `Dockerfile` | CPU/GPU 可选安装的运行镜像 |
| `.github/workflows/ci.yml` | GitHub lint 与 pytest 门禁 |

## 依赖方向

```text
cli -> command/workflow -> domain algorithms -> schema/types/io
                         -> infrastructure adapters
```

- `cli/` 只把 console script 转给一个 workflow `main`。
- `commands/` 负责编排参数、配置、I/O 和 artifact，不承载可复用 dataframe 算法。
- 根级 domain modules 负责数据、特征、标签、模型和分析算法。
- K8s、PVC、ClickHouse、Ceph 等外部适配器不能被纯算法反向依赖。
- 兼容 facade（当前 `features.py`、`model.py`）只做显式 re-export；项目内部新代码直接引用所有者模块。

## 入口层

| 路径 | 职责 |
| --- | --- |
| `cli/` | 每个文件至多一个 command import 和 `main()` 调用 |
| `commands/` | 训练、cache、audit、analysis、plot、K8s、artifact sync 的用例编排 |
| `commands/artifact_sync*.py` | remote fetch、metrics 合并、compact artifact 归档 |
| `commands/k8s_*.py` | training/analysis manifest 渲染，不包含训练算法 |

常用 CLI 按用途分组：

| 用途 | CLI |
| --- | --- |
| 训练 | `osf-train`、`osf-run-experiment` |
| 数据/cache | `osf-probe-clickhouse-data`、`osf-build-labeled-cache`、`osf-build-target-label-cache`、`osf-build-next-close-labels` |
| 核心验收 | `osf-analyze-pool-internal-top100`、`osf-plot-optimization-direction-comparison` |
| 容量/执行 | `osf-audit-capacity`、`osf-analyze-capacity-acceptance`、`osf-extract-execution-context`、`osf-ask-level-attribution`、`osf-analyze-realistic-acceptance` |
| 暴露 | `osf-build-exposure-input`、`osf-audit-exposure` |
| 研究诊断 | feature dependence/hygiene、horizon decay、risk sweep、weekly plots |
| 基础设施 | `osf-render-k8s-job`、`osf-rolling-job-status`、`osf-sync-experiment-artifacts` |
| 契约 | `osf-audit-experiments`、`osf-check-project-contracts` |

## 数据、schema 与 I/O

| 模块 | 所有权 |
| --- | --- |
| `schema.py` | 列名、盘口层级、时钟标准化、timestamp 构造 |
| `io/` | dataframe I/O、原子写与 JSON artifact 序列化 |
| `clickhouse_ticks.py` | `stock.tick` 查询、表名校验、字段标准化 |
| `clickhouse_daily_reference.py` | `stock.daily_bar_jy` 严格 prior-session 市值/股本查询、单位归一化与 many-to-one enrichment |
| `dataset.py` | raw tick → labeled feature frame |
| `sampling.py` | decision-point 采样 |
| `labels.py` | entry delay、buy price、sell VWAP、short label |
| `horizon_*` | local/ClickHouse horizon labels 与报告算法 |
| `cache_lock.py` / `cache_manifest.py` | cache 并发、ready marker、schema/fingerprint manifest |
| `universe.py` / `stock_pool.py` | A 股 universe、S/M/L pool 读取和 mask |

`training_data.py` 是当前迁移中的 workflow 边界：负责 source resolution、ClickHouse 日频 reference enrichment、
PVC projection 与加载；
跨文件历史/截面变换必须在 concat 后统一执行，不能把文件边界当成语义边界。

## 特征与 target

| 模块 | 所有权 |
| --- | --- |
| `features_base.py` | order book、trade flow、momentum、pre-open 与基础 frame |
| `features_postopen.py` | decision-row post-open trajectory、queue/depth state |
| `features_history.py` | prior-date same-minute surprise、path confirmation、historical activity |
| `feature_transforms/cross_sectional.py` | cross-sectional value/rank 与 price-scale transforms |
| `feature_transforms/mechanism.py` | mechanismized v1/v2/v3 规则及共享 reference helpers |
| `feature_config.py` | include/drop/limit 配置解析 |
| `feature_hygiene.py` | correlation cluster、keep/drop 候选 |
| `targets.py` | xs transform、heat/guard/risk-shrunk target |
| `candidates.py` | 只用可见信息的 opening candidate filter |
| `features_relative.py` / `features.py` | 兼容 facade；不新增实现 |

## 模型与训练

| 模块 | 所有权 |
| --- | --- |
| `model_types.py` | prediction model dataclass 与 context columns |
| `model_features.py` | numeric feature selection 与 target cleaning |
| `model_sklearn.py` | Ridge、sklearn GBM、LightGBM fitters |
| `torch_model/architectures.py` | 语义特征分组与网络结构 |
| `torch_model/preprocessing.py` | feature value transforms 与 global/symbol standardization |
| `torch_model/training.py` | device、loss、training loop、early stop 与 gate diagnostics |
| `torch_model/prediction.py` | batch scoring |
| `model_prediction.py` | single/ensemble/clock-segment prediction dispatch |
| `model_metrics.py` | IC、Rank IC 与 grouped metrics |
| `model_torch.py` / `model.py` | 兼容 facade；不新增实现 |
| `training_args.py` | 训练 CLI 参数 |
| `training_windows.py` | chronological/annual/monthly split 与 evaluation settings |
| `training_labeled.py` | label/feature/target/sample-weight 变换管线 |
| `training_modeling.py` | per-split fit/predict/metric orchestration |
| `training.py` | 顶层训练闭环与标准 artifact 写入 |

## 分析与验收

| 模块族 | 职责 |
| --- | --- |
| `pool_internal_*` | pool 内选择、稳定性、company bridge 与 SVG |
| `capacity_audit.py` | 受资金、参与率、深度与集中约束的组合分配 |
| `capacity_acceptance.py` | selected allocation × next-close label 的收益验收 |
| `execution_diagnostics.py` | execution context 与 ask-level attribution |
| `realistic_acceptance.py` | selected-order execution replay；不负责生成候选/refill |
| `exposure_audit.py` | TopN/给定组合相对候选池的暴露与集中度 |
| `optimization_direction_*` | 固定 acceptance data、plot 和 workflow |
| `experiments/scripts/run_top1000_rank_bucket_diagnostics.py` | Top1000 IC、平滑分桶和100 bps收益区间计数 |
| `experiments/scripts/plot_top1000_score_bucket_return_histogram.py` | 从 compact CSV 复画固定坐标的Top1000十曲线收益分布验收图 |
| `evaluation.py` / `reports.py` | score bucket、TopN 与标准汇总 |
| `rolling.py` | 时间切分原语 |

旧 NN multiscale bucket 工具位于 `legacy/`，只用于复现已封存诊断，不参与常规依赖图；原根模块与
command 路径保留薄兼容 facade。

## 实验与产物边界

| 路径 | tracked | 说明 |
| --- | --- | --- |
| `experiments/runs/` | yes | 实验意图与状态 |
| `experiments/jobs/` | yes | 可执行 manifest trace |
| `experiments/results/` | no, default | compact 本地/PVC mirror；必要证据单独 force-add |
| `output/artifacts/` | no | 当前同步副本和 partial metrics |
| `output/legacy/` | no | 历史/debug/可重拉大文件 |
| `docs/archive/` | yes | 冻结的人工历史叙述，不代表当前状态 |

`pvc_layout.py` 统一管理 PVC run 分类、shard 命名和 legacy/v2 双读候选。renderer、artifact
sync、experiment audit 与 prediction reader 不得各自重新拼接路径。

## 工程契约

`tests/test_module_boundaries.py` 和 `osf-check-project-contracts` 应持续保证：

- CLI wrapper 保持薄；
- facade 保持薄且只 re-export；
- command/workflow 不越过行数和依赖边界；
- run id、状态、config、Job 和 entrypoint 对齐；
- 未知实验状态、缺失显式特征和循环依赖直接失败；
- ignored 结果目录不被无意提交。
