# Experiments

本目录保存实验定义与可执行 trace，不保存研究叙述。结果和结论见
[`docs/experiment_log.md`](../docs/experiment_log.md)，操作见 [`docs/runbook.md`](../docs/runbook.md)。

## 目录契约

| 路径 | 所有者 | Git 策略 |
| --- | --- | --- |
| `runs/<run_id>.toml` | 人工维护 | tracked；`run.id` 必须等于文件名 |
| `jobs/<run_id>*_job.yaml` | `osf-render-k8s-job` 生成 | tracked；保留执行 image/config trace |
| `config_templates/` | 人工维护 | tracked；只放可复用片段 |
| `results/metrics/` | artifact sync | ignored；训练指标本地 mirror |
| `results/backtests/` | analysis/audit/plot | ignored；compact CSV/JSON/SVG mirror |

`results/` 不是独立事实源。需要长期审核的小文件可显式 `git add -f`，但优先把 run config、trace、
代码 revision、关键数字和决策写入 experiment log。不要提交 raw prediction、pickle、大 parquet 或
可从 PVC 重拉的缓存。

## Run 生命周期

`[run].status` 只允许：

| status | 含义 |
| --- | --- |
| `queued` | 定义完成，尚未确认开始执行 |
| `running` | 已开始执行，产物不完整 |
| `completed` | 预期产物和审计都完成 |
| `canceled` | 主动终止，不再等待产物 |
| `superseded` | 未必失败，但已被更新 run 替代 |

禁止使用 `submitted`、`done` 等同义状态。`completed` 前应完成对应 artifact sync 和 audit；失败结果也要
保留 config，并在 experiment log 记录原因。

## 命名与最小字段

```toml
[run]
id = "<descriptive_run_id>"
kind = "<run_kind>"
description = "One hypothesis and one controlled change."
status = "queued"
```

- run id 包含模型/窗口/期间/唯一变量/版本，避免只写 `test2`。
- 一个 run 只验证一个核心假设；数据、label、窗口和 gate 尽量继承 incumbent。
- 输出根目录使用同一 run id。
- K8s `job_name` 可缩短，但必须能追溯回 run id。
- 任何会改变研究语义的修复或训练行为使用新版本 run，不覆盖旧结果。

## Run kind 与入口

| run kind | 执行入口 |
| --- | --- |
| `exploration` / standard training | `osf-train` |
| `labeled_cache` / `clickhouse_labeled_cache` | `osf-build-labeled-cache` |
| `cache_transform` / `target_cache` | `osf-build-target-label-cache` |
| `next_close_label_cache` | `osf-build-next-close-labels` |
| `pool_internal_analysis` | `osf-analyze-pool-internal-top100` |
| `capacity_audit` | `osf-audit-capacity` |
| `capacity_acceptance` | `osf-analyze-capacity-acceptance` |
| `execution_context` | `osf-extract-execution-context` |
| `ask_level_attribution` | `osf-ask-level-attribution` |
| `realistic_acceptance` | `osf-analyze-realistic-acceptance` |
| `strategy_acceptance` | `osf-audit-strategy-acceptance` |
| `exposure_input` | `osf-build-exposure-input` |
| `exposure_audit` | `osf-audit-exposure` |
| `feature_audit` | `osf-audit-feature-dependence` |
| `feature_hygiene` | `osf-audit-feature-hygiene` |
| `learned_risk_layer` | `osf-run-learned-risk-layer` |
| `alpha_conditioned_rolling_validation` | `osf-run-alpha-conditioned-rolling-validation` |
| `gap_risk_attribution` | `osf-run-gap-risk-attribution` |
| `score_risk_sweep` | `osf-run-score-risk-sweep` |

并非所有历史 run kind 都有 K8s renderer；`osf-render-k8s-job` 不支持时，按 runbook 使用对应 CLI，
并确保 trace 记录完整参数。

## 完成定义

训练 run：

```text
all shards _SUCCESS
metrics_by_year/month 可读
pool-internal analysis（正式候选）完成
compact sync 完成
osf-audit-experiments --require-metrics 通过
osf-check-project-contracts 通过
experiment log 已记录 decision
```

Artifact-only run 不要求训练 metrics，但必须生成该 kind 约定的 summary、trace 和主要输出；不要借用
训练 run 的 `_SUCCESS` 作为统一完成条件。`realistic_acceptance` 还必须在 trace 中保留 execution
context columns、约束和 modeling limitations；`strategy_acceptance` 必须保留三种 policy 的定义、
capacity/execution/tail 约束及 holding/causality 边界。

## Config templates

| 文件 | 用途 |
| --- | --- |
| `config_templates/stock_pool_selection.toml` | S/M/L selection mask |
| `config_templates/postopen_v2_features.toml` | post-open v2 feature switches |
| `config_templates/gpu_lightgbm.toml` | LightGBM GPU resource switches |
| `config_templates/pvc_layout_v2.toml` | 新 run 的结构化 PVC 输出与单份 prediction 策略 |

## PVC 布局

新 run 使用 `config_templates/pvc_layout_v2.toml`，按模型或任务类型写入
`/mnt/output/opening_strength_fit/runs/<category>/<run_id>`。完整目录规则、legacy 兼容和迁移操作见
`docs/runbook.md`；精确历史映射保存在 PVC 的 `.layout_migrations/`。
