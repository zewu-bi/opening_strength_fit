# Experiments

实验目录保存“意图、执行、证据”，不保存可重建的大型数据。

## 目录契约

| 路径 | 内容 | Git |
| --- | --- | --- |
| `runs/<run_id>.toml` | 完整参数、状态、输入/输出 lineage | tracked |
| `jobs/<run_id>*_job.yaml` | 渲染后的镜像、命令、资源和挂载 trace | tracked |
| `canonical/opening.toml` | 当前 `opening_base`、`opening_cache`、`opening_model` 短名与不可变来源映射 | tracked |
| `config_templates/` | 可复用配置片段 | tracked |
| `scripts/` | 正式或历史诊断入口 | tracked |
| `evidence/` | compact summary、稳健性表和 trace | tracked |
| `archive/` | 已终止路线的配置、实现快照与证据；不参与当前入口和契约 | tracked |
| `results/` | 本地/PVC 结果 mirror | ignored |

cache、prediction、逐行 replay、模型和大 Parquet 不进入 Git；通过 run config、Job、manifest、代码
revision 和 evidence trace 定位。

## Run 契约

最小定义：

```toml
[run]
id = "<与文件名一致>"
kind = "<workflow kind>"
description = "<单一假设和唯一改动>"
status = "queued"
```

状态只允许：

```text
queued -> running -> completed
                 \-> canceled
queued/completed -> superseded
```

历史 run id、Job、trace 和 PVC lineage 不改名。新任务使用
`opening_<window>_<semantic_change>`；语义变化创建新 run，不覆盖历史配置。完整命名规则见
[canonical README](canonical/README.md)。

## 完成条件

训练 run 完成前应满足：

1. 所有 shard 有 `_SUCCESS`；
2. 年/月 metrics 可读；
3. 正式候选完成 pool-internal 分析；
4. compact artifact 已同步并选择为 evidence；
5. `make contracts` 通过；
6. experiment log 记录结论或失败解释。

artifact-only run 不强求训练 metrics，但必须有该 workflow 的 summary、trace 和成功标记。

## Evidence

```bash
osf-sync-experiment-artifacts \
  --config experiments/runs/<run_id>.toml \
  --artifacts --record
```

默认 tracked 目标为 `experiments/evidence/`。单文件上限 1MB，禁止 Parquet、pickle 和模型二进制；大表
先聚合。每个 `evidence/backtests/<run_id>/` 必须有同名 run TOML。正式模型和日内窗口实验沿用固定四图、
compact CSV、trace 与 SHA-256 manifest 契约；具体内容见 [evidence README](evidence/README.md)。

PVC 布局和故障处理见 [runbook](../docs/runbook.md)，当前研究判断见
[project brief](../docs/project_brief.md)。
