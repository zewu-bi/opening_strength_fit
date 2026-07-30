# Experiments

实验目录保存“意图、执行、证据”，不保存可重建的大型数据。

## 目录契约

| 路径 | 内容 | Git |
| --- | --- | --- |
| `runs/<run_id>.toml` | 完整参数、状态、输入/输出 lineage | tracked |
| `jobs/<run_id>*_job.yaml` | 渲染后的镜像、命令、资源和挂载 trace | tracked |
| `config_templates/` | 可复用配置片段 | tracked |
| `scripts/` | 正式或历史诊断入口 | tracked |
| `evidence/` | compact summary、稳健性表和 trace | tracked |
| `archive/` | 已终止路线的配置、实现快照与证据；不参与当前入口和契约 | tracked |
| `results/` | 本地/PVC 结果 mirror | ignored |

cache、prediction、逐行 replay、pickle、模型和大 Parquet 不进入 Git。它们由 run config、Job、manifest、
代码 revision 和 evidence trace 定位。

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

run id 应包含模型、窗口、期间、唯一变量和版本。影响数据、label、特征或训练语义的变化必须创建新 run，
不得覆盖历史配置。

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

默认 tracked 目标为 `experiments/evidence/`。单文件上限 1MB，禁止 Parquet、pickle 和模型二进制；需要
审阅的大表应先聚合。每个 `evidence/backtests/<run_id>/` 必须存在同名 run TOML。

canonical multiden 额外保留 short IC + next excess、Top100 累和、Top1000 平滑分桶和 Top1000 十组
收益分布四图，以及每图的 compact CSV/trace。已有本地 mirror 时运行 `make evidence-four-figures` 刷新；
control 只作为前两图的 ablation baseline。

日内窗口衰减实验也沿用同一四图契约；每个完成窗口使用训练 run id 建独立 evidence 目录，并额外保留
pool summary 与 SHA-256 manifest。明显落后于 09:31 基准的窗口归档为 diagnostic checkpoint，不进入
canonical evidence，也不替换 opening policy。

PVC 布局和故障处理见 [runbook](../docs/runbook.md)，当前研究判断见
[project brief](../docs/project_brief.md)。
