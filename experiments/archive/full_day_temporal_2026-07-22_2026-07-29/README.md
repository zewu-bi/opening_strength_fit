# 全天时序/隔夜 TCN 路线封存

> 状态：terminated / archived
>
> 研究期：2026-07-22 至 2026-07-29
>
> 封存日期：2026-07-29

## 封存原因

这条路线来自对 mentor 要求的误解。误解后的问题是：

```text
全天分钟收益路径
  -> 日级 sequence
  -> D 收盘到 D+1 收盘目标
  -> TCN / attention / backward-path 选股
```

mentor 的实际要求不是建立全天 sequence-to-one 或隔夜模型，而是复用既有
`09:31-09:40` 实验范式，把相同长度的决策窗口移到一天里另外 2–3 个固定时段，在其他条件不变时
比较模型 OOS 选股能力的衰减。

因此，本目录中的路线整体终止，不作为当前候选、基准、诊断主线或后续开发基础。历史结果不支持继续
投入该方向；若未来出现独立的全天时序需求，应重新立项，而不是从当前主线隐式恢复。

## 已封存内容

- `docs/`：全天因果 label/cache、日频目标、序列和 TCN 的完整口径说明；
- `runs/`：已执行、已提交及尚未提交的 cache、label、TCN、linear、backward 和 cross-mask 配置；
- `jobs/`：对应的 K8s Job 与 ConfigMap；
- `implementation/`：路线专用 source、command 和测试在封存时的工作区快照；
- `scripts/`：backward suite 与 cross-mask 归因脚本；
- `evidence/`：forward/backward 1m TCN 的 compact 汇总与解释；
- `docker/`：路线专用 overlay Dockerfile。

封存时工作区尚未提交的 absolute/relative、multiscale、linear 和 cross-mask 变体也已保留，因此这里
不仅是最后一次 Git commit 的副本，而是 2026-07-29 停止时的完整路线快照。

## 历史结论

首个 all-A 1m TCN 的表面 Top100 excess 为 `93.57 bps`，但约 `89.6%` 的 Top100 原始收益来自
D 日已经涨停附近的股票。该结果主要编码涨停状态和有效性 mask，不是可执行的普通强势延续证据。
这个诊断是路线停止前的研究事实，但停止的根本原因是研究问题本身设错，而不是需要继续修补该模型。

## Git 与外部数据

主要 lineage：

- `1e582ce`：首次加入全天因果 temporal label/cache；
- `d02251f`：加入全天时序研究 pipeline；
- `49524c9`：发布 temporal 结果和诊断说明。

上述提交包含当时对共用模块的修改。`implementation/shared_support.patch` 额外保存了封存前尚未提交的
GPU 节点约束 helper。若必须复现历史状态，应以这些提交、本目录 run/job 和 patch 为准；不要让当前
生产代码 import `experiments/archive/`。

PVC 上已经生成的 cache、prediction 和模型没有在本次仓库封存中删除。本目录中的 run/job 保留其路径和
镜像 lineage，可用于审计这些外部产物；它们不再是当前 pipeline 的依赖或完成条件。

## 正确的后续问题

后续实验只研究时间窗口衰减：

1. 以当前 `09:31-09:40` incumbent 为基准；
2. 预先固定另外 2–3 个十分钟窗口；
3. 每个窗口重新生成所需分钟样本并完整重训；
4. 除 `[sample]` 时钟和对应 cache lineage 外，label、feature、模型、seed、股池和 rolling OOS 全部固定；
5. 用相同的 Rank IC、Top100 excess、分期稳定性、容量和成本后指标比较衰减。
