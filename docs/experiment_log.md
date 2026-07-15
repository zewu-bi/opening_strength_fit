# Experiment Log

> Last reconciled: 2026-07-10
> Coverage: 2026-05-20 through 2026-07-10

本文件是人工维护的实验事实账本，严格按发生时间升序记录假设、口径、结果、状态和决策。当前研究
方向见 [project_brief.md](project_brief.md)，命令见 [runbook.md](runbook.md)。旧版逐日长记录原样保存在
[archive/experiment_ledger_2026-05-20_2026-07-10.md](archive/experiment_ledger_2026-05-20_2026-07-10.md)，
仅用于追溯上下文，不再承担当前状态或索引职责。

## 证据规则

发生冲突时按以下顺序判断事实：

1. compact result CSV/JSON/trace 中的数值与时间；
2. `experiments/runs/<run_id>.toml` 中的实验定义；
3. 对应 Git commit 与 K8s manifest；
4. 人工叙述。

`experiments/results/` 和 `output/` 默认不纳入 Git；它们是本地/PVC mirror，不是单独的持久事实源。
持久追溯依赖 run config、Job manifest、代码 revision、trace 与本日志共同完成。完整 run 状态不要手工复制，
使用 `osf-audit-experiments` 从 TOML 生成。

## 在途实验

| date | run | hypothesis | last known status | incumbent impact |
| --- | --- | --- | --- | --- |
| 2026-07-10 | `nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_grouped_gated_v2_mech328_v3_histavg_activity_gelu_mse_v1` | 用 prior-date 60 日历史开盘 activity ratio 替代截面 robust z-score，同时保留 NN train-set z-score | `running`; 8 shards submitted，尚无 metrics | targeted challenger；完成并通过相同 gate 前不改变 mech328 v2 incumbent |

## 决策时间线

### 2026-05：建立信号与因果口径

| date | experiment | controlled change | evidence | decision |
| --- | --- | --- | --- | --- |
| 05-20 | 3 日 Ridge/GBM 小窗 | 首次 opening feature/model smoke | GBM decision Rank IC `0.1426`，Top20 `+41.92 bps` | short signal 值得扩样本；不解释为可交易结果 |
| 05-21 | 2021 train → 2022-01 test | 扩到一年训练、次月测试 | GBM Rank IC `0.1831`，Top20 `+34.33 bps` | short 排序成立；仍无 T+1 结论 |
| 05-22~26 | CPU LightGBM delay0/1/2 | 加真实 entry delay 与 PVC labeled cache | delay2 universe Rank IC `0.1360`，Top20 `+36.75 bps`；越迟越弱 | 固定 delay2 作为保守口径 |
| 05-26 | horizon decay | 同一 score 对日内、close、next-close horizon | 日内快速衰减，close/next-close 基本消失 | 暂停直接 T+1，先强化可见短期信号 |
| 05-26~27 | postopen v1/v2、no-preopen、xs-demean | 只改决策前可见盘口/路径特征 | v2 Rank IC `0.1394`，xs-demean `0.1406`；next Top100 仍弱 | 固定 `09:31-09:40` 样本域；保留 postopen v2 |
| 05-27~28 | heat-neutral、feature core、guards、clean target | 尝试修复 short 正、next 负 | hard guard 可改善 next，但明显损伤 short；模型内 guard/强正则失败 | guard 只作诊断/后处理，不进主模型 |
| 05-28~29 | learned/conditional risk layer | `alpha_rank - λ × risk_rank` | conditional v1 学成 short proxy；alpha-conditioned v2 才有改善 | 进入跨月验证，不在单月继续调参 |

### 2026-06：固定 label、扩展 OOS、收敛特征

| date | experiment | controlled change | evidence | decision |
| --- | --- | --- | --- | --- |
| 06-02 | alpha-conditioned rolling | 6 个月滚动验证 gap risk penalty | `gap_penalty_030_p80` short/next `+21.20/+7.84 bps`，6/6 next 为正 | 证明两层公式可行，但路线封存；经验转入 single mixed label |
| 06-03 | mixed label `w=.10/.20/.30` + S/M/L | 只扫 long-label 权重 | `w=.30` 在三池保住 short，并提高 next internal excess | 固定 `w_long=0.30` |
| 06-04 | 18m feature regroup | 固定 label 后比较 7 组 feature/model | 276-feature `soft_core_reg_light` 唯一稳定晋级 | 固定历史 baseline |
| 06-04~05 | cache v2 + 36m monthly rolling | 迁移到 3 年训练、月度 OOS | 2024 pool_L short/next `+10.4/+4.4 bps` | 扩展半年 fold 和更长 OOS |
| 06-05~08 | 36m train → 6m test | 2018-2025 半年 OOS | 2020-24 pool_L `+12.14/+14.34 bps`；2025 `+6.19/+8.19 bps` | pool 内 next 比 universe 更稳，但存在年份降档 |
| 06-09~11 | 2022-25 baseline + LGBM sweep | 强正则、bagging、删 preopen、特征族、ensemble | baseline pool_L short/next `+8.626/+7.974 bps`；常规方向均无显著增量 | 停止局部 LGBM 调参 |
| 06-12~17 | fullxs、scale、price regime | 历史同分钟、path、尺度处理 | hist/path 有效；`scale_norm` 综合最好；rank-label IC 高但 Top100 退化 | 转向 hist/path 精确并集与 hygiene |
| 06-16 | acceptance fee refresh | 只改实现成本 | 默认成本从 5 bps 调整为 8 bps | 后续验收统一 8 bps |
| 06-18~23 | hist+path exact union | union、rank-centered、zscore | rank-centered union 的 pool_L next 约 `9.45 bps`，本批最高 | 确认历史/路径和尺度是有效方向 |
| 06-18~23 | feature hygiene | 相关簇与 conservative drop | baseline 276 给 17 个 hard-drop；354 union 给 19 个 sensitivity drop | drop 仅作用模型特征，不物理删 cache 基础字段 |
| 06-23 | company API bridge | 将 10 分钟 mean score 接日频 API | full-window mean 使用未来 decision points | 定性为 hindsight diagnostic；正式验收回到分钟级因果口径 |
| 06-25 | `hist_path_pruned_highdup` | 删除 26 个高重复 hist/path 特征 | 328 features；pool_L next `8.8643 bps`，信号基本保留 | 作为干净的 LGBM/NN 输入基线 |
| 06-25 | exposure audit | core + size/industry | 偏 activity/turnover heat、低 spread、中大市值；电子/电力设备/计算机偏高 | 暴露可解释，未发现新的单一押注 |
| 06-25 | split20 capacity audit | 10 亿/20、10% `turnover_diff_30t`、1% 单票上限 | `9690/9690` 截面填满；平均 top124、p95 161、max291 | 容量可行，但固定 Top100 不是容量组合 |
| 06-26 | first NN archive | 同 328 特征换 PyTorch MLP | `mlp_base` next `12.4320 bps`；`mlp_wide_huber` short IC `0.162945` | NN 值得进入结构/损失扫描 |

### 2026-07：NN 收敛与执行约束

| date | experiment | controlled change | evidence | decision |
| --- | --- | --- | --- | --- |
| 07-02 | NN scan + rankblend | 架构、Huber/正则、NN+LGBM blend | `deep_gelu_huber` short IC `0.164169`；rankblend 未胜 NN | deep Huber 作排序锚；停止 NN+LGBM 主线 |
| 07-02 | `mlp_base` capacity acceptance | capacity-selected allocation 加 next-close 与 8 bps | final cumulative net `7656.99 bps` | 容量收益必须来自 selected allocation，不复用 Top100 |
| 07-03 | MSE neighborhood | Huber → MSE 与小型 MLP 邻域 | `deep_gelu_mse` next `12.9610 bps`，capacity net `7916.02 bps` | MSE 更适合 overlay；继续结构化 grouped NN |
| 07-03 | structured NN submitted | residual/cross/gated/group-token | 同数据、label、328 features，只改网络结构 | 保持实验问题单一 |
| 07-03 | realistic acceptance v1/v2 | capacity-selected child orders 加日内单票、状态、价差、深度、最小订单与整手约束 | execctx v2：LGBM/MLP mean fill `0.802/0.809`；8 bps cumulative `4705.3/6113.0 bps`，低于简单 capacity acceptance | 执行约束显著削弱收益；作为 first-pass replay 保留 |
| 07-03 | ask-level attribution | 映射 selected notional 到 ask1-10 | ask1 仅 `37.7%/38.0%`；ask2-10 全 0，约 62% 落入 beyond ask10 | 暴露出上下文字段缺口；不得解释成真实十档可成交性 |
| 07-07 | grouped residual/cross/gated | 语义分组 encoder 与 fusion | next `13.4939/13.3557/13.8491 bps` | grouped gated 成为收益高度锚 |
| 07-07~08 | group-token + gated v2 | 更细机制分组与 per-group embedding | transformer `13.6479 bps`；gated v2 `13.2768 bps`、39/48 正月且 IC/暴露更稳 | gated v2 升为稳定性候选；停止扩大复杂结构 |
| 07-08~09 | symbol z-score | per-symbol train-window 标准化 | short IC 降至 `0.117035`，next `11.6661 bps` | reject；横截面状态被洗掉 |
| 07-09~10 | in-place XS rank | 同 328 列做截面 rank-centered | short IC `0.161260`，next `13.7351 bps` | short-ranking anchor，不取代 overlay incumbent |
| 07-09~10 | mech328 v1 | 机制化后统一 rank | short IC `0.160371`，next 降至 `11.7491 bps` | reject；末端 rank 压掉有用幅度 |
| 07-09~10 | mech328 v2 robust z-score | price tick/bps、volume ratio/share、turnover amount、notional depth、queue share，再做截面 robust z-score | pool_L next `14.3174 bps`；Top100 8 bps cumulative `8508.0 bps`；39/48 正月 | 晋级 overlay incumbent |
| 07-10 | old-NN multiscale buckets | Top1000 粗桶与桶内 IC | 粗桶递减，但 TopK/桶内 IC 为负 | 信号是 head-region selector，不是稳定细排器；旧模型诊断结束 |
| 07-10 | mech328 v3 submit | ratio-style histavg activity，无截面 z-score | 8 shards submitted，尚无 metrics | 保持 targeted challenger，见在途表 |

## 当前决策记录

### Overlay incumbent：mech328 v2

| metric | gated v2 | XS rank | mech328 v1 | mech328 v2 |
| --- | ---: | ---: | ---: | ---: |
| universe short Rank IC | 0.157623 | **0.161260** | 0.160371 | 0.154160 |
| pool_L next excess bps | 13.2768 | 13.7351 | 11.7491 | **14.3174** |
| positive next months | 39/48 | 37/48 | 33/48 | **39/48** |
| Top100 next net cumulative, 8 bps | 8003.8 | 8225.9 | 7263.6 | **8508.0** |

Decision: mech328 v2 是 pool overlay incumbent；XS rank 是 short 排序锚；mech328 v1 和
symbol-zscore 不晋级。v3 只有在相同数据、窗口、成本和 gate 下完成后才可挑战 incumbent。

Artifacts:

```text
experiments/results/backtests/optimization_overlay_acceptance_gated_v2_dimensionless_2022_2025/
experiments/results/backtests/optimization_overlay_acceptance_gated_v2_mech328_v1_v2_2022_2025/
```

### First-pass realistic replay

| model | mean fill | min fill | mean next net bps | cumulative net bps |
| --- | ---: | ---: | ---: | ---: |
| LGBM pruned 328 | 0.802248 | 0.596093 | 9.7116 | 4705.3 |
| MLP base | 0.809290 | 0.635732 | 12.6171 | 6113.0 |

Artifacts:

```text
experiments/results/backtests/realistic_acceptance_lgbm328_2022_2025_t10p20_execctx_v2/
experiments/results/backtests/realistic_acceptance_mlp_base_2022_2025_t10p20_execctx_v2/
experiments/results/backtests/execution_context_{lgbm328,mlp_base}_t10p20_v1/
experiments/results/backtests/ask_level_attribution_{lgbm328,mlp_base}_t10p20_v1/
```

Limitations: selected-order replay 不 refill 低排名股票，不建立同日 turnover budget，不模拟完整持仓/退出/
资金复用；ask2-10 上下文缺失。它是下一阶段实现起点，不是完整策略验收。

## 状态与归档

Run status 只使用：`queued`、`running`、`completed`、`canceled`、`superseded`。未知状态必须使
experiment audit 失败。完成的训练 run 应有 metrics；artifact-only run 按 kind 判断，不强制训练 metrics。

目录约定：

```text
experiments/runs/<run_id>.toml       实验定义；run.id 与文件名一致
experiments/jobs/<run_id>*_job.yaml  可执行 K8s manifest
experiments/results/                 ignored compact mirror
output/artifacts/<run_id>/           ignored 当前同步副本
output/legacy/                       ignored 历史/debug 副本
```

新记录追加顺序：先更新 TOML status 与 trace，再在本文件的对应日期追加一条固定结构记录；不要再维护
重复的全量 run/path 索引。
