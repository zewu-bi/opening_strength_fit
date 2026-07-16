# Experiment Log

> Last reconciled: 2026-07-16
> Coverage: 2026-05-20 through 2026-07-16

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
| 2026-07-10 | `nn_delay2_36m_2022_2025_fullxs_hist_path_pruned_highdup_grouped_gated_v2_mech328_v3_histavg_activity_gelu_mse_v1` | 保留同一 328 特征与 grouped-gated-v2 结构，用 strict ratio-style v3 替代 v2 归一化 | `running`; 2026-07-16 停止旧 cache 高内存重试，切到 T-1 cap/share cache 并以 v2 同规格 `384Gi/768Gi` 重提，8 shards / parallelism 2 | targeted challenger；完成并通过相同 gate 前不改变 mech328 v2 incumbent |
| 2026-07-15 | `build_delay2_{2019..2025}_auction_fresh_cache_v1` | 用修正后的集合竞价、严格上一交易日市值/股本和完整 freshness/readiness 约束重建隔离 base cache | `running`; 7 个年度 CPU Job 已提交并开始写新 lineage | 数据链路修复；不覆盖或改变旧 cache/实验 |
| 2026-07-15 | `build_delay2_{2019..2025}_auction_fresh_mixed_w030_target_v1` | 在新 base cache 上复用既有 next-close label，重建 `w_long=0.30` mixed target | `running`; 7 个 CPU Job 已提交，容器/调度等待对应 base shard 产物 | 只为新 causal-data challenger 提供输入 |
| 2026-07-15 | `nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1` | 同时验证 auction 修正、冗余特征 prune 和三项因果 freshness 控制 | `running`; indexed GPU Job 已提交，等待 7 个 mixed target shard | 完成并通过相同 gate 前不改变 incumbent |

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
| 07-15 | mech328 v2 rank/bucket re-audit | 原始 prediction、next-close label 与 `pool_L` 重新 join 并交叉实现 Spearman | `pool_L` 全池 / Top1000 单股 Rank IC `+0.008054/-0.016463`；Top1000 10×100 per-decision bucket IC `+0.050524`，汇总曲线 IC `+0.975758`；头尾 pair win `48.07%` | 计算无 sign、tie、label 或 excess 口径错误；定位为 conditional-mean/right-tail head overlay，不作单股细排器 |
| 07-15 | auction-fresh causal rebuild | 新竞价口径、两自由度价格基底、派生竞价 path、1m queue horizon、严格 tick freshness/截面 readiness、T-1 市值股本 | `248 passed, 3 skipped`；真实 ClickHouse 抽样和 7 个 base Job 首日日志均确认 prior-session reference，覆盖 `99.94%-100%` | 使用独立 v4 base/v3 mixed lineage；base 已先启动，target/model 随后作为 overnight waiting jobs 提交 |

## 当前决策记录

### Mech328 v3 cap-cache challenger（2026-07-16 重提，在途）

本轮 v3 只作为 `mech328_v2_robust_zscore` 的 targeted challenger，不改变 incumbent。实验固定使用
同一批 328 个模型输入特征名，核心配置为：

```toml
[features]
feature_value_transform = "mechanismized_v3_dimensionless_328"
include_historical_daily_activity_references = false

[model]
name = "torch_mlp"
architecture = "grouped_gated_v2"
loss = "mse"
# feature_standardization 未显式覆盖，Torch NN 入口默认 global_zscore。

[k8s]
shard_parallelism = 2

[k8s.resources]
memory_request = "384Gi"
memory_limit = "768Gi"
```

解释口径：

- `mechanismized_v3_dimensionless_328` 在特征值层继续做原 strict ratio-style 无量纲化：价格转 bps/tick，
  volume/depth 使用 cache 中严格 T-1 `total_shares`，turnover/notional 使用 `total_market_cap`；count 口径不变。
- v3 不做 `date × decision_target_timestamp` 的截面 robust-zscore；代码中该配置归一为
  `mechanismized_v3_none`，即机制化后不再追加横截面 zscore/rank。
- NN 入口仍保留 train-window global zscore，这是优化尺度标准化，不等价于横截面 zscore；它不会按每个
  决策时点抹掉市场共同活跃度或截面幅度。

2026-07-16 复查确认旧 mixed cache 没有市值/股本列，v3 因而为每个模型特征重复探测缺失 reference，
并依赖全量 prior-date 60d volume/turnover fallback。原 768Gi Job 的 8 个 shard 均失败；随后临时提高到
`1Ti/1536Gi` 的重试已按本轮决策停止。新 mixed cache 路径为：

```text
/mnt/output/opening_strength_fit/cache/
opening_2019_2025_delay2_mixed_w030_labeled_v3_auction_fresh_mcap_lag1/
```

2019-2024 已完成 shard 共 69,929,811 行，`total_market_cap` / `total_shares` 仅 18,651 行缺失，
缺失率 `0.0267%`。训练仍保留历史 same-minute surprise 与 path-shape 特征；只关闭不再需要的
`historical_daily_activity_references` denominator 构建。兼容 fallback 代码仍保留供旧 cache 调用。

内存实现同时改为按特征需求懒加载 reference，并在一个 transform 内复用同一组 shares、cap、price、depth
和 count Series；缺失候选列不再分配整列 NaN，book-depth 的 market-cap fallback 只处理 shares 缺失行。
这不改变 328 特征清单、模型结构、窗口、batch、学习率、epoch、loss 或 NN global zscore。

验证与重提：

```text
job:   os-nn-2225-gated-v2-mech328v3-mse
image: registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260716-nn-mech328-v3-capcache-v1
digest: sha256:2dfef0f12017debcfb8758644fca307f72962cb62576ce3dcebfffef4d2b0330
tests: 250 passed, 3 skipped
resources: request 16 CPU / 384Gi / 1 GPU; limit 32 CPU / 768Gi / 1 GPU
status at submit check: Running 0/8; first 2 pods Running, waiting for 2025 mixed cache dependency
```

是否闭环仍以 PVC 输出 `_SUCCESS`、`metrics_by_year.csv`、`predictions.parquet` 和运行期 cgroup memory peak
为准；本轮目标是使用与 v2 相同的内存 request/limit 完成计算。

### Auction-fresh causal rebuild（2026-07-15，在途）

这条线不修改旧 cache 或旧实验。base cache 使用 `stock.tick` 构造分钟样本，并从
`stock.daily_bar_jy` 按 `TradingDay < sample_date` 取得严格最近上一交易日 reference。原表市值、股本
字段分别乘 `10000` 转成元、股，写入：

```text
total_market_cap
float_market_cap
total_shares
float_shares
free_float_shares
market_cap_reference_date
market_cap_reference_lag_sessions
```

缓存与镜像：

```text
base:
/mnt/output/opening_strength_fit/cache/opening_2019_2025_delay2_base_labeled_v4_auction_fresh_mcap_lag1/

mixed w030:
/mnt/output/opening_strength_fit/cache/opening_2019_2025_delay2_mixed_w030_labeled_v3_auction_fresh_mcap_lag1/

image:
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260715-auction-fresh-mcap-lag1-v1
digest: sha256:1ccde06592cf99313112cfc5b0f667a91d448a3818030b072a278cf139d42a33
```

任务依赖和提交状态：

| stage | run/job family | resource | status on 2026-07-15 | dependency |
| --- | --- | --- | --- | --- |
| base | `build_delay2_{2019..2025}_auction_fresh_cache_v1` / `os-cache-auction-fresh-{year}` | CPU，单 Job request `8 CPU / 256Gi` | 7 个 Job `running`、零重启 | ClickHouse `stock.tick` + `stock.daily_bar_jy` |
| mixed target | `build_delay2_{2019..2025}_auction_fresh_mixed_w030_target_v1` / `os-target-auction-fresh-{year}` | CPU，单 Job request `8 CPU / 256Gi` | 7 个 Job 已提交；提交检查时 3 个 Pod Running 等待文件、4 个 Pod Pending 因 CPU/memory | 对应年度 base 完成 + 既有 next-close label |
| model | `nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1` / `os-nn-auction-fresh-pruned-v1` | GPU sharded training，8 completions / parallelism 2，单 Pod request `16 CPU / 512Gi / 1 GPU` | Job 已提交；提交检查时前 2 个 shard Pod Pending 因 GPU/memory | 2019-2025 七个 mixed target 全部完成 |

首个样本交易日的集群日志分别引用 `2018-12-28`、`2019-12-31`、`2020-12-31`、
`2021-12-31`、`2022-12-30`、`2023-12-29`、`2024-12-31`，均严格早于 2019-2025 各年首个
样本日。2019 少 2 个、2021 少 1 个 reference symbol；缺失保持 null，不退回当日或错误前填。

模型配置的 prune 口径保留 `mid_price + spread_bps` 两自由度价格基底，删除模型输入中的
`ask_price_1`、`bid_price_1`、`spread_abs`；原始竞价 min/max 仅用于生成 range/position 派生量后删除；
queue response 使用和名称一致的 1 分钟 horizon。数据层同时要求 entry/outcome tick 最大间隔 5 秒，且
entry 必须晚于同一 decision group 的完整截面 ready timestamp。

五个市值/股本业务字段会保留在 cache 中，既可供其他任务直接使用，也会被当前 mechanismized v3 作为
support reference，把名义金额、盘口量和成交量归一化到公司规模；它们不作为五个 raw feature 直接进入
当前模型矩阵。两个 reference metadata 字段只用于审计并显式列为非特征列。

### Overlay incumbent：mech328 v2

| metric | gated v2 | XS rank | mech328 v1 | mech328 v2 |
| --- | ---: | ---: | ---: | ---: |
| universe short Rank IC | 0.157623 | **0.161260** | 0.160371 | 0.154160 |
| pool_L next excess bps | 13.2768 | 13.7351 | 11.7491 | **14.3174** |
| positive next months | 39/48 | 37/48 | 33/48 | **39/48** |
| Top100 next net cumulative, 8 bps | 8003.8 | 8225.9 | 7263.6 | **8508.0** |

Decision: mech328 v2 是 pool overlay incumbent；XS rank 是 short 排序锚；mech328 v1 和
symbol-zscore 不晋级。v3 只有在相同数据、窗口、成本和 gate 下完成后才可挑战 incumbent。

Rank/bucket boundary：2022-2025 的 9,690 个 decision groups 中，Top1000 单股 pair accuracy
为 `49.45%`，Top100 对 Tail100 为 `48.07%`。第一百股桶均值/单股中位数为
`+14.317/-44.942 bps`，P99 为 `+1624.81 bps`，说明正均值主要来自右尾；每时刻 Top100
组合收益的中位数 `+11.76 bps` 只衡量组合时间稳定性，不能解释单股 IC。下一步按组对单股收益做
P95/P99 winsorize/trim，并拆分非尾部背景、尾部发生率和尾部严重程度。

Artifacts:

```text
experiments/results/backtests/optimization_overlay_acceptance_gated_v2_dimensionless_2022_2025/
experiments/results/backtests/optimization_overlay_acceptance_gated_v2_mech328_v1_v2_2022_2025/
experiments/results/backtests/rank_bucket_reaudit_old3_mech328_v2_2022_2025_v1/
experiments/results/backtests/mech328_v2_top1000_pairwise_audit_v1/
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
