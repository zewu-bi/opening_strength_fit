# Experiment Log

> Last reconciled: 2026-07-17
> Coverage: 2026-05-20 through 2026-07-17

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
| 2026-07-16 | `build_delay2_{2019..2025}_conservative_cap_cache_v1` | 在旧 cache 覆盖率语义上只加 T-1 cap/share、基础竞价变换，并让 delay2 跳过同交易所时间戳重复行 | `completed`; 7 个年度 Job 从 ClickHouse 全量重建，单年约 10.5-18 小时 | 作为 tick-count 对照保留；不再作为新 target/model 的输入 |
| 2026-07-16 | `build_delay2_{2019..2025}_conservative_cap_mixed_w030_target_v1` | 从保守 base 构造既有定义的 mixed-w030 target | `superseded`; 从未提交 | 固定 +6 秒口径确认后取消该下游分支 |
| 2026-07-17 | `build_delay6_clock_state_{2019..2025}_cap_cache_v1` | entry 固定为特征状态后 6 秒，entry/sell 边界取该逻辑时刻最后已知状态，并保留 source timestamp/state age 审计列 | `running`; 7 个年度 ClickHouse base Job 已提交 | 新的 canonical label/cache lineage |
| 2026-07-17 | `build_delay6_clock_state_{2019..2025}_cap_mixed_w030_target_v1` | 从 fixed-clock base 构造既有 mixed-w030 target | `queued`; Job 已渲染和 dry-run，等待 base manifest 与逐日覆盖率审计 | 后续新训练只读该 target lineage |
| 2026-07-17 | `nn_delay6_clock_state_36m_2022_2025_auction_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1` | 固定模型/特征，只把数据切到 fixed-clock cache | `queued`; sharded Job 已渲染和 dry-run，等待 7 个 mixed target | 对 auction-fresh v4 做同规格数据口径 A/B |

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
| 07-16 | conservative cap/unique-tick rebuild | 旧 cache 语义 + T-1 cap/share + indicative-auction；同 `(date,symbol,exchange timestamp)` 只保留一个确定性快照，delay2 继续向后数两个不同时间戳；移除三项严格有效性门 | 2019-04-19 全市场：`2,326,300 -> 1,162,995` unique rows，`38,740/39,115 = 99.0413%` 样本有效；有效 entry 中 `92.51%` 为 6 秒，其余继续读到更后真实 tick；全量 `265 passed, 3 skipped` | 使用独立 v3 base/v2 mixed lineage；先跑 base，完成后按日审计覆盖率再提交 target |
| 07-17 | auction-fresh pruned acceptance archive | 完成 indexed GPU training、pool-internal analysis、artifact sync，并用 runbook overlay acceptance 对比 mech328 v2 / gated v2 | `pool_L` next excess `16.9692 bps`，高于 mech328 v2 `14.3174` 和 gated v2 `13.2768`；Top100 8bps net cumulative `9893.9 bps`，相对 pool_L cumulative excess `8221.6 bps`；universe short IC `0.150489`，低于 mech328 v2/gated v2 | 通过 Top100 收益验收并归档为 causal-data challenger；因 short IC 退化且未做 capacity/realistic，不替换 mech328 v2 incumbent |
| 07-17 | auction-fresh downstream acceptance | 同 runbook t10p20 capacity audit/acceptance、execution-context realistic replay、core/size exposure 和右尾 P95/P99 拆解 | capacity fee8bps cumulative `9244.5 bps`；realistic fill `0.8073`、fee8bps cumulative `7323.9 bps`；size exposure 近中性，activity exposure 仍高；P95 winsor 后 capacity/realistic 分别 `-1.5/-0.3 bps` | execution 后仍强于 LGBM/MLP anchors，但收益质量被 P95 右尾依赖否决；继续作为 archived challenger，不替换 incumbent |
| 07-17 | fixed-clock +6s label patch | 用 event/state 语义替换“向后数两条更新”和 `>5s` freshness 失效：entry 固定 +6s，entry/sell 取边界时刻最后已知状态；同时间戳 revision 仍确定性去重 | 2019-04-19 同一批 2,326,300 ClickHouse 行：conservative `38,740/39,115=99.0413%`，fixed-clock `38,745/39,115=99.0541%`；旧口径 2,901 个有效 entry 晚于 6 秒，新口径全部精确 6 秒，entry state-age P99/max 为 3/6 秒 | 建立独立 delay6-clock-state base/mixed lineage；旧 conservative mixed target superseded；新实验只读新 lineage |

## 当前决策记录

### Fixed-clock +6s state cache（2026-07-17，在途）

ClickHouse `stock.tick` 在状态未变化时可以不写一条 3 秒快照，因此“下一条物理更新距离超过 5 秒”
不能直接解释为 stale 或缺失；反过来，取消阈值后继续数两条更新又会把逻辑 +6 秒入场漂移到
9、12、15 秒甚至更晚。新口径把执行时钟和源状态时间拆开：

```toml
[labels]
entry_tick_delay = 2                 # 只保留为名义/兼容审计字段
entry_alignment = "clock_state"
entry_clock_delay_seconds = 6
future_alignment = "clock_state"
require_entry_after_cross_section_ready = true
```

- `entry_timestamp` 是固定的逻辑 +6 秒；`entry_source_timestamp` 是该时刻之前最后一条状态，
  `entry_state_age_seconds` 记录携带时长。
- sell start/end 同样以逻辑边界做 backward point-in-time state lookup，禁止读取边界之后才出现的 tick。
- 同 `date × symbol × exchange timestamp` 的 revision 仍先按最新本地接收时间和累计成交状态确定性去重。
  固定时钟消除了重复行对 entry 时间的影响，但重复 revision 仍会改变盘口值和 row-lag/diff/path 特征。
- 不再配置 `entry_max_gap_seconds` 或 `max_future_gap_seconds`；真正的数据健康度需要独立 heartbeat、
  sequence 或接收延迟证据，不能由单股票“多久没变化”代理。

新缓存从 ClickHouse 原始行重新计算，不原地修改旧 parquet：

```text
base:
/mnt/output/opening_strength_fit/cache/
opening_2019_2025_label_v4_clock6_state_unique_base_mcap_lag1/

mixed w030:
/mnt/output/opening_strength_fit/cache/
opening_2019_2025_label_v4_clock6_state_unique_mixed_w030_mcap_lag1/

image:
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260717-clock-state-delay6-v4
digest: sha256:4c397a1fb4b505f51c8390c792baae56fa6a57fa3d96927a4348496fef536e75
```

仅 v4 是正式运行镜像。最初 v1 因既有 `.dockerignore` 未排除本地 `.env`，发现后立即删除全部 7 个
刚启动的 Job；v2 从干净基础镜像完整重建并通过“容器内无 `.env`”检查，v3 只在干净 v2 上覆盖
fixed-clock 代码，v4 再加入规范 cache 路径和兼容链接去重，并把 source revision 固定到
`0a91a582fc2149336405c228e7e7825b1ef808c5`。v1 必须从 registry 删除并轮换其中涉及的凭据；
不得再次拉取或运行。

2019-04-19 同日 A/B 使用相同 2,326,300 条原始行和 1,162,995 个去重后状态。fixed-clock 有效覆盖率
为 `38,745/39,115=99.0541%`，conservative tick2 为 `38,740/39,115=99.0413%`；二者
`valid_both/new_only/old_only/invalid_both` 分别为 `38,732/13/8/362`。新口径所有有效 entry
都是 6 秒；source state age 的 P50/P95/P99/max 为 `0/0/3/6` 秒。7 个年度 base Job 已提交；
target 和模型保持 queued，待 base parquet、manifest 和逐日审计完成后再提交。v4 集群首日 smoke 已完成：
2019-01-02 得到 `36,358` 行、`35,570` 个 valid label，T-1 reference 为 2018-12-28；随后已进入
2019-01-03，7 个年度 Pod 均为 0 restart。

同日完成 PVC cache layout v4 迁移：成品只做同文件系统 rename，最终根目录只保留 7 个规范目录；
旧失败目录、13 个遗留 heartbeat lock、92 个退化 alias 产物均删除。PVC 不支持可靠 symlink、hard-link
或 reflink，因此不保留旧路径兼容层，仓库内 run/Job 已统一改写为规范路径。旧 18 个月 mixed-w030
单文件与 v1 年度分片在 `2020-08-03..2022-01-28` 的 16,748,169 行、184 列及关键键/标签哈希完全
一致，已删除并释放 7,593,748,386 bytes；对应旧模型配置改读年度规范目录。

### Conservative cap + unique-tick cache（2026-07-16，base 已完成、target 已终止）

这条线回答 2019-04-19 的重复 tick 问题，不把重复记录本身判成 label 无效。新配置在进入特征和 label
构造前，按 `date × symbol × exchange timestamp` 去重；保留行优先取最新本地接收时间，随后以累计成交
笔数、成交量、成交额和稳定 fingerprint 决定。因而 `entry_tick_delay = 2` 表示向后两个不同的交易所
时间戳，而不是 DataFrame 中向后两条物理记录。最老 cache 的配置不开此开关，历史结果仍可原样复现。

保守 base 只在旧覆盖率语义上增加：

- `stock.daily_bar_jy` 的严格 T-1 `total/float market cap` 与 `total/float/free-float shares`；
- `indicative_quote_v2` 和既有基础特征构造；
- `tick_timestamp_deduplication = "latest_local_timestamp"`。

明确不包含 `entry_max_gap_seconds`、`max_future_gap_seconds` 和
`require_entry_after_cross_section_ready`。缓存路径与执行镜像为：

```text
base:
/mnt/output/opening_strength_fit/cache/
opening_2019_2025_label_v2_tick2_unique_base_mcap_lag1/

mixed w030（尚未提交）:
/mnt/output/opening_strength_fit/cache/
opening_2019_2025_label_v2_tick2_unique_mixed_w030_mcap_lag1/

image:
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260716-conservative-cache-v1
digest: sha256:b9b50bc6ed46a95959e96865fd9ce28750b02cdfeb851571233c9ded0a01f40b
```

真实 2019-04-19 全市场查询得到 2,326,300 条原始记录和 1,162,995 个唯一时间戳；去重后 39,115
个分钟决策样本中 38,740 个有效，覆盖率 99.0413%，T-1 reference date 为 2019-04-18。有效样本中
35,839 个（92.51%）entry delay 为 6 秒；其余样本继续读到 9、12、15 秒等后续真实 tick，并未因超过
5 秒而失效。七个 base Job 均从 ClickHouse 全量重建完成，耗时约 10.5-18 小时；这不是旧 parquet
上的快速补丁。首个交易日 reference date 均早于样本日。fixed-clock 口径确认后，这组 base 只保留为
tick-count 对照；其 mixed target 从未提交，现已标记为 superseded。

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
opening_2019_2025_label_v3_tick2_gap5_ready_mixed_w030_mcap_lag1/
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

### Auction-fresh causal-prune acceptance archive（2026-07-15~17，已归档）

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
/mnt/output/opening_strength_fit/cache/opening_2019_2025_label_v3_tick2_gap5_ready_base_mcap_lag1/

mixed w030:
/mnt/output/opening_strength_fit/cache/opening_2019_2025_label_v3_tick2_gap5_ready_mixed_w030_mcap_lag1/

image:
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260715-auction-fresh-mcap-lag1-v1
digest: sha256:1ccde06592cf99313112cfc5b0f667a91d448a3818030b072a278cf139d42a33
```

任务依赖和提交状态：

| stage | run/job family | resource | final status | dependency |
| --- | --- | --- | --- | --- |
| base | `build_delay2_{2019..2025}_auction_fresh_cache_v1` / `os-cache-auction-fresh-{year}` | CPU，单 Job request `8 CPU / 256Gi` | 7 个 Job `completed`、零重启 | ClickHouse `stock.tick` + `stock.daily_bar_jy` |
| mixed target | `build_delay2_{2019..2025}_auction_fresh_mixed_w030_target_v1` / `os-target-auction-fresh-{year}` | CPU，单 Job request `8 CPU / 256Gi` | 7 个 Job `completed` | 对应年度 base 完成 + 既有 next-close label |
| model | `nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1` / `os-nn-auction-fresh-pruned-v1` | GPU sharded training，8 completions / parallelism 2，单 Pod request `16 CPU / 512Gi / 1 GPU` | 8 个半年 shard completed；metrics 已同步 | 2019-2025 七个 mixed target 全部完成 |
| pool-internal analysis | `os-analyze-nn-auction-fresh-pruned-v1` | CPU，`8 CPU / 256Gi` request | completed；join 后 `44,033,943/44,033,943` prediction rows retained | model predictions + delay2 next-close labels |

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

验收结果（2022-2025，`pool_L` Top100，8 bps fee；对比 mech328 v2 和 gated v2）：

| metric | auction-fresh pruned v1 | mech328 v2 | gated v2 |
| --- | ---: | ---: | ---: |
| universe short Rank IC | 0.150489 | 0.154160 | **0.157623** |
| pool_L short Rank IC | 0.137495 | 0.142022 | **0.144034** |
| pool_L next excess bps | **16.9692** | 14.3174 | 13.2768 |
| positive next months | 38/48 | **39/48** | **39/48** |
| Top100 next net cumulative, 8 bps | **9893.9** | 8508.0 | 8003.8 |
| cumulative excess vs pool_L | **8221.6** | 6936.8 | 6432.6 |
| quarterly next-excess wins | **11/16** | 3/16 | 2/16 |

Interpretation: 这轮收益验收胜出，但 short 排序力退化。日度 `pool_L` next excess 的均值/中位数为
`16.97/15.28 bps`，日胜率 `58.0%`，高于 mech328 v2 的 `14.32/13.52 bps` 和 `57.8%`；但日度标准差
升至 `79.5 bps`，高于 mech328/gated 的约 `70.4 bps`。因此增量更像严格因果数据链路和 payoff
对齐带来的经济收益，而不是更强的 short ranker。

Downstream acceptance（t10p20 capacity-only，8 bps fee；2026-07-17 补充）：

| metric | auction-fresh pruned v1 | LGBM pruned 328 | MLP base |
| --- | ---: | ---: | ---: |
| capacity acceptance cumulative net bps | **9244.5** | 6009.2 | 7657.0 |
| realistic mean fill | 0.8073 | 0.8022 | **0.8093** |
| realistic cumulative net bps | **7323.9** | 4705.3 | 6113.0 |

Exposure：相对 gated v2，auction-fresh 的活跃度暴露略降但仍高，`turnover_diff_10t/30t`
`selected_mean_z` 为 `1.146/1.114`，Top decile share `48.0%/46.3%`；size 暴露从 gated v2 的
log market-cap `z=0.318/0.278` 降到近中性（`market_cap z=-0.024`，`float_market_cap z=-0.020`）。

Tail decomposition：capacity selected raw gross `27.08 bps`（fee 后 `19.08 bps`），P95 winsor 后
`-1.51 bps`、P99 winsor 后 `21.26 bps`；realistic selected raw gross `21.58 bps`（fee 后
`15.12 bps`），P95 winsor 后 `-0.29 bps`、P99 winsor 后 `17.18 bps`。P95 右尾占 notional
约 `5.8%/5.6%`，却贡献 `79.62/61.58 bps`，非尾部背景为 `-52.54/-40.01 bps`。

Decision: 归档为通过 Top100、capacity 和 first-pass realistic 收益 gate 的 causal-data challenger；
但 short IC 退化且 P95 右尾依赖过强，不替换 `mech328_v2_robust_zscore` incumbent。下一步必须做 refill、
完整资金预算、overlap 与 market-state 稳健性拆解。

Artifacts:

```text
experiments/results/metrics/
nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1_metrics_by_year.csv
nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1_metrics_by_month.csv

experiments/results/backtests/
nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_gelu_mse_v1/

experiments/results/backtests/
optimization_overlay_acceptance_auction_fresh_pruned_vs_mech328_v2_gated_v2_2022_2025/

experiments/results/backtests/
capacity_audit_nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_t10p20_capacityonly_v1/
capacity_acceptance_nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_t10p20_capacityonly_fee8bps_v1/
execution_context_auction_fresh_pruned_t10p20_v1/
realistic_acceptance_auction_fresh_pruned_2022_2025_t10p20_execctx_v2/
exposure_audit_nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_core_v1/
exposure_audit_nn_delay2_36m_2022_2025_auction_fresh_pruned_grouped_gated_v2_mech_v3_size_industry_v1/
tail_decomposition_auction_fresh_pruned_t10p20_execctx_v1/

figures:
optimization_directions_overlay_acceptance.svg
optimization_directions_net_alpha_cumulative.svg
```

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
| auction-fresh pruned | 0.807326 | 0.596713 | 15.1165 | 7323.9 |

Artifacts:

```text
experiments/results/backtests/realistic_acceptance_lgbm328_2022_2025_t10p20_execctx_v2/
experiments/results/backtests/realistic_acceptance_mlp_base_2022_2025_t10p20_execctx_v2/
experiments/results/backtests/realistic_acceptance_auction_fresh_pruned_2022_2025_t10p20_execctx_v2/
experiments/results/backtests/execution_context_{lgbm328,mlp_base}_t10p20_v1/
experiments/results/backtests/execution_context_auction_fresh_pruned_t10p20_v1/
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
