# Experiment Log

> Last reconciled: 2026-08-06
>
> Coverage: 2026-05-20 through 2026-08-06

本文件是实验事实账本，按时间记录假设、结果、状态和决策。当前研究口径见
[project_brief.md](project_brief.md)，执行命令见 [runbook.md](runbook.md)，历史长记录见
[archive README](archive/README.md)。

## 证据规则

事实冲突时依次以 compact evidence、run TOML、Git/K8s trace、人工叙述为准。大型结果位于 PVC 或本地
mirror，不进入 Git；持久追溯依赖 run config、Job manifest、代码 revision、compact evidence 和本日志。
run 状态由 `osf-audit-experiments` 从 TOML 审计，不在文档中复制完整索引。

## 近期实验状态

| date | run / decision | status | impact |
| --- | --- | --- | --- |
| 2026-07-22 | fixed-clock v4 control / multiden + unified strategy acceptance | `completed` | multiden 曾晋级 opening policy，保留为旧数据链路对照 |
| 2026-07-30 | `10:01-10:10` 日内窗口重训 | `completed` | next excess `6.5491 bps`，确认较 09:31 明显衰减，不晋级 |
| 2026-07-31 | corrected `opening_model` 09:31-09:40 | `completed` | next excess `17.7934 bps`，成为最新信号基准；策略层仍需同数据重跑 |
| 2026-07-31 | corrected next-close 与 1m/3m label matrix | `completed` / 历史输入保留 | 旧实验继续按原输入复现，不与新数据版本混作严格单变量对照 |
| 2026-08-04 | 三窗口 raw source、350 features、基础 5-label 数据 | `completed` | 原始数据与特征/标签构建分层，combined label 仅作中间产物 |
| 2026-08-04 | 九个 horizon-split label | `completed`; 9 labels × 7 years | 后续实验从三窗口 × 1m/3m/5m 中选择；新版本为权威训练输入 |
| 2026-08-05 | corrected-label 三窗口 × 1m/3m NN 网格 | `completed`; 6 runs × 8 shards | 09:31 的短期/隔夜头部超额最强；10:01、14:01 在信号层归档，不晋级策略验收 |
| 2026-08-05 | 前两窗口 10m/1h/当日收盘 label | `completed`; 6 labels × 7 years | 10m/1h 用持有期后 60 秒 VWAP，收盘 label 用当日收盘价；next-close 逐 key 复用既有数据 |
| 2026-08-05 | 新分层数据 15-label NN 矩阵 | `completed`; 15 runs × 8 shards | 训练入口直接拼同窗口 350-feature 与独立 label；最大 fold 9m15s、H20 SM 83%–85%、主机峰值 202GiB |
| 2026-08-06 | 15-label max-30 收敛实验 | `queued`; 独立结果根目录 | max-10 中 119/120 跑满上限、85/120 best epoch=10；固定其余变量，改为 max 30 / patience 3 |

## 决策时间线

### 2026-05：建立信号与因果口径

| period | controlled change | evidence | decision |
| --- | --- | --- | --- |
| 05-20~21 | 小窗 GBM → 一年训练/次月测试 | decision Rank IC `0.1426 → 0.1831` | opening short 排序值得扩样本；不解释为可交易结果 |
| 05-22~26 | 加真实 entry delay、扩 horizon | delay2 后 short 仍有效，close/next-close 快速衰减 | 固定保守 delay；暂停直接 T+1 主线 |
| 05-26~27 | postopen、no-preopen、XS 处理 | postopen v2 / xs-demean 改善 short，next 仍弱 | 固定 `09:31-09:40` 样本域 |
| 05-28~29 | hard guard、learned/conditional risk | guard 改善 next 但损伤 short；两层公式局部有效 | guard 只作诊断，进入跨月验证 |

### 2026-06：固定 label、扩展 OOS、收敛特征

| period | controlled change | evidence | decision |
| --- | --- | --- | --- |
| 06-02~05 | risk layer → mixed label → 36m rolling | `w_long=.30` 在三池兼顾 short/next | 固定 single mixed label 与 36m→6m OOS |
| 06-05~11 | 扩 2020-2025 OOS、LGBM sweep | pool 内 next 较稳；常规调参无稳定增量 | 停止局部 LGBM 调参 |
| 06-12~23 | hist/path、scale、feature hygiene | rank-centered union 较强；重复簇可删 | 保留 hist/path 精确并集，drop 只作用模型输入 |
| 06-25~26 | 328-feature prune、capacity、首批 NN | 容量截面可填满；MLP next `12.4320 bps` | 328 特征成为干净基线，NN 进入结构扫描 |

### 2026-07：NN 收敛与执行约束

| period | controlled change | evidence | decision |
| --- | --- | --- | --- |
| 07-02~10 | NN 架构、MSE、grouped gated、机制化 | mech328 v2 next `14.3174 bps`；symbol z-score 失败 | grouped-gated + MSE 成为模型主线 |
| 07-15~17 | auction-fresh、T-1 reference、因果重建 | auction-pruned next `16.9692 bps`，realistic `7323.9 bps` | 通过收益 gate，后由 fixed-clock v4 supersede |
| 07-17~22 | fixed `clock+6s`、350-feature multiden | v4 multiden next `17.1714 bps`，Top100 fee8 `9891.7 bps` | 接受 fixed-clock 数据口径和 multiden continuation |
| 07-22~23 | capacity/no-refill/visible-refill 统一验收 | multiden refill fill `99.9970%`，net `8598.7 bps` | 尾部、分期和 overlap 保留为诊断，不设自动否决阈值 |
| 07-29 | 复核 full-day temporal 路线 | 需求应是固定十分钟窗口横向比较 | 归档全天序列/TCN 路线，回到既有范式 |
| 07-30 | 10:01-10:10 同口径重训 | short IC 提高，next 和费后收益显著下降 | 作为衰减 checkpoint，不替换 opening policy |
| 07-31 | corrected decision-state 09:31 重训 | next `17.7934 bps`，8/8 半年为正 | `opening_model` 晋级最新信号基准 |
| 07-31 | canonical naming reset | source lineage 不变，逻辑入口与长 run id 分离 | 使用 `opening_base/cache/model` 短入口 |

## 当前决策记录

### `opening_model` 09:31-09:40 信号验收（2026-07-31，已完成）

唯一变化是 decision sampling：从目标时刻后 5 秒内首条更新，改为目标时刻已可见的最后状态。target、
350 features、模型、seed、`pool_L`、rolling OOS 与 `clock+6s` entry 均不变；47,333,103 行 OOS
prediction 全量匹配 label。

| metric | archived v4 | corrected v6 |
| --- | ---: | ---: |
| universe short Rank IC | 0.156418 | **0.158330** |
| `pool_L` next Rank IC | 0.007140 | **0.007839** |
| `pool_L` next Top100 excess | 17.1714 bps | **17.7934 bps** |
| positive next months / half-years | 37/48; 8/8 | **38/48; 8/8** |
| Top100 fee8 cumulative | 9891.7 bps | **10193.0 bps** |

结果小幅全面提高，但只赢 `26/48` 月，属于同一信号的因果修正，不是强分期胜利。Decision：接受为最新
信号基准；在其上重跑统一策略验收前，不继承 v4 的策略晋级结论。来源映射见
[canonical opening registry](../experiments/canonical/opening.toml)。

### 10:01-10:10 日内窗口衰减（2026-07-30，已完成）

只修改十分钟窗口及对应 cache/label lineage；44,993,233 行 prediction 全量匹配同窗口 label。

| metric | 09:31-09:40 | 10:01-10:10 |
| --- | ---: | ---: |
| universe short Rank IC | 0.156418 | **0.253118** |
| short Top100 absolute return | **3.1805 bps** | 0.1153 bps |
| `pool_L` next Top100 excess | **17.1714 bps** | 6.5491 bps |
| positive next months / half-years | **37/48; 8/8** | 29/48; 6/8 |
| Top100 fee8 cumulative | **9891.7 bps** | 2708.5 bps |

10:01 更稳定地识别“跌得较少”的股票，但 opening edge 的隔夜与费后部分明显衰减。Decision：归档为
completed decay checkpoint，不做 downstream promotion audit，也不替换 09:31 policy。

### Corrected-label 三窗口 × 1m/3m 网格（2026-08-05，已完成）

三组预先固定十分钟窗口各自使用同窗口 corrected feature/label lineage；除 short label horizon 为 1m
或 3m 外，350 features、grouped-gated NN、seed、`pool_L`、36m rolling / 6m OOS 与 corrected
next-close 分量保持一致。六组训练均完成 8/8 shard，pool-internal 分析全量匹配 prediction 与 label。

| window | short horizon | universe short Rank IC | `pool_L` short Top100 excess | `pool_L` overnight Top100 excess |
| --- | --- | ---: | ---: | ---: |
| 09:31-09:40 | 1m | 0.160081 | 11.8129 bps | 18.0814 bps |
| 09:31-09:40 | 3m | 0.108995 | **12.9162 bps** | **20.2087 bps** |
| 10:01-10:10 | 1m | 0.255019 | 7.7006 bps | 6.6667 bps |
| 10:01-10:10 | 3m | 0.185261 | 7.8869 bps | 8.3935 bps |
| 14:01-14:10 | 1m | **0.378929** | 7.6493 bps | 1.0904 bps |
| 14:01-14:10 | 3m | 0.316359 | 8.0766 bps | 2.3101 bps |

窗口后移时，短期 IC 单调升高，但短期头部超额下降、隔夜头部超额强烈单调衰减。3m 相对 1m 在三个
窗口都降低短期 IC（平均 `-0.0611`），但提高短期超额（平均 `+0.5723 bps`）和隔夜超额（平均
`+1.6912 bps`）。Decision：若目标是当前经济指标，09:31-09:40 / 3m 是六格首选；10:01 和
14:01 明显落后，按 runbook 停在信号层归档，不进入 capacity / realistic promotion audit。

### Ordinary 328 mech v3 cap-cache（2026-07-21，已完成）

最终完成的是 `mech328_v3_capcache_896` 单 shard 归因性重跑，而非早期 `histavg_activity` 任务：

- `pool_L` short / next excess：`11.1004 / 16.3318 bps`；
- short / next Rank IC：`0.138516 / 0.006657`；
- next 正月：`38/48`。

它强于 mech328 v2、略低于 auction-pruned。Decision：作为普通 328 对照保留，不再补跑新 cache 主线。

### Fixed-clock +6s state cache 与 auction-pruned 验收（2026-07-17~22，已完成）

`stock.tick` 只在状态变化时写行，因此“向后数两条更新”不能代表固定 6 秒。新口径把执行时钟与源状态
时间拆开：

```toml
[labels]
entry_alignment = "clock_state"
entry_clock_delay_seconds = 6
future_alignment = "clock_state"
require_entry_after_cross_section_ready = true
```

entry 和 sell 边界均做 backward point-in-time lookup，并记录 source timestamp/state age；同时间戳 revision
先确定性去重。7 个年度 base、7 个 mixed target 及两组 8-shard 训练均完成。

| run | short Rank IC | next Rank IC | next excess bps | next 正月 | fee8 cumulative bps |
| --- | ---: | ---: | ---: | ---: | ---: |
| mech328 v2 | 0.154160 | **0.008054** | 14.3174 | **39/48** | 8508.0 |
| v4 control, 325 features | 0.156070 | 0.006411 | 16.8024 | 38/48 | 9713.0 |
| v4 multiden, 350 features | **0.156418** | 0.007140 | **17.1714** | 37/48 | **9891.7** |

multiden 相对 control 仅增加 `0.3690 bps`，月度胜 `25/48`，不是稳定逐期 A/B 胜出。Decision：接受
fixed-clock 因果口径；multiden 作为 continuation candidate，control 保留为单变量基线。

### Unified strategy acceptance toolkit 与 fixed-clock v4 复核（2026-07-22，已完成）

统一工具在同一候选集比较 `capacity_only`、`realistic_no_refill` 和
`visible_pretrade_refill`。refill 只使用决策时点可见信息继续下探，不观察真实失败后再下单。

| run / policy | mean fill | cumulative capital net bps | bootstrap P05 bps |
| --- | ---: | ---: | ---: |
| control capacity-only | 100.0000% | 8989.8 | 805.8 |
| control no-refill | 80.7803% | 7183.6 | 724.9 |
| control visible refill | 99.9969% | 8431.1 | 209.2 |
| multiden capacity-only | 100.0000% | 9217.9 | 1067.3 |
| multiden no-refill | 81.3916% | 7433.4 | 1017.6 |
| multiden visible refill | 99.9970% | **8598.7** | 497.0 |

Decision（2026-07-23）：refill、overlap、tail 和 bootstrap 都是必跑诊断，但不设置自动否决阈值。
multiden 成本后收益为正且 refill 相对 no-refill 增加 `1165.3 bps`，因此当时晋级 opening policy；这不等于
完整持仓账本或实盘批准。

### Conservative cap + unique-tick cache（2026-07-16，base 已完成、target 已终止）

该对照在旧 cache 语义上增加严格 T-1 cap/share、基础竞价变换，并按
`date × symbol × exchange timestamp` 确定性去重。2019-04-19 的 39,115 个样本中 38,740 个有效，
其中 `92.51%` 恰为 6 秒，其余会漂移到更晚真实 tick。7 个年度 base 已全量重建；fixed-clock 确认后，
mixed target 未提交，路线标记为 superseded。

### Mech328 v3 cap-cache challenger（2026-07-16 重提记录；07-21 已完成）

该 challenger 固定 328 特征与 grouped-gated-v2/MSE，只将值变换切到 strict ratio-style v3；不追加截面
z-score，训练入口仍做 global z-score。旧 cache 缺市值/股本导致 reference fallback 和内存问题，低内存
重提后来停止，最终由上文 `mech328_v3_capcache_896` 完成。早期提交状态不再视为活动任务。

### Auction-fresh causal-prune acceptance archive（2026-07-15~17，已归档）

该独立 lineage 使用严格 T-1 cap/share、因果 freshness/readiness 和 pruned feature 输入。base、mixed target、
8 个训练 shard 与 pool analysis 全部完成。

| metric | auction-fresh | mech328 v2 | gated v2 |
| --- | ---: | ---: | ---: |
| universe short Rank IC | 0.150489 | 0.154160 | **0.157623** |
| `pool_L` next excess bps | **16.9692** | 14.3174 | 13.2768 |
| positive next months | 38/48 | **39/48** | **39/48** |
| Top100 fee8 cumulative bps | **9893.9** | 8508.0 | 8003.8 |

| downstream metric | auction-fresh | LGBM 328 | MLP base |
| --- | ---: | ---: | ---: |
| capacity cumulative net bps | **9244.5** | 6009.2 | 7657.0 |
| realistic mean fill | 0.8073 | 0.8022 | **0.8093** |
| realistic cumulative net bps | **7323.9** | 4705.3 | 6113.0 |

Decision：通过 Top100、capacity 和 first-pass realistic 收益 gate；short IC 退化与右尾依赖记录为诊断。
该 lineage 随后由 fixed-clock v4 supersede。

### Historical overlay baseline：mech328 v2

| metric | gated v2 | XS rank | mech328 v1 | mech328 v2 |
| --- | ---: | ---: | ---: | ---: |
| universe short Rank IC | 0.157623 | **0.161260** | 0.160371 | 0.154160 |
| `pool_L` next excess bps | 13.2768 | 13.7351 | 11.7491 | **14.3174** |
| positive next months | 39/48 | 37/48 | 33/48 | **39/48** |
| Top100 fee8 cumulative bps | 8003.8 | 8225.9 | 7263.6 | **8508.0** |

Decision at the time：mech328 v2 为 overlay incumbent，XS rank 为 short 排序锚；之后被 fixed-clock v4
multiden supersede，当前只作历史基线。

### First-pass realistic replay

| model | mean fill | min fill | mean next net bps | cumulative net bps |
| --- | ---: | ---: | ---: | ---: |
| LGBM pruned 328 | 0.802248 | 0.596093 | 9.7116 | 4705.3 |
| MLP base | 0.809290 | 0.635732 | 12.6171 | 6113.0 |
| auction-fresh pruned | 0.807326 | 0.596713 | 15.1165 | 7323.9 |

限制：selected-order replay 不 refill、不维护完整同日预算/持仓/退出，且缺 ask2-10 上下文。它是统一策略
验收的前身，不是完整策略回测。

## 状态与归档

Run status 只允许 `queued`、`running`、`completed`、`canceled`、`superseded`；未知状态必须使 audit 失败。
训练 run 完成时应有 metrics，artifact-only run 按 kind 验收 summary、trace 和成功标记。

新记录先更新 TOML status 与 trace，再在本文件对应日期增加一条结论；不要维护重复的全量 run/path 索引。
