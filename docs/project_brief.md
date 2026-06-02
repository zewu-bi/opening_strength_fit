# 项目简介

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon cross-sectional alpha：只使用
decision point 当时及以前可见的集合竞价、盘口、成交和短期动量信息，预测“当前主动买入并短持有约一分钟”的
future gross return，并检查模型分数是否能稳定识别更强股票或更好入场时刻。

样本粒度固定为 `trading day x symbol x opening decision time`。项目级 opening window 是
`09:30:00-09:40:00` 的整分钟 decision points；后续实验发现 `09:30` 是特殊 opening snapshot，
所以当前优化主线暂时聚焦 `09:31-09:40` post-open decision points。这不是把项目改成只做
`09:31-09:40`，而是把特殊开盘快照从当前建模主目标里旁路出来。

当前 60s label 是 microstructure proxy，不是 A 股 T+1 下的可交易收益。最新主线不再把
short alpha model 和 gap-risk model 拆成两层，而是先固定一个短线为主、少量混入长线成分的单模型
training label。当前阶段仍先把开盘后横截面 short signal / replay 做强；高频和日频的完整联合评估
放到信号变强之后。

## 研究口径

默认数据源：

```text
ClickHouse: ch.db.prod.highfortfunds.com / stock.tick
data window: 09:15:00 - 09:45:00
project sample window: 09:30:00 - 09:40:00 integer-minute decision points
current optimization slice: 09:31:00 - 09:40:00
```

股票池默认是 A 股 `00/30.SZ` 和 `60/68.SH`。X 只能使用 decision point 当时及以前可见的信息。
ClickHouse 原始 tick 偶尔出现 6 秒间隔时，不默认视为中间 tick 缺失；这通常表示上一条 3 秒 tick
所有字段没有变化，后续可作为盘口稳定性信息处理。

外部候选股池目前来自 `lml.bzw@ssd/data/pool_{L,M,S}.parquet`，和隔壁 `xy_fit` 的
`X.parquet` / `Y.parquet` 同目录。三份文件都是 `date x symbol` bool 宽表，`True` 表示当天入池；
当前观察是嵌套池 `pool_S ⊂ pool_M ⊂ pool_L`，不是互斥分组。访问方法见 [runbook](runbook.md#2-外部股池)。

short-horizon proxy label：

```text
decision_t = sampled decision tick
entry_t    = decision_t 之后第 entry_tick_delay 个 tick
buy_price  = ask_price_1[entry_t]
sell_vwap  = VWAP(entry_t + 60s, entry_t + 120s)
label      = sell_vwap / buy_price - 1 - fee_bps / 10000
```

主线 training label：

```text
short_label = xs_norm(short-horizon proxy label | date, decision_time)
long_label  = xs_norm(next-day-close return from the same buy_price | date, decision_time)
train_label = short_label + w_long * long_label
```

`w_long` 只取小权重，起点按 `0.10` 附近做窄扫；目标是让模型仍主要学习一分钟 VWAP
short label，同时带一点持有到第二天收盘的稳定性约束。

`entry_tick_delay` 是研究用成交代理，不等于真实成交。如果价格已经涨上去，真实挂在原位置的买单可能不会成交。
`09:40` 是正式 decision point；它的 label 使用到约 `09:41-09:42` 的 VWAP 是预期口径，不是出界。

主评估：

- `Rank IC`：同一 `date x decision_time` 横截面内的排序能力。
- `Top100 excess`：Top100 相对同横截面均值的 raw short label 超额。
- `replay`：优先强化短线可执行 proxy，不先展开完整交易约束。
- `S/M/L pools`：模型训练仍使用 full universe；评估和报告同时给 `pool_S`、`pool_M`、`pool_L` 三个选择池。
- `next close`：作为少量长线成分进入混合 label，并作为日频 sanity / transfer check 单独报告。

## 当前结论

已经比较稳的判断：

- 开盘短周期信号真实存在。旧 `09:30-09:40` delay2 baseline 和当前 `09:31-09:40` post-open baseline
  都显示 short Rank IC / Top100 excess 为正。
- `09:30` 是特殊 opening snapshot。它强，但主要混合了集合竞价结果、第一张开盘盘口快照、时间坐标和缺失/0 模式；
  当前不围绕它优化。
- `09:31-09:40` 是现在的主战场：这里仍有明确 short alpha，但 raw score 的 Top100 next-close excess
  系统性为负。
- 这个负 next tail 不是市场方向问题，而是 raw short score 混入了“短期正、隔夜回吐”的拥挤追涨风险。
- 可见信息 guard 能显著改善 next tail，说明当前时点已经能看到一部分回吐风险形态。
- sample weight、显式 guard feature、硬 feature core 都没有自然把 Top100 推到干净池子里。
- clean target、risk-shrunk target、learned risk layer、alpha-conditioned gap risk 都证明了
  “短线收益 + 长线稳定性”这个目标有信息，但两模型 `final_score` 公式暂时封存为历史对照。
- 封存原因不是 rolling 失败。`gap_penalty_030_p80` 在 18m cache 的 6 个月 rolling 中保留 Top100
  short excess `+21.20 bps`，并把 next excess 从 alpha baseline 的 `-6.91 bps` 拉到 `+7.84 bps`。
  但它本质上还是用两个模型定义一个短+长目标；主线改为直接训练一个短线为主、少量长线约束的模型。
- 训练仍使用 full universe；评估改成同时看 S/M/L 三个外部股池。后续图表和表格不能只报单一 universe
  Top100，要默认给出三池可比口径。
- 当前优先方向是把短线 label / replay 做强，手段包括特征工程和模型调参。真正做强后，再同时评估
  高频短持有和日频持有到次日收盘。
- 一个真正练好的模型，即使是按某个 label 训练，切到新的评估体系下也应该保持较好的相对表现；
  这会作为后续 transfer check。

当前工作分解：

```text
train_universe = full A-share universe
train_label    = mostly short_label + small long_label component
eval_pools     = pool_S, pool_M, pool_L selection masks
```

## 关键证据

下面按历史路线保留关键实验事实。它们解释为什么要把长线稳定性纳入目标，也解释为什么当前不再优先推进
两模型 `alpha - risk` 公式。

### Opening Baseline

项目早期 baseline 固定看 `09:30-09:40` 分钟曲线。结果显示：

- `09:30` short Top100 excess 最高，next close 也为正，是单独的 opening snapshot regime。
- `09:31-09:40` short Top100 excess 仍全部为正，说明开盘后路径信号成立。
- 同一 raw score 在 `09:31-09:40` 拿到 next close 后，Top100 excess 全部为负。

当前 post-open baseline 是 `lgbm_delay2_postopen_0931_0940_baseline_v1`：

| score | short Top100 excess bps | next Top100 excess bps | next positive minutes | Top100 guard-pass count |
| --- | ---: | ---: | ---: | ---: |
| baseline raw alpha | +22.21 | -32.21 | 0 / 10 | ~1 |

这个结果定义了现在的问题：short signal 很强，但 Top100 里几乎全是 guard-fail，next-close 负 tail 很集中。

### Guard / Risk

`next_flip_guard_10t` 使用 decision point 当时可见的横截面 rank 条件：

- `spread_bps <= p80`
- `turnover_diff_10t in [p10, p80]`
- `return_10t in [p20, p70]`
- `ask_depth_10 >= p40`
- `depth_imbalance_10 in [p20, p70]`

它不是未来函数，只是在当前横截面里排除高 spread、过热成交、追涨、低 ask depth 和极端 depth imbalance。

| baseline score variant | short Top100 excess bps | next Top100 excess bps | next positive minutes |
| --- | ---: | ---: | ---: |
| raw alpha rank | +22.21 | -32.21 | 0 / 10 |
| manual `risk_penalty_075` | +10.91 | +1.99 | 6 / 10 |
| manual `risk_penalty_100` | +9.88 | +1.92 | 7 / 10 |
| hard `next_flip_guard_10t` | +6.77 | +11.88 | 9 / 10 |

这说明 risk penalty 已经能把 next 从系统性负值拉到接近 0 或小正，同时保留一部分 short alpha。
手工公式不是最终策略，但它仍是 learned risk layer 的 teacher / baseline。

### Clean Target

clean target 只削弱 dirty short winner 的训练奖励：

```text
base = median(label | date, decision_time)
positive_excess = max(label - base, 0)

if dirty:
    target_label = label - penalty * positive_excess
else:
    target_label = label
```

| model | short Top100 excess bps | next Top100 excess bps | Top100 guard-pass count |
| --- | ---: | ---: | ---: |
| baseline | +22.21 | -32.21 | ~1 |
| `guard_shrunk_target_050_v1` | +14.55 | -20.98 | ~36 |
| `guard_shrunk_target_060_v1` | +10.47 | -13.13 | ~57 |
| `guard_shrunk_target_065_v1` | +8.49 | -8.92 | ~66 |
| `guard_shrunk_target_075_v1` | +6.21 | +0.07 | ~80 |

结论：guard 信息能改变模型排序，但作为 alpha target 会明显压掉 short alpha。

### Continuous Risk Target

连续 `guard_risk_shrunk` 用 dirty-risk score 替代二元 pass/fail：

```text
target_label = label - lambda * dirty_risk * max(label - group_median, 0)
```

| model | short Top100 excess bps | next Top100 excess bps | Top100 guard-pass count |
| --- | ---: | ---: | ---: |
| `guard_risk_shrunk_target_075_v1` | +19.95 | -25.60 | ~5 |
| `guard_risk_shrunk_target_100_v1` | +18.80 | -16.87 | ~9 |

结论：连续 risk target 比二元 clean target 温和，能保住 short alpha，但没有足够改变 Top100 风险结构。
它更像 target 正则，不像完整风险层。

### Learned Risk Layer v1

第一轮 learned-risk 三个 run 已按 runbook 上集群跑完：

| run | target / score | key result |
| --- | --- | --- |
| `learned_risk_layer_guard_teacher_v1` | 学手工 dirty-risk teacher | group rank IC = 0.9768，几乎完整复现手工规则。 |
| `learned_risk_layer_bad_tail_v1` | 学 `short_rank` 高且 `next_rank` 低的 bad tail | group rank IC = 0.1028，能学到但强度不高。 |
| `score_learned_risk_sweep_v1` | `alpha_rank - lambda * learned_risk_rank` | risk penalty 能把 baseline next tail 从 -32.21 bps 拉回。 |

`score_learned_risk_sweep_v1` 的关键 Top100 excess：

| score | short bps | next bps | next positive minutes |
| --- | ---: | ---: | ---: |
| baseline `alpha_rank` | +22.21 | -32.21 | 0 / 10 |
| learned `guard_teacher`, lambda 0.50 | +9.05 | +3.28 | 6 / 10 |
| learned `guard_teacher`, lambda 1.00 | +7.57 | +7.85 | 9 / 10 |
| learned `bad_tail`, lambda 0.25 | +8.13 | +21.05 | 10 / 10 |
| learned `bad_tail`, lambda 1.00 | +4.67 | +34.87 | 10 / 10 |

解读：

- `guard_teacher` 是平衡但保守的风险层；它证明手工 guard 可以被模型平滑复现，但没有新增信息。
- `bad_tail` 能识别 dirty tail，但当前标签太靠近 next-close 选择器。lambda 稍大就变成低 short、高 next，
  不像“保留可隔夜的短期特征”。
- baseline Top100 内最高 `bad_tail` 桶的确是最脏部分，short 很强但 next 很差；最低 `bad_tail` 桶更干净，
  但它只是 Top100 内的一个窄切片，不能直接和完整 Top100 主模型比较。
- 结论不是“短期强 alpha 能自然隔夜”，而是“短期强势里有可见的回吐风险，需要作为条件风险扣掉”。

### Conditional Risk v1

`conditional_bad_tail_risk_v1` 和 `conditional_bad_tail_binary_risk_v1` 原本想把问题收窄成：
只在短期强势候选中识别回吐风险，避免 `bad_tail` v1 变成全样本 next-close selector。

标签：

```text
gap risk    = 1[short_rank >= p70] * max(short_rank - next_rank, 0)
binary risk = 1[short_rank >= p80 and next_rank <= p50]
```

结果：

| run / score | key result |
| --- | --- |
| `conditional_bad_tail_risk_v1` | risk target group rank IC = 0.6901。 |
| `conditional_bad_tail_binary_risk_v1` | risk target group rank IC = 0.4023。 |
| `score_conditional_risk_sweep_v1` | Top20/50/100 均未通过；risk penalty 吃掉 short alpha，Top100 next tail 还更差。 |

诊断：

- 在 alpha Top100 内，`conditional_gap_score` vs short label 的 group Spearman 约 `+0.7544`，
  vs next close 约 `+0.0587`。
- `conditional_binary_score` vs short label 约 `+0.7463`，vs next close 约 `+0.0584`。
- 因此模型学到的是“短期赢家强度”。这类标签虽然可学，但不是可扣的回吐风险；它会把 alpha 本体当作 risk。

### Alpha-Conditioned Risk v2/v3

v2 改掉 conditional v1 的核心问题：候选不再由真实 `short_rank` 定义，而是先拟合 raw short-label
alpha model，再用 `candidate_alpha_rank >= p80` 定义候选；risk target 只看候选里的 next underperformance。

Risk target 可学但不再明显变成 short-alpha proxy：

| run | target | group rank IC |
| --- | --- | ---: |
| `alpha_conditioned_reversal_binary_risk_v2` | candidate 内 `next_rank <= p40` | 0.4121 |
| `alpha_conditioned_reversal_gap_risk_v2` | candidate 内 bottom-half next severity | 0.4276 |

Top100 v3 细扫结果，以 excess 为主：

| score | short Top100 excess bps | next Top100 excess bps | next positive minutes |
| --- | ---: | ---: | ---: |
| raw alpha baseline | +22.21 | -32.21 | 0 / 10 |
| heat-neutral v2 + `mid_heat_10t` | +9.15 | +2.10 | 8 / 10 |
| `gap penalty 0.30`, p80 | +16.79 | +4.49 | 7 / 10 |
| `gap penalty 0.35`, p80 | +13.24 | +17.86 | 10 / 10 |
| `binary penalty 0.35`, p80 | +19.49 | -2.04 | 4 / 10 |

结论：

- Top100 应看 excess frontier；actual 只作为后续交易成本 sanity check。
- `gap` risk soft penalty 明显优于 hard gate；当时固定 `0.30` 做主 rolling 候选，`0.35` 做更防守候选。
- `binary` risk 能更保 short，但 next 尚未稳定拉正，只作为对照。
- hard gate 不再作为主路线。

### 18m Rolling Validation

`rolling_alpha_conditioned_top100_validation_v1` 在 18m delay2 cache 上做 6 个测试月 rolling：
`2021-08` 至 `2022-01`，每月用前 12 个月重新训练 alpha、gap risk 和 binary risk。
合并结果来自 `output/local/rolling_alpha_conditioned_top100_validation_v1/rolling_summary.csv`。
这段结果保留为“两模型短+长目标”可行性的历史证据；它不再是当前优先实现路线。

| score | short Top100 excess bps | next Top100 excess bps | next positive months | next positive minutes |
| --- | ---: | ---: | ---: | ---: |
| raw alpha baseline | +24.87 | -6.91 | 2 / 6 | 2 / 10 |
| `gap_penalty_030_p80` | +21.20 | +7.84 | 6 / 6 | 8 / 10 |
| `gap_penalty_035_p80` | +17.39 | +13.25 | 6 / 6 | 10 / 10 |
| `gap_penalty_030_p90` | +21.77 | +6.45 | 3 / 6 | 8 / 10 |
| `binary_penalty_035_p80` | +22.45 | +3.64 | 3 / 6 | 5 / 10 |

结论：

- rolling 通过，不再只是 2022-01 单月有效；所有测试月 short excess 均保持为正。
- `gap_penalty_030_p80` 是封存两模型路线里的最好候选：保留约 85% raw alpha excess，同时把 6 个月 next excess 全部拉正。
- `gap_penalty_035_p80` 是两模型路线里的防守候选：short 更低，但 next 更稳，10 个分钟均值全部为正。
- `gap_penalty_030_p90` 和 `binary_penalty_035_p80` 更保 short，但 next 月度稳定性只有 3 / 6，暂作对照。
- rolling 预测诊断显示 gap risk 在 alpha Top100 内与 raw short label 的 Spearman 均值约 `+0.05`，
  与 next close 约 `-0.07`，不像 conditional v1 那样在扣 alpha 本体。

## 下一步任务

下一步目标是先把一个方向做强：以短线 label / replay 为主，训练一个单模型 mixed label。
长线成分只作为小权重稳定性约束，不把模型训练成 next-close selector。

| step | run direction | purpose |
| --- | --- | --- |
| 1 | label freeze | 用 `short_label + w_long * long_label` 作为主线 label，先从 `w_long ~= 0.10` 做窄扫，选一个默认口径。 |
| 2 | short/replay strengthening | 在默认 label 下做特征工程和模型调参，优先提升 short Rank IC、Top100 excess 和 replay。 |
| 3 | S/M/L evaluation | 训练仍用 full universe，评估默认同时报 `pool_S`、`pool_M`、`pool_L`。 |
| 4 | transfer check | 短线方向真正做强后，再同时评估高频短持有和持有到第二天收盘。 |

下一轮 gate：

- short Rank IC / Top100 excess / replay 相比 raw short baseline 有明确提升，且不能只在一个股池里好看。
- next-day-close sanity 不能被明显打坏；长线成分是稳定性约束，不是主收益来源。
- S/M/L 三池评估要成为默认视图。普通 baseline 对比至少 3 个柱子；baseline + 改进模型就是 6 个柱子；
  再加 rolling 维度时很容易到 18 个柱子，优先用分组柱、small multiples 或按池分面，避免挤成一团。
- 如果一个模型在 mixed label 下练得好，切到纯短线、高频 replay 或 next-day-close 检查时也应该相对稳健；
  这是判断“信号真的做强”的重要证据。

## 建模边界

当前不做：

- 不把 `final_score = alpha_rank - lambda * gap_risk_rank` 作为当前主线；它封存为诊断和对照。
- 不把 long label 权重放大到让模型变成 next-close selector；mixed label 仍以短线 VWAP label 为主。
- 不在第一阶段按 S/M/L 股池过滤训练，也不把股池 membership 当特征；股池先只作为评估选择 mask。
- 不在短线 replay 还没做强前，把 fee/slippage、多档容量、同股冷却和 T+1 overlay 混进训练目标。
- 不围绕特殊 `09:30` opening snapshot 做主优化。
- 不继续叠加 clean target、risk-shrunk target 和 risk penalty，避免把同一类 long-risk 约束重复塞进模型。

当前继续坚持：

- 所有 inference features 必须是 decision point 当时及以前可见信息。
- 主指标是 short Rank IC、Top100 excess 和 replay；S/M/L 三池都要报。
- `09:30-09:40` 是项目主窗口；`09:31-09:40` 是当前 post-open 优化子域。
- 大体积 predictions 和图片留在 ignored `output/`；可提交证据保留在 `experiments/results/`。

## 历史归档

这些路线已经可以归档，不再作为近期主线：

- Ridge / sklearn GBM 小窗和一年 baseline。
- CPU LightGBM delay0/1/2 普通 universe 与 strong universe 对比。
- 标准 constrained replay 和 alpha horizon decay。
- post-open v1/v2 feature engineering、feature dependence audit。
- heat-neutral target、feature core、strong regularization。
- guard-filtered、guard-weighted、guard-feature-in-model 尝试。
- two-model learned risk / alpha-conditioned gap-risk `final_score` 路线；数字保留作 short+long 目标可行性证据。

详细实验数字、K8s 输出和配置索引见 [experiment_log.md](experiment_log.md)。
