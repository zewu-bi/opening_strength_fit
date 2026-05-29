# 项目简介

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon cross-sectional alpha：只使用
decision point 当时及以前可见的集合竞价、盘口、成交和短期动量信息，预测“当前主动买入并短持有约一分钟”的
future gross return，并检查模型分数是否能稳定识别更强股票或更好入场时刻。

样本粒度固定为 `trading day x symbol x opening decision time`。项目级 opening window 是
`09:30:00-09:40:00` 的整分钟 decision points；后续实验发现 `09:30` 是特殊 opening snapshot，
所以当前优化主线暂时聚焦 `09:31-09:40` post-open decision points。这不是把项目改成只做
`09:31-09:40`，而是把特殊开盘快照从当前建模主目标里旁路出来。

当前 60s label 是 microstructure proxy，不是 A 股 T+1 下的可交易收益。当前阶段的目标仍然是把开盘后
横截面 short signal 做强；fee/slippage、多档容量、同股冷却和日频 overlay 暂不进入优化目标。

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

`entry_tick_delay` 是研究用成交代理，不等于真实成交。如果价格已经涨上去，真实挂在原位置的买单可能不会成交。
`09:40` 是正式 decision point；它的 label 使用到约 `09:41-09:42` 的 VWAP 是预期口径，不是出界。

主评估：

- `Rank IC`：同一 `date x decision_time` 横截面内的排序能力。
- `Top100 excess`：Top100 相对同横截面均值的 raw short label 超额。
- `next close`：sanity check 和 risk-layer 监督来源；不直接混进 alpha model 的训练目标。

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
- clean target 能把 guard 信息压进模型，但 penalty 越强 short alpha 掉得越多；它现在更适合作诊断和对照。
- learned risk layer v1 证明两层打分方向可行：`guard_teacher` 能复现手工风险，`bad_tail` 能学到一部分
  short-positive / next-negative 成分。
- 但 `bad_tail` v1 不是可直接部署的 overnight-alpha 证据。它太像 next-close selector，强扣之后
  next 远大于 short，说明它混入了“哪些开盘状态第二天收盘更好”的 B 成分。
- `conditional_bad_tail` v1 也不是可用风险层。它在训练标签里直接用了真实 `short_rank` 定义候选和
  risk gap，模型最终主要学到“短期赢家强度”，不是“短期强势里的回吐风险”。扣它会删掉 short alpha
  本体，next tail 也没有改善。
- `alpha_conditioned_reversal` v2 改为用 alpha-score 定义候选，再学习候选里的 next underperformance。
  Top100 v3 细扫显示 soft penalty 是主线：`gap penalty 0.30` 在保留约 `+16.8 bps` short excess 的同时，
  把 next excess 从 `-32.2 bps` 拉到 `+3~4.5 bps`。
- 下一步不继续在单个 2022-01 上调参；用 18 个月 cache 做 6 个月 rolling validation，固定验证
  `gap penalty 0.30/0.35`、`binary penalty 0.35` 和 alpha baseline。

当前工作分解：

```text
alpha_model = raw short-label post-open baseline
risk_model  = learned dirty-risk / next-flip layer
final_score = alpha_score - lambda * risk_score
```

## 关键证据

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
- `gap` risk soft penalty 明显优于 hard gate，`0.30` 是当前 rolling 的主候选，`0.35` 是更防守的候选。
- `binary` risk 能更保 short，但 next 尚未稳定拉正，只作为对照。
- hard gate 不再作为主路线。

## 下一步任务

下一步目标仍是让 alpha 模型继续寻找短周期强势，让 risk 模型只惩罚强势候选里的回吐概率。
但 conditional v1 已经说明：不能用真实 `short_rank` 直接构造 risk 强度，否则 risk layer 会学习
short alpha 本身。

| step | run direction | purpose |
| --- | --- | --- |
| 1 | `rolling_alpha_conditioned_top100_validation_v1` | 每个测试月用前 12 个月重新训练 alpha、gap risk、binary risk，固定验证 Top100 penalty frontier。 |
| 2 | rolling 结果复盘 | 看 6 个测试月中 short/next excess 是否稳定，不再按 2022-01 单月继续调参。 |
| 3 | 若 rolling 通过 | 再进入交易约束：成本、成交、容量、同股冷却和日频 overlay。 |
| 4 | 若 rolling 不通过 | 回到 risk target：候选定义、next residual、OOF alpha conditioning，而不是继续扩大 score sweep。 |

下一轮 gate：

- short Top100 excess 不要从 +22 bps 直接掉到 +5 bps 以下，优先保住约 +10 bps 以上。
- next-close Top100 excess 从 -32 bps 明显收敛，目标是接近 0 或小正，而不是 next 远大于 short。
- risk 层只扣 alpha-score 强势候选中 next residual 差的 A 类 dirty tail；纯 next-close 更好的 B 类不作为额外奖励。
- learned risk 至少接近 manual risk penalty 的 short/next tradeoff，并在小 lambda 区间更平滑。
- risk score 与 short label 的相关不能过高；若 alpha Top100 内 Spearman 接近 `+0.7`，基本就是在扣 alpha 本体。

## 建模边界

当前不做：

- 不把 next-close label 混进 alpha model。next-close 可以监督 risk layer，因为 risk layer 的职责就是识别回吐风险。
- 不把 bad-tail risk 当作 next-close reward；final score 的被减数必须是 conditional reversal-risk。
- 不再用真实 `short_rank` 直接作为 learned risk 的候选或强度来源；它太容易把 risk 学成 short-alpha proxy。
- 不优化 fee/slippage、多档容量、同股冷却或 T+1 overlay。
- 不围绕特殊 `09:30` opening snapshot 做主优化。
- 不继续叠加 clean target 和 risk penalty，避免重复惩罚同一类 dirty tail。

当前继续坚持：

- 所有 inference features 必须是 decision point 当时及以前可见信息。
- 主指标是 short Rank IC 和 Top100 excess。
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

详细实验数字、K8s 输出和配置索引见 [experiment_log.md](experiment_log.md)。
