# Promotion Checklist

本文件定义“候选模型可以进入最终候选/替代 incumbent”的最小证据包。研究判断仍写
[project_brief.md](project_brief.md)，实验数字和归档路径仍写
[experiment_log.md](experiment_log.md)，执行命令仍写 [runbook.md](runbook.md)。

## 使用范围

适用于从 active challenger 晋级为 final candidate 的模型或 score blend。宽扫、smoke、debug
run 不需要填完整 checklist，但如果结论要写入 brief，就应补齐这里的证据。

## 必填元信息

| item | requirement |
| --- | --- |
| candidate id | 对应 `experiments/runs/<run_id>.toml`，且 `run.id` 与文件名一致。 |
| baseline/incumbent | 明确比较对象，例如 `soft_core_reg_light`、`mlp_base` 或当前 final candidate。 |
| data/cache | 写明 labeled cache、next-close cache、pool 版本和日期范围。 |
| decision slice | 与 brief 固定口径一致，默认 `09:31:00-09:40:00`。 |
| label | 写明 short/long/mixed 口径和 `w_long`。 |
| artifact roots | 写明 PVC 输出和本地 compact artifact 归档位置。 |

## Gate

| gate | pass condition | artifact |
| --- | --- | --- |
| universe short Rank IC | 高于或不显著弱于 incumbent，且年度/月度分布没有明显断层。 | `metrics_by_year.csv`、rolling/month summary。 |
| `pool_L` Top100 next internal excess | 高于 incumbent，且不是单一年份或少数日期贡献。 | pool-internal summary、weekly/cumulative plots。 |
| market-relative next alpha | 相对 full-market / pool baseline 后仍为正，回撤和分段表现可解释。 | optimization acceptance figures / daily summary。 |
| exposure audit | size、industry、liquidity、opening heat 等暴露可解释，未退化为不可接受的单因子押注。 | exposure summary、industry summary。 |
| capacity portfolio | 在目标 notional、参与率、单票权重和集中度约束内可装入，收益口径使用 capacity acceptance。 | capacity audit summary、capacity acceptance summary。 |

## 晋级记录模板

```text
candidate:
baseline_or_incumbent:
decision:
  status: promote | keep_challenger | reject | needs_more_evidence
  date:
evidence:
  universe_short_rank_ic:
  pool_L_top100_next_excess:
  market_relative_next_alpha:
  exposure:
  capacity:
artifacts:
  metrics:
  pool_internal:
  market_relative:
  exposure:
  capacity:
notes:
```

## 复核规则

- 不用 Top100 等权收益替代 capacity acceptance 收益。
- 不把 `pool_L` short excess、universe next excess 或 next Rank IC 单独作为主 gate。
- 不使用完整 `09:31-09:40` mean score 压成日频分数做正式验收。
- 如果候选只在一个 gate 上领先，默认保持 challenger，不直接替代 incumbent。
- 最终候选至少要有一条清楚的失败解释路径：如果后续 live / replay 退化，能定位到排序力、股池 overlay、暴露或容量中的哪一项。
