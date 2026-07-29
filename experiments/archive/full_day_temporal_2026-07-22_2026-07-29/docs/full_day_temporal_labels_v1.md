# 全天分钟级因果 label/cache

该数据层保持 fixed-clock v4 的标签语义，只把决策时点从开盘窗口扩展到全天连续竞价。主 cache 是
label-only 窄表，不包含特征、盘口快照或逐边界审计列。

## 固定口径

| 项目 | 定义 |
| --- | --- |
| 样本键 | `date × symbol × decision_target_timestamp` |
| 决策面 | `09:30-11:29`、`13:00-14:59`，共 240 个目标分钟 |
| 决策快照 | 目标分钟后 5 秒内第一条可见快照 |
| 入场 | 实际决策快照 `timestamp + 6s` 时最后已知的 ask1 |
| horizon | `1m`、`10m`、`60m`，按交易秒推进并跳过午休 |
| 退出价 | 持有期结束后 60 个交易秒内的增量成交 VWAP |
| universe | A 股 `00/30.SZ`、`60/68.SH` |
| 日期 | 2019-01-02 至 2025-12-31，按年度 Job 并行回填 |

每个日分片只写 6 列：

```text
date
symbol
decision_target_timestamp
alpha_return_1m
alpha_return_10m
alpha_return_60m
```

无法形成完整退出窗口、入场状态无效或未通过截面 readiness/status 检查时，对应 horizon 写 `NaN`。
构建过程仍在内存中检查逻辑时钟和实际状态的先后关系；manifest 的
`causal_timestamp_violations` 必须为零。

## PVC 与断点续跑

```text
/mnt/output/opening_strength_fit/cache/full_day_clock6_narrow_1m_10m_60m_v1/
  year=YYYY/
    date=YYYY-MM-DD/
      labels.parquet
      summary.json
    full_day_label_cache_manifest.json
    _SUCCESS
```

同一天的 `labels.parquet` 和 `summary.json` 同时存在时自动复用。年度 Job 中断后可直接重提，不覆盖已经
完成的日分片。

正式镜像：

```text
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260723-full-day-narrow-v4
sha256:d24eebeeb6eeb8308cdd50d4a944eb707baa67a1c8847e2d105ae3329150c633
```

年度配置为：

```text
experiments/runs/build_full_day_clock6_narrow_labels_{2019..2025}_v1.toml
```

## 2025-01-02 smoke

Job `os-full-day-narrow-smoke-v4` 于 2026-07-23 完成：

| 检查 | 结果 |
| --- | ---: |
| 输入 tick | `20,941,373` |
| 标签行 / 列 | `1,136,863 / 6` |
| parquet 大小 | `26,740,462 bytes` |
| 股票 / 决策分钟 | `5,105 / 240` |
| 1m valid | `1,096,316` |
| 10m valid | `1,051,899` |
| 60m valid | `812,736` |
| 因果时间戳比较 / 违规 | `7,270,214 / 0` |
| 总耗时 | `5m21s` |
| 观测峰值内存 | 约 `16GB` |

smoke 产物位于：

```text
/mnt/output/opening_strength_fit/cache/full_day_clock6_narrow_1m_10m_60m_v1/
  smoke_vectorized_2025-01-02/
```

2019–2025 七个年度 Job 均已完成，共 `1,699` 个交易日、`1,645,990,902` 行；各年度 `_SUCCESS`
完整。每个年度内部按日写分片并支持断点续跑。

## 日频监督目标

分钟级 cache 中的实现收益不再是这一步的监督目标，而转为按股票展开的日内时序 feature。日频主目标不复用
旧分钟模型的 `short + 0.30 × next_close` 混合目标，而使用满足 A 股现金股票 T+1 的单一持有期收益：

```text
D 日收盘前已完成的 1m/10m/60m 路径
  -> D 日收盘集合竞价建仓
  -> D+1 收盘退出
```

公式为：

```text
alpha_return_close_to_next_close
  = ClosePrice(D+1) / PreClosePrice(D+1) - 1
```

`PreClosePrice(D+1)` 是交易所对 D 日收盘价做除权除息处理后的比较基准，避免把分红、送转等公司行动造成
的机械价格缺口当成 alpha。构建时同时要求该股票在 D 和 D+1 都有记录，且 D 日收盘价有效。这里的
“有效”只表示价格存在且为正，不表示收盘集合竞价有卖盘、排队可成交或满足容量约束。

`D+1` 按交易日历映射，不按自然日加一天。输出以 feature date `D` 为 `date`，物理 Parquet 固定四列，
其中只有最后一列是 label：

```text
date
symbol
target_date
alpha_return_close_to_next_close
```

PVC：

```text
/mnt/output/opening_strength_fit/cache/daily_close_to_next_close_labels_v1/
  year=YYYY/
    labels.parquet
    summary.json
  manifest.json
  _SUCCESS
```

2025-06-11 smoke 由 Job `os-daily-cc-label-smoke-v1` 完成：`5,150` 行、`5,150` 只股票，全部映射到
2025-06-12。该日 `000001.SZ` 的原始收盘价比为 `-1.4346%`，除权除息后的正确 label 为
`+1.6536%`；重复键和因果日期违规均为 `0`。

正式 Job `os-daily-cc-labels-2019-2025-v1` 已完成：七个年度分片共 `7,746,340` 行、
`64,799,533 bytes`，耗时 `2m40s`，零重启；所有物理 Parquet 的四列 schema 一致，每个分片的重复键
和因果日期违规均为 `0`，根目录 `manifest.json` 与 `_SUCCESS` 完整。

正式镜像：

```text
registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260724-daily-cc-label-v2
sha256:f36dd12c79cb8ed77c96be755dadee8bf9028cc2bb8e692eb866e4c366b8a448
```

此前生成的 `daily_next_session_open_close_labels_v1` 保留为诊断 cache，但不再是本路线的主监督目标：
现金股票在 D+1 开盘新买入后不能于 D+1 收盘卖出，因此该 open→close 路径不能直接作为同日可执行 replay。

为了保持 D 收盘集合竞价入场的因果信息边界，后续 feature preparation 必须截断在下单前。第一版采用保守
`14:50` 信息截止；在 `5s` 决策容忍、`+6s` 入场、horizon 后 `60s VWAP` 的口径下，1m/10m/60m
三条路径各自最晚可用的目标分钟约为 `14:47/14:38/13:48`。`14:50` 之后才完成的全天 cache 行只用于
诊断或更晚的交易决策，不得进入 D 收盘买入模型。该 cutoff 只是必要条件；若不同时要求下单时点的
ask/status/depth 和真实成交，它不能单独证明收盘新开仓可执行。

## 无模型诊断与首轮 NN

`full_day_temporal_no_model_2019_2025_v1` 已将每个交易日的三条路径连接到调整后
`close→next-close` label，并同时输出全 A 与 `pool_L` 的逐日 Rank IC、Top100 超额和头尾 10% spread。
最终覆盖 `1,699` 日、`2,446,560` 条日级 clock 指标和 `1,440` 条总体 clock 汇总。

最重要的形状不是平滑 IC：`pool_L` 的 `60m@09:30` 平均 Top100 超额为 `23.58 bps`，2020–2025
六个可用股池年份均为正，但同一点总体 Rank IC 为 `-0.00418`。因此这组路径是明显的
head/right-tail selector；普通全截面 rank regression 只能作为对照，不能成为唯一目标函数。

同一 Job 还物化了日级 sequence cache：

```text
/mnt/output/opening_strength_fit/cache/full_day_return_paths_sequence_v1/
  year=YYYY/
    date=YYYY-MM-DD/
      sequence.npz
  manifest.json
  _SUCCESS
```

序列包含 raw return、全 A 截面 rank、缺失模式、调整后日频 target 和 point-in-time `pool_L`
membership。首轮三个 NN 固定 `36m train -> 3m validation -> 6m test`、2022–2025 八个半年 OOS
fold 和 `parallelism=1`，只比较：

1. residual TCN + pool rank MSE；
2. attention TCN + pool rank MSE；
3. attention TCN + daily pool top-10% BCE。

三个模型均只读取 `14:50` 前已经完成的 horizon，并以 validation pool Top100 excess 选择 epoch。
其输出是 `pool_L` 内的日频 overlay score，不是独立的全 A 策略。

## 全 A 1m TCN 的涨停捷径审计

额外的 all-A plain TCN 基准
`temporal_nn_36m_2022_2025_all_a_rank_1m_tcn_mse_v1` 已完成 2022–2025 八个半年 OOS fold。
原始汇总为全 A Top100 return/base/excess `98.06/4.49/93.57 bps`，`pool_L` 诊断为
`54.41/11.92/42.49 bps`。其中 base 是池内原始 D→D+1 return，不是超额；整个池子相对自身的
excess 恒为 0。

逐行复核确认该结果主要学习了 D 日已经涨停的状态，而不是普通强势延续：

- Top100 平均 `33.58%` 为 D 日涨停附近股票；
- 该组 D+1 原始收益为 `261.51 bps`，对 Top100 return 贡献 `87.82 bps`；
- 非涨停附近股票只贡献 `10.24 bps`；
- 涨停附近贡献约占原始 Top100 return 的 `89.6%`；
- 事后剔除并补足 100 只后，相对未过滤原池 base 的全 A / `pool_L` excess 为
  `10.48/-1.82 bps`；若候选池也正式改为同一非涨停过滤池，则相应 base 为
  `2.21/10.10 bps`，池内 excess 为 `12.76/≈0.00 bps`。

前一种口径固定原 benchmark，用于衡量“拿掉涨停暴露后，原结果还剩多少”；后一种口径才回答
“在重新定义后的非涨停候选池内，选股相对整个池子是否有超额”。整个候选池相对自身的背景严格为
`0 bps`，`4.49 bps` 和 `11.92 bps` 都是对应原始股票池的平均原始收益，不是超额背景。

形成捷径的原因是 sequence 同时向模型暴露分钟 return rank 和有效性 mask，而训练资格只要求当天任意
时刻存在有效 feature。封板前连续上行与封板后长时间无有效 ask1 短标签的 mask 形状，能够直接编码
涨停状态；日频 target 却没有要求 `14:47`/收盘时真实可买。validation Top100 excess 随后奖励模型
集中选择该组。

该 run 现定性为“涨停延续/尾盘已有持仓留仓诊断”，不是收盘新开仓策略，也不是 opening-strength
延续证据。完整数字和口径见
[对应 evidence](../evidence/temporal_nn_36m_2022_2025_all_a_rank_1m_tcn_mse_v1/)。

普通强势延续必须在训练前完成资格修正：信号时已经封死的股票移出主候选池，在剩余可交易股票内重算
target rank 并重训；信号后才涨停的股票仍保留为预测成功。若目标是获取日内上涨至涨停的收益，决策时点
必须提前，监督目标也要改为 `实际 ask 入场价(t)→D+1 可执行退出价`，不能继续使用纯
`close→next-close` 代替策略总收益。

## 被替代的宽表尝试

2026-07-22 的 temporal V1 smoke 生成了 `212` 列，因为实现把整套 feature frame、十档入场状态和
5m/30m/close/next-close 审计字段一起写入 label parquet。它证明了午休交易时钟和因果边界，但偏离了
“只把现有 label 扩展到全天”的需求，已被本窄表 lineage 取代，不用于年度回填。
