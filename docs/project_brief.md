# Project Brief

> Last reviewed: 2026-08-14
>
> Status: frozen research snapshot; no further experiments are scheduled.

## 目标

项目研究分钟级、因果可见的 opening score，能否在 `pool_L` 内产生扣除成本、容量和执行限制后仍稳定的
超额。模型可使用全 A 股训练，但只评价既定股池内的增量。

当前 `09:31-09:40` 基准为 `opening_model`。旧 corrected-label 血缘上的三窗口 × 1m/3m 网格已经
完成：窗口越晚，short Rank IC 越高，但可交易头部尤其隔夜超额越弱；六格中经济指标最强的是
`09:31-09:40 / 3m`。

## 固定研究口径

| 项目 | 口径 |
| --- | --- |
| 样本键 | `date × symbol × decision_target_timestamp` |
| 当前决策面 | `09:31:00-09:40:00`，每分钟一个决策点 |
| 可见性 | feature 的交易所时间不晚于决策时点；receipt-time 与同日 Pool-L 的真实可用时点仍待闭环 |
| 训练 universe | A 股 `00/30.SZ`、`60/68.SH` |
| 选择 universe | `pool_L`；S/M/universe 只作诊断 |
| 目标 | `xs_norm(short_return) + 0.30 × xs_norm(next_close_return)` |
| 验证 | `36m train -> next 6m` rolling OOS，覆盖 2022-2025 |
| canonical base/cache/model | `opening_base` / `opening_cache` / `opening_model` |
| downstream reference | archived v4 multiden；待在 `opening_model` 上重跑 |
| ablation / historical baseline | fixed-clock v4 control / mech328 v2 |

rolling OOS 可用于模型和特征选择，不是 untouched final test。

## 最新数据源合同

后续新实验将原始数据、feature 和 label 分层：ClickHouse 只用于构建 PVC raw-source cache；训练前由
同窗口 raw cache 生成 350 features，并从下列 19 个最终 label 中选择一个。每个 label 单独存放在一个
PVC 目录，表中列出目录名；公共前缀为 `/mnt/output/opening_strength_fit/datasets/`。

| 决策窗口 | 1m label | 3m label | 5m label |
| --- | --- | --- | --- |
| `09:31-09:40` | `opening_0931_0940_labels_h1m_v2` | `opening_0931_0940_labels_h3m_v2` | `opening_0931_0940_labels_h5m_v2` |
| `10:01-10:10` | `opening_1001_1010_labels_h1m_v2` | `opening_1001_1010_labels_h3m_v2` | `opening_1001_1010_labels_h5m_v2` |
| `11:01-11:10` | `opening_1101_1110_labels_h1m_v2` | `opening_1101_1110_labels_h3m_v2` | `opening_1101_1110_labels_h5m_v2` |
| `14:01-14:10` | `opening_1401_1410_labels_h1m_v2` | `opening_1401_1410_labels_h3m_v2` | `opening_1401_1410_labels_h5m_v2` |

首两个窗口各有 3 个长持有期 label；11:01 仅准备了本轮需要的 close label：

| 决策窗口 | 10m VWAP | 1h VWAP | 当日收盘价 |
| --- | --- | --- | --- |
| `09:31-09:40` | `opening_0931_0940_labels_h10m_v1` | `opening_0931_0940_labels_h1h_v1` | `opening_0931_0940_labels_hclose_v1` |
| `10:01-10:10` | `opening_1001_1010_labels_h10m_v1` | `opening_1001_1010_labels_h1h_v1` | `opening_1001_1010_labels_hclose_v1` |
| `11:01-11:10` | - | - | `opening_1101_1110_labels_hclose_v1` |

- 每个实验使用同窗口 `opening_<window>_features_350` 和一个最终 label；不得跨窗口或混用 target。
- label schema 为 3 个样本键加 `label_short`、`label_next_close`、`target_label`；NaN 表示无效。
- `target_label = xs_zscore(label_short) + 0.30 × xs_zscore(label_next_close)`。
- 中间 `opening_<window>_labels_1m_3m_5m_next_mixed` 只用于 lineage/回滚，不直接训练。
- 长持有期中间目录 `opening_{0931_0940,1001_1010}_labels_10m_1h_close_next` 同样只用于
  lineage、审计和拆分，不直接训练。
- 长持有期入场沿用决策时钟状态后 `+6s` 的 Ask1；10m/1h 在持有期结束后用 60 秒累计
  `Turnover/Volume` VWAP 退出，当日收盘使用 `close_reference.ClosePrice`。
- 长持有期 `label_next_close` 逐 key 复用既有 `opening_<window>_labels_h1m_v2`，不重新计算。
- 原 9 个短周期 label、新增 11:01 的 3 个短周期 label，以及 7 个长持有期 label 均已通过
  schema、行数和 `_SUCCESS` 检查。

上述 12 个短周期与 7 个长持有期，共 19 个当前可用的新训练 label。早期 9 个短周期 label 不是旧 6 个
1m/3m target cache 的逐值复制：旧 key 全部包含于新数据，但 key 覆盖、close tie-break 和计算精度不同；
当前旧实验保留原输入完成或复现，后续实验改用上述矩阵。

### 旧 corrected v6 兼容血缘

2026-07-31 启动的旧 1m/3m 实验继续读取各自 run TOML 中固定的 base、standalone 3m、corrected
next-close 和 target cache。六组 3×2 网格已经完成并归档；它们只用于历史复现，不再作为新实验模板。

## 当前结论

| 候选 | `pool_L` next excess | 决策 |
| --- | ---: | --- |
| mech328 v2 | `14.3174 bps` | historical overlay baseline |
| fixed-clock v4 control | `16.8024 bps` | ablation baseline |
| fixed-clock v4 multi-denominator | `17.1714 bps` | archived previous baseline |
| `opening_model` | **`17.7934 bps`** | 最新信号/模型基准；策略层待重跑 |
| 10:01-10:10 multi-denominator | `6.5491 bps` | decay checkpoint；不晋级 |
| corrected 3×2：09:31 / 3m | **`20.2087 bps`** | 六格经济指标最强；旧兼容血缘结论 |
| corrected 3×2：10:01 / 1m~3m | `6.6667~8.3935 bps` | 信号层归档；不晋级 |
| corrected 3×2：14:01 / 1m~3m | `1.0904~2.3101 bps` | 信号层归档；不晋级 |

`opening_model` 的当前证据见
[baseline evidence](../experiments/evidence/baselines/opening_model/)，不可变来源见
[canonical registry](../experiments/canonical/opening.toml)。完整历史数字见
[experiment log](experiment_log.md)。

2026-08-13 完成的 09:31 label 捷径诊断不改变 `opening_model` 的 canonical 状态，但限制了长周期
结果的解释：1m label 分布基本连续，Top100 的 1m 超额以非涨停股为主；close label 中最终涨停是
独立的极端状态，原 mixed target + MSE 会优先学习涨停概率。训练时删除约 `2.7%` 涨跌停样本后，
close Top100 的最终涨停富集从约 `3.9x` 降至 `1.2x`，且 `22.18 bps` 收盘超额中
`16.20 bps` 来自非最终涨停股。详细归因见
[label-shortcut ablation evidence](../experiments/evidence/backtests/ds350_w0931_limit_shortcut_ablation_2022_2025_v1/)。

2026-08-14 补齐的 11:01 1m/3m Baseline 与无涨跌停训练，把同一诊断扩展到四个窗口。Baseline 的
Top100 涨停富集从 09:31 的约 `4.8x`，依次降到 10:01 的约 `2.8x`、11:01 的约 `1.7x`，
14:01 已低于 `1x`；无涨跌停训练对应约 `2.1x~2.6x`、`1.7x~1.8x`、`1.05x~1.19x`
和低于 `1x`。11:01 无涨跌停 1m/3m 仍取得 `7.82/8.01 bps` Label 超额，其中最终涨停贡献仅
`0.24/0.32 bps`。这说明涨停捷径主要集中在临近开盘和 close 目标，短周期连续排序本身不依赖它。
完整归因见 [four-window limit evidence](../experiments/evidence/backtests/ds350_four_window_limit_tables_v1/)。

更精确地说，去除涨跌停训练后，09:31/10:01/14:01 的 1m Rank IC 均略升，四窗口 1m 总超额和
非最终涨停贡献基本不变或改善；变化主要发生在 Top100 涨停率及持有到收盘的收益归因。因此现有证据
强烈支持普通股票上的短期排序能力，同时也确认 Baseline 在临近开盘附加学习了涨停偏好。尚未补做的
是仅在非最终涨停股票中检验全分位收益是否严格平滑单调，不能把“存在排序能力”扩大表述为已证明
每个分位都单调。

未来信息专项审计确认 feature 与 label 在交易所时间上没有直接 overlap，但 2025 样本中有
`68.4791%` 的选中状态在标记决策时点后 `0-1.86s` 到达，且 2022-2024 缺少可用 receipt timestamp；
这些状态仍全部早于 `t+6s` 买入点。当前正确状态是“未发现明确泄露，但 receipt-time、生产推理截止
时点与同日 Pool-L 血缘尚未排除”，不能把 hard-cutoff smoke 当作完整重训结果。详见
[leakage audit](leakage_audit.md)。

close 后续的 2026H1 holdout 诊断已准备但尚未运行，将比较 close-z、bounded rank 以及
non-up/ordinary 训练样本。它目前只有 run/config/Job，不得引用为实验结果。

## 验收逻辑

信号层固定检查：

1. `pool_L` Top100 next internal excess 与费用后累和；
2. 年、半年、月和 decision clock 稳定性；
3. universe short Rank IC 与 Top1000 bucket；
4. capacity、exposure 和成本敏感性。

策略层补充因果入场/退出、同日预算与 overlap、成交约束和尾部/集中度。单边 cap、trim、bootstrap、
overlap 等用于风险解释，不设自动通过或否决门槛。

## 封存边界与未完成项

- visible refill 是决策前重分配，不是真实失败回报后的二次下单；
- 尚无通用的全天持仓、退出和现金复用账本；
- ask2-10 深度在现有输入中无有效信息；
- GPU 复跑依赖 run/Job 中记录的镜像和外部 cache。
- 2022-2024 raw 数据缺少可靠 receipt timestamp；生产推理截止时点和同日 Pool-L 的
  point-in-time 血缘尚未冻结。
- 当前 close label 是普通收益与最终涨停状态的混合；未经 robust loss、rank target 或无涨跌停训练
  对照时，mixed-MSE close 结果不能单独解释为普通股票上的连续 close alpha。
- 当前 350-feature NN 训练只支持高显存 GPU 驻留快路径；80/96GB 节点一次保存约 48GiB 训练
  tensor。主机内存峰值仍约 202GiB，因此资源口径为 8 CPU、256GiB request、1 GPU。

以下事项在冻结时仍未完成，不是当前执行计划：

1. 明确真实生产推理截止时点，并完成 receipt-time hard-cutoff、all-stock raw-safe 与 Pool-L lag
   敏感性闭环；准备好的 kill-test Job 尚未执行。
2. 在 `opening_model` 上重跑 unified capacity/no-refill/visible-refill；
3. 若重启研究，新实验直接使用分层后的 `features_350 + horizon label` 数据，不再现场拼旧 target cache；
4. 如需把 3×2 结论迁移到新权威数据版本，按同一模型/seed/rolling-OOS 口径完整重跑；
5. 只让 09:31 窗口中保留足够经济收益的候选进入完整策略验收；10:01、11:01、14:01 仅作衰减诊断。
6. 若重启 close 路线，以无涨跌停训练为基线，分别对照 robust loss 和有界
   rank/percentile target；不再只调整标准差截断倍数。

项目在上述边界下原样归档；`prepared/not run` 的配置不构成待执行任务，也不应被引用为实验结果。
