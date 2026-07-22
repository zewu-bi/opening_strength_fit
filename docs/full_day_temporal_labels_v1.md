# 全天分钟级因果 label/cache V1

这一步只建立新的研究数据底座，不直接宣称策略改进。它把 fixed-clock v4 的因果入场口径从开盘十分钟
扩展到全天连续竞价，并为后续的全天 score、持仓和现金账本提供统一样本键。

## 固定口径

| 项目 | V1 定义 |
| --- | --- |
| 样本键 | `date × symbol × decision_target_timestamp` |
| 决策面 | `09:31-11:29`、`13:01-14:59`，共 238 个目标分钟 |
| 决策快照 | 目标分钟后 5 秒内第一条可见快照 |
| 入场 | 实际决策快照 `timestamp + 6s` 时的最后已知 ask 状态 |
| 分钟 horizon | `5m`、`30m`，按交易秒推进，自动跳过 `11:30-13:00` 午休 |
| 分钟退出价 | horizon 起点后 60 个交易秒内的增量成交 VWAP |
| 长 horizon | 当日 close、下一交易日 close |
| universe | A 股 `00/30.SZ`、`60/68.SH` |
| 日频引用 | 严格 T-1 market-cap/share |

分钟 horizon 的每个边界都保留三类字段：逻辑 target timestamp、实际 source timestamp 和 state age。
构建器强制 `source <= target`，manifest 中的 `causal_timestamp_violations` 必须为零。午休附近不使用
90 分钟墙钟时间替代交易时间；例如 `11:29:09 + 5m = 13:04:09`。

## 输出与断点续跑

全天原始 tick 和派生特征不能再按全年单文件构建。V1 每次只处理一个交易日，原子写入：

```text
<cache-root>/
  date=YYYY-MM-DD/
    labels.parquet
    summary.json
  full_day_label_cache_manifest.json
  _SUCCESS
```

已同时存在 `labels.parquet` 和 `summary.json` 的日期默认直接复用，不再查询 ClickHouse。需要重建时在
配置中设置 `[cache].overwrite = true`。全年任务中途退出后，可直接重启同一个 Job。

## 使用

本地文件 smoke 可复用同一个正式入口；配置中的 close/next-close 需要 ClickHouse，因此纯本地输入通常
只保留分钟 horizon：

```bash
osf-build-labeled-cache \
  --config experiments/runs/<full-day-config>.toml \
  --input <raw-ticks.parquet> \
  --output-dir output/artifacts/cache_builds/<run-id>
```

集群一日 smoke：

```bash
IMAGE=registry.corp.highfortfunds.com/bizewu/opening-strength-fit:20260722-full-day-label-v1

osf-render-k8s-job \
  --config experiments/runs/build_full_day_clock6_temporal_smoke_20250102_v1.toml \
  --image "$IMAGE"

kubectl apply -f \
  experiments/jobs/build_full_day_clock6_temporal_smoke_20250102_v1_job.yaml
```

smoke 通过后再提交年度配置：

```bash
kubectl apply -f experiments/jobs/build_full_day_clock6_temporal_2025_v1_job.yaml
```

验收时至少检查：Job 为 `Complete`、cache root 有 `_SUCCESS`、manifest 因果违规为 `0`、238 个目标分钟
覆盖早盘/午休两侧/尾盘，以及各 horizon 的 valid rows 随收盘临近按预期下降。该 cache 通过后才进入
全天模型与完整持仓/现金/退出账本阶段。
