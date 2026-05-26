# opening_strength_fit Runbook

日常闭环：预检环境和数据，做小样本 smoke，提交 K8s 训练，拉回 metrics/predictions，跑 replay 或
horizon decay，归档轻量证据。

## 1. 环境和预检

```bash
cd /home/hefu/projects/opening_strength_fit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

set -a
. ./.env
set +a

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

预检用于发现 run config、Job YAML、metrics 归档、脚本索引和文档是否不同步。

## 2. 数据检查和本地 Smoke

ClickHouse schema 和开盘窗口覆盖：

```bash
python scripts/probe_clickhouse_data.py --schema --field-notes
python scripts/probe_clickhouse_data.py \
  --start-date 2021-01-01 \
  --end-date 2022-01-31 \
  --opening-window \
  --a-share-only \
  --year-layout
```

多股票小窗口检查并保存 labeled parquet：

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet
```

本地训练 smoke 只验证训练、预测、评估和写盘链路，不作研究结论：

```bash
python scripts/run_experiment.py \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --input output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet \
  --input-kind labeled \
  --split-mode chronological \
  --test-start-date 2021-09-23 \
  --test-end-date 2021-09-23 \
  --feature-limit 80 \
  --top-n 2 \
  --output-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke

python scripts/summarize_opening_results.py \
  --input-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke
```

不要在本地为一年或多月窗口准备完整 labeled dataset。长窗口数据先在集群从 ClickHouse materialize
到 PVC labeled cache，再由训练 Job 读取。

## 3. 新建实验

复制最接近的已归档 config，修改 run id、数据口径、窗口、模型、输出路径和资源：

```text
ordinary universe: experiments/runs/gbm_opening_1y_next_month.toml
strong candidate: experiments/runs/gbm_opening_1y_next_month_strong.toml
small smoke: experiments/runs/gbm_opening_1m_3d.toml
```

必须检查：

- `run.id` 等于 config 文件名。
- `[run].status` 提交前为 `queued` 或 `running`，确认结果后改为 `completed`。
- `[data].source = "labeled_pvc"`，`[data].labeled_path` 指向最终 `*.parquet` cache。
- `[output].k8s_dir` 位于 `/mnt/output/opening_strength_fit/<run_id>/`。
- `[labels].entry_tick_delay` 是研究口径的一部分；主执行口径 delay1，delay0/2 做敏感性。
- 训练只改影响 label 或样本域的口径；fee/slippage/spread/容量/状态先放 replay。

当前 PVC 路径：

```text
delay cache: /mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay{0,1,2}_labeled.parquet
long cache:  /mnt/output/opening_strength_fit/cache/opening_2013_2024_delay1_labeled.parquet
run output:  /mnt/output/opening_strength_fit/<run_id>/
```

完整 cache 必须是最终 `*.parquet`；`.tmp.parquet`、`*.parquet.lock`、heartbeat 都不是可训练输入。
提交训练前确认 schema 包含 `entry_delay_seconds`、`entry_max_tick_gap_seconds` 和 `entry_delay_ticks`。

如果需要重新 materialize cache，使用专用入口：

```bash
python scripts/materialize_labeled_caches.py --config experiments/runs/<cache_run_id>.toml --help
```

materialize Job 必须调用 `scripts/materialize_labeled_caches.py`，不要把通用训练 renderer 改成
`scripts/run_experiment.py`。

## 4. Build、渲染和提交 Job

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-lgbm-cpu-v1
docker build --build-arg CACHE_BUST=${TAG} -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

python scripts/render_k8s_job.py \
  --config experiments/runs/<run_id>.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}
```

确认 CPU YAML：

```bash
rg -n "nvidia|gpu|nodeSelector|tolerations|envFrom|image:" experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research apply --dry-run=client -f experiments/jobs/<run_id>_job.yaml
```

提交和查看：

```bash
hfcli kubectl --cluster research delete job opening-strength-<run-slug> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research get jobs,pods -n bizewu -o wide
hfcli kubectl --cluster research logs -f job/opening-strength-<run-slug> -n bizewu
```

GPU 路径只在显式设置 `[model].device_type = "gpu"` 且 `[k8s.resources].gpu_limit` 时启用；
默认正式路径是 CPU LightGBM + PVC labeled cache。

## 5. Reader、拉回和归档

```bash
hfcli kubectl --cluster research delete job opening-strength-read-<run-slug> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_reader_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/opening-strength-read-<run-slug> -n bizewu --timeout=300s

python scripts/pull_k8s_metrics.py --config experiments/runs/<run_id>.toml
python scripts/fetch_k8s_predictions.py --config experiments/runs/<run_id>.toml --output-dir output/predictions/<run_id>
python scripts/record_experiment.py --config experiments/runs/<run_id>.toml

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

路径约定：

```text
raw metrics:       output/k8s/metrics/<run_id>_metrics_by_year.csv
raw predictions:  output/predictions/<run_id>/predictions_all.parquet
archived metrics: experiments/results/metrics/<run_id>_metrics_by_year.csv
```

## 6. Metrics 和普通开盘 Replay

单实验和多实验 metrics：

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/<run_id>_metrics_by_year.csv

python scripts/compare_opening_results.py
```

普通开盘短周期 replay：

```bash
python scripts/run_opening_intraday_backtest.py \
  --run gbm=output/predictions/gbm_opening_1y_next_month/predictions_all.parquet \
  --run gbm_strong=output/predictions/gbm_opening_1y_next_month_strong/predictions_all.parquet \
  --fee-bps 5 \
  --slippage-bps 5 \
  --max-symbol-trades-per-day 1 \
  --output-dir output/reports/opening_intraday_top20_1y_next_month_constrained
```

更严格执行约束可由 prediction 自带上下文列，或由 `--context-input` 指向 raw tick / 同 delay labeled
context 补齐。delay 分支 replay 需要 `entry_ask_price_1..N`、`entry_ask_volume_1..N`、
`entry_delay_ticks` 和 entry freshness 字段；缺关键字段默认报错。

## 7. LightGBM Delay Replay

运行前先确认 6 个 prediction 文件已拉回：

```text
output/predictions/lgbm_opening_1y_next_month_delay{0,1,2}/predictions_all.parquet
output/predictions/lgbm_opening_1y_next_month_strong_delay{0,1,2}/predictions_all.parquet
```

接口检查、完整 replay 和无约束 delay 衰减图：

```bash
python scripts/run_lgbm_delay_replays.py --check-interface-only
python scripts/run_lgbm_delay_replays.py
python scripts/plot_lgbm_delay_decay.py
```

只重画 scenario summary 图：

```bash
python scripts/run_lgbm_delay_replays.py --plot-summary-only --summary-plot-delay delay2
```

默认场景：

| scenario | 约束 |
| --- | --- |
| `proxy_top20` | 无额外成本；Top20；同股每日最多一次。 |
| `cost_10bps` | `fee_bps=5`、`slippage_bps=5`。 |
| `tradable_cost` | 成本 + continuous-trading status + decision lag <= 5s + entry path gap <= 10s。 |
| `liquidity_cost` | `tradable_cost` + `spread_bps <= 100` + decision tick 一档买卖量正数。 |
| `capacity_l3_1m` | `liquidity_cost` + 100 万/cycle、5 万/票、5% participation、entry 3 档 sweep。 |
| `capacity_l5_2m` | `liquidity_cost` + 200 万/cycle、10 万/票、5% participation、entry 5 档 sweep。 |

默认输出：

```text
output/reports/opening_intraday_lgbm_delay_replays/scenario_summary.csv
output/reports/opening_intraday_lgbm_delay_replays/delay_scan_proxy_top20.csv
output/reports/opening_intraday_lgbm_delay_replays/replay_l3_l5_single_tradable_delay{0,1,2}.png
```

轻量归档文件在 `experiments/results/backtests/`，PNG 和大 parquet 保留在 `output/`。

## 8. Alpha Horizon Decay

Alpha horizon decay 已阶段归档；默认目标是检查 delay2 opening score 是否在更长 horizon 上仍有
cross-sectional alpha，不是扩大 tick-level replay 资金规模。

固定 `09:30` cohort：

```bash
python scripts/run_alpha_horizon_decay.py \
  --decision-time 09:30:00 \
  --horizon 1m --horizon 2m --horizon 5m --horizon 10m \
  --horizon close --horizon next_close \
  --no-sampled-intraday \
  --clickhouse-intraday-labels \
  --clickhouse-close-labels \
  --allow-missing-horizons \
  --output-root output/reports/opening_alpha_horizon_decay_delay2_clickhouse_point_0930_selected
```

`09:30-09:39` opening window：

```bash
python scripts/run_alpha_horizon_decay.py \
  --decision-time 09:30:00,09:31:00,09:32:00,09:33:00,09:34:00,09:35:00,09:36:00,09:37:00,09:38:00,09:39:00 \
  --timed-target-end-time none \
  --horizon 1m --horizon 2m --horizon 5m --horizon 10m \
  --horizon close --horizon next_close \
  --no-sampled-intraday \
  --clickhouse-intraday-labels \
  --clickhouse-close-labels \
  --allow-missing-horizons \
  --output-root output/reports/opening_alpha_horizon_decay_delay2_clickhouse_point_open10_selected
```

PVC 上运行时显式传入 prediction 路径：

```bash
python scripts/run_alpha_horizon_decay.py \
  --run Universe=/mnt/output/opening_strength_fit/lgbm_opening_1y_next_month_delay2/predictions_all.parquet \
  --run Strong=/mnt/output/opening_strength_fit/lgbm_opening_1y_next_month_strong_delay2/predictions_all.parquet \
  --decision-time 09:30:00 \
  --horizon 1m --horizon 2m --horizon 5m --horizon 10m \
  --horizon close --horizon next_close \
  --no-sampled-intraday \
  --clickhouse-intraday-labels \
  --clickhouse-close-labels \
  --allow-missing-horizons \
  --output-root /mnt/output/opening_strength_fit/alpha_horizon_decay_delay2_clickhouse_point_0930_selected
```

归档结论：固定 `09:30` 到 close / next close 仍有弱正 Rank IC，但 Top20 next close 收益不稳定；
`09:30-09:39` 简单平均后 close / next close 排序基本消失。

## 9. 当前新目标：信号增强

后续先把开盘后信号做强，再考虑交易约束、容量扩展和日频 overlay。主评估口径：

- Rank IC，按 `date x decision_target_timestamp` 分组。
- Top100 选股收益，Top20 只作为尖端 alpha 辅助观察。
- by-minute 诊断，至少拆 `09:30`、`09:31-09:35`、`09:36-09:40`。

优先实验：

1. all features baseline。
2. 去掉显式 `preopen_*` feature。
3. post-open reset：切断或重置 09:30 前累计成交字段对 `volume_diff_*` / `turnover_diff_*` 的影响。
4. post-open order-book only：重点使用 ask/bid 档位、深度、queue 变化、spread 和成交冲击比例。

容量暂只看 ask1 可买量；L3/L5 sweep、fee/slippage、同股冷却等 replay 约束放到信号增强之后。

## 10. 排查和收尾

常见问题：

- `field is immutable`: 删除同名 Job 后重新 apply。
- `FileNotFoundError: experiments/runs/*.toml`: 镜像没有新 config，重新 build/push。
- `python: can't open file scripts/run_experiment.py`: Job 使用旧镜像，重新 build/push/render。
- `No such file` 指向 cache path：确认 PVC 上最终 `*.parquet` cache 已存在。
- 只有旧 `entry_lag_seconds` 的 cache：不要用于新鲜度/延迟拆分后的正式训练。
- replay 场景缺字段：传 `--context-input` 补齐，或只在探索时临时降级 `--missing-constraint warn`。

提交前确认：

- config、Job YAML、metrics 和 replay 轻量证据都落在对应目录。
- config `[run].status` 已更新为 `completed`。
- [docs/experiment_log.md](experiment_log.md)、[docs/project_brief.md](project_brief.md) 和 README 的当前状态一致。
- `output/` 不提交大 parquet、模型和临时图表。
