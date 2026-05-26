# opening_strength_fit Runbook

日常实验闭环：检查环境和 ClickHouse 数据，做极小本地 smoke，提交 K8s 训练，拉回 metrics/predictions，回测，分析，归档。

## 1. 环境准备

```bash
cd /home/hefu/projects/opening_strength_fit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

日常进入项目：

```bash
cd /home/hefu/projects/opening_strength_fit
source .venv/bin/activate

set -a
. ./.env
set +a
```

只读预检：

```bash
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

预检用于发现 config、Job YAML、metrics 和文档索引是否不同步。

## 2. 数据检查

先看 ClickHouse `stock.tick` 是否可读、schema 是否符合预期：

```bash
python scripts/probe_clickhouse_data.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

再看目标实验窗口的数据覆盖。这个命令只读聚合信息，不拉原始 tick：

```bash
python scripts/probe_clickhouse_data.py \
  --start-date 2021-01-01 \
  --end-date 2022-01-31 \
  --opening-window \
  --a-share-only \
  --year-layout
```

最后用多股票小窗口检查 tick 标准化、feature/label 构造、label 分布、决策点可交易性和横截面分组。只检查时不用保存；要跑本地 smoke 时加 `--labeled-output`：

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet
```

默认不要在本地为一年或多月窗口运行 `prepare_research_dataset.py`。长窗口数据先在集群里从 ClickHouse materialize 到 PVC labeled cache；后续训练优先用 `[data].source = "labeled_pvc"` 读取这个 cache。长窗口 label audit / rule baseline 也应放到集群或专门的小结果 Job，只把 CSV/metrics 拉回本地。

## 3. 本地 Smoke

本地 smoke 使用第 2 节保存的 multi-symbol labeled parquet，只确认训练、预测、评估和结果落盘能跑通。小样本结果不作研究结论。

训练 smoke：

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
```

本地训练 smoke 使用 CPU config；后续集群长窗口任务也默认从 PVC labeled cache 读取。

`run_experiment.py` 会训练模型、生成 `predictions.parquet`、写出 `metrics_by_year.csv`，并按 config 中的 evaluation 设置评估一次。这里用 `--top-n 2` 覆盖 config 的 `top_n=20`，避免 smoke 中每个横截面只有 5 个 symbol 时 top-score 变成全选。

查看 smoke metrics：

```bash
python scripts/summarize_opening_results.py \
  --input-dir output/local/gbm_opening_1y_next_month_multi_symbol_smoke
```

`summarize_opening_results.py` 只读取 `metrics_by_year.csv` 做摘要，不重算预测；本地 smoke 和后续集群拉回的年度 metrics 都用这一条查看。

正式 one-year / full-window opening 实验不要在本地准备 labeled dataset。当前推荐路径是：

```text
ClickHouse stock.tick
-> K8s materialize Job
-> /mnt/output/opening_strength_fit/cache/*.parquet
-> CPU LightGBM training Job with [data].source = "labeled_pvc"
-> /mnt/output/opening_strength_fit/<run_id>/
-> local output/k8s/metrics/ and output/predictions/
```

只拉回 `metrics_by_year.csv`、`metrics_by_month.csv`、`predictions.parquet` 或开盘 replay 所需结果。

## 4. 新建实验

当前本地实验注册表只保留 completed baseline。新建实验时复制最接近的已归档 config，再改成新的 run id 和模型/数据口径：

```text
普通 universe: experiments/runs/gbm_opening_1y_next_month.toml
strong candidate: experiments/runs/gbm_opening_1y_next_month_strong.toml
小窗 smoke: experiments/runs/gbm_opening_1m_3d.toml
```

至少修改：

```text
[run].id
[run].description
[run].status
[data].source
[data].labeled_path
[window].test_start_date / test_end_date 或 test_start_month / test_end_month
[model]...
[evaluation].selection_mode
[output].local_dir
[output].k8s_dir
[k8s.resources]...
```

约定：

- `run.id` 必须等于 config 文件名。
- `status` 提交前写 `queued` 或 `running`，拉回 metrics 并确认后写 `completed`。
- 集群训练默认用 `[data].source = "labeled_pvc"` / `[data].labeled_path`，直接读取 PVC 上的 labeled cache，只把模型结果写到 PVC。
- 本地 smoke 可传 `--input` 或设置 `[data].tick_path` 使用小的 prepared parquet/cache。
- `[output].k8s_dir` 必须在 `/mnt/output/opening_strength_fit/` 下，且一个实验一个目录。
- `evaluation.selection_mode` 第一版用 `cross_section`。
- `[labels].entry_tick_delay` 控制决策后第几个 tick 成交；主执行口径用 delay1，delay0/2 只做执行敏感性。
- 训练只改会影响 label 或样本域的口径；fee/slippage/spread/容量/状态/同股一次等执行约束先在 replay 里压测。

LightGBM CPU + PVC cache run 使用这些配置：

```toml
[data]
source = "labeled_pvc"
labeled_path = "/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay<0|1|2>_labeled.parquet"

[model]
name = "lightgbm"
device_type = "cpu"
max_bin = 63
```

说明：

- `[data].labeled_path` 指向 PVC cache；普通 universe 和 strong candidate 共享同一个 delay cache。
- strong 过滤在 labeled cache 读取后再做。
- 提交训练前确认对应 cache 已经完整写好，并且 schema 包含 `entry_delay_seconds` 与 `entry_max_tick_gap_seconds`。完整 cache 是最终 `*.parquet` 文件；`.tmp.parquet`、`*.parquet.lock` 和 heartbeat 只表示 materialize 仍在跑或被中断，不是可用训练输入。
- 只有旧 `entry_lag_seconds` 的 parquet 属于旧口径，不应用作新鲜度/延迟拆分后的正式训练输入。
- 当前 CPU Job YAML 不应包含 `nvidia.com/gpu`、GPU toleration 或 GPU nodeSelector。

当前路径：

```text
ClickHouse source: stock.tick, 09:15:00-09:45:00
1y delay cache: /mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay{0,1,2}_labeled.parquet
long delay1 cache: /mnt/output/opening_strength_fit/cache/opening_2013_2024_delay1_labeled.parquet
run output: /mnt/output/opening_strength_fit/<run_id>/
local metrics: output/k8s/metrics/<run_id>_metrics_by_year.csv
local predictions: output/predictions/<run_id>/predictions_all.parquet
```

PVC cache 由 `scripts/materialize_labeled_caches.py` 生成。若需要在集群里跑 materialize，使用专用 Job 入口；当前仓库不保留活跃 materialize Job YAML：

```bash
python scripts/materialize_labeled_caches.py --config experiments/runs/<cache_run_id>.toml --help
```

materialize Job 必须调用 `scripts/materialize_labeled_caches.py`，不要用通用 `render_k8s_job.py`
把它渲成 `scripts/run_experiment.py`。如果 materialize 被中断，重启前先清理对应
`.opening_*tmp.parquet`、`*.parquet.lock` 和 `*.parquet.lock.done`，但不要删除已经完整落盘的
`opening_1y_next_month_delay*_labeled.parquet`。

## 5. Build 镜像

修改训练代码、config 或依赖后：

```text
改代码/config -> build 新 TAG -> push -> render Job YAML -> 确认 image -> apply
```

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-lgbm-cpu-v1
docker build --build-arg CACHE_BUST=${TAG} -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}
```

本地可以用下面的 smoke 确认镜像里的 LightGBM CPU trainer 可用：

```bash
docker run --rm -i registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} python - <<'PY'
import numpy as np
from lightgbm import LGBMRegressor

x = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([0.0, 1.0, 2.0, 3.0])
model = LGBMRegressor(n_estimators=1, device_type="cpu", max_bin=63, verbosity=-1).fit(x, y)
print(model.predict(x).round(6).tolist())
PY
```

换了 `TAG` 后必须重新 render Job YAML。

GPU 路径当前不作为默认实验路径。只有显式设置 `[model].device_type = "gpu"` 且
`[k8s.resources].gpu_limit` 时，`render_k8s_job.py` 才会渲染 GPU request/toleration/nodeSelector；
这种情况下需要 GPU 兼容镜像和重新 dry-run YAML。没有这些配置时，一律按 CPU Job 处理。

## 6. 生成 Job YAML

单个 opening Job：

```bash
python scripts/render_k8s_job.py \
  --config experiments/runs/<new_run_id>.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}
```

输出路径：

```text
experiments/jobs/<new_run_id>_job.yaml
experiments/jobs/<new_run_id>_reader_job.yaml
```

确认镜像：

```bash
rg -n "image:" experiments/jobs/<new_run_id>_job.yaml
```

确认 CPU YAML：

```bash
rg -n "nvidia|gpu|nodeSelector|tolerations|image:" experiments/jobs/<new_run_id>_job.yaml
hfcli kubectl --cluster research apply --dry-run=client -f experiments/jobs/<new_run_id>_job.yaml
```

## 7. 提交和查看 Job

```bash
hfcli kubectl --cluster research delete job opening-strength-<new-run-slug> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<new_run_id>_job.yaml
hfcli kubectl --cluster research get jobs,pods -n bizewu -o wide
hfcli kubectl --cluster research logs -f job/opening-strength-<new-run-slug> -n bizewu
```

reader Job：

```bash
hfcli kubectl --cluster research delete job opening-strength-read-<new-run-slug> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<new_run_id>_reader_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/opening-strength-read-<new-run-slug> -n bizewu --timeout=300s
```

确认 Job request 和节点：

```bash
hfcli kubectl --cluster research -n bizewu get pods -o wide | rg 'opening-strength'
```

排查要点：

- `field is immutable`: 删除同名 Job 后重新 apply。
- `FileNotFoundError: experiments/runs/*.toml`: 镜像里没有新 config，重新 build/push。
- `python: can't open file scripts/run_experiment.py`: Job 使用的镜像不是当前代码镜像，重新 build/push/render。
- `FileNotFoundError` / `No such file` 指向 cache path：确认对应 delay 的 PVC labeled cache 已存在。
- `field is immutable`: 删除同名 Job 后重新 apply。
- `No tick data path supplied`: 本地/path 模式没填 `[data].tick_path`，或本地 smoke 忘了传 `--input`。
- `input path does not exist`: 容器看不到该路径，检查 PVC、`mount_path` 和 `[data].tick_path`。

## 8. 拉回 Metrics

重复传 `--config` 可一次拉一个或多个实验：

```bash
python scripts/pull_k8s_metrics.py \
  --config experiments/runs/<new_run_id>.toml
```

拉回文件写到：

```text
output/k8s/metrics/<run_id>_metrics_by_year.csv
```

`output/k8s/metrics/` 是本地原始产物；`experiments/results/metrics/` 是收尾时归档的轻量证据。

## 9. 拉回 Predictions 和回测

拉回预测并合并：

```bash
python scripts/fetch_k8s_predictions.py \
  --config experiments/runs/<new_run_id>.toml \
  --output-dir output/predictions/<new_run_id>
```

开盘短周期回测使用 predictions 中的 `decision_target_timestamp` 和 `label`；如果训练配置设置了
`entry_tick_delay`，这里的 `label` 已经是延迟成交后的收益。后续主线只继续 LightGBM 普通 universe 和 LightGBM strong。
默认每天本金从 1.0 开始，在 `09:30/09:32/09:34/09:36/09:38` 五个非重叠两分钟 cycle 上按预测分选
Top20；`--max-symbol-trades-per-day 1` 会阻止同一股票在同一天被重复选中，没选满的资金留作现金。

当前先不要为 fee 立刻重训。fee/slippage 对固定入选交易是确定性收益扣减，应先在 replay 中统一压测；
只有要训练 net label、成本改变样本有效性，或策略定义变成“扣费后排序器”时，才另开 fee 训练分支。

```bash
python scripts/run_opening_intraday_backtest.py \
  --run gbm=output/predictions/gbm_opening_1y_next_month/predictions_all.parquet \
  --run gbm_strong=output/predictions/gbm_opening_1y_next_month_strong/predictions_all.parquet \
  --fee-bps 5 \
  --slippage-bps 5 \
  --max-symbol-trades-per-day 1 \
  --output-dir output/reports/opening_intraday_top20_1y_next_month_constrained
```

replay 可以从 prediction 自带上下文列或额外 context 数据补齐执行约束。`entry_tick_delay` 分支需要使用
`entry_ask_price_1..10` 和 `entry_ask_volume_1..10` 建模 entry tick 卖盘容量；如果旧 prediction
文件缺少这些列，优先传 `--context-input` 指向 raw tick，让 replay 按
`date/symbol/decision_target_timestamp` enrich。labeled context 只适合单个已知 delay 分支，必须和
`--context-entry-tick-delay` 一致。拿到 prediction 和 context 后，可以打开更严格约束：

```bash
python scripts/run_opening_intraday_backtest.py \
  --run gbm=output/predictions/gbm_opening_1y_next_month/predictions_all.parquet \
  --run gbm_strong=output/predictions/gbm_opening_1y_next_month_strong/predictions_all.parquet \
  --context-input <raw_tick_or_same_delay_labeled_context_root> \
  --context-kind auto \
  --context-entry-tick-delay 1 \
  --context-label-mode replace \
  --fee-bps 5 \
  --slippage-bps 5 \
  --tradable-status T0 \
  --tradable-status 20 \
  --tradable-status TRADE \
  --max-spread-bps 100 \
  --min-limit-up-room-bps 5 \
  --min-capacity-notional 50000 \
  --capital-per-cycle 1000000 \
  --max-participation-rate 0.05 \
  --ask-depth-levels 3 \
  --ask-depth-fill-mode sweep \
  --ask-depth-participation-rate 1.0 \
  --max-symbol-trades-per-day 1 \
  --output-dir output/reports/opening_intraday_top20_1y_next_month_constrained
```

`--min-limit-up-room-bps` 只有在 prediction 或 context 已有 `ask1_to_limit_up_bps` 时才能使用；
当前缺字段默认报错，便于发现“场景名写了约束但实际没执行”的问题。

旧归档 predictions 没有 `status`、`spread_bps`、`turnover_diff_30t` 等上下文列；如果不传
`--context-input`，这些文件只能用于成本、滑点和同股重复交易的复算。

如果后续需要检查 execution-delay sensitivity，可以对 LightGBM delay0/1/2 六个 prediction 跑标准 replay 网格：

```bash
python scripts/run_lgbm_delay_replays.py \
  --context-input <raw_tick_context_root> \
  --context-kind auto \
  --context-label-mode replace
```

`run_lgbm_delay_replays.py` 会汇总 `scenario_summary.csv`，并生成
`replay_l3_l5_single_tradable_delay{0,1,2}.png` 作为 6 场景约束衰减图。只重画 summary 图、不重新跑 replay：

```bash
python scripts/run_lgbm_delay_replays.py --plot-summary-only --summary-plot-delay delay2
```

无约束 delay0/1/2 衰减图单独生成：

```bash
python scripts/plot_lgbm_delay_decay.py
```

标准 delay 网格优先用 raw tick context，让 wrapper 对 delay0/1/2 分别派生 entry label 和执行上下文。
如果必须使用 labeled context，需要按 delay 分开传入；wrapper 会检查 `entry_delay_ticks`，不匹配或缺列会直接退出。

默认读取：

```text
output/predictions/lgbm_opening_1y_next_month_delay1/predictions_all.parquet
output/predictions/lgbm_opening_1y_next_month_strong_delay1/predictions_all.parquet
output/predictions/lgbm_opening_1y_next_month_delay2/predictions_all.parquet
output/predictions/lgbm_opening_1y_next_month_strong_delay2/predictions_all.parquet
output/predictions/lgbm_opening_1y_next_month_delay0/predictions_all.parquet
output/predictions/lgbm_opening_1y_next_month_strong_delay0/predictions_all.parquet
```

这些是生成 LightGBM delay predictions 后的预期本地路径，不代表当前已经存在。运行前先用
`test -f` 或 `find output/predictions -maxdepth 2 -type f` 确认文件已拉回。

默认场景：

| scenario | 约束 |
| --- | --- |
| `proxy_top20` | 无额外成本；Top20；同股每日最多一次。 |
| `cost_10bps` | `fee_bps=5`、`slippage_bps=5`。 |
| `tradable_cost` | 成本 + `status/entry_status in T0,20,TRADE` + decision lag 不超过 5 秒 + entry 路径相邻 tick 最大间隔不超过 10 秒。只保留这一版 tradable 口径。 |
| `liquidity_cost` | `tradable_cost` + `spread_bps <= 100` + decision tick 一档买卖量正数。 |
| `capacity_l3_1m` | `liquidity_cost` + 每 cycle 100 万资金、单票目标 5 万、`turnover_diff_30t` 参与率不超过 5%、最低可见容量 5 万，并要求 entry 3 档卖盘能容纳目标金额；收益按 3 档 sweep VWAP 修正。 |
| `capacity_l5_2m` | `liquidity_cost` + 每 cycle 200 万资金、单票目标 10 万、`turnover_diff_30t` 参与率不超过 5%、最低可见容量 10 万，并要求 entry 5 档卖盘能容纳目标金额；收益按 5 档 sweep VWAP 修正。 |

旧 prediction 如果没有 `entry_max_tick_gap_seconds`，entry 新鲜度需要通过重建 prediction 或传
`--context-input` 重新补齐；不要用总等待时间比较 delay0/1/2 的 entry 新鲜度。

输出：

```text
output/reports/opening_intraday_lgbm_delay_replays/scenario_summary.csv
output/reports/opening_intraday_lgbm_delay_replays/replay_l3_l5_single_tradable_delay{0,1,2}.png
output/reports/opening_intraday_lgbm_delay_replays/delay_scan_proxy_top20.csv
output/reports/opening_intraday_lgbm_delay_replays/delay_scan_proxy_top20.png
output/reports/opening_intraday_lgbm_delay_replays/<delay>/<scenario>/intraday_summary.csv
output/reports/opening_intraday_lgbm_delay_replays/<delay>/<scenario>/intraday_cycles.csv
output/reports/opening_intraday_lgbm_delay_replays/<delay>/<scenario>/intraday_selected_trades.csv
```

阶段归档时，把轻量 replay evidence 复制到 `experiments/results/backtests/`：

```text
opening_intraday_lgbm_delay_replays_scenario_summary.csv
opening_intraday_lgbm_delay_replays_delay_scan_proxy_top20.csv
opening_intraday_lgbm_delay_replays_trace.json
```

PNG 和大 parquet 仍保留在 `output/`，默认不提交。

## 10. Alpha Horizon Decay

Alpha horizon decay 已完成阶段归档。这个步骤的目的不是继续扩大 tick-level replay 资金规模，
而是检查 opening score 在更长 horizon 上是否仍有 cross-sectional alpha。归档实验固定使用
delay2 作为保守 entry 口径，比较 Universe 与 Strong 分支在 opening window 的 1m/2m/5m/10m、
same-day close 和 next-day close 上的 TopN mean alpha return 和 rank IC。

归档结论：

- 固定 `09:30` cohort 的 Rank IC 从 1m 到 10m 逐步衰减，到 close / next close 仍有弱正排序。
- 固定 `09:30` 的 next close Top20 mean alpha return 为负，不能解释成隔夜 Top20 已可交易。
- `09:30-09:39` 十分钟简单平均后，close / next close 排序效果基本消失，不支持直接取十分钟均值作为日频特征。
- 下一步 active work 是已有日频候选池内的 opening score 重排序 / 辅助排序，而不是继续扩大 opening replay。

主口径直接从 ClickHouse 拉 target-minute bid/ask mid price。timed horizon 一律用当前 delay2
`buy_price` 到未来整分钟有效 `mid_price` 的 point return；`1m` 不再混用旧 60s VWAP proxy label。
固定 `09:30:00-09:39:00` 这 10 个开盘分钟作为 decision cohort，并设置
`--timed-target-end-time none`，让 1m/2m/5m/10m 都使用同一组 opening cohorts：

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

如果只看固定 `09:30` opening score，与早期 smooth decay 图对齐，可加 `--decision-time 09:30:00`。
这个读法回答的是“09:30 这一批股票持有更久以后排序 IC 如何衰减”，不是每个开盘分钟平均：

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

PVC 上跑同样命令时显式指定 prediction cache 路径：

```bash
python scripts/run_alpha_horizon_decay.py \
  --run Universe=/mnt/output/opening_strength_fit/lgbm_opening_1y_next_month_delay2/predictions_all.parquet \
  --run Strong=/mnt/output/opening_strength_fit/lgbm_opening_1y_next_month_strong_delay2/predictions_all.parquet \
  --decision-time 09:30:00,09:31:00,09:32:00,09:33:00,09:34:00,09:35:00,09:36:00,09:37:00,09:38:00,09:39:00 \
  --timed-target-end-time none \
  --horizon 1m --horizon 2m --horizon 5m --horizon 10m \
  --horizon close --horizon next_close \
  --no-sampled-intraday \
  --clickhouse-intraday-labels \
  --clickhouse-close-labels \
  --allow-missing-horizons \
  --output-root /mnt/output/opening_strength_fit/alpha_horizon_decay_delay2_clickhouse_point_open10_selected
```

输出：

```text
alpha_horizon_decay_summary.csv
alpha_horizon_decay_buckets.csv
alpha_horizon_decay_mean_return.png
alpha_horizon_decay_rank_ic.png
alpha_horizon_decay_trace.json
```

阶段归档路径：

```text
output/reports/opening_alpha_horizon_decay_delay2_clickhouse_point_open10_selected
output/reports/opening_alpha_horizon_decay_delay2_clickhouse_point_0930_selected
output/reports/opening_alpha_horizon_decay_delay2_compare_selected

experiments/results/backtests/opening_alpha_horizon_decay_delay2_0930_summary.csv
experiments/results/backtests/opening_alpha_horizon_decay_delay2_0930_trace.json
experiments/results/backtests/opening_alpha_horizon_decay_delay2_open10_summary.csv
experiments/results/backtests/opening_alpha_horizon_decay_delay2_open10_trace.json
experiments/results/backtests/opening_alpha_horizon_decay_delay2_0930_vs_open10_summary.csv
experiments/results/backtests/opening_alpha_horizon_decay_delay2_close_next_close_by_decision_minute.csv
```

Daily candidate reranking / overlay 的下一步使用口径：

```text
输入: 上游日频策略给出的当天候选池、日频分数或目标持仓
开盘辅助信号: 优先使用固定 09:30 或最早可交易 opening score
对照: 原始日频候选排序 vs 加 opening score 后的 rerank / overlay
评估: 候选池内 Rank IC、TopK 命中率、最终持仓收益、换手和未成交样本
暂不主推: 09:30-09:39 opening score 简单平均
```

如果只想预览任务而不读取 parquet：

```bash
python scripts/run_lgbm_delay_replays.py --dry-run
```

拉回新的 CPU LightGBM predictions 后，先跑接口检查：

```bash
python scripts/run_lgbm_delay_replays.py --check-interface-only
```

这一步会检查 `output/predictions/<run_id>/predictions_all.parquet` 是否包含 replay 默认场景需要的
core prediction、delay metadata、freshness、状态、spread、容量和 entry 3/5 档卖盘字段，并确认
`entry_delay_ticks` 与 delay0/1/2 分支一致。检查通过后再跑完整 replay。

默认 `--missing-constraint error`，所以请求了某个约束但 prediction 和 `--context-input` 都缺字段时会直接报错。
这能防止 replay 名字里写着约束、实际却被静默跳过。只有做探索性诊断、明确接受缺字段时，才临时降级：

```bash
python scripts/run_lgbm_delay_replays.py --missing-constraint warn
```

涨停距离现在不是默认网格的一部分；需要额外压测时显式运行
`--scenario limit_up_room_10s`，并确保 context/prediction 已有 `ask1_to_limit_up_bps`。

真实约束检查表：

replay 实现上不要为每个真实限制单独造一套模型，优先合并成五类：

```text
成本 haircut: fee / slippage / 平均冲击成本
时延和新鲜度: data latency / signal latency / gateway latency / entry tick delay / entry tick gap
可交易性: status / entry_status / 涨跌停距离 / 停牌临停
流动性和成交: spread / 一档深度 / 多档 sweep / capacity / participation / partial fill
组合调度: TopN / 单票权重 / 现金 / 同股重复 / cooldown / 换手
```

| 约束 | 字段/参数 | 是否需要重训 |
| --- | --- | --- |
| 成交延迟 | `--context-entry-tick-delay` / `[labels].entry_tick_delay` | replay 可用 raw tick context 重算 realized label；若要让模型训练目标也变成 delay label，再单独训练 delay run。 |
| fee/slippage | `--fee-bps` / `--slippage-bps` | 通常不需要，先 replay。 |
| 交易状态 | `status` / `entry_status` / `--tradable-status` | 固定部署硬过滤可重训；压力测试先 replay。 |
| tick 新鲜度 | `decision_lag_seconds` / `entry_max_tick_gap_seconds` / `--max-entry-tick-gap-seconds` | 不需要，replay。成交延迟本身用 `entry_delay_seconds` 单独审计。 |
| spread | `spread_bps` / `--max-spread-bps` | strong candidate 可进样本域；执行压测先 replay。 |
| 涨停距离 | `ask1_to_limit_up_bps` / `--min-limit-up-room-bps` | 不需要，除非定义固定交易池。 |
| 一档挂量 | `ask_volume_1` / `bid_volume_1` | 不需要，replay。 |
| entry 卖盘容量 | `entry_ask_price_1..N` / `entry_ask_volume_1..N` / `--ask-depth-levels` | 不需要重训模型；prediction 可自带 entry 盘口上下文，也可由 replay `--context-input` enrich；用于 ask1-only 过滤、部分成交或多档 sweep。 |
| 容量/参与率 | `turnover_diff_30t` / `--max-participation-rate` | 不需要，replay。 |
| TopN/单票/现金 | `--top-n` / `--max-symbol-weight` | 不需要，replay。 |
| 同股重复/冷却 | `--max-symbol-trades-per-day` / `--symbol-cooldown-minutes` | 不需要，replay。 |
| T+1 | close/next-open/next-close label | 需要新的 horizon label，不是当前 60s replay 能解决。 |

Ask-depth replay 参数：

```text
--context-input <path>  # prediction 很瘦时，从 raw tick 或同 delay labeled context 补执行字段
--context-kind auto     # auto/raw_ticks/labeled
--context-label-mode replace  # 用 context label 作为 replay PnL，保留 prediction_label 便于审计
--ask-depth-levels 1      # ask1-only
--ask-depth-levels 3      # 默认小容量 L3 场景
--ask-depth-levels 5      # 默认小容量 L5 场景
--ask-depth-fill-mode filter  # 深度不足则不交易
--ask-depth-fill-mode scale   # 深度不足则按可成交比例降权，剩余留现金
--ask-depth-fill-mode sweep   # 深度足够时扫档成交，用 sweep VWAP 修正 label
--ask-depth-participation-rate 0.5  # 只允许使用可见卖盘的一部分，取值 (0, 1]
```

默认不要开 `--allow-decision-depth-fallback`。用了 `entry_tick_delay` 后，decision tick 的
`ask_volume_1` 不能代表真实 entry tick 的可成交量；fallback 只适合旧预测文件的粗略诊断。

## 11. 分析 Metrics

单个实验：

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/gbm_opening_1y_next_month_metrics_by_year.csv
```

比较已归档 full-window 实验：

```bash
python scripts/compare_opening_results.py
```

指定实验：

```bash
python scripts/compare_opening_results.py \
  --run gbm=output/k8s/metrics/gbm_opening_1y_next_month_metrics_by_year.csv
```

重点看：

```text
cross_section IC: group_rank_ic_mean
cross_section IC IR: group_rank_ic_ir
daily_rank_ic_mean
model_test_r2
top_score_mean_return
```

当前主配置的 `ic_mode` 是 `cross_section`。

## 12. 分析开盘短周期回测

重点看：

```text
mean_cycle_return_bps
cycle_win_rate
mean_day_final_return_bps
positive_day_rate
compounded_month_return
candidate_count
eligible_count
cash_weight
```

逐日曲线在 `output/reports/opening_intraday_top20_1y_next_month/daily_curves/`。

## 13. 收尾：记录和审计

一轮实验包括：训练完成、reader 合并完成、metrics 拉回、需要的 predictions/opening replay 分析完成，并且相关 config 的 `status` 已更新为 `completed`。
如果某条 `docs/project_brief.md` 的 active research route 已满足阶段目标，还需要同步更新
`docs/project_brief.md`、`docs/experiment_log.md` 和 README 的当前状态说明，明确“已归档”和下一步 active work。

```bash
python scripts/record_experiment.py \
  --config experiments/runs/<new_run_id>.toml
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

提交前确认：

- 新实验 config、Job YAML、metrics 记录和 opening replay 记录都落在对应目录。
- `docs/experiment_log.md` 写入实验状态或结论。
- `output/` 只保留本地运行产物，不提交大 parquet / pkl。
- 新 Job YAML 使用刚 build/push 的镜像 tag。
