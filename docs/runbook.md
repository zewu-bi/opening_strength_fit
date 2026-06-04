# 运行手册

这个文件只放操作命令和配置口径。研究逻辑见 [project_brief.md](project_brief.md)，实验事实见
[experiment_log.md](experiment_log.md)。

标准闭环：

```text
precheck -> render job -> apply/wait -> sync artifacts -> audit/coverage -> analysis
```

## 当前维护分层

- Active mainline: single mixed label，当前暂定 `w_long=0.30`；short label 为主体，小权重 long /
  next-day close 约束。训练仍用 full universe，S/M/L 只作为 TopN selection mask，指标在不同 mask
  下分别汇总。
- Terminology: 主线不要简称为 “postopen 口径”。当前口径是 decision-time visible / causal feature
  set：只要在下单决策时已经可见，`preopen_*` 集合竞价摘要、`09:30` 开盘快照、以及 `09:31-09:40`
  的开盘后轨迹都可以进入特征。`postopen_*` 只是历史特征前缀，特指开盘后轨迹/盘口响应特征，不表示删除
  集合竞价信息。
- Cache rebuild line: `build_delay2_2024_cache_v1` 已按用户要求停止；旧 2023/2024 v1 cache 和过期派生
  cache 已从 PVC 清掉。新的基础 cache 线是 `build_delay2_2015_cache_v2` 至
  `build_delay2_2024_cache_v2`，统一写入
  `/mnt/output/opening_strength_fit/cache/opening_10y_201501_202412_delay2_base_labeled_v2/`。
  `opening-strength-fit-20260603-cache-v2` 镜像已 build/push；`2021-2024` 四个年度 shard 已完成并有
  manifest，`2015-2020` 仍待分批启动。
- Next rolling target: 先用已完成的 `2021-2024` base labeled cache v2 做 `36m train -> next 1m test`
  monthly rolling，覆盖 `2024-01` 至 `2024-12` 这 12 个 OOS 月份；每个 fold 只用测试月之前的 36
  个自然月训练。后续补齐 `2015-2020` 后再把同一口径往更早 OOS 月份扩展。
- Historical evidence: `docs/experiment_log.md` 已记录、`experiments/results/**` 有轻量证据、或文档明确引用的
  run/job/config 都保留。guard / clean target / two-model alpha-risk / risk penalty 路线现在是 historical /
  superseded evidence，不是当前主实现。
- Generated retained trace: 已运行过的 `experiments/jobs/*.yaml` 是轻量 K8s manifest trace；即使对应路线
  已封存，也保留作可复现索引。
- Stale cleanup: 只有同时满足“无 docs 引用、无 `experiments/results` 证据、非当前 running、非计划中 active”
  的 run/job/config 才能删除。本地 `__pycache__`、`.pytest_cache`、`*.egg-info` 可以直接清；`.venv`、`.env`
  和 ignored `output/` 不作为项目级瘦身目标。

## 1. 预检

```bash
cd /home/hefu/projects/opening_strength_fit
source .venv/bin/activate
set -a; . ./.env; set +a

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
python scripts/probe_clickhouse_data.py --schema --field-notes
```

## 2. 外部股池

mentor 发来的股池和隔壁 `xy_fit` 项目的 `X.parquet` / `Y.parquet` 在同一个 Ceph S3 目录：

```text
bucket:   lml.bzw@ssd
endpoint: http://ceph-s3-ssd.prod.highfortfunds.com
prefix:   data/

data/pool_L.parquet
data/pool_M.parquet
data/pool_S.parquet
```

本地读取需要司令部 LDAP 凭据。把真实值写在项目根目录的 `.env`，不要提交；`.gitignore` 已忽略
`.env`。

```bash
CEPH_LDAP_ID='your_headquarter_username'
CEPH_LDAP_KEY='your_headquarter_password'
```

读取前加载环境变量：

```bash
cd /home/hefu/projects/opening_strength_fit
set -a; . ./.env; set +a
```

项目原生支持读取 `bucket@ssd/path.parquet` 形式的股池路径。新环境安装依赖后可以直接在 run config
里启用；如果当前本地 venv 还没重装依赖，也可以临时复用隔壁 `xy_fit` 的 venv 快速核对：

```bash
cd /home/hefu/projects/xy_fit
set -a; . /home/hefu/projects/opening_strength_fit/.env; set +a

.venv/bin/python - <<'PY'
from xyfit.io import build_client

client = build_client()
resp = client.list_objects_v2(Bucket="lml.bzw", Prefix="data/")
for item in sorted(resp.get("Contents", []), key=lambda x: x["LastModified"], reverse=True):
    print(item["LastModified"], item["Size"], item["Key"])
PY
```

推荐先把股池作为 TopN selection mask，不改变训练 universe。日常快速试验可以直接用 CLI：

```bash
python scripts/run_experiment.py \
  --config experiments/runs/lgbm_delay2_postopen_0931_0940_baseline_v1.toml \
  --pool S \
  --output-dir output/local/lgbm_delay2_postopen_pool_s_selection
```

`--pool S|M|L` 会自动映射到：

```text
S -> lml.bzw@ssd/data/pool_S.parquet
M -> lml.bzw@ssd/data/pool_M.parquet
L -> lml.bzw@ssd/data/pool_L.parquet
```

默认语义是 `filter_train=false`、`filter_selection=true`：模型仍在 full universe 上训练和打分，
最终 TopN 只从池内候选里选。当前主线评估默认要同时产出 `pool_S`、`pool_M`、`pool_L` 三个口径；
不要只看单个池子的 TopN 结果。保守无未来检验可以加：

```bash
--pool S --pool-date-lag-sessions 1
```

`scripts/run_alpha_conditioned_rolling_validation.py` 目前只支持这个 selection-mask 语义：alpha/risk
训练和 risk target 仍使用 full universe，TopN 从股池内候选里选；不要给 rolling 入口传
`--pool-filter-train` 或 `--pool-add-feature`。后续单模型 mixed-label rolling 也沿用同一原则：
训练 full universe，在 S/M/L selection masks 下分别汇总同一组指标。

当前主线不使用 `--pool-filter-train` 或 `--pool-add-feature`；股池只做 TopN selection mask。

正式实验归档时，把同样口径落进 TOML：

```toml
[stock_pool]
enabled = true
path = "lml.bzw@ssd/data/pool_S.parquet"
name = "pool_S"

# 推荐第一阶段：训练仍用 full universe，最终 TopN 只在股池里选。
filter_train = false
filter_selection = true

# 如果确认股池当日盘前可知，用 0；如果不确定生成时点，先用 1 做保守无未来检验。
date_lag_sessions = 0

# 输出预测文件里会带这个 0/1 列，方便 audit。
membership_col = "stock_pool_member"
annotate_predictions = true

# 只有想让模型显式使用“是否在股池”这个特征时才打开。
add_feature = false
```

开启 `filter_selection=true` 后，`metrics_by_year.csv` 的 TopN 汇总使用池内候选行；
`predictions*.parquet` 会保留全 universe 打分并额外写出 `stock_pool_member`，方便事后比较。
同时会额外输出池内分桶文件：

```text
score_buckets_<period>_stock_pool.csv
score_buckets_stock_pool.csv
```

三份股池文件都是 `date x symbol` 的 bool 宽表：

- index 是交易日，范围 `2020-01-02` 到 `2025-12-31`，共 `1455` 天。
- columns 是股票代码，当前共 `5420` 列，例如 `000001.SZ`。
- cell 为 `True` 表示该股票当天在对应池子里。
- `pool_S`、`pool_M`、`pool_L` 不是互斥分组；当前看起来是嵌套候选池：`pool_S ⊂ pool_M ⊂ pool_L`。
- 2025-12-31 当天大约有 `pool_S=1497`、`pool_M=2494`、`pool_L=3491` 只股票入池。

快速 inspection 命令：

```bash
cd /home/hefu/projects/xy_fit
set -a; . /home/hefu/projects/opening_strength_fit/.env; set +a

.venv/bin/python - <<'PY'
from io import BytesIO

import numpy as np
import pyarrow.parquet as pq
from xyfit.io import build_client

client = build_client()
for key in ("data/pool_L.parquet", "data/pool_M.parquet", "data/pool_S.parquet"):
    body = client.get_object(Bucket="lml.bzw", Key=key)["Body"].read()
    df = pq.ParquetFile(BytesIO(body)).read().to_pandas()
    counts = df.to_numpy(dtype=bool, copy=False).sum(axis=1)
    print(
        key,
        "shape=", df.shape,
        "date_range=", (df.index.min(), df.index.max()),
        "members_median=", float(np.median(counts)),
        "members_last=", int(counts[-1]),
    )
PY
```

## 3. 本地 Smoke

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
  --preview-rows 3 \
  --label-preview-rows 3 \
  --labeled-output output/local/inspect_smoke/multi_symbol_2021-09-22_2021-09-23_labeled.parquet

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

不要在本地构造多月或一年级别 labeled dataset；正式长窗口使用已有 PVC cache 或专门训练任务。

## 4. 实验配置

每个正式实验至少对应：

```text
experiments/runs/<run_id>.toml
experiments/jobs/<run_id>_job.yaml
experiments/results/metrics/<run_id>_metrics_by_year.csv
```

例外：

- `[run].kind = "feature_audit"`：运行 grouped importance、permutation 和 drop-retrain ablation，
  audit CSV 写到 run output dir。
- `[run].kind = "labeled_cache"`：从 ClickHouse 读取 tick、构造 labeled rows，并写出单个 PVC cache，
  不训练模型。`scripts/build_labeled_cache.py` 会同时写 `<cache>.manifest.json` 和 run output 下的
  `labeled_cache_manifest.json`，manifest 由 `opening_strength_fit.cache_manifest` 生成。
- `[run].kind = "cache_transform"` 或 `"target_cache"`：运行 target-label cache 构建，
  output 通常是 `/mnt/output/opening_strength_fit/cache/*.parquet`。
- `[run].kind = "next_close_label_cache"`：运行 `scripts/build_next_close_labels.py`，从 labeled decision rows
  的 `buy_price` 和 ClickHouse close price 缓存 `alpha_return_next_close`。
- `[run].kind = "learned_risk_layer"`：训练 learned dirty-risk / next-flip risk model，
  输出 risk predictions，供后续 score-risk sweep 读取。
- `[run].kind = "alpha_conditioned_rolling_validation"`：每个测试月用前 N 个月重新训练
  alpha 和 alpha-conditioned risk model，并固定评估 Top100 penalty variants。
- `[run].kind = "gap_risk_attribution"`：比较 baseline Top100、risk-penalized 剔除、替换和保留样本。
- `[run].kind = "score_risk_sweep"`：对已有 prediction 做 score/risk penalty 或 hard gate 扫描，
  不训练新模型。
- `[run].kind = "exploration"`：可以先保持 active/running，不要求立刻有 metrics；确认后再归档成正式实验。
- `postopen_*` 特征实验仍走 `scripts/run_experiment.py`；`[features].include_postopen_decision = true`
  打开开盘后轨迹/盘口响应特征，v2 特征再加 `[features].include_postopen_v2 = true`。这只是特征族开关，
  不等于删除 `preopen_*` 集合竞价摘要。

PVC 约定：

```text
cache:      /mnt/output/opening_strength_fit/cache/*.parquet
run output: /mnt/output/opening_strength_fit/<run_id>/
local pull: output/predictions/<run_id>/predictions_all.parquet
```

`*.tmp.parquet`、`*.parquet.lock` 和 heartbeat 文件都是运行中状态，不能当训练输入。

## 5. 构建和 K8s

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-lgbm-cpu-v1
docker build --build-arg CACHE_BUST=${TAG} -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

python scripts/render_k8s_job.py \
  --config experiments/runs/<run_id>.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

hfcli kubectl --cluster research apply --dry-run=client -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research delete job opening-strength-<run-slug> --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/opening-strength-<run-slug> -n bizewu --timeout=24h
```

默认正式路径是 CPU LightGBM + PVC labeled cache。GPU 只在显式设置
`[model].device_type = "gpu"` 和 `[k8s.resources].gpu_limit` 时使用。

rolling monthly 或长窗口任务如果单 Job 内存压力过大，使用 sharded Job。
monthly sharded manifest 使用 Kubernetes Indexed Job，每个月一个独立 Pod，默认
`parallelism=1`；如需并行且节点内存足够，可在 `[k8s]` 设 `shard_parallelism`。

```bash
python scripts/render_k8s_job.py \
  --config experiments/runs/<run_id>.toml \
  --sharded \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

hfcli kubectl --cluster research delete job opening-strength-<run-slug>-sharded --ignore-not-found -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_sharded_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/<rendered-sharded-job-name> -n bizewu --timeout=24h
```

`alpha_conditioned_rolling_validation` 的 sharded Job 按月写
`month_YYYY-MM/rolling_*.csv` 和 `month_YYYY-MM/predictions.parquet`。训练完成后直接运行
`sync_experiment_artifacts.py --all`；sync 会在 root summary 缺失时自动拉取月度 shard 并本地合并。

## 6. 同步产物

metrics 拉回、predictions 拉回、shard metrics 合并和轻量归档统一使用：

```bash
python scripts/sync_experiment_artifacts.py \
  --config experiments/runs/<run_id>.toml \
  --all

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

默认输出：

```text
output/k8s/metrics/<run_id>_metrics_by_year.csv
output/predictions/<run_id>/predictions_all.parquet
experiments/results/metrics/<run_id>_metrics_by_year.csv
```

`score_risk_sweep` run 没有标准 metrics/predictions；`--all` 会改为拉取轻量 sweep artifact：

```text
output/local/<run_id>/score_risk_summary.csv
output/local/<run_id>/score_risk_minute_summary.csv
output/local/<run_id>/score_risk_group_metrics.csv
output/local/<run_id>/score_risk_trace.json
experiments/results/backtests/<run_id>_summary.csv
```

`alpha_conditioned_rolling_validation` run 也属于非标准 artifact sync；`--all` 会优先拉取 root-level
`rolling_summary.csv` / `rolling_month_summary.csv` / `rolling_group_metrics.csv`，如果 root summary 缺失，
则按 `month_YYYY-MM/` shards 拉取并在本地合并。score-risk 和 rolling artifact 拉取共用同一套
artifact fetch trace，结果写到 `output/local/<run_id>/artifact_fetch_trace.json`；summary / month summary
会归档到 `experiments/results/backtests/<run_id>_*.csv`。`clickhouse_next_close_labels.parquet` 属于
可重建 label cache，sync 不再把它作为 artifact 拉回本地。

`gap_risk_attribution` run 只拉取和归档 outcome / exposure / residual-control 的轻量 CSV，不拉 bulky
group-level diagnostics。

## 7. 分析命令

当前报告默认在 universe / S / M / L selection mask 下分别汇总同一组指标。画模型对比图时按 pool 分组或分面：

- baseline 一组 3 个柱子：`S`、`M`、`L`。
- baseline + 改进模型一组 6 个柱子。
- baseline + 改进模型 + rolling 窗口很容易到 18 个柱子；优先用 small multiples、按 pool 分面，
  或固定横轴为 rolling window、颜色表示模型，避免一张图里同时塞满模型、池子和月份。

Metrics：

```bash
python scripts/summarize_opening_results.py \
  --metrics-csv output/k8s/metrics/<run_id>_metrics_by_year.csv

python scripts/compare_opening_results.py
```

标准 LightGBM delay replay：

```bash
python scripts/run_lgbm_delay_replays.py --check-interface-only
python scripts/run_lgbm_delay_replays.py
python scripts/plot_lgbm_delay_decay.py
```

Horizon decay：

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

Target-label cache build:

```bash
python scripts/build_labeled_cache.py \
  --config experiments/runs/<build_labeled_cache_run_id>.toml

python scripts/inspect_labeled_cache.py \
  --input /mnt/output/opening_strength_fit/cache/<cache>.parquet \
  --config experiments/runs/<build_labeled_cache_run_id>.toml \
  --output output/local/<build_labeled_cache_run_id>/cache_manifest.json

python scripts/build_target_label_cache.py \
  --config experiments/runs/<build_target_run_id>.toml
```

Existing-score TopN guard sweep:

```bash
python scripts/run_score_tail_guards.py \
  --input output/predictions/lgbm_delay2_postopen_0931_0940_baseline_v1/predictions_all.parquet \
  --output-dir output/reports/lgbm_delay2_postopen_tail_guards_v1
```

Existing-score risk penalty sweep:

```bash
python scripts/run_score_risk_sweep.py \
  --config experiments/runs/score_risk_sweep_guard_shrunk_v1.toml \
  --output-dir output/local/score_risk_sweep_guard_shrunk_v1
```

Rolling short-vs-next tradeoff chart:

```bash
python scripts/plot_rolling_validation_tradeoff.py \
  --input experiments/results/backtests/rolling_alpha_conditioned_top100_validation_v1_month_summary.csv \
  --output-dir output/reports/rolling_alpha_conditioned_top100_validation_v1
```

Feature dependence audit：

```bash
python scripts/audit_feature_dependence.py \
  --config experiments/runs/lgbm_delay2_feature_dependence_v1.toml \
  --output-dir output/local/lgbm_delay2_feature_dependence_v1
```

## 8. 排查

- `field is immutable`：删除同名 Job 后重新 apply。
- K8s 内找不到新 config：重新 build/push 镜像，并重新 render Job。
- PVC cache 缺失：等待最终 `*.parquet`，不要使用 `.tmp` 或 lock 文件。
- replay 缺少上下文字段：传 `--context-input`，或先运行 interface check。
- completed config 没有 metrics：运行 `sync_experiment_artifacts.py --all`，然后 audit。
