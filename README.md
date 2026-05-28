# opening_strength_fit

`opening_strength_fit` 研究 A 股开盘阶段的 short-horizon alpha。项目从
ClickHouse `stock.tick` 或本地 tick parquet 构造
`date x symbol x opening decision time` 样本，只使用 decision point 当时及以前可见的信息，
预测一个约 60 秒的 buy-at-ask / sell-VWAP return proxy。

当前结论：opening proxy signal 真实存在，但 `09:31-09:40` raw short-alpha 里混有明显
“短正长负”的 dirty tail。下一阶段主线是两层建模：baseline / raw-label 模型继续把短周期信号做强，
另建 learned risk layer 或 reranker，在只使用当时可见信息的前提下扣掉拥挤追涨、低深度和失衡 tail。

```text
ClickHouse / local ticks
-> 标准 schema + A-share universe
-> opening features + entry-delay label
-> Ridge / GBM / LightGBM training
-> IC、score bucket、TopN metrics
-> constrained replay + horizon decay diagnostics
-> lightweight evidence in experiments/results/
```

## 文档分工

| 文件 | 职责 |
| --- | --- |
| [docs/project_brief.md](docs/project_brief.md) | 研究目标、当前结论、四宫格解释和下一步路线。 |
| [docs/runbook.md](docs/runbook.md) | 本地 smoke、K8s Job、artifact sync、replay 和归档命令。 |
| [docs/experiment_log.md](docs/experiment_log.md) | 已完成和进行中实验的事实来源。 |
| [docs/project_map.md](docs/project_map.md) | 文件、模块和脚本索引。 |
| [experiments/results/README.md](experiments/results/README.md) | 可提交轻量结果目录约定。 |

## 核心口径

- 采样窗口：`09:30:00` 到 `09:40:00` 的整分钟 decision points。
- 默认数据窗口：`09:15:00` 到 `09:45:00`。
- 股票池：A 股 `00/30.SZ` 和 `60/68.SH`，除非 config 显式指定 symbols file。
- 当前主评估：short-horizon Rank IC 和 Top100 excess；next close 只做 sanity check。
- 当前主线：增强开盘后信号，不把容量、fee/slippage、多档 sweep 和日频 overlay 提前作为优化目标。

Label 定义：

```text
decision_t = sampled decision tick
entry_t = decision_t + entry_tick_delay ticks
buy_price = ask_price_1[entry_t]
sell_vwap = VWAP(entry_t + 60s, entry_t + 120s)
label = sell_vwap / buy_price - 1 - fee_bps / 10000
```

LightGBM delay 分支默认使用 PVC labeled cache：

```text
/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay{0,1,2}_labeled.parquet
```

大体积 predictions、models、PNGs 和临时报告保留在 ignored `output/`；
可提交证据放在 `experiments/results/`。

## 快速开始

```bash
cd /home/hefu/projects/opening_strength_fit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

set -a
. ./.env
set +a

python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

本地 smoke：

```bash
python scripts/inspect_dataset.py \
  --symbol 000001.SZ 000925.SZ 600519.SH 601318.SH 300750.SZ \
  --date 2021-09-22 2021-09-23 \
  --config experiments/runs/gbm_opening_1y_next_month.toml \
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
```

K8s 训练闭环：

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-lgbm-cpu-v1
docker build --build-arg CACHE_BUST=${TAG} -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

python scripts/render_k8s_job.py \
  --config experiments/runs/<run_id>.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research wait --for=condition=complete job/opening-strength-<run-slug> -n bizewu --timeout=24h

python scripts/sync_experiment_artifacts.py --config experiments/runs/<run_id>.toml --all
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

当前活跃工作和已归档指标见 [docs/experiment_log.md](docs/experiment_log.md)。
