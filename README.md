# opening_strength_fit

`opening_strength_fit` studies short-horizon A-share opening alpha. It builds
`date x symbol x opening decision time` samples from ClickHouse `stock.tick` or local
tick parquet, uses only information visible at the decision point, and predicts a
rough 60s buy-at-ask / sell-VWAP return proxy.

Current project spine:

```text
ClickHouse / local ticks
-> standard schema + A-share universe
-> opening features + entry-delay label
-> Ridge / GBM / LightGBM training
-> IC, score bucket, TopN metrics
-> constrained replay + horizon decay diagnostics
-> lightweight evidence in experiments/results/
```

Current conclusion: the opening proxy signal is real, but it is not a complete
T+1 trading strategy. The next research task is signal strengthening: raise
cross-sectional Rank IC and Top100 excess return with post-open orderbook,
queue-depth, depth-change, and trade-impact features.

## Documents

| file | role |
| --- | --- |
| [docs/project_brief.md](docs/project_brief.md) | Research target, current conclusion, next decision gates. |
| [docs/runbook.md](docs/runbook.md) | Commands for local smoke, K8s jobs, artifact sync, replay, and archive. |
| [docs/experiment_log.md](docs/experiment_log.md) | Fact source for completed and active experiments. |
| [docs/project_map.md](docs/project_map.md) | File/module/script index. |
| [experiments/results/README.md](experiments/results/README.md) | Tracked lightweight result contract. |

## Key Contract

- Sample window: `09:30:00` through `09:40:00` integer-minute decision points.
- Default source window: `09:15:00` through `09:45:00`.
- Universe: A-share `00/30.SZ` and `60/68.SH`, unless a symbols file overrides it.
- Label:

```text
decision_t = sampled decision tick
entry_t = decision_t + entry_tick_delay ticks
buy_price = ask_price_1[entry_t]
sell_vwap = VWAP(entry_t + 60s, entry_t + 120s)
label = sell_vwap / buy_price - 1 - fee_bps / 10000
```

LightGBM delay branches use PVC labeled caches:

```text
/mnt/output/opening_strength_fit/cache/opening_1y_next_month_delay{0,1,2}_labeled.parquet
```

Large predictions, models, PNGs, and scratch reports stay in ignored `output/`.
Tracked evidence stays in `experiments/results/`.

## Quick Start

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

Local smoke:

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

K8s training loop:

```bash
TAG=opening-strength-fit-$(date +%Y%m%d)-lgbm-cpu-v1
docker build --build-arg CACHE_BUST=${TAG} -t registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG} .
docker push registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

python scripts/render_k8s_job.py \
  --config experiments/runs/<run_id>.toml \
  --image registry.corp.highfortfunds.com/bizewu/opening-strength-fit:${TAG}

hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_job.yaml
hfcli kubectl --cluster research logs -f job/opening-strength-<run-slug> -n bizewu
hfcli kubectl --cluster research apply -f experiments/jobs/<run_id>_reader_job.yaml

python scripts/sync_experiment_artifacts.py --config experiments/runs/<run_id>.toml --all
python scripts/audit_experiments.py
python scripts/check_workflow_coverage.py
```

Current active work and archived metrics are summarized in
[docs/experiment_log.md](docs/experiment_log.md).
