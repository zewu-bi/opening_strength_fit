# Project Brief

## Goal

Build and evaluate an A-share opening-stage cross-sectional signal. The model may
use only data visible at each decision point: call auction state, orderbook,
trade flow, and short-term momentum. The primary research metric is now
cross-sectional Rank IC plus Top100 excess return on the 60s proxy label.

The proxy label is a microstructure discovery label, not a full T+1 trading
return. Trading constraints, daily overlay, and capacity expansion come after the
opening signal itself improves.

## Current Conclusion

Completed evidence supports four claims:

1. The data path is working: ClickHouse ticks, A-share filtering, opening-window
   sampling, feature building, entry-delay labels, K8s training, PVC outputs, and
   lightweight archives are connected.
2. Ridge/GBM baselines and CPU LightGBM delay branches show positive
   cross-sectional sorting value in 2022-01.
3. Ordinary universe LightGBM is stronger than the hand-built strong-candidate
   branch. Longer entry delay weakens IC and replay returns.
4. Horizon decay is steep. Fixed `09:30` scores retain weak close/next-close Rank
   IC, but next-close Top20 return is unstable; averaging `09:30-09:39` largely
   erases longer-horizon value.

Therefore the next mainline is feature work, not execution modeling.

## Baseline

Current conservative baseline is `lgbm_opening_1y_next_month_delay2`, evaluated by
minute. Short proxy label is still useful across the opening window; longer
labels are only sanity checks.

| minute | short Rank IC | short Top100 excess bps | next-close Rank IC | next-close Top100 excess bps |
| --- | ---: | ---: | ---: | ---: |
| 09:30 | 0.196 | +49.0 | 0.039 | +42.7 |
| 09:31 | 0.085 | +16.6 | -0.026 | -28.6 |
| 09:32 | 0.087 | +16.6 | -0.021 | -16.9 |
| 09:33 | 0.127 | +19.1 | -0.026 | -38.8 |
| 09:34 | 0.138 | +18.3 | -0.026 | -36.5 |
| 09:35 | 0.142 | +20.7 | -0.026 | -37.8 |
| 09:36 | 0.143 | +21.4 | -0.034 | -17.5 |
| 09:37 | 0.140 | +18.3 | -0.033 | -24.5 |
| 09:38 | 0.163 | +21.6 | -0.026 | -34.7 |
| 09:39 | 0.148 | +13.8 | -0.035 | -52.9 |
| 09:40 | 0.127 | +16.8 | -0.037 | -42.7 |

## Next Gates

| gate | target |
| --- | --- |
| Post-open orderbook features | Strengthen ask/bid gap, depth slope, queue change, depth concentration, spread change, and trade-impact signals. |
| Feature dependence audit | Confirm `preopen_*` and cumulative volume/turnover are not the dominant source of alpha. |
| Objective alignment | Try label de-mean/z-score or ranking-oriented variants if Top100 and Rank IC diverge. |
| Long-horizon sanity | Recheck close/next-close after signal improvement, without optimizing for it yet. |

Capacity stays at ask1 availability for now. L3/L5 sweep, fee/slippage grids,
same-symbol cooldown, and T+1 overlay are parked until signal strength improves.
