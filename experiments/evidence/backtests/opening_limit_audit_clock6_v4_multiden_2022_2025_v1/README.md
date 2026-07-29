# Opening-window point-in-time limit-up audit

This audit checks the canonical fixed-clock `09:31-09:40` multi-denominator opening
model over all `9,690` 2022-2025 `pool_L` decision cross-sections. Each cross-section
contains an average of `3,333.89` eligible names and the published selection is Top100.

The upper-limit reference uses `stock.daily_bar_jy` previous close and ST status:
ST `5%`, ChiNext/STAR `20%`, and other supported A shares `10%`, rounded to one fen.
When the daily table marks a name as up-limit, its exact daily close is used. The
decision snapshot and the clock+6-second entry snapshot are audited separately.

## Main result

The published Top100 is **not** materially selecting stocks that are already sealed or
currently at the upper limit:

| Point-in-time state | Candidate share | Top100 share | Enrichment | Selected rows | Remove and reselect: excess change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sealed at decision, no ask offer | `0.00230%` | `0.000103%` | `0.045x` | `1 / 969,000` | `+0.0007 bps` |
| At upper limit at decision | `0.00544%` | `0.000929%` | `0.171x` | `9 / 969,000` | `+0.0008 bps` |
| Within `50 bps` at decision | `0.02475%` | `0.03282%` | `1.326x` | `318 / 969,000` | `-0.0139 bps` |
| Within `100 bps` at decision | `0.04650%` | `0.13024%` | `2.801x` | `1,262 / 969,000` | `-0.0254 bps` |
| Within `200 bps` at decision | `0.10956%` | `0.55418%` | `5.058x` | `5,370 / 969,000` | `+0.0537 bps` |
| No ask offer at clock+6s entry | `0%` | `0%` | — | `0 / 969,000` | `0 bps` |
| At upper limit with offer at entry | `0.00568%` | `0.00485%` | `0.854x` | `47 / 969,000` | `+0.0063 bps` |

The one Top100 row sealed at the decision snapshot reopened before the clock+6-second
entry snapshot and had a valid ask offer at entry. Thus none of the published Top100
entries lacked an executable ask at the actual entry snapshot.

The model does over-select stocks that had touched the upper limit earlier in the
opening window but had reopened by the decision snapshot:

- candidate share `0.1345%`, Top100 share `0.6827%`, enrichment `5.07x`;
- average `0.683` names per Top100;
- their next-close return is `-68.11 bps`, contributing `-0.465 bps` to Top100;
- removing and reselecting them improves excess by `+0.327 bps`.

This is a real selection tendency, but it is small in absolute count and is a slight
drag rather than the source of the positive result.

## Important separate finding

The model strongly selects stocks that are **not currently near the limit**, but later
finish the same day at the upper limit:

| Attribution | Candidate share | Top100 share | Enrichment | Top100 return contribution |
| --- | ---: | ---: | ---: | ---: |
| Daily table marks same-day up-limit | `0.7737%` | `4.6772%` | `6.05x` | `38.85 bps` |
| Same-day up-limit, but not within `100 bps` at decision | `0.7496%` | `4.5894%` | `6.12x` | `38.61 bps` |

Original Top100 next-close return/excess are `28.42 / 17.17 bps`. Removing the
same-day up-limit names and filling Top100 by score changes them to
`-7.79 / -19.03 bps` relative to the original pool baseline.

Therefore the old opening result is highly dependent on predicting stocks that will
limit up later, but not on selecting stocks already sealed or near-limit at the
09:31-09:40 decision/entry time. This is economically different from a late-day
strategy selecting an already unbuyable limit-up stock.

## Artifact boundary

Every persisted `pool_L` prediction row already has a valid one-minute execution label,
so the `published_top100` and `scoreable_pool` summaries are identical. The artifacts
can establish what the published backtest selected, but cannot show what the trained
model would score on rows omitted before prediction because their future execution
label was invalid. A counterfactual inference pass would be required for that question.

Tracked aggregate artifacts:

- `opening_limit_overall.csv`: overall prevalence, enrichment, contribution, and
  exclusion/reselection results.
- `opening_limit_by_clock.csv`: `09:31` through `09:40`.
- `opening_limit_by_year.csv` and `opening_limit_by_month.csv`: stability.
- `opening_limit_trace.json`: inputs and definitions.

The row-level `opening_limit_selected_events.parquet` and decision-group
`opening_limit_group_metrics.parquet` are generated under the run config's ignored
`output/artifacts/` directory. They remain available for local case inspection but are
not tracked as compact evidence.
