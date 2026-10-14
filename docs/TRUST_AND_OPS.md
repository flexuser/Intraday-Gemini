# Trust, scoring, and operations

## What is proven vs experimental

| Component | Status |
|-----------|--------|
| Pipeline (fetch, JSON, UI, 30m cron) | Operational |
| Open→close **quant** (ridge + gates) | Walk-forward tested offline; **live sample still thin** |
| Direction **TRADE** flags | Rare by design (P(up) thresholds + gates) |
| 15/30/60 intraday model | **Unvalidated** — display only |
| LLM text | Commentary; day-ahead opening only if before **10:00 IST** |

## Scoring rules (after this update)

1. **Primary track record** prefers **quant** when a quant forecast exists.
2. **LLM opening** is stored only if the first LLM call is before **10:00 IST** (`day_ahead: true`).
3. Later LLM calls are **checkpoints** only — not the official direction score.
4. `enough_data` in `performance_summary.json` requires **≥ 20 scored session days**.

## Model score P(up)

`p_up` is a **model score** in [0, 1], **not** a calibrated probability of an up day.  
Risk thresholds 0.58 / 0.42 are conservative filters, not “58% chance”.

## Per-symbol gates

Gates on a large training universe can pass by chance. Prefer **global** gates + live Track Record over “this symbol was tradeable in backtest”.

## Retrain

```bash
# Continuity with current live weights shape
python -m muthoot_poc.backtest --feature-set v1 --out public/data_store

# After sector/US live fetches look healthy
python -m muthoot_poc.backtest --feature-set v2 --out public/data_store
```

Commit new `quant_model.json` + `backtest_report.json` only after reading the summary.  
Direction gate may **fail** — that can be the correct scientific outcome.

## Ops

- External cron: `*/30 9-16 * * 1-5` (IST) — ensure runs actually fire from open.
- Supabase is optional; JSON under `public/data_store` is the source of truth for Pages.
- Re-check performance after ~20 trading days; do not loosen TRADE thresholds to force activity.

## Disclaimer

Educational / testing only. Not financial advice. Not a guarantee of profit.
