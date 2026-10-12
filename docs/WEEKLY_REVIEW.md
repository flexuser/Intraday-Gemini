# Weekly review protocol (free-tier professional use)

Run every weekend for 1–2 months before changing thresholds again.

## 1. Plumbing check
- [ ] Did Mon–Fri ~16:00 runs write rows to `history.json`?
- [ ] Each row has `close_source` (`5m_bar` or `daily_bar`)?
- [ ] New symbols present: RELIANCE, HDFCBANK, ICICIBANK, SBIN, INFY, TCS, BHARTIARTL, TATASTEEL, ITC, AXISBANK

## 2. Score two sets separately
| Set | Filter | Purpose |
|-----|--------|---------|
| A. All sessions | every completed row | Calibration / error size |
| B. Trade-enabled only | `trade_approved == true` | **This is the belief set** |

If set B has fewer than ~10 samples in a month, do **not** judge accuracy yet.

## 3. Metrics to log (spreadsheet is fine)
- Direction hit rate (set B)
- Mean \|error_pct\| (set A and B)
- Mean error vs flat baseline
- Count of trade-approved signals
- Top rejection reasons (from `rejection_reason`)

## 4. Decision rules for the month
- **Do not loosen** gates if set B hit rate is weak — tighten or stay.
- **Do not expand** the 10-name universe in month 1–2.
- Only consider threshold changes after ≥ 40 trade-approved samples **or** 8 full weeks.

## 5. What “good enough to believe” looks like
- Stable process (settle + history every day)
- Set B expectancy positive after a simple cost assumption (e.g. 0.1–0.2% round trip)
- No single-week miracle required — consistency matters
