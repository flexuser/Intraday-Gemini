"""Single source of truth for the free-tier liquid watchlist and context maps.

Selected for: deep cash + F&O liquidity, tight spreads, volume through the day.
Revisit quarterly; do not expand casually (more names dilute history quality).
"""

# NSE cash symbols (no .NS suffix). archive_once appends .NS for Yahoo.
SYMBOLS = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "INFY",
    "TCS",
    "BHARTIARTL",
    "TATASTEEL",
    "ITC",
    "AXISBANK",
]

DEFAULT_SYMBOL = "RELIANCE"

# Professional free-tier signal thresholds
MIN_PROB_EDGE = 0.08
MIN_P_UP_BULLISH = 0.58
MAX_P_UP_BEARISH = 0.42

# Sector bucket -> Yahoo index (best-effort free). If a fetch fails, quant falls back
# to Nifty-only features and the *existing* trained model still works unchanged.
SYMBOL_SECTOR = {
    "RELIANCE": "ENERGY",
    "HDFCBANK": "BANK",
    "ICICIBANK": "BANK",
    "SBIN": "BANK",
    "AXISBANK": "BANK",
    "INFY": "IT",
    "TCS": "IT",
    "BHARTIARTL": "TELCOM",
    "TATASTEEL": "METAL",
    "ITC": "FMCG",
}

# Yahoo Finance tickers for India sector indices (may be unavailable; handled softly).
SECTOR_YAHOO = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "METAL": "^CNXMETAL",
    "ENERGY": "^CNXENERGY",
    "FMCG": "^CNXFMCG",
    "TELCOM": "^CNXIT",  # no liquid free telecom index; IT is a soft proxy only
}

# Overnight / global proxy known before the Indian cash open (prior US session).
US_YAHOO = "^GSPC"  # S&P 500 cash; prior close is available by 09:15 IST
