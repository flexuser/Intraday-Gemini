"""Single source of truth for the free-tier liquid watchlist and context maps.

Selected for: deep cash + F&O liquidity, tight spreads, volume through the day.
Revisit quarterly; do not expand casually (more names dilute history quality).
"""

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

MIN_PROB_EDGE = 0.08
MIN_P_UP_BULLISH = 0.58
MAX_P_UP_BEARISH = 0.42
MIN_SCORED_DAYS_FOR_TRUST = 20

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

# TELCOM intentionally omitted — no honest free Yahoo proxy (do not map to IT).
SECTOR_YAHOO = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "METAL": "^CNXMETAL",
    "ENERGY": "^CNXENERGY",
    "FMCG": "^CNXFMCG",
}

US_YAHOO = "^GSPC"
