"""Single source of truth for the free-tier liquid watchlist.

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

# Default UI focus
DEFAULT_SYMBOL = "RELIANCE"

# Professional free-tier signal thresholds (shared documentation for operators)
MIN_PROB_EDGE = 0.08          # |P(up)-0.5| required before bias can be non-NEUTRAL
MIN_P_UP_BULLISH = 0.58       # risk engine bullish floor
MAX_P_UP_BEARISH = 0.42       # risk engine bearish ceiling
