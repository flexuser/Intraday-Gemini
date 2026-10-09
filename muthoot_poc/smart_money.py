"""Free smart-money + daily delivery features (no API keys).

Sources
-------
- NSE bulk / block deal CSVs (official, free)
- jugaad-data daily bars for Delivery % (official-ish NSE scrape, free)
- Causal proxies only — never uses future information

All failures degrade to neutral defaults so the live pipeline never crashes.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "public/data_store")
CACHE_PATH = os.path.join(DATA_DIR, "smart_money_cache.json")
CACHE_MAX_AGE_HOURS = 18          # refresh at most once per trading day

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

BULK_URL = "https://www.nseindia.com/api/historical/bulkrd?index=bulk&from={from_d}&to={to_d}"
BLOCK_URL = "https://www.nseindia.com/api/historical/bulkrd?index=block&from={from_d}&to={to_d}"

# Neutral defaults when a data source is unavailable
NEUTRAL = {
    "oi_change_pct": 0.0,
    "buildup_code": 0,          # -2 short-build, -1 short-cover, 0 neutral, 1 long-build, 2 long-unwinding
    "oi_vs_avg": 1.0,
    "bulk_buy_flag_5d": 0,
    "bulk_sell_flag_5d": 0,
    "bulk_net_value_cr": 0.0,
    "block_buy_flag_5d": 0,
    "insider_like_buy_flag": 0,
    "promoter_delta_q": 0.0,
    "fii_delta_q": 0.0,
    "dii_delta_q": 0.0,
    "delivery_pct": 0.0,        # latest available delivery %
    "delivery_vs_avg": 1.0,     # latest / 5-day average (1.0 = average)
}


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default


def _today_ist() -> datetime.date:
    try:
        import pytz
        return datetime.datetime.now(pytz.timezone("Asia/Kolkata")).date()
    except Exception:
        return datetime.date.today()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _load_cache() -> Dict[str, Any]:
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, CACHE_PATH)
    except Exception as e:
        print(f"[smart_money] cache write warning: {e}")


def _cache_fresh(cache: Dict[str, Any]) -> bool:
    try:
        ts = cache.get("fetched_at")
        if not ts:
            return False
        fetched = datetime.datetime.fromisoformat(ts)
        age_h = (datetime.datetime.now(fetched.tzinfo) - fetched).total_seconds() / 3600.0
        return age_h < CACHE_MAX_AGE_HOURS
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NSE bulk / block deals (free official CSVs via JSON API)
# ---------------------------------------------------------------------------
def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=8)
    except Exception:
        pass
    return s


def _fetch_bulk_block(days: int = 7) -> pd.DataFrame:
    """Return combined bulk+block deals for the last `days` calendar days."""
    end = _today_ist()
    start = end - datetime.timedelta(days=days)
    from_d = start.strftime("%d-%m-%Y")
    to_d = end.strftime("%d-%m-%Y")
    rows = []
    sess = _nse_session()
    for url_tmpl, deal_type in ((BULK_URL, "bulk"), (BLOCK_URL, "block")):
        try:
            r = sess.get(url_tmpl.format(from_d=from_d, to_d=to_d), timeout=12)
            if r.status_code != 200:
                continue
            data = r.json()
            for item in data.get("data") or data.get("full") or []:
                item = dict(item)
                item["_deal_type"] = deal_type
                rows.append(item)
        except Exception as e:
            print(f"[smart_money] {deal_type} deals fetch warning: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _bulk_block_features(df: pd.DataFrame, symbol: str) -> Dict[str, float]:
    out = {
        "bulk_buy_flag_5d": 0,
        "bulk_sell_flag_5d": 0,
        "bulk_net_value_cr": 0.0,
        "block_buy_flag_5d": 0,
        "insider_like_buy_flag": 0,
    }
    if df is None or df.empty:
        return out

    # Normalise column names that NSE sometimes changes
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get("symbol") or cols.get("sm_name") or cols.get("scripsymbol")
    buy_sell_col = cols.get("buy/sell") or cols.get("client_type") or cols.get("bs")
    value_col = cols.get("trade_val") or cols.get("value") or cols.get("trd_val") or cols.get("tradevalue")
    deal_col = "_deal_type"

    if not sym_col:
        return out

    sub = df[df[sym_col].astype(str).str.upper() == symbol.upper()].copy()
    if sub.empty:
        return out

    def _is_buy(row) -> bool:
        v = str(row.get(buy_sell_col, "")).upper()
        return "BUY" in v or v in ("B", "BUY")

    def _is_sell(row) -> bool:
        v = str(row.get(buy_sell_col, "")).upper()
        return "SELL" in v or v in ("S", "SELL")

    bulk = sub[sub[deal_col] == "bulk"] if deal_col in sub.columns else sub
    block = sub[sub[deal_col] == "block"] if deal_col in sub.columns else pd.DataFrame()

    buy_val = 0.0
    sell_val = 0.0
    for _, row in bulk.iterrows():
        val = _safe_float(row.get(value_col), 0.0)
        # NSE value is often already in Rs; convert crudely to crore
        if val > 1e7:          # looks like raw rupees
            val = val / 1e7
        elif val > 100:        # already in lakh / something else — leave as-is scale
            val = val / 100.0
        if _is_buy(row):
            buy_val += val
            out["bulk_buy_flag_5d"] = 1
        elif _is_sell(row):
            sell_val += val
            out["bulk_sell_flag_5d"] = 1

    out["bulk_net_value_cr"] = round(buy_val - sell_val, 2)

    for _, row in block.iterrows():
        if _is_buy(row):
            out["block_buy_flag_5d"] = 1
            # Heuristic: large block buy can be insider-like
            val = _safe_float(row.get(value_col), 0.0)
            if val > 5:        # rough threshold
                out["insider_like_buy_flag"] = 1
            break

    return out


# ---------------------------------------------------------------------------
# jugaad-data — daily bars + Delivery %
# ---------------------------------------------------------------------------
def _fetch_delivery_features(symbol: str, lookback_days: int = 12) -> Dict[str, float]:
    """Return delivery_pct and delivery_vs_avg using jugaad-data (free)."""
    out = {"delivery_pct": 0.0, "delivery_vs_avg": 1.0}
    try:
        from jugaad_data.nse import stock_df
        from datetime import date, timedelta
        end = _today_ist()
        start = end - timedelta(days=lookback_days + 5)
        df = stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ")
        if df is None or df.empty:
            return out

        # Keep only EQ series if multiple series present
        if "SERIES" in df.columns:
            df = df[df["SERIES"].astype(str).str.upper() == "EQ"]
        if df.empty:
            return out

        # Sort by date ascending
        date_col = "DATE" if "DATE" in df.columns else df.columns[0]
        df = df.sort_values(date_col)

        del_col = None
        for candidate in ("DELIVERY %", "DELIVERY%", "DELIVERY_PCT", "DELIVERY PCT"):
            if candidate in df.columns:
                del_col = candidate
                break
        if del_col is None:
            return out

        series = pd.to_numeric(df[del_col], errors="coerce").dropna()
        if series.empty:
            return out

        latest = float(series.iloc[-1])
        avg = float(series.tail(5).mean()) if len(series) >= 2 else latest
        out["delivery_pct"] = round(latest, 2)
        out["delivery_vs_avg"] = round(latest / avg, 3) if avg > 0 else 1.0
    except Exception as e:
        print(f"[smart_money] delivery fetch ({symbol}): {e}")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def features_for_symbol(bundle: Dict[str, Any], symbol: str) -> Dict[str, float]:
    """Extract the feature dict for one symbol from the bundle returned by fetch_smart_money."""
    if not bundle:
        return dict(NEUTRAL)
    return dict(bundle.get(symbol) or NEUTRAL)


def fetch_smart_money(symbols: List[str], force: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Fetch (or load from cache) smart-money + delivery features for every symbol.

    Returns
    -------
    dict[symbol] -> feature dict (always complete, never None values)
    """
    cache = _load_cache()
    if not force and _cache_fresh(cache) and cache.get("symbols"):
        print(f"[smart_money] using cache from {cache.get('fetched_at')}")
        return cache["symbols"]

    print("[smart_money] refreshing features (free sources only)…")
    result: Dict[str, Dict[str, float]] = {s: dict(NEUTRAL) for s in symbols}

    # --- bulk / block deals (one call for all symbols) ---
    try:
        deals = _fetch_bulk_block(days=7)
        for sym in symbols:
            result[sym].update(_bulk_block_features(deals, sym))
    except Exception as e:
        print(f"[smart_money] bulk/block disabled: {e}")

    # --- delivery % via jugaad-data (per symbol, cached for the day) ---
    for sym in symbols:
        try:
            result[sym].update(_fetch_delivery_features(sym))
        except Exception as e:
            print(f"[smart_money] delivery ({sym}): {e}")

    # OI / buildup / shareholding remain neutral until a free reliable source appears.
    # They are kept in the schema so the model can learn them later without a version bump.

    payload = {
        "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "symbols": result,
    }
    _save_cache(payload)
    return result
