"""Causal five-minute observations for training and evaluating intraday models.

This module deliberately records features and outcomes only.  It does not invent a
forecast before a model has earned the right to publish one.

FEATURE_VERSION is bumped when the feature vector changes so old rows are ignored
until fresh observations are collected.
"""
import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


HORIZONS_MINUTES = (15, 30, 60)
FEATURE_VERSION = "intraday-observation-v3"          # bumped for smart-money features
MAX_SESSIONS = 60


def _round(value, digits=4):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _pct_change(series, bars):
    if len(series) <= bars or float(series.iloc[-bars - 1]) == 0:
        return None
    return (float(series.iloc[-1]) / float(series.iloc[-bars - 1]) - 1.0) * 100.0


def capture(
    symbol: str,
    frame: pd.DataFrame,
    technical: Dict[str, Any],
    session: Dict[str, Any],
    smart_money: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Capture one feature row using data through the last completed five-minute bar.

    smart_money is an optional dict of institutional / F&O features (already causal).
    Missing keys are filled with safe neutral defaults so the pipeline never breaks.
    """
    if frame is None or frame.empty or "Close" not in frame or "Volume" not in frame:
        return {}

    last = frame.iloc[-1]
    closes = frame["Close"].astype(float)
    prior_volume = frame["Volume"].iloc[:-1]
    volume_baseline = float(prior_volume.mean()) if len(prior_volume) else 0.0
    close = float(last["Close"])
    open_price = float(frame["Open"].iloc[0]) if "Open" in frame else close
    bar_range = ((float(last["High"]) - float(last["Low"])) / close * 100.0) if close > 0 else None
    # Yahoo's five-minute timestamp marks the bar *start*. Its close is only
    # observable five minutes later, so use the bar end as the decision time.
    bar_time = pd.Timestamp(frame.index[-1]) + datetime.timedelta(minutes=5)

    sm = smart_money or {}

    features = {
        # --- original technical / price features ---
        "return_5m_pct": _round(_pct_change(closes, 1)),
        "return_15m_pct": _round(_pct_change(closes, 3)),
        "return_30m_pct": _round(_pct_change(closes, 6)),
        "move_since_open_pct": _round((close / open_price - 1.0) * 100.0 if open_price > 0 else None),
        "vwap_distance_pct": _round(technical.get("vwap_distance_pct")),
        "rsi_14": _round(technical.get("rsi_14"), 2),
        "macd_hist": _round(technical.get("macd_hist"), 6),
        "atr_pct": _round(technical.get("atr_pct")),
        "last_bar_range_pct": _round(bar_range),
        "last_bar_volume_ratio": _round(float(last["Volume"]) / volume_baseline if volume_baseline > 0 else None),
        "session_volume_ratio": _round(session.get("vol_ratio"), 2),
        "minutes_since_open": int(max(0, (bar_time.hour * 60 + bar_time.minute) - (9 * 60 + 15))),

        # --- smart-money / institutional features (extra) ---
        "oi_change_pct": _round(sm.get("oi_change_pct"), 2) or 0.0,
        "buildup_code": int(sm.get("buildup_code") or 0),
        "oi_vs_avg": _round(sm.get("oi_vs_avg"), 2) or 1.0,
        "bulk_buy_flag_5d": int(sm.get("bulk_buy_flag_5d") or 0),
        "bulk_sell_flag_5d": int(sm.get("bulk_sell_flag_5d") or 0),
        "bulk_net_value_cr": _round(sm.get("bulk_net_value_cr"), 3) or 0.0,
        "block_buy_flag_5d": int(sm.get("block_buy_flag_5d") or 0),
        "insider_like_buy_flag": int(sm.get("insider_like_buy_flag") or 0),
        "promoter_delta_q": _round(sm.get("promoter_delta_q"), 3) or 0.0,
        "fii_delta_q": _round(sm.get("fii_delta_q"), 3) or 0.0,
        "dii_delta_q": _round(sm.get("dii_delta_q"), 3) or 0.0,
    }

    outcomes = {
        str(minutes): {
            "target_at": (bar_time + datetime.timedelta(minutes=minutes)).isoformat(),
            "status": "pending",
        }
        for minutes in HORIZONS_MINUTES
    }
    return {
        "key": f"{symbol}|{bar_time.isoformat()}",
        "symbol": symbol,
        "bar_time": bar_time.isoformat(),
        "entry_price": _round(close, 2),
        "feature_version": FEATURE_VERSION,
        "features": features,
        "outcomes": outcomes,
    }


def update_outcomes(observations: List[Dict[str, Any]], frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Attach realised prices once the relevant horizon is complete."""
    if frame is None or frame.empty or "Close" not in frame:
        return observations or []
    # Match outcomes against completed-bar times for the same reason capture()
    # records a decision at bar end rather than bar start.
    timestamps = pd.DatetimeIndex(frame.index) + datetime.timedelta(minutes=5)
    closes = frame["Close"].astype(float).tolist()
    updated = []
    for observation in observations or []:
        row = dict(observation)
        outcomes = {name: dict(value) for name, value in (row.get("outcomes") or {}).items()}
        entry_price = float(row.get("entry_price") or 0)
        for outcome in outcomes.values():
            if outcome.get("status") == "scored" or entry_price <= 0:
                continue
            target_at = pd.Timestamp(outcome["target_at"])
            matching = timestamps >= target_at
            if not matching.any():
                continue
            index = int(matching.argmax())
            actual_price = float(closes[index])
            outcome.update({
                "status": "scored",
                "actual_at": pd.Timestamp(timestamps[index]).isoformat(),
                "actual_price": _round(actual_price, 2),
                "actual_return_pct": _round((actual_price / entry_price - 1.0) * 100.0),
            })
        row["outcomes"] = outcomes
        updated.append(row)
    return updated


def upsert_observation(observations: List[Dict[str, Any]], observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keep one record per symbol and completed five-minute bar."""
    records = list(observations or [])
    if not observation:
        return records
    key = observation["key"]
    for existing in records:
        if existing.get("key") == key:
            return records
    records.append(observation)
    return records


def merge_history(history: List[Dict[str, Any]], observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add scored rows to the rolling training set and retain recent sessions only."""
    merged = {row.get("key"): row for row in (history or []) if row.get("key")}
    for row in observations or []:
        outcomes = row.get("outcomes") or {}
        if any(value.get("status") == "scored" for value in outcomes.values()):
            merged[row["key"]] = row

    rows = list(merged.values())
    session_dates = sorted({str(pd.Timestamp(row["bar_time"]).date()) for row in rows})[-MAX_SESSIONS:]
    allowed = set(session_dates)
    return sorted(
        [row for row in rows if str(pd.Timestamp(row["bar_time"]).date()) in allowed],
        key=lambda row: row["key"],
    )