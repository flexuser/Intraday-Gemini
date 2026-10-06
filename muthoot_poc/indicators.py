"""muthoot_poc/indicators.py

Deterministic intraday technical indicator computation pipeline.
Calculates standard quantitative metrics (VWAP, RSI, MACD, ATR, Bollinger Bands)
using pure pandas and numpy to prevent LLM hallucinations and enrich model context.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Computes Volume Weighted Average Price (VWAP) reset daily."""
    if df.empty or not all(col in df.columns for col in ["High", "Low", "Close", "Volume"]):
        return pd.Series(index=df.index, dtype=float)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol = typical_price * df["Volume"]

    # Group by date to reset VWAP calculation daily
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = df["Volume"].groupby(dates).cumsum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = df["Volume"].cumsum()

    return cum_tp_vol / cum_vol.replace(0, np.nan)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes Relative Strength Index (RSI) using Exponential Moving Average."""
    if series.empty or len(series) < period:
        return pd.Series(index=series.index, dtype=float)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, pd.Series]:
    """Computes Moving Average Convergence Divergence (MACD), Signal line, and Histogram."""
    if series.empty or len(series) < slow:
        empty_s = pd.Series(index=series.index, dtype=float)
        return {"macd": empty_s, "signal": empty_s, "hist": empty_s}

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {"macd": macd_line, "signal": signal_line, "hist": histogram}


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Computes Average True Range (ATR) for volatility and stop-loss placement."""
    if df.empty or len(df) < period or not all(col in df.columns for col in ["High", "Low", "Close"]):
        return pd.Series(index=df.index, dtype=float)

    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Dict[str, pd.Series]:
    """Computes Bollinger Bands (Middle, Upper, Lower, and %B bandwidth)."""
    if series.empty or len(series) < period:
        empty_s = pd.Series(index=series.index, dtype=float)
        return {"middle": empty_s, "upper": empty_s, "lower": empty_s, "percent_b": empty_s}

    sma = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()

    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    bandwidth = upper_band - lower_band
    percent_b = (series - lower_band) / bandwidth.replace(0, np.nan)

    return {
        "middle": sma,
        "upper": upper_band,
        "lower": lower_band,
        "percent_b": percent_b,
    }


def calculate_technical_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    """Engineers full technical indicator summary dictionary for current bar.

    Args:
        df: Pandas DataFrame containing OHLCV history with columns
            ['Open', 'High', 'Low', 'Close', 'Volume'].

    Returns:
        Dict containing latest indicator values and signal interpretations.
    """
    if df is None or df.empty or len(df) < 5:
        return {
            "vwap": None,
            "rsi_14": 50.0,
            "macd_hist": 0.0,
            "atr_14": 0.0,
            "atr_pct": 0.0,
            "bollinger_pct_b": 0.5,
            "vwap_distance_pct": 0.0,
            "technical_bias": "NEUTRAL",
        }

    close_series = df["Close"]
    latest_close = float(close_series.iloc[-1])

    vwap_s = compute_vwap(df)
    rsi_s = compute_rsi(close_series)
    macd_dict = compute_macd(close_series)
    atr_s = compute_atr(df)
    bb_dict = compute_bollinger_bands(close_series)

    vwap_val = float(vwap_s.iloc[-1]) if not pd.isna(vwap_s.iloc[-1]) else latest_close
    rsi_val = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0
    macd_hist_val = float(macd_dict["hist"].iloc[-1]) if not pd.isna(macd_dict["hist"].iloc[-1]) else 0.0
    atr_val = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
    pct_b_val = float(bb_dict["percent_b"].iloc[-1]) if not pd.isna(bb_dict["percent_b"].iloc[-1]) else 0.5

    vwap_dist = ((latest_close - vwap_val) / vwap_val * 100) if vwap_val > 0 else 0.0
    atr_pct = (atr_val / latest_close * 100) if latest_close > 0 else 0.0

    # Determine aggregated technical bias
    bullish_signals = sum([
        latest_close > vwap_val,
        rsi_val > 52.0,
        macd_hist_val > 0,
        pct_b_val > 0.5,
    ])

    if bullish_signals >= 3:
        tech_bias = "BULLISH"
    elif bullish_signals <= 1:
        tech_bias = "BEARISH"
    else:
        tech_bias = "NEUTRAL"

    return {
        "vwap": round(vwap_val, 2),
        "rsi_14": round(rsi_val, 2),
        "macd_hist": round(macd_hist_val, 4),
        "atr_14": round(atr_val, 2),
        "atr_pct": round(atr_pct, 2),
        "bollinger_pct_b": round(pct_b_val, 3),
        "vwap_distance_pct": round(vwap_dist, 2),
        "technical_bias": tech_bias,
    }