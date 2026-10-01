import datetime
import json
import math
import os
import re
import pytz
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import PchipInterpolator
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

WATCHLIST = [
    {"symbol": "MUTHOOTFIN", "ticker": "MUTHOOTFIN.NS"},
    {"symbol": "RELIANCE", "ticker": "RELIANCE.NS"},
    {"symbol": "TMPV", "ticker": "TMPV.NS"},
    {"symbol": "INFY", "ticker": "INFY.NS"},
    {"symbol": "HDFCBANK", "ticker": "HDFCBANK.NS"},
    {"symbol": "ICICIBANK", "ticker": "ICICIBANK.NS"},
    {"symbol": "TCS", "ticker": "TCS.NS"},
    {"symbol": "SBIN", "ticker": "SBIN.NS"},
    {"symbol": "BHARTIARTL", "ticker": "BHARTIARTL.NS"},
    {"symbol": "LT", "ticker": "LT.NS"},
    {"symbol": "SUNPHARMA", "ticker": "SUNPHARMA.NS"}
]

TIMEFRAMES = ["09:15", "09:45", "10:30", "11:15", "12:00", "12:45", "13:30", "14:15", "15:00", "15:30"]

SLOT_MINUTES = {
    "09:15": 9 * 60 + 15,
    "09:45": 9 * 60 + 45,
    "10:30": 10 * 60 + 30,
    "11:15": 11 * 60 + 15,
    "12:00": 12 * 60 + 0,
    "12:45": 12 * 60 + 45,
    "13:30": 13 * 60 + 30,
    "14:15": 14 * 60 + 15,
    "15:00": 15 * 60 + 0,
    "15:30": 15 * 60 + 30
}

def get_ist_now():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.datetime.now(ist)

def extract_series(df, col_name):
    """Safely extracts a 1D Pandas Series from yfinance DataFrames."""
    if df is None or df.empty or col_name not in df:
        return None
    data = df[col_name]
    if hasattr(data, 'ndim') and data.ndim > 1:
        return data.iloc[:, 0]
    return data

def calculate_technical_indicators(df_5m: pd.DataFrame) -> dict:
    """Calculates VWAP, RSI (14), ATR (14), and Bollinger Bands."""
    default_res = {
        "vwap": None, "rsi": 50.0, "atr": None,
        "bb_upper": None, "bb_lower": None, "vwap_signal": "NEUTRAL"
    }
    if df_5m is None or df_5m.empty or len(df_5m) < 3:
        return default_res

    high = extract_series(df_5m, 'High')
    low = extract_series(df_5m, 'Low')
    close = extract_series(df_5m, 'Close')
    volume = extract_series(df_5m, 'Volume')

    if close is None or high is None or low is None or volume is None:
        return default_res

    try:
        typical_price = (high + low + close) / 3.0
        valid_vol = volume.replace(0, np.nan).fillna(1.0)
        cum_tp_vol = (typical_price * valid_vol).cumsum()
        cum_vol = valid_vol.cumsum()
        vwap_series = cum_tp_vol / cum_vol
        latest_vwap = round(float(vwap_series.iloc[-1]), 2)
        latest_close = float(close.iloc[-1])

        vwap_signal = "ABOVE_VWAP" if latest_close >= latest_vwap else "BELOW_VWAP"

        if len(close) >= 14:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-9)
            rsi_series = 100.0 - (100.0 / (1.0 + rs))
            latest_rsi = round(float(rsi_series.iloc[-1]), 2)
        else:
            latest_rsi = 50.0

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        latest_atr = round(float(tr.rolling(window=min(14, len(tr)), min_periods=1).mean().iloc[-1]), 2)

        window = min(20, len(close))
        sma = close.rolling(window=window, min_periods=1).mean()
        std = close.rolling(window=window, min_periods=1).std().fillna(0)
        bb_upper = round(float((sma + 2 * std).iloc[-1]), 2)
        bb_lower = round(float((sma - 2 * std).iloc[-1]), 2)

        return {
            "vwap": latest_vwap,
            "rsi": latest_rsi,
            "atr": latest_atr,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "vwap_signal": vwap_signal
        }
    except Exception as e:
        print(f"Technical calculation notice: {e}")
        return default_res

def fetch_nse_option_chain_signals(symbol: str) -> dict:
    """Fetches real-time PCR and ATM Implied Volatility directly from NSE API."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain"
    }
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=4)
        resp = session.get(url, headers=headers, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            records = data.get('filtered', {}).get('data', [])
            tot_ce_oi = sum(item.get('CE', {}).get('openInterest', 0) for item in records if 'CE' in item)
            tot_pe_oi = sum(item.get('PE', {}).get('openInterest', 0) for item in records if 'PE' in item)
            pcr = round(tot_pe_oi / tot_ce_oi, 2) if tot_ce_oi > 0 else 1.0

            underlying_price = data.get('records', {}).get('underlyingValue', 0)
            atm_iv = 18.5
            if records and underlying_price > 0:
                atm_item = min(records, key=lambda x: abs(x.get('strikePrice', 0) - underlying_price))
                atm_iv = atm_item.get('CE', {}).get('impliedVolatility', 0.0) or atm_item.get('PE', {}).get('impliedVolatility', 0.0) or 18.5

            return {"pcr": pcr, "implied_volatility": round(float(atm_iv), 2)}
    except Exception as e:
        print(f"Option chain info for {symbol}: {e}")

    return {"pcr": 1.0, "implied_volatility": 18.5}

def load_historical_feedback(symbol: str, output_dir: str) -> dict:
    """Reads eod_evaluation.json to calculate recent model performance and feedback scaling."""
    audit_path = os.path.join(output_dir, "eod_evaluation.json")
    if not os.path.exists(audit_path):
        return {"avg_error_pct": 0.0, "past_verdict": "NO_HISTORY", "volatility_scaling": 1.0}

    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            matrix = json.load(f)

        ticker_hist = [row for row in matrix if row.get("symbol") == symbol]
        if ticker_hist:
            last_row = ticker_hist[0]
            err = last_row.get("error_pct", 0.0)
            verdict = last_row.get("verdict", "IN_PROGRESS")
            scaling = 0.75 if verdict == "FAILED" else (0.9 if verdict == "PARTIAL" else 1.0)
            return {
                "avg_error_pct": err,
                "past_verdict": verdict,
                "volatility_scaling": scaling
            }
    except Exception as e:
        print(f"Feedback loop load notice for {symbol}: {e}")

    return {"avg_error_pct": 0.0, "past_verdict": "NO_HISTORY", "volatility_scaling": 1.0}

def build_intraday_pchip_curve(base_price: float, target_pct: float, archetype: str, scaling: float = 1.0) -> list:
    """
    Constructs a continuous intraday price curve using Monotone Cubic Hermite Spline (PCHIP).
    Keyframe time pivots:
      t = 0.00  (09:15 Open)
      t = 0.22  (10:30 Morning Peak/Trough)
      t = 0.67  (13:30 Midday VWAP Anchor)
      t = 1.00  (15:30 EOD Close Target)
    """
    # Scale target percent based on historical feedback scaling
    effective_target_pct = max(min(target_pct * scaling, 3.5), -3.5) / 100.0
    end_price = base_price * (1.0 + effective_target_pct)

    # Time values normalized between 0.0 and 1.0 for 10 slots
    t_timeframes = np.linspace(0.0, 1.0, len(TIMEFRAMES))

    # Define keyframe pivot times and prices based on archetypes
    t_pivots = [0.0, 0.22, 0.67, 1.0]

    if archetype == "MORNING_SPIKE_FADE":
        p1 = base_price * (1.0 + max(effective_target_pct * 1.5, 0.008))
        p2 = base_price * (1.0 + effective_target_pct * 0.4)
        price_pivots = [base_price, p1, p2, end_price]

    elif archetype == "DIP_AND_RECOVERY":
        p1 = base_price * (1.0 + min(effective_target_pct * 1.5, -0.008))
        p2 = base_price * (1.0 + effective_target_pct * 0.5)
        price_pivots = [base_price, p1, p2, end_price]

    elif archetype == "RANGE_BOUND":
        p1 = base_price * 1.002
        p2 = base_price * 0.998
        price_pivots = [base_price, p1, p2, end_price]

    else:  # STEADY_DRIFT (Default)
        p1 = base_price + (end_price - base_price) * 0.35
        p2 = base_price + (end_price - base_price) * 0.70
        price_pivots = [base_price, p1, p2, end_price]

    # Perform Piecewise Cubic Hermite Interpolating Polynomial fit
    pchip = PchipInterpolator(t_pivots, price_pivots)
    curve_prices = pchip(t_timeframes)

    return [round(float(p), 2) for p in curve_prices]

def fetch_market_signals(ticker_symbol: str):
    news_items = []
    volume_ratio = 1.0
    has_volume_spike = False
    recent_return = 0.0
    prev_close = None
    df_intraday = None

    try:
        t = yf.Ticker(ticker_symbol)
        raw_news = t.news or []
        for n in raw_news[:3]:
            title = n.get('title') or n.get('content', {}).get('title', '')
            publisher = n.get('publisher') or n.get('content', {}).get('provider', {}).get('displayName', 'Market News')
            link = n.get('link') or n.get('content', {}).get('canonicalUrl', '#')
            if title:
                news_items.append({"title": title, "publisher": publisher, "link": link})

        hist = t.history(period="5d", interval="1d")
        df_intraday = t.history(period="5d", interval="5m")

        if not hist.empty:
            closes = extract_series(hist, 'Close')
            if closes is not None and len(closes) >= 2:
                prev_close = float(closes.iloc[-2])
            elif closes is not None and not closes.empty:
                prev_close = float(closes.iloc[-1])

        if not hist.empty and df_intraday is not None and not df_intraday.empty:
            vols = extract_series(hist, 'Volume')
            intra_vols = extract_series(df_intraday, 'Volume')
            intra_closes = extract_series(df_intraday, 'Close')

            if vols is not None and not vols.empty and intra_vols is not None and not intra_vols.empty:
                avg_daily_vol = float(vols.mean() / 75)
                latest_vol = float(intra_vols.iloc[-1])
                if avg_daily_vol > 0:
                    volume_ratio = round(latest_vol / avg_daily_vol, 2)
                    has_volume_spike = bool(volume_ratio >= 1.5)

            if intra_closes is not None and len(intra_closes) > 1:
                c0 = float(intra_closes.iloc[0])
                cN = float(intra_closes.iloc[-1])
                if c0 > 0:
                    recent_return = (cN - c0) / c0

    except Exception as e:
        print(f"Signal fetch warning for {ticker_symbol}: {e}")

    if not news_items:
        news_items = [{"title": f"No major structural announcements recorded for {ticker_symbol}.", "publisher": "NSE Surveillance", "link": "#"}]

    return news_items, volume_ratio, has_volume_spike, recent_return, prev_close, df_intraday

def query_gemini_regime(symbol: str, news_items: list, volume_ratio: float, has_spike: bool,
                         base_price: float, recent_return: float, techs: dict, options: dict, feedback: dict) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    scaling = feedback.get("volatility_scaling", 1.0)

    # Defaults if Gemini API is unavailable
    default_bias = "BULLISH" if recent_return > 0.0015 else ("BEARISH" if recent_return < -0.0015 else "NEUTRAL")
    default_target_pct = round(recent_return * 100 * 1.2, 2) if abs(recent_return) > 0.001 else (0.6 if default_bias == "BULLISH" else -0.6)
    default_archetype = "STEADY_DRIFT"

    if not GEMINI_AVAILABLE or not api_key:
        return {
            "bias": default_bias,
            "target_pct": default_target_pct,
            "archetype": default_archetype,
            "reasoning": f"Quantitative fallback applied based on technical drift (VWAP: {techs.get('vwap')}, RSI: {techs.get('rsi')})."
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.8-flash')
        news_summary = "\n".join([f"- {n['title']} ({n['publisher']})" for n in news_items])

        prompt = f"""
        Act as a Quantitative Analyst for NSE Stock: {symbol}
        Current Price: ₹{base_price} | Intraday Shift: {recent_return*100:.2f}%

        === TECHNICAL & QUANT SIGNALS ===
        - VWAP Benchmark: ₹{techs.get('vwap') or 'N/A'} (Signal: {techs.get('vwap_signal')})
        - RSI (14-Period): {techs.get('rsi')}
        - ATR (14-Period): {techs.get('atr')} | Bollinger Upper: ₹{techs.get('bb_upper')} | Lower: ₹{techs.get('bb_lower')}
        - Put-Call Ratio (PCR): {options.get('pcr')} | Implied Volatility: {options.get('implied_volatility')}%
        - Volume Ratio: {volume_ratio}x (Spike Active: {has_spike})

        === SELF-LEARNING AUDIT HISTORY ===
        - Past Session Error: {feedback.get('avg_error_pct')}%
        - Last Session Verdict: {feedback.get('past_verdict')}

        === NEWS FEED ===
        {news_summary}

        TASK:
        Evaluate macro direction and determine:
        1. "bias": "BULLISH", "BEARISH", or "NEUTRAL"
        2. "target_pct": Expected percentage price change by 3:30 PM Close (Float between -3.0 and +3.0). Grounded by ATR & Bollinger Bands.
        3. "archetype": Choose exactly ONE structural profile:
           - "STEADY_DRIFT": Consistent trend momentum toward target close.
           - "MORNING_SPIKE_FADE": Early aggressive move followed by mean-reversion/pullback.
           - "DIP_AND_RECOVERY": Initial morning drop finding support and recovering toward EOD.
           - "RANGE_BOUND": Tight oscillation around VWAP within ATR limits.
        4. "reasoning": Concise 2-sentence rationale synthesizing VWAP, PCR, and RSI.

        Return strictly JSON:
        {{
          "bias": "BULLISH",
          "target_pct": 1.25,
          "archetype": "STEADY_DRIFT",
          "reasoning": "Price remains above VWAP with a bullish PCR wall at 1.25, supporting a steady afternoon trend expansion."
        }}
        """

        res = model.generate_content(prompt)
        match = re.search(r'\{.*\}', res.text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "bias": str(data.get("bias", default_bias)),
                "target_pct": float(data.get("target_pct", default_target_pct)),
                "archetype": str(data.get("archetype", default_archetype)),
                "reasoning": str(data.get("reasoning", "Analysis grounded in technical indicators."))
            }
    except Exception as e:
        print(f"Gemini API parse notice for {symbol}: {e}")

    return {
        "bias": default_bias,
        "target_pct": default_target_pct,
        "archetype": default_archetype,
        "reasoning": f"Quant fallback trajectory applied (Feedback scaling: {scaling})."
    }

def fetch_actual_intraday_prices_and_volumes(ticker: str):
    actuals = [None] * len(TIMEFRAMES)
    volumes = [None] * len(TIMEFRAMES)
    ist_now = get_ist_now()
    current_minutes = ist_now.hour * 60 + ist_now.minute

    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)

        if not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')

            latest_date = df.index.date[-1]
            df_latest = df[df.index.date == latest_date]

            close_series = extract_series(df_latest, 'Close')
            vol_series = extract_series(df_latest, 'Volume')

            if close_series is not None and not close_series.empty:
                for idx, slot in enumerate(TIMEFRAMES):
                    slot_min = SLOT_MINUTES[slot]
                    if latest_date < ist_now.date() or current_minutes >= slot_min:
                        slot_time = datetime.time(slot_min // 60, slot_min % 60)
                        filtered_close = close_series[close_series.index.time <= slot_time]
                        filtered_vol = vol_series[vol_series.index.time <= slot_time] if vol_series is not None else None

                        if not filtered_close.empty:
                            val = float(filtered_close.iloc[-1])
                            if not math.isnan(val):
                                actuals[idx] = round(val, 2)

                        if filtered_vol is not None and not filtered_vol.empty:
                            v_val = float(filtered_vol.iloc[-1])
                            if not math.isnan(v_val):
                                volumes[idx] = int(v_val)
    except Exception as e:
        print(f"yfinance download error for {ticker}: {e}")

    return actuals, volumes

def process_single_ticker(item: dict, output_dir: str, today_str: str, now_str: str) -> dict:
    symbol = item["symbol"]
    ticker = item["ticker"]
    file_path = os.path.join(output_dir, f"{symbol}_prediction.json")

    actual_curve, actual_volume = fetch_actual_intraday_prices_and_volumes(ticker)
    news_items, volume_ratio, has_spike, recent_return, prev_close, df_intraday = fetch_market_signals(ticker)

    techs = calculate_technical_indicators(df_intraday)
    options = fetch_nse_option_chain_signals(symbol)
    feedback = load_historical_feedback(symbol, output_dir)

    valid_actuals = [p for p in actual_curve if p is not None]
    base_price = valid_actuals[0] if valid_actuals else (prev_close if prev_close else 1000.0)

    existing_predicted_curve = None
    existing_bias = None
    existing_reasoning = None

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if old_data.get("curves", {}).get("date") == today_str:
                    existing_predicted_curve = old_data["curves"].get("predicted_curve")
                    existing_bias = old_data["prediction"].get("bias")
                    existing_reasoning = old_data["prediction"].get("reasoning")
        except Exception:
            pass

    if existing_predicted_curve and len(existing_predicted_curve) == 10:
        predicted_curve = existing_predicted_curve
        bias = existing_bias or "NEUTRAL"
        reasoning = existing_reasoning or "Forecast locked for current session."
        regime_data = {"bias": bias, "reasoning": reasoning}
    else:
        regime_data = query_gemini_regime(symbol, news_items, volume_ratio, has_spike, base_price, recent_return, techs, options, feedback)
        scaling = feedback.get("volatility_scaling", 1.0)
        
        # Build mathematical PCHIP continuous curve
        predicted_curve = build_intraday_pchip_curve(
            base_price=base_price,
            target_pct=regime_data.get("target_pct", 0.0),
            archetype=regime_data.get("archetype", "STEADY_DRIFT"),
            scaling=scaling
        )

    final_pred = predicted_curve[-1]
    final_actual = valid_actuals[-1] if valid_actuals else None

    error_pct = 0.0
    if final_actual and base_price:
        error_pct = round(float(abs(final_actual - final_pred) / base_price * 100), 2)
        verdict = "SUCCESS" if error_pct <= 0.8 else ("PARTIAL" if error_pct <= 1.8 else "FAILED")
    else:
        verdict = "IN_PROGRESS"

    audit_verdict = verdict
    audit_error_pct = error_pct
    if verdict == "IN_PROGRESS" and feedback.get("past_verdict") in ["SUCCESS", "PARTIAL", "FAILED"]:
        audit_verdict = feedback.get("past_verdict")
        audit_error_pct = feedback.get("avg_error_pct", 0.0)

    payload = {
        "prediction": {
            "bias": str(regime_data.get("bias", "NEUTRAL")),
            "reasoning": str(regime_data.get("reasoning", "Analysis active.")),
            "volume_ratio": float(volume_ratio),
            "has_volume_spike": bool(has_spike),
            "news_feed": news_items,
            "quant_signals": {
                "vwap": techs.get("vwap"),
                "rsi": techs.get("rsi"),
                "atr": techs.get("atr"),
                "bb_upper": techs.get("bb_upper"),
                "bb_lower": techs.get("bb_lower"),
                "pcr": options.get("pcr"),
                "implied_volatility": options.get("implied_volatility"),
                "past_verdict": feedback.get("past_verdict")
            }
        },
        "curves": {
            "symbol": symbol,
            "date": today_str,
            "generated_at": now_str,
            "timeframes": TIMEFRAMES,
            "bias": str(regime_data.get("bias", "NEUTRAL")),
            "predicted_curve": predicted_curve,
            "actual_curve": actual_curve,
            "actual_volume": actual_volume,
            "final_pred": float(final_pred),
            "final_actual": float(final_actual) if final_actual else None,
            "error_pct": float(error_pct)
        }
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    eod_row = {
        "symbol": symbol,
        "bias": str(regime_data.get("bias", "NEUTRAL")),
        "final_pred": float(final_pred),
        "final_actual": float(final_actual) if final_actual else "--",
        "error_pct": float(audit_error_pct),
        "verdict": audit_verdict
    }

    print(f" Synced {symbol:12} | Bias: {regime_data.get('bias'):7} | Archetype: {regime_data.get('archetype', 'STEADY_DRIFT'):20} | PCR: {options.get('pcr')}")
    return eod_row

def run_archive():
    output_dir = os.path.join(os.getcwd(), "public", "data_store")
    os.makedirs(output_dir, exist_ok=True)

    ist_now = get_ist_now()
    today_str = ist_now.strftime("%Y-%m-%d")
    now_str = ist_now.strftime("%I:%M %p IST")

    print(f"=== Quantitative Intraday AI Pipeline Execution [{today_str} {now_str}] ===")

    eod_matrix = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_single_ticker, item, output_dir, today_str, now_str)
            for item in WATCHLIST
        ]
        for future in as_completed(futures):
            try:
                row = future.result()
                if row:
                    eod_matrix.append(row)
            except Exception as e:
                print(f"Error processing item: {e}")

    audit_file = os.path.join(output_dir, "eod_evaluation.json")
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(eod_matrix, f, indent=2)

    print("=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    run_archive()
