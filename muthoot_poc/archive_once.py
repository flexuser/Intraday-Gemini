import os
import sys
import json
import time
import datetime
import pytz
import numpy as np
import yfinance as yf
from scipy.interpolate import PchipInterpolator
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

IST = pytz.timezone("Asia/Kolkata")

SYMBOLS = ["MUTHOOTFIN", "RELIANCE", "TMPV", "INFY", "HDFCBANK", "ICICIBANK", "TCS", "SBIN", "BHARTIARTL", "LT", "SUNPHARMA"]

def is_live_session(symbol="RELIANCE.NS") -> bool:
    """
    Probes anchor ticker to check if today is an active trading day 
    and current time is within market hours (09:15 - 15:50 IST).
    """
    now_ist = datetime.datetime.now(IST)
    
    # Check Weekend (Mon = 0, Sun = 6)
    if now_ist.weekday() >= 5:
        print(f"[SessionGuard] Today is {now_ist.strftime('%A')} (Weekend). Market closed.")
        return False
        
    market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now_ist.replace(hour=15, minute=50, second=0, microsecond=0)
    
    if not (market_start <= now_ist <= market_end):
        print(f"[SessionGuard] Current IST time ({now_ist.strftime('%H:%M')}) is outside market hours (09:15 - 15:50).")
        return False

    try:
        df = yf.download(symbol, period="1d", interval="5m", progress=False)
        if df is None or df.empty:
            print(f"[SessionGuard] Could not fetch probe ticker {symbol}.")
            return False
            
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index
        last_bar_date = idx.tz_convert("Asia/Kolkata")[-1].date()
        
        if last_bar_date != now_ist.date():
            print(f"[SessionGuard] Holiday or stale data detected. Last bar date: {last_bar_date}, Today: {now_ist.date()}")
            return False
            
    except Exception as e:
        print(f"[SessionGuard] Error probing anchor symbol: {e}")
        return False

    return True

def compute_session_vwap_and_spikes(df_5m):
    """
    Computes session-grouped VWAP and checks for volume spikes on the last COMPLETED bar.
    """
    if df_5m.empty or len(df_5m) < 2:
        return df_5m, 1.0, False

    df = df_5m.copy()
    
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    df['vol'] = df['Volume'].replace(0, np.nan).fillna(1.0)
    
    df['date_group'] = df.index.date
    df['pv'] = tp * df['vol']
    df['cum_pv'] = df.groupby('date_group')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date_group')['vol'].cumsum()
    df['vwap'] = (df['cum_pv'] / df['cum_vol']).round(2)

    last_completed_vol = float(df['Volume'].iloc[-2]) if len(df) >= 2 else float(df['Volume'].iloc[-1])
    avg_session_vol = float(df['Volume'].mean()) if len(df) > 0 else 1.0
    
    vol_ratio = round(last_completed_vol / max(avg_session_vol, 1.0), 2)
    has_spike = vol_ratio >= 2.0

    return df, vol_ratio, has_spike

def query_gemini_regime(symbol, df_5m, news_items, vol_ratio, retries=3):
    """
    Queries Gemini using Structured Outputs with 3 retries & exponential backoff.
    Falls back strictly to NEUTRAL / 0.0% / RANGE_BOUND on failure.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = "gemini-2.5-flash"
    
    if not api_key:
        return {
            "bias": "NEUTRAL",
            "target_pct": 0.0,
            "archetype": "RANGE_BOUND",
            "reasoning": "GEMINI_API_KEY environment variable missing.",
            "source": "fallback",
            "model": "none",
            "llm_error": "Missing GEMINI_API_KEY"
        }

    client = genai.Client(api_key=api_key)
    
    last_close = float(df_5m['Close'].iloc[-1])
    open_price = float(df_5m['Open'].iloc[0])
    vwap_val = float(df_5m['vwap'].iloc[-1]) if 'vwap' in df_5m.columns else last_close
    intraday_shift_pct = round(((last_close - open_price) / open_price) * 100, 2)
    
    news_text = "\n".join([f"- {item['title']} ({item.get('publisher', 'News')})" for item in news_items[:3]]) or "No major corporate catalysts reported."

    prompt = f"""
Perform intraday quantitative analysis for stock symbol: {symbol} (NSE India)

Market Technicals:
- Open Price: ₹{open_price:.2f}
- Current Price: ₹{last_close:.2f}
- Session VWAP: ₹{vwap_val:.2f}
- Intraday Shift (Open to Current): {intraday_shift_pct}%
- Volume Spike Ratio (Last completed bar): {vol_ratio}x

Recent Corporate Headlines:
{news_text}

Task: Determine the expected price trajectory for the remainder of the session.
"""

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "bias": {"type": "STRING", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
            "target_pct": {"type": "NUMBER"},
            "archetype": {"type": "STRING", "enum": ["MOMENTUM_BREAKOUT", "MEAN_REVERSION", "RANGE_BOUND", "BREAKDOWN"]},
            "reasoning": {"type": "STRING"}
        },
        "required": ["bias", "target_pct", "archetype", "reasoning"]
    }

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.2
    )

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            data = json.loads(res.text.strip())
            
            target_pct = max(min(float(data.get("target_pct", 0.0)), 3.5), -3.5)
            
            return {
                "bias": data.get("bias", "NEUTRAL"),
                "target_pct": target_pct,
                "archetype": data.get("archetype", "RANGE_BOUND"),
                "reasoning": data.get("reasoning", "Analysis completed."),
                "source": "llm",
                "model": model_name,
                "llm_error": None
            }
        except Exception as e:
            last_error = str(e)
            print(f"[Gemini Retry {attempt}/{retries}] Failed for {symbol}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)

    return {
        "bias": "NEUTRAL",
        "target_pct": 0.0,
        "archetype": "RANGE_BOUND",
        "reasoning": "LLM query timeout/error. Defaulted to neutral bounds.",
        "source": "fallback",
        "model": model_name,
        "llm_error": last_error
    }

def generate_pchip_curves(df_5m, target_pct):
    """
    Generates time-aligned 5-minute timeframes (09:15 to 15:30) and calculates
    PCHIP trajectory curves for predicted vs actual market prices.
    """
    base_date = datetime.datetime.now(IST).date()
    start_time = datetime.datetime.combine(base_date, datetime.time(9, 15), tzinfo=IST)
    all_slots = [start_time + datetime.timedelta(minutes=5 * i) for i in range(75)]
    timeframes = [t.strftime("%H:%M") for t in all_slots]

    df_actual = df_5m.copy()
    if df_actual.index.tz is None:
        df_actual.index = df_actual.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df_actual.index = df_actual.index.tz_convert("Asia/Kolkata")

    actual_map = {t.strftime("%H:%M"): round(float(p), 2) for t, p in zip(df_actual.index, df_actual['Close'])}
    vol_map = {t.strftime("%H:%M"): int(v) for t, v in zip(df_actual.index, df_actual['Volume'])}

    actual_curve = [actual_map.get(tf, None) for tf in timeframes]
    actual_volume = [vol_map.get(tf, 0) for tf in timeframes]

    valid_actuals = [p for p in actual_curve if p is not None]
    if not valid_actuals:
        return timeframes, [], [], [], 0.0, 0.0

    start_price = valid_actuals[0]
    current_price = valid_actuals[-1]
    target_price = round(start_price * (1 + target_pct / 100.0), 2)

    current_idx = len(valid_actuals) - 1
    x_points = [0, current_idx, 74]
    y_points = [start_price, current_price, target_price]

    if current_idx == 0:
        x_points = [0, 74]
        y_points = [start_price, target_price]
    elif current_idx >= 74:
        x_points = [0, 74]
        y_points = [start_price, current_price]

    pchip = PchipInterpolator(x_points, y_points)
    x_all = np.arange(75)
    predicted_curve = [round(float(p), 2) for p in pchip(x_all)]

    error_pct = round(abs((current_price - predicted_curve[current_idx]) / current_price) * 100, 2)

    return timeframes, actual_curve, predicted_curve, actual_volume, target_price, error_pct

def process_symbol(symbol):
    print(f"Processing {symbol}...")
    ticker = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
    clean_sym = symbol.replace(".NS", "")
    
    try:
        df = yf.download(ticker, period="1d", interval="5m", progress=False)
        if df.empty:
            print(f"No market data available for {symbol}")
            return None

        try:
            yf_obj = yf.Ticker(ticker)
            raw_news = yf_obj.news or []
            news_list = []
            for n in raw_news[:3]:
                title = n.get("title") or n.get("content", {}).get("title", "")
                link = n.get("link") or n.get("content", {}).get("canonicalUrl", {}).get("url", "")
                publisher = n.get("publisher") or n.get("content", {}).get("provider", {}).get("displayName", "")
                if title:
                    news_list.append({"title": title, "link": link, "publisher": publisher})
        except Exception:
            news_list = []

        df_calc, vol_ratio, has_spike = compute_session_vwap_and_spikes(df)
        pred = query_gemini_regime(clean_sym, df_calc, news_list, vol_ratio)

        timeframes, actual_curve, predicted_curve, actual_volume, final_pred, error_pct = generate_pchip_curves(
            df_calc, pred.get("target_pct", 0.0)
        )

        valid_actuals = [p for p in actual_curve if p is not None]
        latest_actual = valid_actuals[-1] if valid_actuals else 0.0

        out_payload = {
            "symbol": clean_sym,
            "prediction": {
                "bias": pred.get("bias", "NEUTRAL"),
                "target_pct": pred.get("target_pct", 0.0),
                "archetype": pred.get("archetype", "RANGE_BOUND"),
                "reasoning": pred.get("reasoning", ""),
                "source": pred.get("source", "fallback"),
                "model": pred.get("model", "none"),
                "llm_error": pred.get("llm_error"),
                "locked": (pred.get("source") == "llm"),
                "volume_ratio": vol_ratio,
                "has_volume_spike": has_spike,
                "news_feed": news_list
            },
            "curves": {
                "date": str(datetime.datetime.now(IST).date()),
                "generated_at": datetime.datetime.now(IST).strftime("%I:%M %p IST"),
                "symbol": clean_sym,
                "bias": pred.get("bias", "NEUTRAL"),
                "timeframes": timeframes,
                "actual_curve": actual_curve,
                "predicted_curve": predicted_curve,
                "actual_volume": actual_volume,
                "final_pred": final_pred,
                "error_pct": error_pct
            }
        }

        os.makedirs("public/data_store", exist_ok=True)
        with open(f"public/data_store/{clean_sym}_prediction.json", "w") as f:
            json.dump(out_payload, f, indent=2)

        return {
            "symbol": clean_sym,
            "bias": pred.get("bias", "NEUTRAL"),
            "final_pred": final_pred,
            "final_actual": latest_actual,
            "error_pct": error_pct,
            "verdict": "SUCCESS" if error_pct <= 1.0 else ("PARTIAL" if error_pct <= 2.5 else "FAILED")
        }

    except Exception as e:
        print(f"Error executing processing pipeline for {symbol}: {e}")
        return None

def update_eod_evaluation(eval_results):
    eval_results = [r for r in eval_results if r is not None]
    if not eval_results:
        return

    os.makedirs("public/data_store", exist_ok=True)
    with open("public/data_store/eod_evaluation.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print("Updated public/data_store/eod_evaluation.json successfully.")

if __name__ == "__main__":
    if not is_live_session():
        print("[SessionGuard] Execution halted. Non-trading session or market closed.")
        sys.exit(0)

    print("Live session confirmed. Starting thread pool executor...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_symbol, SYMBOLS))

    update_eod_evaluation(results)
    print("Archive run completed successfully.")