import os
import sys
import json
import time
import datetime
import pytz
import yfinance as yf
from google import genai

IST = pytz.timezone("Asia/Kolkata")

def is_live_session(symbol="RELIANCE.NS") -> bool:
    """
    Checks whether current IST time is within trading hours (09:15 - 15:50 IST, Mon-Fri)
    and verifies that Yahoo Finance returns data timestamped for today.
    """
    now_ist = datetime.datetime.now(IST)
    
    # Mon = 0, Sun = 6
    if now_ist.weekday() >= 5:
        print(f"[SessionGuard] Today is {now_ist.strftime('%A')} (Weekend). Market closed.")
        return False
        
    market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now_ist.replace(hour=15, minute=50, second=0, microsecond=0)
    
    if not (market_start <= now_ist <= market_end):
        print(f"[SessionGuard] Current IST time ({now_ist.strftime('%H:%M')}) is outside live market hours (09:15 - 15:50).")
        return False

    try:
        df = yf.download(symbol, period="1d", interval="5m", progress=False)
        if df is None or df.empty:
            print(f"[SessionGuard] Could not fetch anchor ticker {symbol}.")
            return False
            
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index
        last_bar_date = idx.tz_convert("Asia/Kolkata")[-1].date()
        
        if last_bar_date != now_ist.date():
            print(f"[SessionGuard] Market holiday or stale data detected. Last bar date: {last_bar_date}, Today: {now_ist.date()}")
            return False
            
    except Exception as e:
        print(f"[SessionGuard] Error probing anchor symbol: {e}")
        return False

    return True

def query_gemini_regime(symbol, df_5m, news_items, retries=3):
    """
    Queries Gemini with 3 retries and backoff.
    If calls fail, defaults to NEUTRAL / 0.0% / RANGE_BOUND without inventing swings.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = "gemini-2.5-flash"
    
    if not api_key:
        return {
            "bias": "NEUTRAL",
            "target_pct": 0.0,
            "archetype": "RANGE_BOUND",
            "reasoning": "Missing GEMINI_API_KEY",
            "source": "fallback",
            "model": "none",
            "llm_error": "Missing GEMINI_API_KEY"
        }

    client = genai.Client(api_key=api_key)
    last_close = float(df_5m['Close'].iloc[-1])
    news_text = "\n".join([f"- {item['title']}" for item in news_items[:3]]) or "No headlines."
    
    prompt = f"""
Analyze intraday technicals for {symbol}:
Current Price: {last_close}
Recent Headlines:
{news_text}

Respond strictly in valid JSON format:
{{
  "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
  "target_pct": float (e.g. 0.5 for +0.5%, -0.3 for -0.3%, 0.0 for NEUTRAL),
  "archetype": "MOMENTUM_BREAKOUT" | "MEAN_REVERSION" | "RANGE_BOUND" | "BREAKDOWN",
  "reasoning": "brief 1-2 sentence explanation"
}}
"""

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            raw_text = res.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw_text)
            
            return {
                "bias": data.get("bias", "NEUTRAL"),
                "target_pct": float(data.get("target_pct", 0.0)),
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

    # Strictly neutral fallback when retries are exhausted
    return {
        "bias": "NEUTRAL",
        "target_pct": 0.0,
        "archetype": "RANGE_BOUND",
        "reasoning": "LLM unavailable. Falling back to neutral range.",
        "source": "fallback",
        "model": model_name,
        "llm_error": last_error
    }

def process_symbol(symbol):
    print(f"Processing {symbol}...")
    ticker = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
    clean_sym = symbol.replace(".NS", "")
    
    df = yf.download(ticker, period="1d", interval="5m", progress=False)
    if df.empty:
        print(f"No data for {symbol}")
        return

    # Extract news
    try:
        yf_obj = yf.Ticker(ticker)
        raw_news = yf_obj.news or []
        news_list = [{"title": n.get("title", ""), "link": n.get("link", ""), "publisher": n.get("publisher", "")} for n in raw_news[:3]]
    except Exception:
        news_list = []

    pred = query_gemini_regime(clean_sym, df, news_list)
    
    timeframes = [t.strftime("%H:%M") for t in df.index.tz_convert("Asia/Kolkata")]
    actual_prices = [round(float(p), 2) for p in df['Close']]
    
    out_data = {
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
            "news_feed": news_list
        },
        "curves": {
            "date": str(datetime.datetime.now(IST).date()),
            "generated_at": datetime.datetime.now(IST).strftime("%H:%M IST"),
            "bias": pred.get("bias", "NEUTRAL"),
            "timeframes": timeframes,
            "actual_curve": actual_prices,
            "predicted_curve": actual_prices,
            "final_pred": round(actual_prices[-1] * (1 + pred.get("target_pct", 0.0)/100), 2) if actual_prices else 0
        }
    }
    
    os.makedirs("public/data_store", exist_ok=True)
    with open(f"public/data_store/{clean_sym}_prediction.json", "w") as f:
        json.dump(out_data, f, indent=2)

if __name__ == "__main__":
    if not is_live_session():
        print("[SessionGuard] Execution stopped. Not a live session.")
        sys.exit(0)

    symbols = ["MUTHOOTFIN", "RELIANCE", "TMPV", "INFY", "HDFCBANK", "ICICIBANK", "TCS", "SBIN", "BHARTIARTL", "LT", "SUNPHARMA"]
    for sym in symbols:
        process_symbol(sym)
