import json
import os
import logging
from datetime import datetime
import yfinance as yf

from muthoot_poc.data_sources.announcements import (
    fetch_corporate_announcements,
    deduplicate_announcements
)
from muthoot_poc.engine.event_classifier import classify_event_and_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WATCHLIST = [
    "MUTHOOTFIN", "RELIANCE", "TMPV", "INFY", "HDFCBANK",
    "ICICIBANK", "TCS", "SBIN", "BHARTIARTL", "LT", "SUNPHARMA"
]

TIMEFRAMES = ['09:15', '09:45', '10:30', '11:15', '12:00', '12:45', '13:30', '14:15', '15:00', '15:30']

def fetch_actual_prices(symbol: str) -> list:
    nse_symbol = f"{symbol}.NS"
    logger.info(f"Fetching live market prices for {nse_symbol}...")
    
    try:
        ticker = yf.Ticker(nse_symbol)
        df = ticker.history(period="1d", interval="30m")
        
        if df.empty:
            # Weekend / Pre-market fallback
            df = ticker.history(period="5d", interval="30m").tail(10)
            actuals = [round(float(p), 2) for p in df['Close'].tolist()]
            return actuals[:len(TIMEFRAMES)]

        # Live trading day: Extract available candles so far
        live_prices = [round(float(p), 2) for p in df['Close'].tolist()]
        
        # Pad unreached future timeframe slots with None (null in JSON)
        actual_prices = []
        for i in range(len(TIMEFRAMES)):
            if i < len(live_prices):
                actual_prices.append(live_prices[i])
            else:
                actual_prices.append(None)

        return actual_prices

    except Exception as e:
        logger.error(f"Failed to fetch market data for {symbol}: {e}")
        return [None] * len(TIMEFRAMES)

def generate_morning_prediction_curve(symbol: str, base_price: float, ai_sentiment: str, confidence_score: float) -> list:
    if ai_sentiment == "BULLISH":
        expected_return_pct = (confidence_score / 100.0) * 0.02
    elif ai_sentiment == "BEARISH":
        expected_return_pct = -(confidence_score / 100.0) * 0.02
    else:
        # Micro-drift for NEUTRAL sentiment
        expected_return_pct = 0.002

    target_price = base_price * (1 + expected_return_pct)
    
    predicted_prices = []
    total_steps = max(1, len(TIMEFRAMES) - 1)
    
    for i in range(len(TIMEFRAMES)):
        progress = i / total_steps
        # Subtle intraday curve trajectory
        pred_val = base_price + ((target_price - base_price) * (progress ** 0.8))
        predicted_prices.append(round(pred_val, 2))
        
    return predicted_prices

def run_archive_cycle():
    logger.info("Running Real Market Data & AI Prediction Cycle...")
    os.makedirs("public/data_store", exist_ok=True)
    
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%I:%M %p IST")

    eod_summary = []

    for symbol in WATCHLIST:
        announcements = fetch_corporate_announcements(symbol)
        clean_ann = deduplicate_announcements(announcements)
        prediction_res = classify_event_and_predict(symbol, clean_ann)
        
        ai_bias = prediction_res.get("sentiment", "NEUTRAL")
        confidence = prediction_res.get("confidence", 50.0)

        actual_prices = fetch_actual_prices(symbol)
        
        # Base price at 09:15
        valid_prices = [p for p in actual_prices if p is not None]
        base_price = valid_prices[0] if valid_prices else 1000.0
        final_actual = valid_prices[-1] if valid_prices else base_price

        predicted_prices = generate_morning_prediction_curve(
            symbol=symbol,
            base_price=base_price,
            ai_sentiment=ai_bias,
            confidence_score=confidence
        )
        final_pred = predicted_prices[-1]

        error_pct = round(abs((final_actual - final_pred) / final_actual) * 100, 2)
        direction_matched = (
            (final_actual >= base_price and ai_bias == "BULLISH") or
            (final_actual < base_price and ai_bias == "BEARISH") or
            (ai_bias == "NEUTRAL" and error_pct <= 1.5)
        )

        verdict = "SUCCESS" if direction_matched and error_pct <= 3.0 else ("PARTIAL" if direction_matched else "FAIL")

        curve_data = {
            "symbol": symbol,
            "date": today_str,
            "generated_at": time_str,
            "timeframes": TIMEFRAMES,
            "actual_curve": actual_prices,
            "predicted_curve": predicted_prices,
            "bias": ai_bias,
            "final_actual": final_actual,
            "final_pred": final_pred,
            "error_pct": error_pct,
            "verdict": verdict
        }

        with open(f"public/data_store/{symbol}_prediction.json", "w") as f:
            json.dump({"prediction": prediction_res, "curves": curve_data}, f, indent=2)

        eod_summary.append(curve_data)

    with open("public/data_store/eod_evaluation.json", "w") as f:
        json.dump(eod_summary, f, indent=2)

if __name__ == "__main__":
    run_archive_cycle()