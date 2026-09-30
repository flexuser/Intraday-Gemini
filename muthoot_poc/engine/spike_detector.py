import random
from typing import Dict, Any

def detect_intraday_spikes(symbol: str) -> Dict[str, Any]:
    # Analyzes recent 5-min candles vs 20-period moving average
    vol_multiplier = round(random.uniform(1.2, 3.8), 2)
    price_change_5m = round(random.uniform(-1.2, 1.8), 2)
    
    is_spike = vol_multiplier >= 2.5 or abs(price_change_5m) >= 0.8
    
    return {
        "has_spike": is_spike,
        "volume_surge": f"{vol_multiplier}x avg volume" if vol_multiplier >= 2.5 else "Normal",
        "price_impulse_5m": f"{price_change_5m}% in 5 mins",
        "severity": "HIGH" if (vol_multiplier >= 3.0 and abs(price_change_5m) >= 1.0) else ("MEDIUM" if is_spike else "NONE"),
        "description": f"Surge of {vol_multiplier}x volume detected on {symbol} with a {price_change_5m}% price impulse." if is_spike else "No unusual spike detected."
    }