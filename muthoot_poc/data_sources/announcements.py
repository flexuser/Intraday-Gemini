import re
import logging
from datetime import datetime
from typing import List, Dict, Any
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    boilerplate_patterns = [
        r"bse\s+limited\s+has\s+informed\s+the\s+exchange\s+regarding",
        r"nse\s+limited\s+has\s+informed\s+the\s+exchange\s+regarding",
        r"submission\s+of",
        r"intimation\s+under\s+regulation\s+\d+",
        r"reg\.\s*\d+\(?\d*\)?",
        r"\(?scrip\s*code:?\s*\d+\)?",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)
    return re.sub(r"[^a-z0-9\s]", "", text).strip()

def is_duplicate_filing(item1: Dict[str, Any], item2: Dict[str, Any], time_window_minutes: int = 45) -> bool:
    time_fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t1 = datetime.strptime(str(item1.get("timestamp", "")), time_fmt)
        t2 = datetime.strptime(str(item2.get("timestamp", "")), time_fmt)
    except ValueError:
        return False

    if abs((t1 - t2).total_seconds()) > (time_window_minutes * 60):
        return False

    text1 = normalize_text(item1.get("caption") or item1.get("headline") or "")
    text2 = normalize_text(item2.get("caption") or item2.get("headline") or "")

    if not text1 or not text2:
        return False

    return SequenceMatcher(None, text1, text2).ratio() >= 0.80

def deduplicate_announcements(announcements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_items = sorted(
        announcements,
        key=lambda x: str(x.get("timestamp", "")),
        reverse=True
    )
    unique_records: List[Dict[str, Any]] = []
    for item in sorted_items:
        duplicate_match = None
        for existing in unique_records:
            if is_duplicate_filing(item, existing):
                duplicate_match = existing
                break
        
        if duplicate_match:
            exchanges = duplicate_match.get("exchanges", [duplicate_match.get("source", "UNKNOWN")])
            new_source = item.get("source", "UNKNOWN")
            if new_source not in exchanges:
                exchanges.append(new_source)
            duplicate_match["exchanges"] = exchanges
        else:
            item["exchanges"] = [item.get("source", "UNKNOWN")]
            unique_records.append(item)
            
    return unique_records

def fetch_corporate_announcements(symbol: str) -> List[Dict[str, Any]]:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        {
            "caption": f"BSE Limited HAS informed the Exchange regarding Scrutinizers report of Annual General Meeting for {symbol}",
            "timestamp": now_str,
            "source": "BSE"
        },
        {
            "caption": f"Scrutinizer Report of Annual General Meeting held for {symbol}",
            "timestamp": now_str,
            "source": "NSE"
        }
    ]

def fetch_nse_bulk_block_deals(symbol: str) -> List[Dict[str, Any]]:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        {
            "symbol": symbol,
            "client_name": "CAPITAL GROUP FUNDS",
            "buy_sell": "BUY",
            "quantity": 250000,
            "price": 1825.50,
            "timestamp": now_str
        }
    ]
