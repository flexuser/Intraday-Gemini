import logging
from muthoot_poc.data_sources.announcements import (
    fetch_corporate_announcements,
    fetch_nse_bulk_block_deals,
    deduplicate_announcements
)
from muthoot_poc.engine.event_classifier import classify_event_headline
from muthoot_poc.infra.storage import StorageEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WATCHLIST = ["MUTHOOTFIN", "RELIANCE", "TATAMOTORS", "INFY", "HDFCBANK"]

def run_archive_cycle(symbol: str) -> None:
    logger.info(f"Starting archiving cycle for target: {symbol}")
    storage = StorageEngine()
    
    raw_announcements = fetch_corporate_announcements(symbol)
    clean_announcements = deduplicate_announcements(raw_announcements)
    
    for item in clean_announcements:
        headline = item.get("caption") or item.get("headline") or ""
        item["category"] = classify_event_headline(headline)
        
    bulk_block_deals = fetch_nse_bulk_block_deals(symbol)
    
    storage.save_announcements(symbol, clean_announcements)
    storage.save_bulk_block_deals(symbol, bulk_block_deals)

def main():
    for symbol in WATCHLIST:
        run_archive_cycle(symbol)

if __name__ == "__main__":
    main()
