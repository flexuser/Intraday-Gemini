import json
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class StorageEngine:
    def __init__(self, data_dir: str = "public/data_store"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def save_announcements(self, symbol: str, items: List[Dict[str, Any]]) -> None:
        path = os.path.join(self.data_dir, f"{symbol}_announcements.json")
        with open(path, "w") as f:
            json.dump(items, f, indent=2)

    def save_bulk_block_deals(self, symbol: str, items: List[Dict[str, Any]]) -> None:
        path = os.path.join(self.data_dir, f"{symbol}_bulk_block.json")
        with open(path, "w") as f:
            json.dump(items, f, indent=2)

    def save_stock_prediction(self, symbol: str, prediction: dict, spike: dict) -> None:
        path = os.path.join(self.data_dir, f"{symbol}_prediction.json")
        with open(path, "w") as f:
            json.dump({"prediction": prediction, "spike": spike}, f, indent=2)

    def save_morning_outlook(self, outlook_items: List[Dict[str, Any]]) -> None:
        path = os.path.join(self.data_dir, "morning_outlook.json")
        with open(path, "w") as f:
            json.dump(outlook_items, f, indent=2)