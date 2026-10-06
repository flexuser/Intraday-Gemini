"""muthoot_poc/telemetry.py

Audit, prompt logging, and prediction drift monitoring suite.
Persists raw inputs, LLM responses, quant outputs, and risk decisions
to public/data_store/audit_telemetry.json for trade reviews.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger("telemetry")


class AuditLogger:
    """Persists model interaction logs and trade decisions."""

    def __init__(self, log_filepath: str = "public/data_store/audit_telemetry.json"):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)

    def log_prediction_event(
        self,
        symbol: str,
        session_date: str,
        technical_snapshot: Dict[str, Any],
        quant_prediction: Optional[Dict[str, Any]],
        llm_raw_prompt: Optional[str],
        llm_raw_response: Optional[str],
        risk_plan: Dict[str, Any],
    ) -> bool:
        """Records raw prompt, model output, indicators, and risk decision."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_date": session_date,
            "symbol": symbol,
            "indicators": technical_snapshot,
            "quant_prediction": quant_prediction,
            "llm_prompt_snippet": (llm_raw_prompt[:300] + "...") if llm_raw_prompt else None,
            "llm_raw_response": llm_raw_response,
            "risk_plan": risk_plan,
        }

        try:
            records = []
            if os.path.exists(self.log_filepath):
                with open(self.log_filepath, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if not isinstance(records, list):
                    records = []

            # Append new record and retain latest 500 events
            records.append(entry)
            records = records[-500:]

            with open(self.log_filepath, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"[telemetry] Failed to write audit event for {symbol}: {e}")
            return False