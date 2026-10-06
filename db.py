"""db.py

Supabase persistence layer for forecast history and system health logging.
Defensively handles network timeouts, missing credentials, and offline fallback.
"""

import os
import logging
from typing import List, Dict, Any, Optional

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = Any  # type: ignore

logger = logging.getLogger("db")


def get_supabase() -> Optional[Client]:
    """Initializes and returns Supabase client.
    
    Returns None gracefully if credentials or package are missing,
    allowing offline/local file generation without crashing.
    """
    if not SUPABASE_AVAILABLE:
        logger.warning("[db] 'supabase' package is not installed.")
        return None

    url = os.environ.get("SUPABASE_URL")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
    )
    
    if not url or not key:
        logger.warning("[db] Missing SUPABASE_URL or SUPABASE_KEY. Operating in offline mode.")
        return None

    try:
        return create_client(url, key)
    except Exception as e:
        logger.error(f"[db] Failed to initialize Supabase client: {e}")
        return None


def upsert_forecast_record(
    session_date: str,
    symbol: str,
    prompt_version: str,
    opening_forecast: Dict[str, Any],
    baseline_targets: Dict[str, Any],
    closing_actuals: Optional[Dict[str, Any]] = None,
    score: Optional[Dict[str, Any]] = None,
) -> bool:
    """Upserts opening forecast and baseline targets into forecast_history."""
    supabase = get_supabase()
    if not supabase:
        return False

    payload = {
        "session_date": session_date,
        "symbol": symbol,
        "prompt_version": prompt_version,
        "opening_forecast": opening_forecast,
        "baseline_targets": baseline_targets,
    }
    if closing_actuals is not None:
        payload["closing_actuals"] = closing_actuals
    if score is not None:
        payload["score"] = score

    try:
        supabase.table("forecast_history").upsert(
            payload, on_conflict="session_date,symbol"
        ).execute()
        return True
    except Exception as e:
        logger.error(f"[db] Failed to upsert record for {symbol} ({session_date}): {e}")
        return False


def update_closing_score(
    session_date: str, symbol: str, closing_actuals: Dict[str, Any], score: Dict[str, Any]
) -> bool:
    """Updates closing actuals and evaluated score for a completed session."""
    supabase = get_supabase()
    if not supabase:
        return False

    payload = {
        "closing_actuals": closing_actuals,
        "score": score,
    }
    try:
        supabase.table("forecast_history").update(payload).eq(
            "session_date", session_date
        ).eq("symbol", symbol).execute()
        return True
    except Exception as e:
        logger.error(f"[db] Failed to update closing score for {symbol} ({session_date}): {e}")
        return False


def save_health_status(status: str, details: Dict[str, Any]) -> bool:
    """Upserts current system health status to row ID 1."""
    supabase = get_supabase()
    if not supabase:
        return False

    payload = {
        "id": 1,
        "status": status,
        "details": details,
    }
    try:
        supabase.table("health_status").upsert(payload, on_conflict="id").execute()
        return True
    except Exception as e:
        logger.error(f"[db] Failed to save health status: {e}")
        return False


def fetch_all_history() -> List[Dict[str, Any]]:
    """Fetches all forecast history ordered by date descending."""
    supabase = get_supabase()
    if not supabase:
        return []

    try:
        res = (
            supabase.table("forecast_history")
            .select("*")
            .order("session_date", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.error(f"[db] Failed to fetch forecast history: {e}")
        return []