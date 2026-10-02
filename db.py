import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client


def get_supabase() -> Client:
    """Initialize Supabase client using environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment variables."
        )
    return create_client(url, key)


def upsert_forecast_record(
    session_date: str,
    symbol: str,
    prompt_version: str,
    opening_forecast: Dict[str, Any],
    baseline_targets: Dict[str, Any],
    closing_actuals: Optional[Dict[str, Any]] = None,
    score: Optional[Dict[str, Any]] = None,
) -> None:
    """Upserts opening forecast and baseline targets into forecast_history."""
    supabase = get_supabase()
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

    supabase.table("forecast_history").upsert(
        payload, on_conflict="session_date,symbol"
    ).execute()


def update_closing_score(
    session_date: str, symbol: str, closing_actuals: Dict[str, Any], score: Dict[str, Any]
) -> None:
    """Updates closing actuals and evaluated score for a completed session."""
    supabase = get_supabase()
    payload = {
        "closing_actuals": closing_actuals,
        "score": score,
    }
    supabase.table("forecast_history").update(payload).eq(
        "session_date", session_date
    ).eq("symbol", symbol).execute()


def save_health_status(status: str, details: Dict[str, Any]) -> None:
    """Upserts current system health status to row ID 1."""
    supabase = get_supabase()
    payload = {
        "id": 1,
        "status": status,
        "details": details,
    }
    supabase.table("health_status").upsert(payload, on_conflict="id").execute()


def fetch_all_history() -> List[Dict[str, Any]]:
    """Fetches all forecast history ordered by date descending."""
    supabase = get_supabase()
    res = (
        supabase.table("forecast_history")
        .select("*")
        .order("session_date", desc=True)
        .execute()
    )
    return res.data or []