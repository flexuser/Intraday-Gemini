import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

EVENT_CATEGORY_MAP: Dict[str, List[str]] = {
    "Shareholders Meeting / Governance": [
        "shareholders meeting", "annual general meeting", "extraordinary general meeting",
        "agm", "egm", "scrutinizer report", "voting results", "proceedings of agm"
    ],
    "Earnings & Financial Results": [
        "financial results", "quarterly results", "audited results", "un-audited results"
    ],
    "Fund Raising & Capital Structure": [
        "allotment of equity shares", "allotment of ncd", "commercial paper", "rights issue"
    ],
    "Management & Leadership Updates": [
        "appointment", "resignation", "re-appointment", "key managerial personnel", "cfo", "ceo"
    ]
}

def rule_based_classify(headline: str) -> str:
    text_clean = headline.lower()
    for category, keywords in EVENT_CATEGORY_MAP.items():
        if any(kw in text_clean for kw in keywords):
            return category
    return "General Corporate Update"

def classify_event_headline(headline: str) -> str:
    if not headline:
        return "Unclassified Update"

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            prompt = f"Categorize this Indian stock headline into exactly ONE category: Shareholders Meeting / Governance, Earnings & Financial Results, Fund Raising & Capital Structure, Management & Leadership Updates, or General Corporate Update. Headline: '{headline}'"
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=30)
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini API fallback to rule engine: {e}")

    return rule_based_classify(headline)
