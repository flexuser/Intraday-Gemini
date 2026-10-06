"""muthoot_poc/gemini_engine.py

Thread-safe, quota-resilient Gemini engine module for `archive_once_2.py`.
Reuses standard `google-genai` Client sessions and enforces inter-request delay pacing.
"""

import os
import time
import threading
from typing import Optional, Any, Dict
from google import genai
from google.genai import types


class QuotaSafeGeminiEngine:
    """Persistent Gemini API engine providing thread-safe request pacing,
    reusable client sessions, and structured schema generation.
    """

    def __init__(self, model_name: str = "gemini-3.8-flash", min_delay_seconds: float = 4.0):
        self.model_name = model_name
        self.min_delay_seconds = min_delay_seconds
        self._lock = threading.Lock()
        self._last_call_time = 0.0

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    def _pace(self) -> None:
        """Enforces thread-safe inter-request spacing to prevent rate limit (429) bursts."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self.min_delay_seconds:
                time.sleep(self.min_delay_seconds - elapsed)
            self._last_call_time = time.time()

    def generate(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None) -> str:
        """Executes a content generation request using the persistent client session.

        Args:
            prompt: Text input prompt for the model.
            response_schema: Optional dict schema for structured JSON output.

        Returns:
            str: Raw JSON text response from the Gemini model.
        """
        self._pace()

        config_kwargs = {}
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        return response.text or ""