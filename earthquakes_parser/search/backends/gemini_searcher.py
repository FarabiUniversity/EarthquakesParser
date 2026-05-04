"""Gemini-based searcher using Google Generative AI with Search Grounding."""

import os
import time
from typing import List, Optional

from google import genai
from google.genai import types

from earthquakes_parser.search.backends.base_searcher import BaseSearcher
from earthquakes_parser.search.backends.search_result import SearchResult


class GeminiSearcher(BaseSearcher):
    """Gemini-based searcher using Google Search Grounding via Generative AI API."""

    def __init__(
        self,
        delay: float = 1.0,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
    ):
        """Initialize the GeminiSearcher.

        Parameters:
        - delay: Time in seconds to wait between requests (default: 1.0).
        - api_key: Gemini API key. If not provided, loaded from GEMINI_API_KEY env var.
        - model: Gemini model name to use (default: 'gemini-2.0-flash').

        Raises:
        - ValueError: If API key is missing and not found in environment.
        """
        self.delay = delay
        self.model_name = model

        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("Missing required environment variable: GEMINI_API_KEY")

        self._client = genai.Client(api_key=resolved_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        site_filter: Optional[str] = None,
        offset: int = 0,
    ) -> List[SearchResult]:
        """Perform a search using Gemini with Google Search Grounding.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            site_filter: Optional site filter (e.g., 'example.com').
            offset: Number of results to skip (used for pagination).

        Returns:
            List of SearchResult objects extracted from grounding metadata.
        """
        search_query = f"site:{site_filter} {query}" if site_filter else query

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=search_query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

        results: List[SearchResult] = []

        if response.candidates:
            candidate = response.candidates[0]
            grounding = getattr(candidate, "grounding_metadata", None)
            if grounding:
                chunks = getattr(grounding, "grounding_chunks", None) or []
                for chunk in chunks[offset : offset + max_results]:
                    web = getattr(chunk, "web", None)
                    if web:
                        results.append(
                            SearchResult(
                                query=query,
                                link=getattr(web, "uri", ""),
                                title=getattr(web, "title", ""),
                            )
                        )

        time.sleep(self.delay)
        return results
