"""Google-based searcher using Custom Search API (synchronous version)."""

import os
import time
from typing import List, Optional

import httpx

from earthquakes_parser.search.base_searcher import BaseSearcher
from earthquakes_parser.search.search_result import SearchResult


class GoogleSearcher(BaseSearcher):
    """Google-based searcher using synchronous HTTP requests."""

    def __init__(
        self,
        delay: float = 1.0,
        key: Optional[str] = None,
        endpoint: Optional[str] = None,
        cx: Optional[str] = None,
    ):
        """Initialize the GoogleSearcher.

        Parameters:
        - delay: Time in seconds to wait between requests (default: 1.0).
        - key: Google Search API key. If not provided, loaded from environment.
        - endpoint: API endpoint URL. If not provided, loaded from environment.
        - cx: Custom Search Engine ID. If not provided, loaded from environment.

        Raises:
        - ValueError: If any required parameter is missing and not found in environment.
        """
        self.delay = delay

        self.GOOGLE_SEARCH_API_KEY = key or os.getenv("GOOGLE_SEARCH_API_KEY")
        self.GOOGLE_SEARCH_ENDPOINT = endpoint or os.getenv("GOOGLE_SEARCH_ENDPOINT")
        self.CX = cx or os.getenv("CX")

        if not all([self.GOOGLE_SEARCH_API_KEY, self.GOOGLE_SEARCH_ENDPOINT, self.CX]):
            raise ValueError(
                "Missing required environment variables: "
                "GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENDPOINT, CX"
            )

    def search(
        self,
        query: str,
        max_results: int = 5,
        site_filter: Optional[str] = None,
        offset: int = 1,
    ) -> List[SearchResult]:
        """Perform a Google Custom Search with optional site filter and offset."""
        search_query = f"site:{site_filter} {query}" if site_filter else query
        results_returned = 0
        items: List[dict] = []

        with httpx.Client(timeout=10.0) as client:
            while results_returned < max_results:
                count = min(max_results - results_returned, 10)
                params = {
                    "q": search_query,
                    "key": self.GOOGLE_SEARCH_API_KEY,
                    "cx": self.CX,
                    "num": count,
                    "start": offset,
                }

                response = client.get(self.GOOGLE_SEARCH_ENDPOINT, params=params)
                if response.status_code == 200:
                    data = response.json()
                    batch = data.get("items", [])
                    if not batch:
                        break  # No more results
                    items.extend(batch)
                    results_returned += len(batch)
                    offset += len(batch)
                else:
                    error_msg = (
                        f"Google Search API error {response.status_code}: "
                        f"{response.text}"
                    )
                    raise RuntimeError(error_msg)

                time.sleep(self.delay)

        return [
            SearchResult(
                query=query,
                link=item.get("link", ""),
                title=item.get("title", "No title"),
            )
            for item in items
        ]
