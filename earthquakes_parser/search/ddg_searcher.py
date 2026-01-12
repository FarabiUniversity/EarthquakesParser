"""DuckDuckGo-based searcher implementation."""

import time
from typing import List, Optional

from ddgs import DDGS

from earthquakes_parser.search.base_searcher import BaseSearcher
from earthquakes_parser.search.search_result import SearchResult


class DDGSearcher(BaseSearcher):
    """DuckDuckGo searcher using ddgs library."""

    def __init__(self, delay: float = 1.0):
        """Initialize the DuckDuckGo searcher.

        Args:
            delay: Delay in seconds between search requests to avoid rate limiting.
        """
        self.ddgs = DDGS()
        self.delay = delay

    def search(
        self,
        query: str,
        max_results: int = 5,
        site_filter: Optional[str] = None,
        offset: int = 0,
    ) -> List[SearchResult]:
        """Perform DuckDuckGo search with optional site filter and offset support.

        Args:
            query: Search query string.
            max_results: Number of results to return.
            site_filter: Optional site filter (e.g., 'instagram.com').
            offset: Number of results to skip (used for pagination).

        Returns:
            List of SearchResult objects.
        """
        results = []
        search_query = f"site:{site_filter} {query}" if site_filter else query

        try:
            all_items = list(self.ddgs.text(search_query))
            filtered_items = []

            for item in all_items:
                link = item.get("href", "")
                title = item.get("title", "")

                if site_filter and site_filter not in link:
                    continue

                filtered_items.append(SearchResult(query=query, link=link, title=title))

            # Apply offset and limit
            results = filtered_items[offset : offset + max_results]

        except Exception as e:
            print(f"DDG search error for '{query}': {e}")

        time.sleep(self.delay)
        return results
