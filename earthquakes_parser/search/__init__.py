"""Search module for keyword-based web searches."""

from earthquakes_parser.search.base_searcher import BaseSearcher
from earthquakes_parser.search.ddg_searcher import DDGSearcher
from earthquakes_parser.search.google_searcher import GoogleSearcher
from earthquakes_parser.search.search_manager import SearchManager
from earthquakes_parser.search.search_result import SearchResult

__all__ = [
    "SearchResult",
    "BaseSearcher",
    "GoogleSearcher",
    "DDGSearcher",
    "SearchManager",
]
