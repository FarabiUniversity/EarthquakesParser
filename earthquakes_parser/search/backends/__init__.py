"""Search backends package containing concrete searcher implementations."""

from earthquakes_parser.search.backends.base_searcher import BaseSearcher
from earthquakes_parser.search.backends.ddg_searcher import DDGSearcher
from earthquakes_parser.search.backends.gemini_searcher import GeminiSearcher
from earthquakes_parser.search.backends.google_searcher import GoogleSearcher
from earthquakes_parser.search.backends.search_result import SearchResult

__all__ = [
    "SearchResult",
    "BaseSearcher",
    "GoogleSearcher",
    "DDGSearcher",
    "GeminiSearcher",
]
